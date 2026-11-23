#!/usr/bin/env python3
"""Flow-DPO LoRA trainer for the native Wan2.2-TI2V-5B DiT (cached inputs), single-node DDP.

Pairs (winner, loser) share the conditioning context (same training-pool first frame + caption) and, per step, the same
timestep and noise. Loss (Flow-DPO, as in flow_grpo/scripts/train_sd3_dpo.py):
    d_w = mse(v_theta(x_w,t), u_w) - mse(v_ref(x_w,t), u_w);  d_l likewise
    L = -log sigmoid(-0.5 * beta * (d_w - d_l))  [+ sft_lambda * mse(v_theta(x_w,t), u_w)]
The reference is the frozen base (LoRA adapters disabled), identical to the anchor reference of the E3 trainer.
First latent frame (the conditioning frame) is excluded from all terms. Re-uses the verified E3 trainer pieces.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_e3_lora as tr  # noqa: E402

require = tr.require


def load_pairs(pairs_path, cache_index, check_shard=None):
    idx = tr.load_cache_index(cache_index, 'dpo', check_shard=check_shard)
    by_id = {e['sample_id']: e for e in idx['samples']}
    pairs = [json.loads(l) for l in Path(pairs_path).read_text().splitlines() if l.strip()]
    require(pairs and all(p['winner'] in by_id and p['loser'] in by_id and p['winner'] != p['loser'] for p in pairs),
            'Every pair must reference two distinct cached samples')
    return [(by_id[p['winner']], by_id[p['loser']]) for p in pairs], idx


def dpo_losses(pipe, sw, sl, tid, noise, beta, sft_lambda, autocast):
    import torch
    out = {}
    for tag, s in (('w', sw), ('l', sl)):
        inputs = tr.cached_inputs(s, pipe.device, offload=False)
        timestep, latents, target = tr.native_state(pipe, inputs, tid, noise)
        base_inputs = dict(inputs, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)
        with torch.no_grad(), tr.adapters_disabled(pipe.dit), autocast():
            ref = tr.predict(pipe, base_inputs, latents.clone(), timestep)[:, :, 1:].float()
        with autocast():
            pred = tr.predict(pipe, inputs, latents, timestep)[:, :, 1:]
        tgt = target[:, :, 1:].float()
        out[tag] = (torch.nn.functional.mse_loss(pred.float(), tgt), torch.nn.functional.mse_loss(ref, tgt).detach())
    (mw, rw), (ml, rl) = out['w'], out['l']
    inside = -0.5 * beta * ((mw - rw) - (ml - rl))
    loss = -torch.nn.functional.logsigmoid(inside)
    if sft_lambda > 0:
        loss = loss + sft_lambda * mw
    return dict(total=loss, inside=inside.detach(), mw=mw.detach(), ml=ml.detach(), rw=rw, rl=rl,
                acc=float(inside.detach() > 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs', required=True); ap.add_argument('--cache-index', required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--models-dir', type=Path, required=True); ap.add_argument('--framework', type=Path, default=tr.FRAMEWORK_DEFAULT)
    ap.add_argument('--max-steps', type=int, required=True); ap.add_argument('--seed', type=int, default=20261123)
    ap.add_argument('--lr', type=float, default=2e-5); ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--weight-decay', type=float, default=0.0); ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=100.0); ap.add_argument('--sft-lambda', type=float, default=0.0)
    ap.add_argument('--max-gpu-hours', type=float, required=True)
    ap.add_argument('--multi-nodes', type=int, default=1, help='user-authorized N nodes x 8 ranks')
    a = ap.parse_args()
    import torch
    comm = tr.setup_distributed(multi_nodes=a.multi_nodes)
    started = time.monotonic()
    out = a.out_dir
    if comm.rank == 0:
        require(not out.exists(), 'New --out-dir required')
        (out / 'private').mkdir(parents=True); (out / 'export').mkdir()
    comm.barrier()
    pairs, idx = load_pairs(a.pairs, a.cache_index, check_shard=(comm.rank, comm.world))
    allocation = tr.budget(a.max_gpu_hours, comm.world, 60, 240, 90, multi_nodes=a.multi_nodes)
    shards = tr.verify_base_shards(a.models_dir, hash_content=(comm.rank == 0))
    pipe = tr.build_pipe(str(a.framework), shards, a.seed, device=comm.device)
    frozen_before = tr.ts.frozen_state_hash(pipe.dit) if comm.rank == 0 else None
    if a.multi_nodes > 1:
        tr.require_same_frozen_base_across_nodes(comm, pipe, frozen_before)
    params = tr.adapter_list(pipe.dit)
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    order = tr.DataOrder(a.seed, len(pairs))
    autocast = lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)  # noqa: E731
    log = open(out / 'progress.jsonl', 'a') if comm.rank == 0 else None
    estimate, stop, step = 90.0, None, 0
    for step in range(a.max_steps):
        if comm.any_flag(tr.should_stop(comm.max_duration(time.monotonic() - started), estimate, allocation)):
            stop = 'budget_exhausted_before_max_steps'; break
        t0 = time.perf_counter()
        index = order.index(step, comm.rank, comm.world)
        ew, el = pairs[index]
        sw, sl = tr.load_sample(ew), tr.load_sample(el)
        tid, noise = tr.draw_t_noise(a.seed, step, index, sw['x0'].shape)
        optimizer.zero_grad(set_to_none=True)
        L = dpo_losses(pipe, sw, sl, tid, noise, a.beta, a.sft_lambda, autocast)
        ok = bool(torch.isfinite(L['total']).all())
        if ok:
            L['total'].backward(); ok = tr.check_grads(pipe.dit)
        if comm.any_flag(not ok):
            raise FloatingPointError('Nonfinite DPO loss/grad at step %d' % (step + 1))
        comm.all_reduce_mean_([p.grad for p in params])
        lr = a.lr * tr.lr_factor(step + 1, a.warmup, a.max_steps)
        for g in optimizer.param_groups:
            g['lr'] = lr
        gn = float(torch.nn.utils.clip_grad_norm_(params, max_norm=a.clip, error_if_nonfinite=True))
        optimizer.step(); torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        estimate = max(estimate if step else 0, comm.max_duration(sec))
        rows = comm.gather([float(L['total']), float(L['inside']), L['acc'], float(L['mw'] - L['rw']), float(L['ml'] - L['rl']), tid])
        if log:
            log.write(json.dumps(dict(step=step + 1, lr=lr, grad_norm=gn, seconds=sec,
                                      loss=sum(r[0] for r in rows) / len(rows), implicit_acc=sum(r[2] for r in rows) / len(rows),
                                      inside=sum(r[1] for r in rows) / len(rows), dw=sum(r[3] for r in rows) / len(rows),
                                      dl=sum(r[4] for r in rows) / len(rows))) + '\n'); log.flush()
    completed = step + 1 if stop is None else step
    checks = [r[0] for r in comm.gather([tr.adapter_checksum(params)])]
    require(len(set(checks)) == 1, 'Adapters differ across DDP ranks')
    comm.barrier()
    if comm.rank == 0:
        state = tr.checkpoint_state(pipe.dit, optimizer, completed, dict(algo='flow_dpo', beta=a.beta, lr=a.lr, steps=completed), cuda=True)
        rec = tr.save_checkpoint(out / 'private' / tr.CKPT_NAME, state)
        export = tr.export_lora(out / 'export' / 'dpo_lora.safetensors', state['adapters'])
        frozen_after = tr.ts.frozen_state_hash(pipe.dit)
        require(frozen_after == frozen_before, 'Frozen base changed')
        (out / 'summary.json').write_text(json.dumps(dict(status='completed' if stop is None else 'stopped', stop_reason=stop,
            completed_steps=completed, pairs=len(pairs), cache_index_sha256=idx['index_sha256'], beta=a.beta, lr=a.lr,
            sft_lambda=a.sft_lambda, world=comm.world, checkpoint=rec, export=export, frozen_base_sha256=frozen_after,
            adapter_checksum_all_ranks=checks, elapsed_seconds=time.monotonic() - started), indent=1))
    comm.barrier()
    require(stop is None, 'Stopped before --max-steps: ' + str(stop))


if __name__ == '__main__':
    main()
