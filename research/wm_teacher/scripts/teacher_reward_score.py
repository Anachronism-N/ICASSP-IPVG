#!/usr/bin/env python3
"""Reward scoring of generated videos for DPO pair construction (training data only; never eval benchmarks).

Per video: (1) WMReward V-JEPA2 surprise with the official facebookresearch/WMReward code
(compute_vjepa_loss_sliding_window, ViT-g 256, causal masking, window 16, context 8, stride 8, 49 frames uniformly
sampled, mean over chunks; lower = more physically predictable); (2) VideoReward (KwaiVGI, Qwen2-VL-2B) VQ/MQ/TA with
its published normalisation; (3) a simple motion magnitude (mean abs difference of consecutive 64x36 gray frames).
One process per GPU; resumable per-video JSON outputs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

WMR = os.environ.get('WMREWARD_REPO', '/dockerdata/wm_eval_20261121/official/WMReward')
VALIGN = os.environ.get('VIDEOALIGN_REPO', '/dockerdata/wm_eval_20261121/official/VideoAlign')
VR_CKPT = os.environ.get('VIDEOREWARD_DIR', '/dockerdata/wm_models_videoreward_20261122')
QWEN2VL = os.environ.get('QWEN2VL2B_DIR', '/dockerdata/wm_models_qwen2vl2b_20261122')


def motion_magnitude(path):
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image
    r = imageio.get_reader(path)
    prev, diffs = None, []
    for fr in r:
        g = np.asarray(Image.fromarray(fr).convert('L').resize((64, 36), Image.BILINEAR), dtype=np.float32)
        if prev is not None:
            diffs.append(float(np.abs(g - prev).mean()))
        prev = g
    r.close()
    return sum(diffs) / max(1, len(diffs))


def load_vjepa():
    import copy
    import torch
    import importlib.util
    # official WMReward utils under a unique name (VideoAlign also ships a top-level 'utils')
    spec = importlib.util.spec_from_file_location('wmreward_utils', os.path.join(WMR, 'utils.py'))
    wmr = importlib.util.module_from_spec(spec); spec.loader.exec_module(wmr)  # adds WMReward/vjepa2 to sys.path
    enc, pred = torch.hub.load(os.path.join(WMR, 'vjepa2'), 'vjepa2_vit_giant', source='local')
    tgt = copy.deepcopy(enc)
    return wmr, enc.cuda().eval(), tgt.cuda().eval(), pred.cuda().eval()


def vjepa_surprise(wmr, enc, tgt, pred, path):
    import torch
    from torchvision.transforms.functional import resize
    v = torch.from_numpy(wmr.get_video(path, max_frames=49)).permute(3, 0, 1, 2).float()
    v = resize(v.permute(1, 0, 2, 3), [256, 256]).permute(1, 0, 2, 3)
    v = ((v / 127.5) - 1.0).unsqueeze(0).cuda()
    with torch.no_grad():
        loss = wmr.compute_vjepa_loss_sliding_window(video_tensor=v, encoder=enc, target_encoder=tgt, predictor=pred,
                                                     img_size=256, window_size=16, loss_exp=2, masking_mode='causal',
                                                     context_frames=8, is_vae_output=True, seed=42, stride=8, mode='mean')
    return float(loss)


def load_videoreward():
    import json as _j
    import torch
    sys.path.insert(0, VALIGN)
    import types
    import importlib.machinery
    import transformers  # noqa: F401  real package probing happens before any stub exists
    import transformers.trainer  # noqa: F401
    # Training-only imports of VideoAlign (trl/datasets) are not used for scoring: inert stubs, no numerics involved.
    if 'trl' not in sys.modules:
        trl = types.ModuleType('trl'); trl.__spec__ = importlib.machinery.ModuleSpec('trl', None)
        trl.get_kbit_device_map = lambda: None
        trl.get_quantization_config = lambda cfg: (None if not (cfg.load_in_8bit or cfg.load_in_4bit) else
                                                   (_ for _ in ()).throw(RuntimeError('quantization not supported')))
        trl.RewardTrainer = type('RewardTrainer', (object,), {})
        sys.modules['trl'] = trl
    if 'datasets' not in sys.modules:
        ds = types.ModuleType('datasets'); ds.__spec__ = importlib.machinery.ModuleSpec('datasets', None)
        ds.load_dataset = ds.concatenate_datasets = None
        ds.Dataset = type('Dataset', (object,), {})
        sys.modules['datasets'] = ds
    import inference as vr
    real_load = vr.load_configs_from_json
    def patched(path):
        data, _, model, peft, inf = real_load(path)
        model = dict(model, model_name_or_path=QWEN2VL)
        return data, _, model, peft, inf
    vr.load_configs_from_json = patched
    real_tc = vr.TrainingConfig
    vr.TrainingConfig = lambda **kw: real_tc(**dict(kw, disable_flash_attn2=True))
    return vr.VideoVLMRewardInference(VR_CKPT, device='cuda:0', dtype=torch.bfloat16)


def worker(a):
    import torch
    assert torch.cuda.device_count() == 1
    items = [json.loads(l) for l in Path(a.items).read_text().splitlines() if l.strip()][a.shard::a.nshards]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (out / (it['id'] + '.json')).exists()]
    if not todo:
        return
    wmr, enc, tgt, pred = load_vjepa()
    inf = load_videoreward()
    for it in todo:
        rec = dict(id=it['id'], video=it['video'])
        try:
            rec['vjepa_surprise'] = vjepa_surprise(wmr, enc, tgt, pred, it['video'])
            with torch.no_grad():
                r = inf.reward([it['video']], [it['prompt']], use_norm=True)[0]
            rec.update({'vr_' + k: float(v) for k, v in r.items()})
            rec['motion'] = motion_magnitude(it['video'])
            rec['ok'] = True
        except Exception as e:
            rec.update(ok=False, error=type(e).__name__ + ': ' + str(e)[:200])
        tmp = out / (it['id'] + '.json.tmp')
        tmp.write_text(json.dumps(rec)); os.replace(tmp, out / (it['id'] + '.json'))
        print(json.dumps(rec), flush=True)


def spawn(a):
    uuids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    off = int(os.environ.get('SPAWN_SHARD_OFFSET', '0')); tot = int(os.environ.get('SPAWN_SHARD_TOTAL', str(len(uuids))))
    procs = []
    for k, u in enumerate(uuids):
        log = open(Path(a.out_dir) / f'worker{off + k}.log', 'ab')
        procs.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--items', a.items,
                                       '--out-dir', a.out_dir, '--shard', str(off + k), '--nshards', str(tot)],
                                      env=dict(os.environ, CUDA_VISIBLE_DEVICES=u), stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    print(json.dumps(dict(returncodes=codes)), flush=True)
    raise SystemExit(0 if not any(codes) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=('worker', 'spawn'))
    ap.add_argument('--items', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    worker(a) if a.cmd == 'worker' else spawn(a)


if __name__ == '__main__':
    main()
