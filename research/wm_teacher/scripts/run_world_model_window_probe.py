#!/usr/bin/env python3
"""One-shot synthetic-input Wan window/branch GPU admission; no training or quality claim."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/dockerdata/venv_qwen32b_20261103/bin/python3.12'
SITE = '/dockerdata/wm_a800_fa2_runtime_20261103/site'
CAP = 3600
PHASE = 'technical_admission'
MAX_CALLS = 3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def record(path):
    from world_model_harness import file_record
    return file_record(path)


def write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + chr(10)).encode()
    with Path(path).open('xb') as handle:
        os.chmod(path, 0o600)
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    return record(path)


def prepare(out):
    import teacher_gpu_exclusive as guard
    out = Path(out).absolute()
    require(out.resolve() == out and out.parent == ROOT/'eval/teacher_progress_campaign_20261104'
            and not out.exists(), 'New campaign allocation directory required')
    campaign = out.parent/'campaign.authorization.json'
    authorization = json.loads(campaign.read_bytes())
    budget = json.loads((out.parent/'budget.live.json').read_bytes())
    require(authorization['stage_caps_GPUh']['technical_admission'] == 8
            and budget['current_technical_reserved_upper_GPUh'] + 1 <= 8,
            'Unborrowed synthetic technical budget insufficient')
    base = json.loads(Path('/dockerdata/wm_models_teacher_base_20261103/base_content.proof.json').read_bytes())
    encoders = json.loads((out.parent/'encoder_originalpins.independent_audit.json').read_bytes())
    require(base['all_three_original_pins_matched'] and encoders['success']
            and encoders['all_six_original_pins_independently_rehashed'], 'Original model pins required')
    models = [dict(path=r['local_path'], bytes=r['bytes'], sha256=r['sha256']) for r in base['files']]
    models += [dict(path=r['path'], bytes=r['bytes'], sha256=r['sha256']) for r in encoders['files']]
    require(len(models) == 9, 'Three DiT shards and six encoder/tokenizer files required')
    for row in models:
        require(record(row['path']) == row, 'Model bytes differ before allocation')
    framework = ROOT/'DiffSynth-Studio'
    source_names = ('run_world_model_window_probe.py', 'interactive_world_model.py',
                    'world_model_harness.py', 'teacher_gpu_exclusive.py',
                    'run_teacher_generation.py', 'test_world_model_window_probe.py')
    sources = {}
    for name in source_names:
        path = ROOT/('tests' if name.startswith('test_') else 'scripts')/name
        sources[str(path.relative_to(ROOT))] = record(path)
    framework_files = [record(p) for p in sorted((framework/'diffsynth').rglob('*.py'))]
    require(framework_files, 'Frozen framework missing')
    prior_runtime = json.loads((out.parent/'a800_exact_resume003/plan.json').read_bytes())['runtime_files']
    for row in prior_runtime:
        require(record(row['path']) == row, 'Frozen FA2 runtime drift')
    modes = guard.query_modes()
    gpu_uuid = next(u for u, row in modes.items() if row['index'] == 4)
    out.mkdir(mode=0o700)
    plan = dict(schema='world_model_window_probe_plan_v1', phase=PHASE,
        campaign_authorization=record(campaign), budget_before=record(out.parent/'budget.live.json'),
        prior_technical_GPUh=budget['current_technical_reserved_upper_GPUh'], cap_seconds=CAP,
        allocated_cards=1, upper_GPUh=1, GPU_index=4, gpu_uuid=gpu_uuid,
        source=sources, framework=str(framework), framework_files=framework_files,
        models=models, runtime_files=prior_runtime, control_runtime=guard.control_runtime(),
        output=str(out), hostname=os.uname().nodename, max_calls=MAX_CALLS,
        synthetic_input=True, automatic_retry_allowed=False, model_updates=0,
        supervised_cache_admitted=False, training_started=False, quality_claim=False, sota_proven=False)
    return write_new(out/'plan.json', plan)


def verify(path, sha, *, assets=True):
    row = record(path); require(row['sha256'] == sha, 'Frozen plan SHA differs')
    p = json.loads(Path(path).read_bytes())
    require(p['schema'] == 'world_model_window_probe_plan_v1' and p['phase'] == PHASE
            and p['hostname'] == os.uname().nodename and p['cap_seconds'] == CAP
            and p['allocated_cards'] == 1 and p['upper_GPUh'] == 1 and p['max_calls'] == MAX_CALLS
            and p['synthetic_input'] is True and p['automatic_retry_allowed'] is False
            and not p['training_started'] and not p['quality_claim'], 'Synthetic one-shot scope differs')
    require(Path(p['output']) == Path(path).parent and Path(p['output']).parent == ROOT/'eval/teacher_progress_campaign_20261104', 'Output path drift')
    require(record(p['campaign_authorization']['path']) == p['campaign_authorization'], 'Campaign changed')
    require(json.loads(Path(p['campaign_authorization']['path']).read_bytes())['stage_caps_GPUh'][PHASE] == 8
            and p['prior_technical_GPUh'] + p['upper_GPUh'] <= 8, 'Phase cap exceeded')
    for row in p['source'].values(): require(record(row['path']) == row, 'Source drift')
    if assets:
        import teacher_gpu_exclusive as guard
        require(guard.control_runtime() == p['control_runtime'], 'Resource controller drift')
        require({str(f) for f in (Path(p['framework'])/'diffsynth').rglob('*.py')} == {r['path'] for r in p['framework_files']}, 'Framework file set drift')
        for row in [*p['models'], *p['framework_files'], *p['runtime_files']]:
            require(record(row['path']) == row, 'Asset/runtime bytes drift')
    return p


def publication(p, receipt_path):
    receipt = json.loads(Path(receipt_path).read_bytes())
    require(receipt['remote_equal'] is True and receipt['public_tests_passed'] > 0, 'Independent remote/test proof required')
    public = Path('/dockerdata/icaspp_ipvg_publication_20261018/research/wm_teacher')
    require(hashlib.sha256((public/'PUBLICATION_MANIFEST.json').read_bytes()).hexdigest() == receipt['manifest_sha256'], 'Public manifest drift')
    manifest = json.loads((public/'PUBLICATION_MANIFEST.json').read_bytes())
    for name, row in p['source'].items():
        require(next(r for r in manifest['files'] if r['path'] == name)['source_sha256'] == row['sha256'], 'Published source differs')
    head = subprocess.check_output(['git', '-C', str(public), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == receipt['commit'], 'Publication commit drift')


def worker(args):
    p = verify(args.plan, args.sha, assets=False)
    out = Path(p['output']); launch = json.loads((out/'launch.json').read_bytes())
    import run_teacher_generation as process
    require(process.same_process(launch['parent']) and os.getppid() == launch['parent']['pid'], 'Live bound parent required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == p['gpu_uuid']
            and os.environ.get('PYTHONPATH') == SITE
            and os.environ.get('HF_HUB_OFFLINE') == '1', 'Worker environment differs')
    import torch
    import flash_attn
    from PIL import Image, ImageDraw
    require(torch.cuda.device_count() == 1 and torch.cuda.get_device_capability(0) == (8, 0), 'A800 single card required')
    original = flash_attn.flash_attn_func
    attention_calls = []
    def attention(*a, **kw):
        kw['deterministic'] = True; attention_calls.append(1)
        return original(*a, **kw)
    flash_attn.flash_attn_func = attention
    sys.path.insert(0, p['framework'])
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.utils.data import save_video
    from interactive_world_model import InteractiveSession, TeacherWindowBackend
    from world_model_harness import validate_video
    def budget_check():
        require(time.monotonic() - launch['start_monotonic'] < CAP-120, 'Window probe work deadline reached')
    budget_check()
    files = [r['path'] for r in p['models']]
    pipeline = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[ModelConfig(path=files[:3], skip_download=True),
            ModelConfig(path=next(f for f in files if '/models_t5_' in f), skip_download=True),
            ModelConfig(path=next(f for f in files if '/Wan2.2_VAE.' in f), skip_download=True)],
        tokenizer_config=ModelConfig(path=str(Path(next(f for f in files if f.endswith('/spiece.model'))).parent), skip_download=True),
        redirect_common_files=False)
    require(pipeline.dit is not None and pipeline.vae is not None and pipeline.text_encoder is not None, 'Complete native pipeline required')
    specs = dict(width=1280, height=704, frames=121, fps=24)
    observed = Image.new('RGB', (1280, 704), 'white')
    ImageDraw.Draw(observed).ellipse((200, 250, 340, 390), fill='red')
    durations = []; references = []; outputs = []
    async def scenario():
        backend = TeacherWindowBackend(pipeline, 'blurry, distorted, flicker, text', budget_check, max_calls=MAX_CALLS)
        session = InteractiveSession(backend)
        session.push('observation', 0, dict(frame=observed, entities={}), evidence='synthetic_image_constructed_locally')
        async def generate(s, instruction, seed, name):
            ref = s.predicted_reference if s.predicted_reference is not None else s.observed_reference
            references.append(hashlib.sha256(ref.tobytes()).hexdigest())
            start = time.monotonic(); result = await s.rollout(instruction, seed=seed)
            require(result['status'] == 'predicted', 'Window was not committed')
            durations.append(time.monotonic()-start)
            path = out/(name+'.mp4'); require(not path.exists(), 'Never overwrite output')
            save_video(list(result['result'].frames), str(path), fps=24, quality=5)
            validate_video(path, specs); outputs.append(record(path))
            return hashlib.sha256(result['result'].frames[-1].tobytes()).hexdigest()
        last = await generate(session, 'A red disk slides gently to the right on a plain white surface.', 0, 'window0')
        left, right = session.fork('left'), session.fork('right')
        left.push('action', 1, dict(raw_action=None, semantic_action='The disk slides left.', expected_outcome={}), evidence='synthetic_request')
        right.push('action', 1, dict(raw_action=None, semantic_action='The disk slides right.', expected_outcome={}), evidence='synthetic_request')
        await generate(left, 'A red disk slides left on a plain white surface.', 0, 'branch_left')
        await generate(right, 'A red disk slides right on a plain white surface.', 1, 'branch_right')
        require(references[1] == references[2] == last, 'Branches must start from exactly the same first window endpoint')
        require(hashlib.sha256(session.predicted_reference.tobytes()).hexdigest() == last
                and session.observed_reference.tobytes() == observed.tobytes(), 'Branch or prediction overwrote parent/observation')
        require(backend.calls == MAX_CALLS and all(len(s.predictions) == 1 for s in (session, left, right)), 'Unexpected extra generation')
    try:
        asyncio.run(scenario()); torch.cuda.synchronize(); require(attention_calls, 'Native FA2 not observed')
        write_new(out/'worker.json', dict(success=True, native_pipeline=True, model_updates=0,
            windows=MAX_CALLS, frame_count_per_window=121, outputs=outputs,
            same_parent_endpoint_for_branches=True, observation_prediction_separation=True,
            durations_seconds=durations, reference_pixel_sha256=references, attention_calls=len(attention_calls),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), synthetic_input=True,
            physical_action_correctness_measured=False, model_quality_measured=False, sota_proven=False))
    finally:
        flash_attn.flash_attn_func = original


def run(args):
    require(args.gpu_authorized and args.publication, 'Separate explicit GPU execution and publication proof required')
    p = verify(args.plan, args.sha); publication(p, args.publication)
    import teacher_gpu_exclusive as guard
    import run_teacher_generation as process
    out = Path(p['output']); uuid = p['gpu_uuid']
    initial = guard.query_modes()[uuid]
    require(initial['index'] == 4 and initial['mode'] in ('Default', 'Exclusive_Process'), 'GPU allocation differs')
    lease = Path('/dockerdata/wm_gpu_exclusivity')/(uuid+'.json')
    require(not lease.exists(), 'Existing lease; never steal it')
    require(all(guard.holder_identity(pid, 4) is not None for pid in guard.apps().get(uuid, set())), 'Unknown GPU owner; no signals')
    budget = json.loads((out.parent/'budget.live.json').read_bytes())
    allocation = next((r for r in budget['allocations'] if r['id'] == out.name), None)
    require(allocation is not None and allocation['phase'] == PHASE
            and allocation.get('reserved_upper_GPUh') == 1 and not allocation['terminal']
            and budget['current_technical_reserved_upper_GPUh'] <= 8, 'Budget reservation absent or exceeded')
    start = time.monotonic(); write_new(out/'attempt.claim', dict(pid=os.getpid(), plan_sha256=args.sha, start_monotonic=start))
    child = None; created = False; error = None; signals = []
    def timeout(*_): raise TimeoutError('Synthetic technical cap reached')
    old = {s:signal.signal(s,timeout) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL, CAP-60)
    try:
        write_new(lease, dict(gpu_uuid=uuid, owner_pid=os.getpid(), expires_epoch=time.time()+CAP)); created=True
        subprocess.run(['nvidia-smi','-i',uuid,'-c','3'], check=True, capture_output=True, timeout=10)
        for pid in guard.apps().get(uuid,set()):
            identity=guard.signal_holder(pid,4); require(identity is not None,'Unknown owner; never signal'); signals.append(identity)
        require(not guard.apps().get(uuid), 'GPU owner remains')
        write_new(out/'launch.json', dict(parent=process.identity(os.getpid()), start_monotonic=start))
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONPATH=SITE,
                 HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1')
        with (out/'worker.log').open('xb') as log:
            child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'worker','--plan',args.plan,'--sha',args.sha],
                stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            child.wait(timeout=max(1,CAP-120-(time.monotonic()-start)))
        require(child.returncode==0 and json.loads((out/'worker.json').read_bytes())['success'], 'Worker failed; no automatic retry')
    except BaseException as exc:
        error=type(exc).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if child is not None and child.poll() is None:
            owner=process.identity(child.pid); process.cleanup_owned(child, owner, time.monotonic()+30)
        released=all(not guard.apps().get(uuid) for _ in range(2))
        subprocess.run(['nvidia-smi','-i',uuid,'-c','0' if initial['mode']=='Default' else '3'],check=True,capture_output=True,timeout=10)
        restored=guard.query_modes()[uuid]['mode']==initial['mode']
        if created:
            require(json.loads(lease.read_bytes())['owner_pid']==os.getpid(),'Lease owner changed'); lease.unlink()
        elapsed=time.monotonic()-start
        terminal=dict(success=error is None and released and restored and elapsed<=CAP,terminal=True,
            error_type=error, actual_GPUh=elapsed/3600, charged_upper_GPUh=max(1,elapsed/3600),
            released=released,modes_restored=restored,holder_signals=signals,model_updates=0,
            phase=PHASE,synthetic_input=True,training_started=False,model_quality_measured=False,sota_proven=False)
        write_new(out/'terminal.json',terminal)
        for s,h in old.items(): signal.signal(s,h)
    require(terminal['success'], 'Window probe failed; retained with charge, no retry')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run','worker'))
    parser.add_argument('--output');parser.add_argument('--plan');parser.add_argument('--sha')
    parser.add_argument('--publication');parser.add_argument('--gpu-authorized',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='prepare': print(json.dumps(prepare(args.output),sort_keys=True))
    elif args.command=='verify': verify(args.plan,args.sha); print('CPU verified; not GPU authorization')
    elif args.command=='worker': worker(args)
    else: run(args)


if __name__=='__main__':
    main()
