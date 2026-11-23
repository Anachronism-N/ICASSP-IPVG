#!/usr/bin/env python3
"""Protocol-009 grounded labels over a frozen window list (not semantic approval or training admission).

009 = allocation007 decoding (bounded facts JSON, free four-string caption JSON, deny vocabulary, NO
per-sentence word regex) + allocation008 entity-specific action facts prompt + one required
camera_motion enum + a fail-closed caption text-integrity gate. Original facts/caption validators are
unchanged. Single-GPU model replicas claim windows dynamically; a failed window is excluded (never
repaired) and the parent stops all replicas when the approved failure-rate rule trips. Model text of
failed windows is not stored. Also runs the same independent 8B AI review protocol as review007/008.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'eval/teacher_progress_campaign_20261104'
POOL = ROOT / 'training_metadata/e3_recaption_20261022/pool1856_dual_manifest.jsonl'
PYTHON = '/dockerdata/venv_qwen32b_20261103/bin/python3.12'
MODEL = '/opt/models/Qwen3-VL-32B-Instruct'
REVIEW_MODEL = '/dockerdata/wm_models_qwen8b_review_20261103'
DECODER_SITE = Path('/dockerdata/wm_json_decoder_CPU_20261104/site')
LEASE_DIR = Path('/dockerdata/wm_gpu_exclusivity')
PROTOCOL = 'label009b_camera_actor_free_caption_v1'
CAPTION_WORDS = (40, 150)
FACTS_LIMIT, CAPTION_LIMIT = 1024, 384
LENGTH_REASON = 'Generation reached token limit without EOS; window excluded'
DECISIONS = CAMPAIGN / 'training_route_resume_20261121.authorization.json'
STOP_FAILURE_RATE = 0.10
STOP_MIN_WINDOWS = 128
TAIL_SECONDS = 90
MAX_ALPHA_RUN = 18
SOURCES = ('scripts/run_teacher_label_scale.py', 'tests/test_teacher_label_scale.py',
           'scripts/run_teacher_temporal_labels.py', 'scripts/teacher_structured_decoder.py',
           'scripts/run_teacher_qwen3vl32b_recovery.py', 'scripts/teacher_gpu_exclusive.py',
           'scripts/run_teacher_label_semantic_review.py')

CAMERA_PROMPT = """
Also include the key camera_motion with exactly one of: static, pan_left, pan_right, tilt_up,
tilt_down, zoom_in_or_push_in, zoom_out_or_pull_back, tracking_or_orbit, handheld_shake,
mixed_or_other. Judge camera_motion only from consistent shifting or scaling of the whole
background and stationary objects across early, middle and late images, independently of any
entity's own motion. An entity moving across a fixed background means static. When the whole
view gradually enlarges use zoom_in_or_push_in; when more surroundings are gradually revealed use
zoom_out_or_pull_back. Never describe camera movement as an entity's action, and never report an
entity as entering or leaving when only the camera reveals or hides it.
"""
CAMERA_CAPTION = """If camera_motion is not static, mention that camera movement once in plain words, such as the
camera slowly zooms in or pans left; if it is static, do not claim any camera movement. Do not
describe an entity as entering or leaving when only the camera reveals or hides it.
"""


class IntegrityError(ValueError):
    pass


INTEGRITY_REASONS = ('Caption contains a non-ASCII letter', 'Caption contains an implausibly long alphabetic run',
                     'Caption contains a hyphen chain', 'Camera motion enum required', 'Caption must contain 40..150 words')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def modules():
    sys.path.insert(0, str(ROOT / 'scripts'))
    if str(DECODER_SITE) not in sys.path:
        sys.path.insert(0, str(DECODER_SITE))
    import run_teacher_qwen3vl32b_recovery as r
    import run_teacher_temporal_labels as tl
    import teacher_gpu_exclusive as g
    return r, tl, g


def configure_decoding():
    """Set the shared decoding flags; only in the worker process (module globals, so never at import/test time)."""
    _, tl, _ = modules()
    tl.SCHEMA_DECODE = tl.BOUNDARY_DECODE = tl.BOUNDED_DECODE = True
    tl.WORD_ACTOR_DECODE = False


def facts_prompt():
    r, tl, _ = modules()
    return r.FACTS_PROMPT_V2 + tl.ACTOR_PROMPT + CAMERA_PROMPT


def caption_prompt():
    _, tl, _ = modules()
    marker = 'Grounded facts:\n'
    require(tl.CAPTION_PROMPT.endswith(marker), 'Original 007 caption prompt layout changed')
    return tl.CAPTION_PROMPT[:-len(marker)] + CAMERA_CAPTION + marker


def text_integrity(caption):
    """Fail-closed orthography gate added after the 008 glued-word defect; never edits text."""
    for ch in caption:
        if ord(ch) > 127 and unicodedata.category(ch).startswith('L'):
            raise IntegrityError(INTEGRITY_REASONS[0])
    if any(len(run) >= MAX_ALPHA_RUN for run in re.findall(r'[A-Za-z]+', caption)):
        raise IntegrityError(INTEGRITY_REASONS[1])
    if re.search(r'[A-Za-z]+(?:-[A-Za-z]+){3,}', caption):
        raise IntegrityError(INTEGRITY_REASONS[2])
    return caption


def validate_camera_facts(value):
    r, _, _ = modules()
    import teacher_structured_decoder as sd
    require(type(value) is dict, 'Exact typed facts root schema required')
    rest = dict(value)
    camera = rest.pop('camera_motion', None)
    if camera not in sd.CAMERA_MOTION_VALUES:
        raise IntegrityError(INTEGRITY_REASONS[3])
    r.validate_facts(rest)
    return value


def parse_camera_facts(text):
    r, _, _ = modules()
    prep, _, _, rc = r.helpers()
    r.require(type(text) is str and len(text) <= 24000 and not prep.UNSAFE.search(text), 'Sensitive/malformed facts output withheld')
    return validate_camera_facts(rc.extract_json(text))


def validate_caption_009b(text):
    """Original validate_caption checks with the approved 40..150 word range (2026-11-21); never edits text."""
    r, _, _ = modules()
    r.safe_text(text)
    match = r.FRAME_META.search(text)
    if match:
        raise r.DenyMatch('Single grounded paragraph without frame/sample metadata required', match)
    r.require('\n' not in text.strip(), 'Single grounded paragraph without frame/sample metadata required')
    count = len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", text))
    if not CAPTION_WORDS[0] <= count <= CAPTION_WORDS[1]:
        raise IntegrityError(INTEGRITY_REASONS[4])
    return text.strip()


def parse_caption(text):
    value = json.loads(text)
    require(type(value) is list and len(value) == 4 and all(type(s) is str and s == s.strip() and s and '\n' not in s
                                                           for s in value), 'Exact four-string caption JSON required')
    return text_integrity(validate_caption_009b(' '.join(value)))


def config():
    r, tl, _ = modules()
    import teacher_structured_decoder as sd
    return dict(protocol=PROTOCOL, model=MODEL, dtype='bfloat16', replica_cards=1, do_sample=False,
        source_frames=16, source_fps=24, source_indices=list(range(0, 121, 8)), resize=[640, 352],
        do_sample_frames=False, temporal_patch_size=2, facts_max_tokens=1024, caption_max_tokens=384,
        caption_words=list(CAPTION_WORDS), token_limit_without_EOS='window_excluded', facts_prompt_sha256=hashlib.sha256(facts_prompt().encode()).hexdigest(),
        caption_prompt_sha256=hashlib.sha256(caption_prompt().encode()).hexdigest(),
        caption_render='exact_json_array4_join_single_spaces_no_edit',
        decoding='JSON_schema_and_original_deny_vocabulary', facts_schema=sd.bounded_camera_facts_schema(compact=True),
        caption_schema=sd.CAPTION_SCHEMA, caption_word_count_grammar=False, entity_specific_action_prompt=True,
        camera_motion_values=list(sd.CAMERA_MOTION_VALUES), text_integrity=dict(max_alpha_run=MAX_ALPHA_RUN,
        non_ascii_letters=False, hyphen_chain_max=2), denied_facts_words=list(r.DENY_DECODE_WORDS),
        denied_caption_words=list(r.DENY_DECODE_WORDS + r.CAPTION_DECODE_WORDS),
        decoder_boundary_tokens='character_validated_quote_crossing_trie_v1',
        stop_on_generation_without_EOS_or_AssertionError=True, failed_windows='excluded_fail_closed_never_repaired',
        stop_failure_rate=STOP_FAILURE_RATE, stop_min_windows=STOP_MIN_WINDOWS, formal_training_allowed=False)


# ----------------------------------------------------------------------------- windows

def pool_rows():
    rows = [json.loads(line) for line in POOL.read_text().splitlines()]
    require(len(rows) == 1856 and len({r['sample_id'] for r in rows}) == 1856, 'Frozen 1856 pool required')
    return rows


def dataset_of(video):
    m = re.search(r'/wan22_5s/([^/]+)_5s/', video)
    require(m is not None, 'Unparseable dataset path')
    return m.group(1)


def select_pilot(seed, exclude_videos, count=32):
    rows = [r for r in pool_rows() if r['video'] not in set(exclude_videos)]
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(rows)), count))
    return [dict(sample_id=rows[i]['sample_id'], video=rows[i]['video'], dataset=dataset_of(rows[i]['video'])) for i in chosen]


def scale_windows(exclude_sample_ids):
    return [dict(sample_id=r['sample_id'], video=r['video'], dataset=dataset_of(r['video']))
            for r in pool_rows() if r['sample_id'] not in set(exclude_sample_ids)]


def sha_record(path):
    h = hashlib.sha256(); n = 0
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block); n += len(block)
    return dict(path=str(path), bytes=n, sha256=h.hexdigest())


def bind_videos(windows, workers=16):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(lambda w: sha_record(w['video']), windows))
    return [dict(index=i, sample_id=w['sample_id'], dataset=w['dataset'], video=rec) for i, (w, rec) in enumerate(zip(windows, recs))]


# ----------------------------------------------------------------------------- stop rule

def stop_decision(rows, total, rate=STOP_FAILURE_RATE, minimum=STOP_MIN_WINDOWS):
    """rows: completed window outputs. Returns a stop reason or None (approved 2026-11-21 rule)."""
    if any(r.get('decoder_stop') for r in rows):
        return 'decoder_admission_failed_stop_without_retry'
    done = len(rows)
    failures = sum(1 for r in rows if not r['success'])
    if done >= min(minimum, total) and failures > rate * done:
        return 'failure_rate_exceeded'
    return None


def pilot_pass(rows, total, rate=STOP_FAILURE_RATE):
    failures = sum(1 for r in rows if not r['success'])
    return len(rows) == total and failures <= rate * total


# ----------------------------------------------------------------------------- plan

def save_new(path, value):
    path = Path(path)
    with open(path, 'x') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')
    path.chmod(0o600)
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def load_bound(row):
    data = Path(row['path']).read_bytes()
    require(hashlib.sha256(data).hexdigest() == row['sha256'] and len(data) == row['bytes'], 'Bound file drift')
    return json.loads(data)


def prepare(a):
    r, tl, g = modules()
    stage = Path(a.stage_dir)
    require(stage.is_absolute() and stage.parent == CAMPAIGN, 'Stage dir must be a new campaign subdirectory')
    stage.mkdir(mode=0o700)
    windows = json.loads(Path(a.windows).read_text())
    require(windows and all(set(w) >= {'sample_id', 'video', 'dataset'} for w in windows), 'Window list required')
    if a.kind == 'label':
        jobs = bind_videos(windows)
    else:
        jobs = windows
        for j in jobs:
            require(sha_record(j['label']['path']) == j['label'] and sha_record(j['video']['path']) == j['video'], 'Review binding drift')
    model_root = MODEL if a.kind == 'label' else REVIEW_MODEL
    uuids = {int(v['index']): u for u, v in g.query_modes().items()}
    gpus = [int(x) for x in a.gpus.split(',')]
    require(len(gpus) == len(set(gpus)) and all(0 <= x < 8 for x in gpus), 'Distinct local GPU indices required')
    decisions = sha_record(DECISIONS)
    budget = json.loads((CAMPAIGN / 'budget.live.json').read_text())
    prior = budget['current_label_reserved_upper_GPUh']
    upper = len(gpus) * a.cap_seconds / 3600.0
    require(prior + upper <= budget['stage_caps_GPUh']['label_and_cache'], 'Label/cache stage cap exceeded')
    model_files = [dict(sha_record(p), mtime_ns=p.stat().st_mtime_ns)
                   for p in sorted(Path(model_root).iterdir()) if p.is_file() and not p.name.startswith('.')]
    plan = dict(schema='teacher_label_scale_plan_v1', kind=a.kind, protocol=PROTOCOL, mode=a.mode,
        hostname=os.uname().nodename, gpu_indices=gpus, gpu_uuids=[uuids[i] for i in gpus],
        cap_seconds=a.cap_seconds, upper_GPUh=upper, prior_label_reserved_upper_GPUh=prior,
        stage_cap_GPUh=budget['stage_caps_GPUh']['label_and_cache'], decisions=decisions,
        jobs=jobs, model_root=model_root, model_files=model_files, runtime=r.runtime(),
        source={n: sha_record(ROOT / n) for n in SOURCES}, config=config() if a.kind == 'label' else review_config(),
        resource_control_runtime=g.control_runtime(), output_root=str(stage / 'run01'),
        automatic_retry_allowed=False, semantic_approval=False, formal_training_allowed=False)
    if a.kind == 'review':
        plan['reviewed_label_plan'] = sha_record(a.label_plan)
    if a.kind == 'label':
        plan['decoder_runtime_files'] = [sha_record(f) for f in sorted(DECODER_SITE.rglob('*'))
                                         if f.is_file() and '__pycache__' not in f.parts and not f.name.endswith('.pyc')]
    snap = stage / 'executed_sources'
    for n in SOURCES:
        dst = snap / n; dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / n).read_bytes()); dst.chmod(0o600)
        require(sha_record(dst)['sha256'] == plan['source'][n]['sha256'], 'Source snapshot drift')
    row = save_new(stage / 'plan.json', plan)
    print(json.dumps(dict(plan=row, jobs=len(jobs), upper_GPUh=upper, prior=prior)))
    return row


def verify(path, digest):
    r, _, g = modules()
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'Plan SHA mismatch')
    p = json.loads(raw)
    require(p['hostname'] == os.uname().nodename and p['protocol'] == PROTOCOL, 'Host/protocol drift')
    require(p['config'] == (config() if p['kind'] == 'label' else review_config()), 'Config drift')
    require(p['runtime'] == r.runtime() and p['resource_control_runtime'] == g.control_runtime(), 'Runtime drift')
    for n, row in p['source'].items():
        require(sha_record(ROOT / n) == row, 'Source drift: ' + n)
    require(sha_record(DECISIONS) == p['decisions'], 'Decision record drift')
    require(p['automatic_retry_allowed'] is False and p['semantic_approval'] is False and p['formal_training_allowed'] is False,
            'Plan may not grant approvals')
    return p


# ----------------------------------------------------------------------------- worker

def label_one(model, processor, job, tools, end):
    r, tl, _ = modules()
    began = time.monotonic(); stage = 'frame_decode'; trace = {}; counts = {}
    base = dict(index=job['index'], sample_id=job['sample_id'], dataset=job['dataset'], video=job['video'], protocol=PROTOCOL)
    try:
        r.check(job['video'], end)
        images, frames = r.sample_images(job['video']['path'], tools, end)
        require(frames['source_indices'] == list(range(0, 121, 8)), 'Ordered real source-frame stage identity differs')
        stage = 'facts_generate'; trace = {}
        raw = tl.generate(model, processor, images, facts_prompt(), FACTS_LIMIT, trace)
        counts['facts'] = trace['new_tokens']; stage = 'facts_parse'
        facts = parse_camera_facts(raw)
        stage = 'caption_generate'; trace = {}
        raw = tl.generate(model, processor, images, caption_prompt() + json.dumps(facts, sort_keys=True), CAPTION_LIMIT, trace)
        counts['caption'] = trace['new_tokens']; stage = 'caption_validate'
        caption = parse_caption(raw)
        return dict(base, facts_json=facts, caption=caption, success=True,
                    duration_seconds=time.monotonic() - began, new_tokens=counts)
    except Exception as exc:
        if trace.get('new_tokens') is not None and stage.endswith('generate'):
            counts['facts' if stage == 'facts_generate' else 'caption'] = trace['new_tokens']
        value = dict(base, success=False, error_type=type(exc).__name__, failure_stage=stage,
                     duration_seconds=time.monotonic() - began, new_tokens=counts)
        value.update(r.case_failure(exc, stage, time.monotonic() - began, counts))
        if isinstance(exc, IntegrityError):
            value['failure_reason'] = str(exc)
        value['decoder_diagnostic'] = dict(trace)
        limit = FACTS_LIMIT if stage == 'facts_generate' else CAPTION_LIMIT
        length_only = (isinstance(exc, AssertionError) and trace.get('ended_with_EOS') is False
                       and trace.get('new_tokens') == limit and stage.endswith('generate'))
        if length_only:
            value['failure_reason'] = LENGTH_REASON
        value['decoder_stop'] = not length_only and (trace.get('ended_with_EOS') is False or isinstance(exc, AssertionError))
        return value


def claim(run_dir, index):
    try:
        with open(Path(run_dir) / 'claims' / f'{index}.claim', 'x') as f:
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        return False


def worker(a):
    r, tl, _ = modules()
    p = verify(a.plan, a.sha)
    run_dir = Path(p['output_root']); launch = json.loads((run_dir / 'launch.json').read_text())
    require(os.getppid() == launch['parent_pid'], 'Owned parent required')
    require(os.environ['CUDA_VISIBLE_DEVICES'] == p['gpu_uuids'][a.slot], 'Exact replica GPU required')
    require(os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1', 'Offline loading required')
    end = time.monotonic() + (launch['started_wall'] + p['cap_seconds'] - TAIL_SECONDS - time.time())
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    require(torch.cuda.device_count() == 1, 'One replica GPU required')
    started = time.monotonic()
    if p['kind'] == 'review':
        return review_worker(p, run_dir, a.slot, end)
    from teacher_structured_decoder import StructuredDecoder
    configure_decoding()
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    eos = json.loads(Path(MODEL, 'generation_config.json').read_bytes())['eos_token_id']
    tl.STRUCTURED_INSTANCE = StructuredDecoder(processor.tokenizer, eos, bounded=True, camera=True, compact=True)
    admission = tl.STRUCTURED_INSTANCE.cuda_admission('cuda:0')
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={'': 0}, attn_implementation='sdpa')
    require(all(v.device.type == 'cuda' and v.device.index == 0 and v.dtype == torch.bfloat16 for v in model.parameters()),
            'Dense BF16 single-GPU replica required')
    load_seconds = time.monotonic() - started; worst = 0.; done = []
    stop = 'completed'
    for job in p['jobs']:
        if (run_dir / 'STOP').exists():
            stop = 'parent_stop'; break
        if time.monotonic() + max(worst, 60) + 30 > end:
            stop = 'projected_budget_insufficient'; break
        out = run_dir / 'labels' / f"{job['index']}.json"
        if out.exists() or not claim(run_dir, job['index']):
            continue
        value = label_one(model, processor, job, p['runtime']['tools'], end)
        value['replica'] = a.slot
        save_new(out, value); done.append(job['index'])
        worst = max(worst, value['duration_seconds'])
        print(json.dumps(dict(window=job['index'], success=value['success'],
                              stage=None if value['success'] else value['failure_stage'])), flush=True)
        if value.get('decoder_stop'):
            stop = 'decoder_stop'; break
    peak = torch.cuda.max_memory_allocated(0) / 2**30
    del model
    save_new(run_dir / f'worker{a.slot}.json', dict(slot=a.slot, windows=done, load_seconds=load_seconds,
        stop_reason=stop, decoder_admission=admission, peak_allocated_GiB=peak, formal_training_allowed=False))


# ----------------------------------------------------------------------------- AI review (same protocol as 007/008)

def review_config():
    sys.path.insert(0, str(ROOT / 'scripts'))
    import run_teacher_label_semantic_review as sr
    return dict(protocol=PROTOCOL, reviewer=REVIEW_MODEL, fields=list(sr.FIELDS),
                prompt_sha256=hashlib.sha256(sr.PROMPT.encode()).hexdigest(), controls=sr.controls(),
                review_kind='independent_model_same_Qwen_family_AI_diagnostic_not_human_gold')


def review_worker(p, run_dir, slot, end):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    r, _, _ = modules()
    import run_teacher_label_semantic_review as sr
    model = Qwen3VLForConditionalGeneration.from_pretrained(REVIEW_MODEL, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
    processor = AutoProcessor.from_pretrained(REVIEW_MODEL, local_files_only=True)
    controls = []
    for c in p['config']['controls']:
        value = sr.validate_review(r.helpers()[3].extract_json(sr.judge(model, processor, sr.control_images(c['moving']), c['caption'])))
        controls.append(dict(control=c['id'], assessment=value, verdict=sr.verdict(value), expected=c['expected']))
    passed = all(c['verdict'] == c['expected'] for c in controls)
    save_new(run_dir / 'controls.json', dict(results=controls, passed=passed))
    require(passed, 'Reviewer controls failed; no real label scores trusted')
    for job in p['jobs']:
        require(time.monotonic() < end, 'Allocated work window exhausted')
        began = time.monotonic()
        try:
            label = load_bound(job['label'])
            r.check(job['video'], end)
            images, frames = r.sample_images(job['video']['path'], p['runtime']['tools'], end)
            value = sr.validate_review(r.helpers()[3].extract_json(sr.judge(model, processor, images, label['caption'])))
            item = dict(index=job['index'], sample_id=job['sample_id'], assessment=value, verdict=sr.verdict(value),
                        label=job['label'], frames=frames)
        except Exception as exc:
            item = dict(index=job['index'], sample_id=job['sample_id'], verdict='error', error_type=type(exc).__name__, label=job['label'])
        item['duration_seconds'] = time.monotonic() - began
        save_new(run_dir / 'labels' / f"review{job['index']}.json", item)
    save_new(run_dir / f'worker{slot}.json', dict(slot=slot, reviewed=len(p['jobs']), controls_passed=passed,
        review_kind='independent_model_same_Qwen_family_AI_diagnostic_not_human_gold'))


def review_windows(label_plan_path):
    p = json.loads(Path(label_plan_path).read_text())
    run_dir = Path(p['output_root'])
    terminal = json.loads((run_dir / 'terminal.json').read_text())
    require(terminal['terminal'] and terminal['released'] and terminal['modes_restored'], 'Released terminal label run required')
    jobs = []
    for job in p['jobs']:
        f = run_dir / 'labels' / f"{job['index']}.json"
        if not f.exists():
            continue
        value = json.loads(f.read_text())
        if value['success']:
            text_integrity(value['caption'])
            jobs.append(dict(index=len(jobs), source_index=job['index'], sample_id=job['sample_id'], dataset=job['dataset'],
                             video=job['video'], label=sha_record(f)))
    return jobs


# ----------------------------------------------------------------------------- parent

def run(a):
    _, _, g = modules()
    require(a.gpu_authorized and Path(sys.argv[0]).is_absolute(), 'Explicit GPU authorization and absolute runner path required')
    started = time.monotonic(); started_wall = time.time()
    p = verify(a.plan, a.sha)
    for m in p['model_files']:
        st = Path(m['path']).stat()
        require((st.st_size, st.st_mtime_ns) == (m['bytes'], m['mtime_ns']), 'Model file changed since SHA binding')
    budget = json.loads((CAMPAIGN / 'budget.live.json').read_text())
    require(budget['current_label_reserved_upper_GPUh'] == p['prior_label_reserved_upper_GPUh'], 'Budget ledger moved since plan')
    run_dir = Path(p['output_root']); run_dir.mkdir(mode=0o700)
    (run_dir / 'labels').mkdir(mode=0o700); (run_dir / 'claims').mkdir(mode=0o700)
    save_new(run_dir / 'attempt.claim', dict(plan_sha256=a.sha, started_wall=started_wall))
    modes = g.query_modes(); initial = {u: modes[u] for u in p['gpu_uuids']}
    require([initial[u]['index'] for u in p['gpu_uuids']] == p['gpu_indices'], 'GPU index/UUID drift')
    leases, children, signals, intrusions = [], [], [], []
    error = None; stop_reason = None
    def interrupted(*_):
        raise TimeoutError('Finite label cap or owned termination')
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(s, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(.01, p['cap_seconds'] - 60 - (time.monotonic() - started)))
    try:
        LEASE_DIR.mkdir(mode=0o700, exist_ok=True)
        for u in p['gpu_uuids']:
            f = LEASE_DIR / (u + '.json')
            with f.open('x') as out:
                json.dump(dict(gpu_uuid=u, expires_epoch=started_wall + p['cap_seconds'], owner_pid=os.getpid()), out)
            f.chmod(0o600); leases.append(f)
        for i in p['gpu_indices']:
            Path('/tmp/od_reserve_%d' % i).touch()  # hold.sh idle watchdog skips reserved GPUs (<1200 s)
        for u in p['gpu_uuids']:
            subprocess.run(['nvidia-smi', '-i', u, '-c', '3'], check=True, capture_output=True, timeout=10)
            for pid in g.apps().get(u, set()):
                v = g.signal_holder(pid, initial[u]['index'])
                require(v is not None, 'Unknown GPU owner; never signal')
                signals.append(v)
        require(all(not g.apps().get(u) for u in p['gpu_uuids']), 'GPUs not free after known-holder release')
        save_new(run_dir / 'launch.json', dict(parent_pid=os.getpid(), started_wall=started_wall, plan_sha256=a.sha))
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4')
        for slot, u in enumerate(p['gpu_uuids']):
            log = (run_dir / f'worker{slot}.log').open('xb')
            cmd = [PYTHON, str(Path(__file__).resolve()), 'worker', '--plan', a.plan, '--sha', a.sha, '--slot', str(slot)]
            children.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(env, CUDA_VISIBLE_DEVICES=u),
                                             start_new_session=True))
        total = len(p['jobs'])
        while any(c.poll() is None for c in children):
            time.sleep(5)
            for i in p['gpu_indices']:
                Path('/tmp/od_reserve_%d' % i).touch()
            mine = {c.pid for c in children}
            for u in p['gpu_uuids']:
                for pid in g.apps().get(u, set()) - mine:
                    intrusions.append(dict(gpu_uuid=u, pid=pid, observed=time.time()))
            if intrusions:
                stop_reason = 'external_gpu_process_observed'
                (run_dir / 'STOP').write_text(stop_reason); break
            if p['kind'] == 'label':
                rows = [json.loads(f.read_text()) for f in (run_dir / 'labels').glob('*.json')]
                reason = stop_decision(rows, total)
                if reason and not (run_dir / 'STOP').exists():
                    stop_reason = reason; (run_dir / 'STOP').write_text(reason)
        for c in children:
            c.wait(timeout=60)
        require(all(c.returncode == 0 for c in children), 'Replica worker failed; no retry')
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)[:200]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for c in children:
            if c.poll() is None:
                os.killpg(c.pid, signal.SIGTERM)
                try:
                    c.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(c.pid, signal.SIGKILL); c.wait(timeout=10)
        queries = [g.apps(), g.apps()]
        released = all(not q.get(u) for q in queries for u in p['gpu_uuids'])
        for u, v in initial.items():
            subprocess.run(['nvidia-smi', '-i', u, '-c', '0' if v['mode'] == 'Default' else '3'], check=True, capture_output=True, timeout=10)
        restored = all(g.query_modes()[u]['mode'] == v['mode'] for u, v in initial.items())
        for f in leases:
            if json.loads(f.read_text())['owner_pid'] == os.getpid():
                f.unlink()
        elapsed = time.monotonic() - started
        outs = [json.loads(f.read_text()) for f in sorted((run_dir / 'labels').glob('*.json'))] if (run_dir / 'labels').exists() else []
        summary = summarize(outs, p) if p['kind'] == 'label' else dict(reviewed=len(outs))
        terminal = dict(terminal=True, kind=p['kind'], protocol=PROTOCOL, error=error, stop_reason=stop_reason,
            released=released, modes_restored=restored, holder_signals=signals, intrusions=intrusions,
            elapsed_seconds=elapsed, actual_GPUh=elapsed * len(p['gpu_uuids']) / 3600.0,
            charged_upper_GPUh=p['upper_GPUh'], prior_label_reserved_upper_GPUh=p['prior_label_reserved_upper_GPUh'],
            cumulative_label_reserved_upper_GPUh=p['prior_label_reserved_upper_GPUh'] + p['upper_GPUh'],
            semantic_approval=False, formal_training_allowed=False, sota_proven=False, **summary)
        save_new(run_dir / 'terminal.json', terminal)
        print(json.dumps({k: terminal[k] for k in ('error', 'stop_reason', 'released', 'modes_restored', 'actual_GPUh')}))


def summarize(outs, p):
    by_ds = {}
    for o in outs:
        d = by_ds.setdefault(o['dataset'], dict(windows=0, labels=0, excluded=0))
        d['windows'] += 1; d['labels' if o['success'] else 'excluded'] += 1
    reasons = {}
    for o in outs:
        if not o['success']:
            reasons[o.get('failure_reason', o['error_type'])] = reasons.get(o.get('failure_reason', o['error_type']), 0) + 1
    camera = {}
    for o in outs:
        if o['success']:
            camera[o['facts_json']['camera_motion']] = camera.get(o['facts_json']['camera_motion'], 0) + 1
    total = len(p['jobs'])
    return dict(windows_total=total, windows_attempted=len(outs), labels=sum(o['success'] for o in outs),
                excluded=sum(not o['success'] for o in outs), by_dataset=by_ds, failure_reasons=reasons,
                camera_motion_counts=camera, pilot_failure_rule_pass=pilot_pass(outs, total) if p['mode'] == 'pilot' else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select-pilot'); s.add_argument('--seed', type=int, required=True); s.add_argument('--exclude-plan', required=True)
    s.add_argument('--out', required=True); s.add_argument('--count', type=int, default=32)
    s = sub.add_parser('prepare'); s.add_argument('--kind', choices=('label', 'review'), default='label')
    s.add_argument('--mode', choices=('pilot', 'scale'), required=True); s.add_argument('--windows', required=True)
    s.add_argument('--stage-dir', required=True); s.add_argument('--gpus', required=True)
    s.add_argument('--cap-seconds', type=float, required=True)
    s.add_argument('--label-plan')
    s = sub.add_parser('review-windows'); s.add_argument('--label-plan', required=True); s.add_argument('--out', required=True)
    for name in ('run', 'worker'):
        s = sub.add_parser(name); s.add_argument('--plan', required=True); s.add_argument('--sha', required=True)
        if name == 'run':
            s.add_argument('--gpu-authorized', action='store_true')
        else:
            s.add_argument('--slot', type=int, required=True)
    a = ap.parse_args()
    if a.cmd == 'select-pilot':
        old = json.loads(Path(a.exclude_plan).read_text())['jobs']
        windows = select_pilot(a.seed, [j['video']['path'] for j in old], a.count)
        Path(a.out).write_text(json.dumps(windows, indent=1)); Path(a.out).chmod(0o600)
        print(json.dumps(dict(count=len(windows), datasets={d: sum(w['dataset'] == d for w in windows) for d in sorted({w['dataset'] for w in windows})})))
    elif a.cmd == 'prepare':
        prepare(a)
    elif a.cmd == 'review-windows':
        jobs = review_windows(a.label_plan)
        Path(a.out).write_text(json.dumps(jobs, indent=1)); Path(a.out).chmod(0o600)
        print(json.dumps(dict(labels=len(jobs))))
    elif a.cmd == 'run':
        run(a)
    else:
        worker(a)


if __name__ == '__main__':
    main()
