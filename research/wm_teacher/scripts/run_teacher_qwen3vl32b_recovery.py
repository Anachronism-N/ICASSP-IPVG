#!/usr/bin/env python3
"""One approved new-cluster Qwen3-VL-32B recovery of the fixed 32-case label pilot.

Same cases/frames/prompts/decoding as the first pilot; adds safe per-case stage codes,
per-case durations and an explicit stop reason. Structural coverage is not training approval.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path('/dockerdata/wm_tmp/teacher_qwen3vl32b_recovery_20261103_run01')
STATE_DIR = Path('/dockerdata/wm_logs/teacher_qwen3vl32b_recovery_20261103')
MODEL_ROOT = Path('/opt/models/Qwen3-VL-32B-Instruct')
OLD_MODEL_ROOT = '/dockerdata/ms_stage2_resources_20261028_v1/Qwen3-VL-32B-Instruct'
PYTHON = '/dockerdata/venv_qwen32b_20261103/bin/python3.12'
EVIDENCE = ROOT / 'eval/teacher_label_pilot_20261102'
RECOVERY_EVIDENCE = ROOT / 'eval/teacher_label_recovery_20261103'
AUTH = ROOT / 'eval/diag/teacher_training_admission_authorization_20261101.json'
AUTH_SHA = '245e14b3b7a4d3073c6914b7c9c8fba8907261aa85b6f6a3e9a59be7f400c2d9'
NEW_AUTH = EVIDENCE / 'new_captioner_pilot.authorization.json'
NEW_AUTH_SHA = '3d33077608b2837b1ead2eadb3e7ed5e50afc0850f341588da3c79255284d9fd'
SCOPE_AUTH = EVIDENCE / 'qwen3vl32b_pilot.authorization.json'
SCOPE_AUTH_SHA = '0e17eed8322202e3981666f4073c6ce31a0b75ed1af6c226f9d431fab1cd0567'
RECOVERY_AUTH = EVIDENCE / 'qwen3vl32b_one_recovery.authorization.json'
RECOVERY_AUTH_SHA = '61ead2c77621bdabdcaeedd006ab6ada252fbf541730322ae73a641f0a554d8a'
CLUSTER_AUTH = RECOVERY_EVIDENCE / 'new_cluster_recovery.authorization.json'
CLUSTER_AUTH_SHA = '767b4e1e861264987220b50d336f8af9606f4ef9a11101cc6293d8feb3e34083'
FIRST_PLAN = ROOT / 'eval/diag/teacher_qwen3vl32b_plan_source_final_20261102.json'
FIRST_PLAN_SHA = 'aa80636d65e828c86db9130cf54f1d122ef421649f56b22d0c78947220433d90'
OLD_PLAN = EVIDENCE / 'plan.lease_fix.json'
OLD_PLAN_SHA = '38236595a602b8e91946d76ddfe07d4cab79c569fd6c40c77714c390ab87d2ad'
MANIFEST = ROOT / 'training_metadata/e3_recaption_20261022/pool1856_dual_manifest.jsonl'
SCHEMA = 'teacher_qwen3vl32b_recovery_plan_v1'
PHASE = 'label_rebuild_and_cache'
PROMPT_VERSION = 'qwen3vl32b-grounded-entity-events-v1'
CAP, TAIL, CARDS, RANK, SEED = 5400, 120, 4, 0, 20261102
ALLOCATION_GPUH = 6
FIRST_PILOT_PRIOR_GPUH = 0.1658982108534676
# Earlier label GPUh plus the first 32B pilot's retained 2 GPUh upper bound; never refunded.
PRIOR_GPUH = 2.1658982108534675
CUMULATIVE_CAP_GPUH = 8.165898210853468
ORIGINAL_PRIOR_GPUH = PRIOR_GPUH
ORIGINAL_CUMULATIVE_CAP_GPUH = CUMULATIVE_CAP_GPUH
SUPPLEMENTAL = False
PROMPT_V2 = False
CONSTRAINED = False
CONSTRAINED_AUTH = ROOT / 'eval/teacher_label_constrained_decode_20261103/constrained.authorization.json'
CONSTRAINED_AUTH_SHA = 'a1f093dd1e9b3dbf65be26b300dca4788f5a7c39b24f175b6a29690ddc71198c'
PROMPT_V2_AUTH = ROOT / 'eval/teacher_label_prompt_v2_20261103/prompt_v2.authorization.json'
PROMPT_V2_AUTH_SHA = '5a70b129fb053dce7c3dde62542adcc064e47543b84ef844de88079c13d4111c'
SUPPLEMENTAL_AUTH = ROOT / 'eval/teacher_label_supplemental_20261103/supplemental.authorization.json'
SUPPLEMENTAL_AUTH_SHA = 'd3c7e6ddc2dfffd0f63f2bb8cfd9f614412b8b764dd56a4a144c81dfacc41fe5'
WEIGHT_BYTES = 66714912872
WEIGHTS = tuple(f'model-{i:05d}-of-00014.safetensors' for i in range(1, 15))
# The actual index/config/processor/tokenizer set is checked against the local directory.
REQUIRED_CONFIGS = {'README.md', 'chat_template.json', 'config.json', 'generation_config.json',
    'merges.txt', 'model.safetensors.index.json', 'preprocessor_config.json', 'tokenizer.json',
    'tokenizer_config.json', 'video_preprocessor_config.json', 'vocab.json'}
SOURCE_NAMES = ('scripts/run_teacher_qwen3vl32b_recovery.py', 'tests/test_teacher_qwen3vl32b_recovery.py',
 'scripts/run_teacher_qwen3vl32b_pilot.py', 'tests/test_teacher_qwen3vl32b_pilot.py',
 'scripts/run_teacher_label_pilot.py', 'scripts/run_teacher_single_gpu_admission.py',
 'scripts/recaption_e3.py', 'scripts/prepare_teacher_label_admission.py',
 'scripts/run_teacher_generation.py', 'scripts/run_teacher_mechanism_probe.py',
 'scripts/prepare_teacher_recovery.py', 'scripts/run_teacher_devset.py',
 'scripts/run_teacher_inference_smoke.py', 'scripts/measure_teacher_throughput.py',
 'scripts/prepare_teacher_comparison.py', 'scripts/run_teacher_base.py',
 'scripts/verify_teacher_history.py', 'scripts/gpu_hold.py', 'scripts/train_e3_lora.py',
 'scripts/run_teacher_training_smoke.py', 'scripts/prepare_teacher_training_admission.py')
GROUNDING_DENY = re.compile(r'\b(name(?:d)?|identity|brand|logo|ocr|reads? (?:as|the words)|'
 r'intends?|wants?|plans? to|hopes?|probably|perhaps|apparently|seems?|likely|'
 r'invisible|unseen|off.screen|imagines?|feels?|thinks?|camera operator)\b', re.I)
FRAME_META = re.compile(r'\b(frame|sampled|sampling|timestamps?|clip length|video duration|'
 r'fps|resolution)\b|\b\d+\s*(?:seconds?|pixels?)\b', re.I)

CASE_STAGES = ('local_copy_verify', 'frame_decode', 'facts_generate', 'facts_parse',
    'caption_generate', 'caption_validate')
# Only these literal protocol messages may be stored; model text/paths/stderr never are.
SAFE_REASONS = frozenset((
    'Nonempty bounded visible description required', 'Sensitive output withheld',
    'Ungrounded identity/intent/object claim withheld', 'Exact typed facts root schema required',
    'Typed visible facts arrays required', 'No global held-object schema', 'Generic entity ID required',
    'Hands alone cannot transfer actions across entities', 'Distinct sequential generic entity IDs required',
    'Exact event schema required', 'Action must refer to its actual entity ID',
    'Strict displayed frame range required', 'Cross-entity global hand/object transfer withheld',
    'Sensitive/malformed facts output withheld', 'no parseable complete top-level JSON object in model output',
    'Single grounded paragraph without frame/sample metadata required', 'Caption must contain 40..120 words',
    'Actual sixteen resized images required', 'Generation produced no new tokens',
    'Actual model generation config must provide strict EOS token IDs',
    'Generation did not finish with EOS; truncated facts or caption withheld',
    'Ordered real source-frame stage identity differs', 'Actual local-copy/source-video SHA differs',
    'Actual decoded source frame indices differ', 'CPU source frame probe failed',
    'Exactly one source video stream required', 'Actual fixed 121/24/1280x704 media required',
    'Partial CPU source frame decode',
    'CPU must decode every real source frame; exact sixteen selected images required',
    'Allocated work window exhausted'))
STOP_REASONS = ('completed', 'projected_budget_insufficient', 'allocated_window_exhausted',
    'model_load_failed', 'worker_exception')


class DenyMatch(ValueError):
    """A fixed deny-vocabulary hit; only the normalized fixed token is retained."""
    def __init__(self, message, match):
        super().__init__(message)
        self.token = re.sub(r'[0-9]+', '#', re.sub(r'[^A-Za-z0-9 ]', ' ', match.group(0)).lower()).strip()[:24]


def reason_code(exc):
    message = str(exc)
    return message if message in SAFE_REASONS else 'unlisted_' + bounded_error(exc)['error_type']


FACTS_PROMPT = """Use only these sixteen images, in displayed chronological order 0..15. Return ONLY JSON
with EXACT keys scene (string), rendered_or_screen (boolean), entities (array), events (array),
start_end_difference (string). Each entity has EXACT id, description, held_object. IDs are
unique E1,E2,...; description uses only generic visible appearance, never names or inferred
identity. held_object is a visible generic description string or null; do not guess object type
from occlusion. Each event has EXACT entity_id, frame_start, frame_end, action; the entity_id
must reference a visible entity, and integer 0<=frame_start<=frame_end<16. Describe only
visible changes, clearly associate head/arm/hand/object motion with its actual entity. Never
transfer global hands or held objects across people. A stable background does not mean a
motionless subject; pose is not evidence of motion. Preserve actual head, arm and object
changes instead of a blanket no-change claim. Omit uncertain/invisible hands and objects,
intent/emotion, identity, proper names, brands, logos and OCR. Do not invent events. Empty
events is allowed only when no visible entity motion is supported. Events describe the
trajectory; start_end_difference describes only the net first-to-last state. Visible motion
may return to its initial position, so matching endpoints never erase supported events.
Compare first and last images honestly; all descriptions must be nonempty. No additional keys or prose."""
CAPTION_PROMPT = """Using the SAME sixteen images and the typed facts below, write ONE grounded English
paragraph of 40..120 words. Associate every action/held object with the correct visible entity.
Describe supported appearance, setting, changes and start/end relationship, including rendered
or on-screen content only when visible. A static background does not erase entity motion.
Preserve supported outward-and-return motion even when first and last states are the same.
No names, inferred identity/intent, brands, OCR, hidden hands/objects, object-type guesses,
unsupported dynamics, or frame/sample indices, sampling/clip metadata. Do not pad with
invented events or automatically insert a motionless/no-change sentence. Return paragraph only.
Facts:
"""


ORIGINAL_FACTS_PROMPT_SHA = hashlib.sha256(FACTS_PROMPT.encode()).hexdigest()
ORIGINAL_CAPTION_PROMPT_SHA = hashlib.sha256(CAPTION_PROMPT.encode()).hexdigest()
FACTS_PROMPT_V2 = FACTS_PROMPT + """
Positive output specification: describe the depicted scene and generic visible entities directly.
Write descriptions as ordinary appearance, location, pose and visible motion; never narrate how
images were sampled or how text/identity could be read. Keep each entity description concise.
For an event, describe only the action of its entity_id; split simultaneous actions into separate
entity-specific events, with no speculation about anyone else. For a still scene, events=[] and
state only the visible arrangement in start_end_difference. Report what is supported, not a
list of unavailable information. The JSON's numeric frame_start/frame_end fields are sufficient
for chronology; prose descriptions use ordinary scene language without sampling terminology.
"""
CAPTION_PROMPT_V2 = """Write exactly ONE English paragraph describing the depicted scene using the SAME sixteen
images and the grounded facts below. Use four sentences, each with 12 to 22 words, for a total
of 48 to 88 words. Plain text only, no title, list, newline, JSON, quotation marks or word counts.
Sentence 1: generic visible subject and appearance in its setting.
Sentence 2: supported pose or action, attributed to the correct subject and visible object.
Sentence 3: another supported scene detail or visible progression without inventing an event.
Sentence 4: the visible end relationship; preserve an outward-and-return trajectory if supported.
For a still scene, describe appearance, spatial arrangement and supported unchanged state;
never invent movement merely to reach the length. Choose a few relevant details rather than
listing every object. Stay concise, and describe the scene directly instead of the input images.
Use normal chronology words such as initially, then and finally when useful. Never use frame,
frames, sampled, sampling, timestamps, fps, resolution, names, brands, OCR, identity, inferred
intent, unseen objects or a claim about an unavailable view. State only supported visible facts,
not what could not be read or inferred. Plain paragraph only. Facts:
"""


def configure_prompt_v2():
    global PROMPT_V2, LOCAL, STATE_DIR, RECOVERY_EVIDENCE, PRIOR_GPUH, CUMULATIVE_CAP_GPUH
    global FACTS_PROMPT, CAPTION_PROMPT, PROMPT_VERSION
    PROMPT_V2 = True
    LOCAL = Path('/dockerdata/wm_tmp/teacher_qwen32b_prompt_v2_20261103_run01')
    STATE_DIR = Path('/dockerdata/wm_logs/teacher_qwen32b_prompt_v2_20261103')
    RECOVERY_EVIDENCE = PROMPT_V2_AUTH.parent
    PRIOR_GPUH = 14.165898210853467
    CUMULATIVE_CAP_GPUH = 20.165898210853467
    FACTS_PROMPT, CAPTION_PROMPT = FACTS_PROMPT_V2, CAPTION_PROMPT_V2
    PROMPT_VERSION = 'qwen3vl32b-positive-output-policy-v2'


def prompt_v2_authorization(end=None):
    row = pinned(PROMPT_V2_AUTH, PROMPT_V2_AUTH_SHA, end)
    a = read(row, end)
    expected = dict(allocation_ordinal=3, new_allocation_GPUh_cap=6, wall_cap_seconds=5400,
        allocated_cards=4, dtype='bfloat16', cases=32, fixed_groups=[16, 16], frames_per_case=16,
        prior_charged_upper_GPUh=PRIOR_GPUH, cumulative_label_upper_GPUh_cap=CUMULATIVE_CAP_GPUH,
        label_stage_cap_GPUh=24, automatic_retry_allowed=False, original_validation_gates_preserved=True,
        original_cases_preserved=True, original_raw_failures_and_costs_preserved=True,
        full_rebuild_allowed=False, formal_training_allowed=False, AR_allowed=False, student_distillation_allowed=False)
    require(a.get('schema') == 'teacher_qwen32b_prompt_policy_v2_authorization_v1'
        and a.get('user_directive') == '请持续推进实验,直到模型能接近或者超越sota'
        and all(type(a.get(k)) is type(v) and a[k] == v for k, v in expected.items()),
        'Exact new prompt-policy allocation required; no previous allocation reset')
    prior = ROOT / 'eval/teacher_label_supplemental_20261103'
    require(a['previous_experiment']['path'] == str(prior / 'run01_raw_snapshot/cost.failure.json')
        and a['previous_claim']['path'] == str(prior / 'attempt.claim'), 'Previous charged allocation records required')
    failure, claim = read(a['previous_experiment'], end), read(a['previous_claim'], end)
    require(failure['terminal'] is True and failure['success'] is False
        and failure['released_observed'] is True and failure['allocated_GPUh'] == 6.0
        and failure['cumulative_label_upper_GPUh'] == PRIOR_GPUH and type(claim.get('pid')) is int,
        'Previous failure and claim must remain consumed and fully charged')
    return row


def require(ok, message):
    if not ok:
        raise ValueError(message)


def helpers():
    import prepare_teacher_label_admission as prep
    import run_teacher_generation as process
    import run_teacher_mechanism_probe as m1
    import recaption_e3 as rc
    return prep, process, m1, rc


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def deadline(end):
    if end is not None and time.monotonic() >= end:
        raise TimeoutError('Allocated work window exhausted')


def record(path, end=None):
    deadline(end)
    p = Path(path).absolute()
    require(p.resolve() == p and p.is_file(), 'Actual non-symlink file required')
    before = p.stat()
    _, _, m1, _ = helpers()
    r = m1.file_record(p, deadline=end, allow_empty=before.st_size == 0)
    after = p.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'Content changed during hashing')
    deadline(end)
    return {k: r[k] for k in ('path', 'bytes', 'sha256')}


def check(row, end=None):
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}
            and type(row['bytes']) is int and row['bytes'] >= 0
            and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Exact typed file binding required')
    require(record(row['path'], end) == row, 'Current content differs from frozen bytes')


def read(row, end=None):
    deadline(end)
    prep, _, _, _ = helpers()
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}
            and type(row['bytes']) is int and 0 < row['bytes'] <= 16 * 1024 * 1024,
            'Small exact JSON binding required')
    bound, raw = prep.bound(row['path'])
    require(bound == row, 'Actual raw JSON bytes changed')
    value = prep.strict_json(raw)
    deadline(end)
    return value


def write_new(path, value):
    _, _, m1, _ = helpers()
    m1.atomic_json(path, value)
    return record(path)


def pinned(path, digest, end=None):
    row = record(path, end)
    require(row['sha256'] == digest, 'Original immutable evidence SHA differs')
    return row


def fixed_config():
    config = dict(model='Qwen3-VL-32B-Instruct', model_root=str(MODEL_ROOT), dtype='bfloat16',
        quantization=False, device_map='balanced', max_memory={str(i): '34GiB' for i in range(4)},
        attn_implementation='sdpa', local_files_only=True, CPU_disk_weight_offload_allowed=False,
        frames=16, source_frames=121, source_fps=24, source_size=[1280, 704], resize=[640, 352],
        include_first_last=True, do_sample=False, max_new_tokens_facts=1024,
        max_new_tokens_caption=384, caption_words=[40, 120], prompt_version=PROMPT_VERSION,
        facts_prompt_sha256=hashlib.sha256(FACTS_PROMPT.encode()).hexdigest(),
        caption_prompt_sha256=hashlib.sha256(CAPTION_PROMPT.encode()).hexdigest(),
        stage2_same_images=True, automatic_retry_allowed=False)
    if CONSTRAINED:
        config['decoder_constraints'] = dict(deny_words=list(DENY_DECODE_WORDS),caption_deny_words=list(CAPTION_DECODE_WORDS),caption_eos_min_words=40)
    return config


DENY_DECODE_WORDS = ('name','named','identity','brand','logo','ocr','intends','wants',
    'hopes','probably','perhaps','apparently','seems','likely','invisible','unseen',
    'off-screen','imagines','feels','thinks','camera operator')
CAPTION_DECODE_WORDS = ('frame','frames','sampled','sampling','timestamp','timestamps','fps','resolution','pixels','seconds')


def configure_constrained():
    global CONSTRAINED, PROMPT_V2, LOCAL, STATE_DIR, RECOVERY_EVIDENCE, CAP, ALLOCATION_GPUH
    global PRIOR_GPUH, CUMULATIVE_CAP_GPUH, PROMPT_VERSION
    configure_prompt_v2()
    PROMPT_V2, CONSTRAINED = False, True
    LOCAL = Path('/dockerdata/wm_tmp/teacher_qwen32b_constrained_20261103_run01')
    STATE_DIR = Path('/dockerdata/wm_logs/teacher_qwen32b_constrained_20261103')
    RECOVERY_EVIDENCE = CONSTRAINED_AUTH.parent
    CAP, ALLOCATION_GPUH = 2400, 4*2400/3600
    PRIOR_GPUH = 21.165898210853467
    CUMULATIVE_CAP_GPUH = PRIOR_GPUH + ALLOCATION_GPUH
    PROMPT_VERSION = 'qwen3vl32b-positive-policy-constrained-v3'


def constrained_authorization(end=None):
    row = pinned(CONSTRAINED_AUTH, CONSTRAINED_AUTH_SHA, end)
    a = read(row, end)
    expected = dict(allocation_ordinal=4, new_allocation_GPUh_cap=ALLOCATION_GPUH,
        wall_cap_seconds=CAP, allocated_cards=4, frames_per_case=16, cases=32,
        fixed_groups=[16,16], dtype='bfloat16', prior_charged_upper_GPUh=PRIOR_GPUH,
        cumulative_label_upper_GPUh_cap=CUMULATIVE_CAP_GPUH, label_stage_cap_GPUh=24,
        automatic_retry_allowed=False, original_cases_preserved=True, original_validation_gates_preserved=True,
        semantic_review_followup_cap_GPUh=0.15, formal_training_allowed=False,AR_allowed=False,student_distillation_allowed=False)
    require(a.get('schema')=='teacher_qwen32b_constrained_decode_scope_v1'
        and all(type(a.get(k)) is type(v) and a[k]==v for k,v in expected.items()),
        'New prospective decoder protocol and exact remaining budget required')
    review, failed = read(a['review_terminal'],end), read(a['v2_failure'],end)
    require(review['terminal'] is True and review['success'] is True and review['charged_upper_GPUh']==1
        and review['cumulative_label_upper_GPUh']==PRIOR_GPUH and review['GPUs_released'] is True
        and failed['terminal'] is True and failed['success'] is False and failed['allocated_GPUh']==6,
        'Previous original failure and independent review costs must remain charged')
    require(CUMULATIVE_CAP_GPUH+0.15<=24, 'Decoder and review must fit the unborrowed remaining label budget')
    return row


def minimum_caption_words(prompt_length, processor, eos):
    from transformers import LogitsProcessor
    class MinimumWords(LogitsProcessor):
        def __call__(self, input_ids, scores):
            text = processor.batch_decode(input_ids[:,prompt_length:],skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            if len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b",text))<40:
                scores[:,eos]=float('-inf')
            return scores
    return MinimumWords()


def configure_supplemental():
    global SUPPLEMENTAL, LOCAL, STATE_DIR, RECOVERY_EVIDENCE, PRIOR_GPUH, CUMULATIVE_CAP_GPUH
    SUPPLEMENTAL = True
    LOCAL = Path('/dockerdata/wm_tmp/teacher_qwen32b_supplemental_20261103_execution01')
    STATE_DIR = Path('/dockerdata/wm_logs/teacher_qwen32b_supplemental_20261103')
    RECOVERY_EVIDENCE = SUPPLEMENTAL_AUTH.parent
    PRIOR_GPUH = 8.165898210853467
    CUMULATIVE_CAP_GPUH = 14.165898210853467


def canonical_launch(argv):
    require(type(argv) is list and len(argv) >= 2 and argv[1] == str(Path(__file__).resolve()),
            'Launch the parent with the absolute canonical runner path before any allocation')
    require(argv.count('--supplemental') == int(SUPPLEMENTAL)
        and argv.count('--prompt-v2') == int(PROMPT_V2) and argv.count('--constrained') == int(CONSTRAINED),
        'Exact experimental launch scope required')
    return True


def supplemental_authorization(end=None):
    row = pinned(SUPPLEMENTAL_AUTH, SUPPLEMENTAL_AUTH_SHA, end)
    a = read(row, end)
    expected = dict(new_allocation_GPUh_cap=6, wall_cap_seconds=5400, allocated_cards=4,
        fixed_cases=32, fixed_groups=[16, 16], frames_per_case=16, dtype='bfloat16',
        prior_charged_upper_GPUh=PRIOR_GPUH, cumulative_label_upper_GPUh_cap=CUMULATIVE_CAP_GPUH,
        label_stage_cap_GPUh=24, previous_consumed_recovery_attempts=1, allocation_ordinal=2,
        original_recovery_claim_preserved=True, previous_failure_costs_preserved=True,
        automatic_retry_allowed=False, sample_substitution_allowed=False, frame_reduction_allowed=False,
        precision_gate_relaxation_allowed=False, formal_training_allowed=False, AR_allowed=False,
        student_distillation_allowed=False)
    require(a.get('schema') == 'teacher_qwen32b_supplemental_allocation_authorization_v1'
        and a.get('user_reply') == '我给你所有权限,请继续进行推进'
        and all(type(a.get(k)) is type(v) and a[k] == v for k, v in expected.items()),
        'New explicit supplemental allocation authorization required; original attempt stays consumed')
    old_root = ROOT / 'eval/teacher_label_recovery_20261103'
    require(a['prior_failure']['path'] == str(old_root / 'run01_raw_snapshot/cost.failure.json')
        and a['prior_claim']['path'] == str(old_root / 'attempt.claim'), 'Original consumed allocation bindings required')
    failure, claim = read(a['prior_failure'], end), read(a['prior_claim'], end)
    require(failure['terminal'] is True and failure['success'] is False
        and failure['released_observed'] is True and failure['allocated_GPUh'] == 6.0
        and failure['prior_charged_label_GPUh'] == ORIGINAL_PRIOR_GPUH
        and failure['cumulative_label_upper_GPUh'] == PRIOR_GPUH
        and type(claim.get('pid')) is int and claim['pid'] > 0, 'Prior failure must remain fully charged and consumed')
    return row


def authorization_rows(end=None):
    rows = [pinned(p, d, end) for p, d in ((AUTH, AUTH_SHA), (NEW_AUTH, NEW_AUTH_SHA),
                                         (SCOPE_AUTH, SCOPE_AUTH_SHA))]
    original, new, scope = [read(r, end) for r in rows]
    require(original['schema'] == 'teacher_training_admission_authorization_v1'
        and original['stage_caps_GPUh'] == {'label_rebuild_and_cache': 24,
            'single_gpu_training_resume': 2, 'multi_node_technical_validation': 6}
        and type(original['total_cap_GPUh']) is int and original['total_cap_GPUh'] == 32
        and original['failed_allocations_charged'] is True
        and all(original[k] is False for k in ('formal_candidate_training_budget_approved',
            'phase_borrowing_allowed', 'AR_allowed', 'student_or_distillation_allowed',
            'previous_experiment_budgets_reused', 'training_started')), 'Original 24+2+6 scope drift')
    require(new['schema'] == 'teacher_new_captioner_pilot_authorization_v1'
        and new['user_choice'] == '换标注器做新pilot'
        and type(new['original_stage_cap_GPUh']) is int and new['original_stage_cap_GPUh'] == 24
        and type(new['prior_label_GPUh']) is float and new['prior_label_GPUh'] == FIRST_PILOT_PRIOR_GPUH
        and all(new[k] is False for k in ('phase_borrowing_allowed', 'automatic_retry_allowed',
            'full_rebuild_allowed', 'formal_training_allowed'))
        and all(new[k] is True for k in ('failed_evidence_preserved',
            'pilot_must_freeze_source_budget_and_samples_before_GPU', 'semantic_review_required')),
        'New-captioner authorization drift')
    expected = dict(schema='teacher_stronger_captioner_pilot_scope_v1',
        authorized_user_choice='换标注器做新pilot', original_authorization=str(NEW_AUTH),
        stage=PHASE, stage_cap_GPUh=24, prior_charged_label_GPUh=FIRST_PILOT_PRIOR_GPUH,
        allocation_GPUh_cap=2, allocated_cards=4, wall_cap_seconds=1800, node_rank=1,
        gpu_indices=[0, 1, 2, 3], model='Qwen3-VL-32B-Instruct', model_root=OLD_MODEL_ROOT,
        dtype='bfloat16', weight_quantization=False, model_shards=14, frames_per_case=16,
        frame_resize=[640, 352], decoding='greedy,do_sample=false', cases=32,
        case_selection='16previous_diagnostic+16newblind_no_benchmark_scores', selection_seed=SEED,
        max_new_tokens_facts=1024, max_new_tokens_caption=384,
        stages=['grounded_entity_ID_and_visible_events', 'caption_with_same_images_and_facts'],
        full_model_CPU_or_disk_offload_allowed=False, automatic_retry_allowed=False,
        all_failure_costs_charged=True, new_output_root_required=True,
        actual_GPU_entry_requires_source_publication_and_asset_lease_checks=True,
        independent_semantic_review_required=True, full_rebuild_allowed=False,
        formal_training_allowed=False, AR_allowed=False, student_distillation_allowed=False)
    require(set(scope) == set(expected) and all(type(scope[k]) is type(v) and scope[k] == v
        for k, v in expected.items()), 'Exact four-card/2GPUh label scope drift')
    recovery_row, cluster_row = pinned(RECOVERY_AUTH, RECOVERY_AUTH_SHA, end), pinned(CLUSTER_AUTH, CLUSTER_AUTH_SHA, end)
    recovery, cluster = read(recovery_row, end), read(cluster_row, end)
    expected = dict(schema='teacher_qwen3vl32b_one_recovery_authorization_v1', user_choice='批准一次6 GPUh恢复',
        original_scope_authorization=str(SCOPE_AUTH), stage=PHASE, stage_cap_GPUh=24,
        new_allocation_GPUh_cap=6, allocated_cards=4, wall_cap_seconds=5400, node_rank=1,
        gpu_indices=[0, 1, 2, 3], frames_per_case=16, cases=32, original_fixed_cases_preserved=True,
        old_pilot_charged_upper_GPUh=2, all_previous_label_charged_upper_GPUh=ORIGINAL_PRIOR_GPUH,
        cumulative_label_upper_GPUh_cap=ORIGINAL_CUMULATIVE_CAP_GPUH, one_recovery_attempt=True,
        automatic_retry_allowed=False, new_output_root_required=True,
        old_raw_logs_labels_and_costs_preserved=True, safe_failure_stage_and_case_duration_required=True,
        new_source_runtime_publication_gate_required=True, full_rebuild_allowed=False,
        formal_training_allowed=False, AR_allowed=False, student_distillation_allowed=False)
    require(set(recovery) == set(expected) and all(type(recovery[k]) is type(v) and recovery[k] == v
        for k, v in expected.items()), 'Exact one-recovery 6GPUh scope drift')
    require(cluster.get('schema') == 'teacher_qwen3vl32b_new_cluster_recovery_authorization_v1'
        and cluster.get('original_recovery_authorization') == {'path': str(RECOVERY_AUTH), 'sha256': RECOVERY_AUTH_SHA}
        and cluster.get('first_pilot_plan') == {'path': str(FIRST_PLAN), 'sha256': FIRST_PLAN_SHA}
        and all(type(cluster.get(k)) is int and cluster[k] == v for k, v in (('nodes', 1), ('allocated_cards', CARDS),
            ('cases', 32), ('frames_per_case', 16), ('wall_cap_seconds', 5400), ('new_allocation_GPUh_cap', 6),
            ('stage_cap_GPUh', 24), ('recovery_attempts_before_migration', 0)))
        and cluster.get('all_previous_label_charged_upper_GPUh') == ORIGINAL_PRIOR_GPUH
        and cluster.get('cumulative_label_upper_GPUh_cap') == ORIGINAL_CUMULATIVE_CAP_GPUH
        and cluster.get('dtype') == 'bfloat16' and cluster.get('fixed_groups') == [16, 16]
        and all(cluster.get(k) is True for k in ('one_recovery_attempt', 'new_output_root_required',
            'safe_failure_stage_case_duration_and_stop_reason_required', 'new_host_runtime_model_source_freeze_required',
            'publication_gate_required'))
        and all(cluster.get(k) is False for k in ('old_host_gpu_paths_are_execution_credentials',
            'recovery_count_reset_by_migration', 'automatic_retry_allowed', 'sample_substitution_allowed',
            'frame_reduction_allowed', 'foreign_gpu_workloads_may_be_killed', 'full_rebuild_allowed',
            'formal_training_allowed', 'AR_allowed', 'student_distillation_allowed')),
        'New-cluster recovery freeze drift')
    require(CAP == (2400 if CONSTRAINED else 5400) and TAIL == 120 and CARDS == 4 and ALLOCATION_GPUH == CARDS * CAP / 3600
            and abs(PRIOR_GPUH + ALLOCATION_GPUH - CUMULATIVE_CAP_GPUH) < 1e-12
            and CUMULATIVE_CAP_GPUH <= 24, 'No phase borrowing')
    return rows + [recovery_row, cluster_row] + ([supplemental_authorization(end)] if SUPPLEMENTAL else []) + ([prompt_v2_authorization(end)] if PROMPT_V2 else []) + ([constrained_authorization(end)] if CONSTRAINED else [])


def runtime(end=None):
    packages, files = {}, {}
    for name, module in (('torch', 'torch'), ('transformers', 'transformers'),
        ('accelerate', 'accelerate'), ('safetensors', 'safetensors'), ('numpy', 'numpy'),
        ('pillow', 'PIL'), ('tokenizers', 'tokenizers'), ('torchvision', 'torchvision')):
        dist = importlib.metadata.distribution(name)
        packages[name] = dist.version
        paths = [Path(dist.locate_file(module + '/__init__.py')).resolve()]
        metadata = next((p for p in dist.files or () if str(p).endswith('.dist-info/METADATA')), None)
        require(metadata is not None, 'Runtime package metadata missing')
        paths.append(Path(dist.locate_file(metadata)).resolve())
        if name == 'torch':
            native = list(Path(dist.locate_file('torch')).glob('_C*.so'))
            require(native, 'Actual torch C extension missing')
            paths.extend(p.resolve() for p in native)
        for p in paths:
            files[str(p)] = record(p, end)
    tools = {n: shutil.which(n) for n in ('ffmpeg', 'ffprobe', 'timeout', 'nvidia-smi', 'git')}
    require(all(tools.values()), 'CPU/GPU boundary tools missing')
    # Launch through the venv entry (keeps its site-packages); hash the resolved interpreter and venv config.
    python = str(Path(sys.executable).absolute())
    require(python == PYTHON and packages['torch'].startswith('2.7.1')
            and packages['transformers'] == '4.57.1', 'Actual new-cluster selected runtime required')
    for p in [python, str(Path(PYTHON).parents[1] / 'pyvenv.cfg'), *tools.values()]:
        files[str(Path(p).resolve())] = record(Path(p).resolve(), end)
    return dict(python=python, packages=packages, tools=tools, files=list(files.values()),
        exhaustive_attestation=False, caveat='Selected actual entries, metadata, torch C and tools only')


def model_identity(rows):
    return [{k: r[k] for k in ('relative', 'bytes', 'sha256')} for r in rows]


def validate_model_proof(proof, end=None, current=True):
    require(proof.get('schema') == 'teacher_qwen3vl32b_model_content_v1'
        and type(proof.get('rank')) is int and proof['rank'] == RANK
        and type(proof.get('hostname')) is str and proof['hostname']
        and proof.get('GPU_used') is False and proof.get('model_type') == 'qwen3_vl'
        and proof.get('architectures') == ['Qwen3VLForConditionalGeneration']
        and proof.get('weight_shards') == 14 and type(proof['weight_shards']) is int
        and proof.get('weight_bytes') == WEIGHT_BYTES and type(proof['weight_bytes']) is int
        and proof.get('model_root') == str(MODEL_ROOT), 'Actual dense 32B rank1 model proof required')
    require(MODEL_ROOT.resolve() == MODEL_ROOT, 'Canonical local model root required')
    rows = proof['models']
    require(type(rows) is list and rows and all(type(r) is dict
        and set(r) == {'relative', 'path', 'bytes', 'sha256'} for r in rows), 'Exact actual model file records required')
    names = [r['relative'] for r in rows]
    require(names == sorted(set(names)) and set(names) == set(WEIGHTS) | REQUIRED_CONFIGS
        and set(n for n in names if n.endswith('.safetensors')) == set(WEIGHTS)
        and all(Path(n).name == n and not n.startswith('.') for n in names), 'Fourteen shards and all local configs required')
    require(sum(r['bytes'] for r in rows if r['relative'] in WEIGHTS) == WEIGHT_BYTES,
            'Actual fourteen-shard byte coverage differs')
    require(hashlib.sha256(encoded(model_identity(rows))).hexdigest() == proof['content_sha256'],
            'Actual full model content digest differs')
    for r in rows:
        require(r['path'] == str(MODEL_ROOT / r['relative']) and type(r['bytes']) is int
                and r['bytes'] > 0 and re.fullmatch('[0-9a-f]{64}', r['sha256']), 'Loaded file identity/root drift')
        if current:
            check({k: r[k] for k in ('path', 'bytes', 'sha256')}, end)
    if current:
        actual = sorted(p.name for p in MODEL_ROOT.iterdir() if (p.is_file() or p.is_symlink()) and not p.name.startswith('.'))
        require(all(not p.is_symlink() for p in MODEL_ROOT.iterdir()), 'No symlinked model-root entries')
        require(actual == names, 'Freeze all actual model-root files, no symlinks or omitted assets')
        prep, _, _, _ = helpers()
        config = prep.strict_json((MODEL_ROOT / 'config.json').read_bytes())
        index = prep.strict_json((MODEL_ROOT / 'model.safetensors.index.json').read_bytes())
        require(config['model_type'] == 'qwen3_vl'
            and config['architectures'] == ['Qwen3VLForConditionalGeneration']
            and config.get('torch_dtype', config.get('dtype', config.get('text_config', {}).get('dtype'))) == 'bfloat16'
            and 'quantization_config' not in config, 'Actual config is not dense BF16 Qwen3 VL')
        require(type(index['weight_map']) is dict and index['weight_map']
            and set(index['weight_map'].values()) == set(WEIGHTS), 'Actual index must cover all fourteen shards')
    return proof


def selected_rows(rows, old):
    prep, _, _, _ = helpers()
    require(type(rows) is list and len(rows) == 1856 and type(old.get('jobs')) is list
        and len(old['jobs']) == 16 and old.get('schema') == 'teacher_label_pilot_execution_v2'
        and old.get('selection_seed') == 20261101, 'Exact old sixteen diagnostic identities required')
    eligible = [r for r in rows if not prep.UNSAFE.search(' '.join(
        [r['sample_id'], Path(r['video']).name, *r['captions'].values()]))]
    require(len(eligible) == 1815 and len(rows) - len(eligible) == 41, 'Metadata prescreen count drift')
    by_id = {r['sample_id']: r for r in eligible}
    require(len(by_id) == 1815, 'Duplicate safe metadata identity')
    diagnostic = []
    for i, job in enumerate(old['jobs']):
        require(job['index'] == i and job['sample_id'] in by_id, 'Old diagnostic order/safety drift')
        row = by_id[job['sample_id']]
        require(row['video'] == job['video']['path']
            and hashlib.sha256(row['captions']['orig'].encode()).hexdigest() == job['original_prompt_sha256'],
            'Old diagnostic video/original-prompt identity drift')
        diagnostic.append(row)
    used = {r['sample_id'] for r in diagnostic}
    require(len(used) == 16, 'Old sixteen must be distinct')
    blind = random.Random(SEED).sample([r for r in eligible if r['sample_id'] not in used], 16)
    return diagnostic + blind


def jobs_for(rows, old, end=None):
    selected = selected_rows(rows, old)
    jobs = []
    for i, row in enumerate(selected):
        require(re.fullmatch(r'[A-Za-z0-9_.+-]+', row['sample_id']) and row['sample_id'] not in ('.', '..'),
                'Safe generic media basename required')
        video = record(row['video'], end)
        if i < 16:
            require(video == old['jobs'][i]['video'], 'Immutable diagnostic source bytes drift')
        jobs.append(dict(index=i, rank=RANK, group='diagnostic' if i < 16 else 'blind',
            sample_id=row['sample_id'], video=video,
            original_prompt_sha256=hashlib.sha256(row['captions']['orig'].encode()).hexdigest(),
            local_video=str(LOCAL / f'video{i}.mp4')))
    return jobs


def first_pilot_identity(jobs, end=None):
    first = read(pinned(FIRST_PLAN, FIRST_PLAN_SHA, end), end)
    keys = ('index', 'group', 'sample_id', 'video', 'original_prompt_sha256')
    require(first.get('schema') == 'teacher_qwen3vl32b_pilot_plan_v1' and len(first['jobs']) == 32
        and [{k: j[k] for k in keys} for j in jobs] == [{k: j[k] for k in keys} for j in first['jobs']]
        and first['config']['facts_prompt_sha256'] == (ORIGINAL_FACTS_PROMPT_SHA if (PROMPT_V2 or CONSTRAINED) else fixed_config()['facts_prompt_sha256'])
        and first['config']['caption_prompt_sha256'] == (ORIGINAL_CAPTION_PROMPT_SHA if (PROMPT_V2 or CONSTRAINED) else fixed_config()['caption_prompt_sha256']),
        'Recovery must use the same 32 cases, bytes and prompts as the first pilot')
    return True


def model_proof(output):
    require('torch' not in sys.modules, 'CPU model proof must not initialize model runtime')
    rows = []
    for name in sorted(p.name for p in MODEL_ROOT.iterdir() if p.is_file() and not p.name.startswith('.')):
        r = record(MODEL_ROOT / name)
        rows.append(dict(relative=name, **r))
    proof = dict(schema='teacher_qwen3vl32b_model_content_v1', rank=RANK, hostname=os.uname().nodename,
        GPU_used=False, model_type='qwen3_vl', architectures=['Qwen3VLForConditionalGeneration'],
        weight_shards=len(WEIGHTS), weight_bytes=WEIGHT_BYTES, model_root=str(MODEL_ROOT), models=rows,
        ignored_hidden_entries=sorted(p.name for p in MODEL_ROOT.iterdir() if p.name.startswith('.')),
        content_sha256=hashlib.sha256(encoded(model_identity(rows))).hexdigest())
    validate_model_proof(proof)
    return write_new(output, proof)


def prepare(model_proof_path, output, gpu_uuids):
    require('torch' not in sys.modules, 'CPU preparation must not initialize model runtime')
    require(not LOCAL.exists() and not LOCAL.is_symlink(), 'New local root required; never reset')
    auths = authorization_rows()
    old_row = pinned(OLD_PLAN, OLD_PLAN_SHA)
    old = read(old_row)
    prep, _, _, _ = helpers()
    manifest, rows = prep.manifest_rows(MANIFEST)
    require(manifest == old['manifest'], 'Original private 1856 manifest SHA drift')
    model_row = record(model_proof_path)
    model = validate_model_proof(read(model_row))
    require(model['hostname'] == os.uname().nodename, 'Prepare on actual rank1 model host')
    require(type(gpu_uuids) is list and len(gpu_uuids) == 4 and len(set(gpu_uuids)) == 4
        and all(re.fullmatch(r'GPU-[0-9a-fA-F-]+', u) for u in gpu_uuids), 'Four explicit physical UUIDs required')
    import gpu_hold
    mapping = gpu_hold.query_gpus()
    require([mapping[i]['uuid'] for i in range(4)] == gpu_uuids, 'Actual rank1 GPUs0..3 UUID order differs')
    value = dict(schema=SCHEMA, private=True, phase=PHASE, authorizations=auths,
        old_plan=old_row, manifest=manifest, jobs=jobs_for(rows, old), model_proof=model_row,
        model=model, source={n: record(ROOT / n) for n in SOURCE_NAMES}, runtime=runtime(),
        first_pilot_plan=pinned(FIRST_PLAN, FIRST_PLAN_SHA), node_ip=os.environ.get('NODE_IP'),
        hostname=os.uname().nodename, node_rank=RANK, gpu_indices=list(range(4)), gpu_uuids=gpu_uuids,
        output_root=str(LOCAL), state_dir=str(STATE_DIR), config=fixed_config(),
        selection_seed=SEED, selection_uses_benchmark_scores=False, fixed_groups=[16, 16],
        metadata_eligible_rows=1815, metadata_rejected_rows=41,
        cap_seconds=CAP, tail_seconds=TAIL, allocated_cards=CARDS, allocation_GPUh_cap=ALLOCATION_GPUH,
        prior_charged_label_GPUh=PRIOR_GPUH, stage_cap_GPUh=24, single_attempt=True,
        automatic_retry_allowed=False, full_rebuild_allowed=False, formal_training_allowed=False,
        AR_allowed=False, student_distillation_allowed=False, GPU_execution_started=False,
        semantic_review_pass=False, structural_validation_is_semantic_approval=False)
    first_pilot_identity(value['jobs'])
    return write_new(output, value)


def verify(path, digest, end=None, local_assets=True):
    bound = pinned(path, digest, end)
    p = read(bound, end)
    require(p.get('schema') == SCHEMA and p.get('private') is True and p.get('phase') == PHASE
        and type(p.get('node_rank')) is int and p['node_rank'] == RANK
        and p.get('hostname') == os.uname().nodename
        and type(p.get('gpu_indices')) is list and all(type(i) is int for i in p['gpu_indices'])
        and p['gpu_indices'] == [0, 1, 2, 3] and encoded(p.get('config')) == encoded(fixed_config())
        and type(p.get('node_ip')) is str and p['node_ip'] == os.environ.get('NODE_IP')
        and p.get('output_root') == str(LOCAL) and p.get('state_dir') == str(STATE_DIR), 'New fixed rank1 scope/root/config required')
    fixed = dict(cap_seconds=CAP, tail_seconds=TAIL, allocated_cards=CARDS, allocation_GPUh_cap=ALLOCATION_GPUH,
        prior_charged_label_GPUh=PRIOR_GPUH, stage_cap_GPUh=24, selection_seed=SEED,
        metadata_eligible_rows=1815, metadata_rejected_rows=41, fixed_groups=[16, 16],
        single_attempt=True, automatic_retry_allowed=False, full_rebuild_allowed=False,
        formal_training_allowed=False, AR_allowed=False, student_distillation_allowed=False,
        GPU_execution_started=False, semantic_review_pass=False,
        structural_validation_is_semantic_approval=False, selection_uses_benchmark_scores=False)
    require(all(type(p.get(k)) is type(v) and p[k] == v for k, v in fixed.items()), 'Typed reservation/quality gates drift')
    require(LOCAL.resolve() == LOCAL and STATE_DIR.resolve() == STATE_DIR
        and len(p['gpu_uuids']) == 4 and len(set(p['gpu_uuids'])) == 4
        and all(re.fullmatch(r'GPU-[0-9a-fA-F-]+', u) for u in p['gpu_uuids']), 'Canonical new roots/four UUIDs required')
    require(p['authorizations'] == authorization_rows(end), 'All three original raw auth bindings required')
    require(p['old_plan'] == pinned(OLD_PLAN, OLD_PLAN_SHA, end), 'Old private selection plan SHA differs')
    prep, _, _, _ = helpers()
    old = read(p['old_plan'], end)
    manifest, rows = prep.manifest_rows(MANIFEST)
    require(p['manifest'] == manifest == old['manifest'], 'Current original private pool bytes required')
    require(p['jobs'] == jobs_for(rows, old, end), 'Exact immutable diagnostic+seeded blind selection differs')
    require(p.get('first_pilot_plan') == pinned(FIRST_PLAN, FIRST_PLAN_SHA, end), 'First pilot plan binding differs')
    first_pilot_identity(p['jobs'], end)
    require(set(p['source']) == set(SOURCE_NAMES), 'Full required current source coverage differs')
    for name, row in p['source'].items():
        require(row['path'] == str(ROOT / name), 'Source root drift')
        check(row, end)
    check(p['model_proof'], end)
    require(read(p['model_proof'], end) == p['model'], 'Actual model proof raw binding differs')
    validate_model_proof(p['model'], end, current=local_assets)
    require(p['model']['hostname'] == p['hostname'], 'Actual model host drift')
    if local_assets:
        require(runtime(end) == p['runtime'], 'Current actual selected runtime drift')
    return p, bound


def source_indices(count):
    require(type(count) is int and count >= 16, 'At least sixteen real source frames required')
    indices = [i * (count - 1) // 15 for i in range(16)]
    require(indices[0] == 0 and indices[-1] == count - 1 and len(set(indices)) == 16,
            'Exact chronological first/last uniform source frames required')
    return indices


def probe_video(path, tools, end=None):
    deadline(end)
    result = subprocess.run([tools['ffprobe'], '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames,r_frame_rate,avg_frame_rate,width,height,duration',
        '-of', 'json', str(path)], capture_output=True, timeout= min(60, max(.01, end-time.monotonic())) if end else 60)
    require(result.returncode == 0, 'CPU source frame probe failed')
    prep, _, _, _ = helpers()
    stream = prep.strict_json(result.stdout)['streams']
    require(len(stream) == 1, 'Exactly one source video stream required')
    s = stream[0]
    from fractions import Fraction
    fps, average = Fraction(s['r_frame_rate']), Fraction(s['avg_frame_rate'])
    count, duration = int(s['nb_read_frames']), float(s['duration'])
    require(count == 121 and fps == average == 24 and (s['width'], s['height']) == (1280, 704)
        and math.isfinite(duration) and abs(duration-count/24) <= .002, 'Actual fixed 121/24/1280x704 media required')
    return dict(source_frames=count, fps_numerator=fps.numerator, fps_denominator=fps.denominator,
        duration_seconds=duration, source_size=[s['width'], s['height']],
        displayed_indices=list(range(16)), source_indices=source_indices(count), resize=[640, 352])


def sample_images(path, tools, end=None):
    from PIL import Image
    info = probe_video(path, tools, end)
    selected = set(info['source_indices'])
    command = [tools['ffmpeg'], '-nostdin', '-v', 'error', '-threads', '1', '-i', str(path),
        '-map', '0:v:0', '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    images, count, size = [], 0, 1280 * 704 * 3
    try:
        while True:
            deadline(end)
            data = child.stdout.read(size)
            if not data:
                break
            require(len(data) == size, 'Partial CPU source frame decode')
            if count in selected:
                image = Image.frombytes('RGB', (1280, 704), data)
                images.append(image.resize((640, 352), Image.Resampling.BILINEAR))
            count += 1
        child.wait(timeout=min(20, max(.01, end-time.monotonic())) if end else 20)
        require(child.returncode == 0 and count == info['source_frames'] and len(images) == 16,
                'CPU must decode every real source frame; exact sixteen selected images required')
        return images, info
    finally:
        child.stdout.close()
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)


def stage(path, digest):
    p, bound = verify(path, digest)
    require(not LOCAL.exists() and not LOCAL.is_symlink(), 'Stage never reuses an old local root')
    LOCAL.mkdir(mode=0o700)
    records = []
    for job in p['jobs']:
        check(job['video'])
        local = Path(job['local_video'])
        with Path(job['video']['path']).open('rb') as src, local.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8*1024*1024)
            dst.flush()
            os.fsync(dst.fileno())
        require(record(local)['sha256'] == job['video']['sha256'], 'Actual local/source bytes differ')
        images, frames = sample_images(local, p['runtime']['tools'])
        del images
        records.append(dict(index=job['index'], local_video=record(local), frames=frames))
    return write_new(LOCAL / 'stage.json', dict(schema='teacher_qwen3vl32b_recovery_stage_v1', plan=bound,
        hostname=p['hostname'], GPU_used=False, records=records))


def safe_text(text):
    prep, _, _, _ = helpers()
    require(type(text) is str and text.strip() and len(text) <= 12000 and '\0' not in text,
            'Nonempty bounded visible description required')
    require(not prep.UNSAFE.search(text), 'Sensitive output withheld')
    match = GROUNDING_DENY.search(text)
    if match:
        raise DenyMatch('Ungrounded identity/intent/object claim withheld', match)
    return text


def validate_facts(value):
    require(type(value) is dict and set(value) == {'scene', 'rendered_or_screen', 'entities',
        'events', 'start_end_difference'}, 'Exact typed facts root schema required')
    safe_text(value['scene'])
    safe_text(value['start_end_difference'])
    require(type(value['rendered_or_screen']) is bool and type(value['entities']) is list
        and 0 < len(value['entities']) <= 32 and type(value['events']) is list
        and len(value['events']) <= 128, 'Typed visible facts arrays required')
    ids = []
    for entity in value['entities']:
        require(type(entity) is dict and set(entity) == {'id', 'description', 'held_object'}, 'No global held-object schema')
        require(type(entity['id']) is str and re.fullmatch(r'E[1-9][0-9]*', entity['id']), 'Generic entity ID required')
        ids.append(entity['id'])
        safe_text(entity['description'])
        require(not re.fullmatch(r'(?:a pair of |two |the |a )?hands?', entity['description'], re.I),
                'Hands alone cannot transfer actions across entities')
        if entity['held_object'] is not None:
            safe_text(entity['held_object'])
    require(ids == [f'E{i+1}' for i in range(len(ids))], 'Distinct sequential generic entity IDs required')
    for event in value['events']:
        require(type(event) is dict and set(event) == {'entity_id', 'frame_start', 'frame_end', 'action'}, 'Exact event schema required')
        require(type(event['entity_id']) is str and event['entity_id'] in ids,
                'Action must refer to its actual entity ID')
        require(type(event['frame_start']) is int and type(event['frame_end']) is int
            and 0 <= event['frame_start'] <= event['frame_end'] < 16, 'Strict displayed frame range required')
        safe_text(event['action'])
        require(not re.search(r'\b(other|another|someone else|global)\b.*\b(hands?|held|holding)\b', event['action'], re.I),
                'Cross-entity global hand/object transfer withheld')
    return value


def parse_facts(text):
    prep, _, _, rc = helpers()
    require(type(text) is str and len(text) <= 24000 and not prep.UNSAFE.search(text), 'Sensitive/malformed facts output withheld')
    return validate_facts(rc.extract_json(text))


def validate_caption(text):
    safe_text(text)
    match = FRAME_META.search(text)
    if match:
        raise DenyMatch('Single grounded paragraph without frame/sample metadata required', match)
    require('\n' not in text.strip(), 'Single grounded paragraph without frame/sample metadata required')
    count = len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", text))
    require(40 <= count <= 120, 'Caption must contain 40..120 words')
    return text.strip()


def stage_messages(images, facts=None):
    require(len(images) == 16 and all(image.size == (640, 352) for image in images), 'Actual sixteen resized images required')
    if facts is not None:
        validate_facts(facts)
    prompt = FACTS_PROMPT if facts is None else CAPTION_PROMPT + json.dumps(facts, sort_keys=True)
    content = [{'type': 'image', 'image': image} for image in images]
    content.append({'type': 'text', 'text': prompt})
    return [{'role': 'user', 'content': content}]


def generate_text(model, processor, images, device, max_tokens, facts=None, trace=None):
    import torch
    messages = stage_messages(images, facts)
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt').to(device)
    constraints = {}
    if CONSTRAINED:
        words = DENY_DECODE_WORDS + (CAPTION_DECODE_WORDS if facts is not None else ())
        forms = {prefix+form for word in words for form in (word,word.title(),word.upper()) for prefix in ('',' ')}
        ids = [processor.tokenizer.encode(form,add_special_tokens=False) for form in sorted(forms)]
        constraints['bad_words_ids'] = [tokens for tokens in ids if tokens]
        if facts is not None:
            from transformers import LogitsProcessorList
            eos_ids = model.generation_config.eos_token_id
            eos_ids = [eos_ids] if type(eos_ids) is int else eos_ids
            require(type(eos_ids) is list and eos_ids and all(type(t) is int for t in eos_ids), 'Strict configured decoder EOS required')
            constraints['logits_processor'] = LogitsProcessorList([minimum_caption_words(inputs['input_ids'].shape[1],processor,eos_ids)])
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, **constraints)
    new_tokens = output[:, inputs['input_ids'].shape[1]:]
    require(new_tokens.ndim == 2 and new_tokens.shape[0] == 1 and new_tokens.shape[1] > 0,
            'Generation produced no new tokens')
    if trace is not None:
        trace['new_tokens'] = int(new_tokens.shape[1])
    eos = model.generation_config.eos_token_id
    if type(eos) is int:
        eos = [eos]
    require(type(eos) is list and eos and all(type(token) is int and token >= 0 for token in eos),
            'Actual model generation config must provide strict EOS token IDs')
    require(new_tokens[0, -1].item() in eos,
            'Generation did not finish with EOS; truncated facts or caption withheld')
    return processor.batch_decode(new_tokens, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]


def validate_device_map(mapping):
    require(type(mapping) is dict and mapping, 'Actual hf_device_map required')
    devices = set()
    for value in mapping.values():
        if type(value) is int:
            devices.add(value)
        elif type(value) is str and re.fullmatch(r'cuda:[0-3]', value):
            devices.add(int(value.split(':')[1]))
        else:
            raise ValueError('All weights must remain on the approved four CUDA devices; CPU/disk forbidden')
    require(devices == {0, 1, 2, 3}, 'Actual model must use all four approved CUDA devices')
    return True


def bounded_error(exc):
    # Do not echo model text, prompts, file paths, names or subprocess stderr.
    kind = type(exc).__name__
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,79}', kind):
        kind = 'PilotError'
    return dict(error_type=kind, error_message='Pilot case failed; raw sensitive/ungrounded content withheld')


def record_base(job, plan, frames=None):
    return dict(index=job['index'], group=job['group'], sample_id=job['sample_id'],
        video_sha256=job['video']['sha256'], original_prompt_sha256=job['original_prompt_sha256'],
        prompt_version=PROMPT_VERSION, frames=frames,
        model=dict(name='Qwen3-VL-32B-Instruct', root=str(MODEL_ROOT), dtype='bfloat16'),
        model_content_sha256=plan['model']['content_sha256'], source=plan['source'],
        runtime=plan['runtime'], semantic_review_pass=False, formal_training_allowed=False)


def case_failure(exc, stage, seconds, tokens):
    value = dict(bounded_error(exc), failure_stage=stage, failure_reason=reason_code(exc),
                 deny_token=getattr(exc, 'token', None), duration_seconds=seconds, new_tokens=tokens)
    require(stage in CASE_STAGES + ('not_attempted',), 'Fixed safe stage code required')
    return value


def complete_error_records(p, reason, stop_reason='worker_exception'):
    require(stop_reason in STOP_REASONS[1:], 'Fixed safe stop reason required')
    for job in p['jobs']:
        path = LOCAL / f"label{job['index']}.json"
        if not path.exists():
            value = record_base(job, p)
            value.update(bounded_error(reason), failure_stage='not_attempted', failure_reason=stop_reason,
                         deny_token=None, duration_seconds=None, new_tokens={})
            write_new(path, value)


def aggregate_records(p):
    bound = []
    groups = {g: dict(cases=16, captions=0, errors=0, not_attempted=0, semantic_review_pass=False)
              for g in ('diagnostic', 'blind')}
    failure_keys = {'error_type', 'error_message', 'failure_stage', 'failure_reason', 'deny_token',
                    'duration_seconds', 'new_tokens'}
    for job in p['jobs']:
        row = record(LOCAL / f"label{job['index']}.json")
        value = read(row)
        base = record_base(job, p, value.get('frames'))
        require(all(value.get(k) == v for k, v in base.items()), 'Exact full 32-record provenance coverage required')
        if 'error_type' in value:
            require(set(value) == set(base) | failure_keys
                and value['error_message'] == bounded_error(ValueError())['error_message']
                and value['failure_stage'] in CASE_STAGES + ('not_attempted',)
                and (value['failure_reason'] in SAFE_REASONS or value['failure_reason'] in STOP_REASONS
                     or re.fullmatch(r'unlisted_[A-Za-z][A-Za-z0-9_]{0,79}', value['failure_reason']))
                and (value['deny_token'] is None or re.fullmatch(r'[a-z# ]{1,24}', value['deny_token'])),
                'Safe bounded error record required')
            groups[job['group']]['errors'] += 1
            groups[job['group']]['not_attempted'] += value['failure_stage'] == 'not_attempted'
        else:
            require(set(value) == set(base) | {'facts_json', 'caption', 'duration_seconds', 'new_tokens'},
                    'Exact grounded result record required')
            validate_facts(value['facts_json'])
            validate_caption(value['caption'])
            require(value['frames']['displayed_indices'] == list(range(16))
                and value['frames']['source_indices'] == source_indices(121), 'Real source/display indices required')
            groups[job['group']]['captions'] += 1
        bound.append(row)
    return bound, groups


def verify_approval(path, digest, bound, p, end):
    # Reuse the existing publication format, not its single-training phase admission.
    import run_teacher_single_gpu_admission as single
    approval_row = pinned(path, digest, end)
    a = read(approval_row, end)
    keys = {'schema', 'private', 'GPU_authorized', 'plan', 'source_git_root', 'source_commit',
        'publication_manifest', 'remote_receipt', 'phase', 'formal_training_allowed', 'branch', 'snapshot_prefix'}
    require(set(a) == keys and a['schema'] == 'teacher_qwen3vl32b_recovery_execution_approval_v1'
        and a['private'] is True and a['GPU_authorized'] is True and a['plan'] == bound
        and a['phase'] == PHASE and a['formal_training_allowed'] is False,
        'Independent new label-scope execution approval required')
    commit = a['source_commit']
    require(type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit)
        and a['source_git_root'] == str(single.GIT_ROOT) and single.GIT_ROOT.resolve() == single.GIT_ROOT
        and a['branch'] == single.BRANCH and a['snapshot_prefix'] == single.SNAPSHOT_PREFIX,
        'Fixed actual full Git commit/root/branch/prefix required')
    manifest, receipt = read(a['publication_manifest'], end), read(a['remote_receipt'], end)
    require(receipt.get('commit', receipt.get('source_commit')) == commit
        and receipt.get('remote_equal') is True and receipt.get('manifest_sha256') == a['publication_manifest']['sha256'],
        'Actual new remote byte receipt required')
    def git(*args):
        deadline(end)
        r = subprocess.run([p['runtime']['tools']['git'], '-C', str(single.GIT_ROOT), *args],
            capture_output=True, timeout=min(30, max(.01, end-time.monotonic())))
        require(r.returncode == 0 and not r.stderr, 'Actual read-only Git verification failed')
        return r.stdout
    require(git('remote', 'get-url', 'origin').decode().strip() == single.ORIGIN,
            'Only the approved publication origin is permitted')
    require(git('rev-parse', commit + '^{commit}').decode().strip() == commit, 'Actual source commit missing')
    prefix = single.SNAPSHOT_PREFIX
    require(a['publication_manifest']['path'] == str(single.GIT_ROOT / prefix / 'PUBLICATION_MANIFEST.json')
        and hashlib.sha256(git('show', commit + ':' + prefix + '/PUBLICATION_MANIFEST.json')).hexdigest()
            == a['publication_manifest']['sha256'], 'Actual committed new manifest bytes differ')
    rows = manifest.get('files')
    require(type(rows) is list and rows and all(type(r) is dict and set(r) == {'path', 'source_sha256', 'snapshot_sha256'} for r in rows),
            'Exact full publication file manifest required')
    published = {r['path']: r for r in rows}
    require(len(published) == len(rows), 'Duplicate publication path')
    for name, row in p['source'].items():
        require(name in published and published[name]['source_sha256'] == row['sha256'],
                'New manifest does not cover all current needed source bytes')
        check(row, end)
    for r in rows:
        require(type(r['path']) is str and not Path(r['path']).is_absolute() and '..' not in Path(r['path']).parts
            and re.fullmatch('[0-9a-f]{64}', r['source_sha256'])
            and re.fullmatch('[0-9a-f]{64}', r['snapshot_sha256']), 'Malformed sanitized manifest row')
        require(hashlib.sha256(git('show', commit + ':' + prefix + '/' + r['path'])).hexdigest()
            == r['snapshot_sha256'], 'Actual full sanitized public snapshot bytes differ')
    changed = [os.fsdecode(raw) for raw in git('diff-tree', '--root', '--no-commit-id', '--name-only',
        '-r', '-z', commit, '--', prefix).split(b'\0') if raw]
    require(prefix + '/scripts/run_teacher_qwen3vl32b_recovery.py' in changed,
            'A new published runner commit is required, not a historical phase approval')
    require(type(receipt.get('manifest_files_verified')) is int and receipt['manifest_files_verified'] == len(rows)
        and type(receipt.get('changed_files_remote_bytes_verified')) is int
        and receipt['changed_files_remote_bytes_verified'] == len(changed) > 0,
        'Actual full-manifest and changed remote byte counts differ')
    require(all(n == prefix + '/PUBLICATION_MANIFEST.json' or n.startswith(prefix + '/')
        and n[len(prefix)+1:] in published for n in changed), 'Changed snapshot file omitted from manifest')
    require(git('ls-remote', 'origin', 'refs/heads/' + single.BRANCH).decode().splitlines()
        == [commit + '\trefs/heads/' + single.BRANCH], 'Actual current remote branch differs')
    return approval_row


def exact_flags(args, flags):
    return all(args.count(flag) == 1 and args.index(flag)+1 < len(args)
               and args[args.index(flag)+1] == value for flag, value in flags)


def same_identity(a, b):
    return all(a.get(k) == b.get(k) for k in ('pid', 'session', 'start_ticks', 'argv'))


def verify_worker_ancestry(launch, p, bound, ancestors, supervised):
    require(launch['plan'] == bound and launch['cap_seconds'] == CAP
        and launch['hostname'] == p['hostname'] == os.uname().nodename
        and launch['gpu_uuids'] == p['gpu_uuids'] and launch['state_dir'] == str(STATE_DIR),
        'Actual worker launch/host/four UUID scope differs')
    for key in ('start_monotonic', 'start_epoch'):
        require(type(launch[key]) in (int, float) and math.isfinite(launch[key]) and launch[key] > 0,
                'Actual allocation clocks required')
    parent = launch['parent']
    require(any(same_identity(a, parent) for a in ancestors), 'Actual approved parent ancestor required')
    args = parent['argv']
    require(args.count('--supplemental') == int(SUPPLEMENTAL) and args.count('--prompt-v2') == int(PROMPT_V2) and args.count('--constrained') == int(CONSTRAINED),
        'Actual experimental parent command binding differs')
    require(str(Path(__file__).resolve()) in args and 'run' in args and args.count('--gpu-authorized') == 1
        and exact_flags(args, (('--plan', bound['path']), ('--approved-plan-sha256', bound['sha256']),
            ('--approval', launch['approval']['path']), ('--approved-approval-sha256', launch['approval']['sha256']))),
        'Actual parent command must bind the exact independent approval and plan')
    require(supervised['launch'] == record(LOCAL / 'launch.json')
        and supervised['parent'] == parent, 'Actual immutable owned dispatch binding required')
    child = supervised['child']
    require(child['pid'] == child['session'] and child['ppid'] == parent['pid']
        and child['start_ticks'] > 0 and any(same_identity(a, child) for a in ancestors),
        'Actual newly supervised owned child session required')
    leases = []
    for a in ancestors:
        prefix = a['argv'][:a['argv'].index('--')] if '--' in a['argv'] else a['argv']
        if str(ROOT / 'scripts/gpu_hold.py') in prefix:
            leases.append(prefix)
    require(any(exact_flags(args, (('--action', 'exec'), ('--gpus', '0,1,2,3'),
        ('--state-dir', str(STATE_DIR)))) for args in leases),
        'All four exact GPU lease flags must occur in one actual ancestor prefix')
    require(any(Path(a['argv'][0]).name == 'timeout' and a['argv'][0] == p['runtime']['tools']['timeout']
        and '--signal=TERM' in a['argv'] and '--kill-after=20s' in a['argv']
        and f'{launch["workload_seconds"]}s' in a['argv'] for a in ancestors), 'Actual bounded GNU timeout ancestor required')
    require(type(launch['workload_seconds']) is int and 0 < launch['workload_seconds'] <= CAP-TAIL,
            'Workload may not spend cleanup reservation')
    return True


def boundary(p, idle=True, initial=None):
    _, process, _, _ = helpers()
    rows = [process.manager_boundary(p['state_dir'], i, u, require_idle=idle)
            for i, u in enumerate(p['gpu_uuids'])]
    manager = rows[0]['manager']
    require(all(same_identity(r['manager'], manager) for r in rows), 'One fresh manager PID/start required for all four GPUs')
    require(str(ROOT / 'scripts/gpu_hold.py') in manager['argv']
        and Path(manager['argv'][0]).resolve() == Path(PYTHON).resolve()
        and exact_flags(manager['argv'], (('--action', 'run'), ('--gpus', '0,1,2,3'),
        ('--state-dir', str(STATE_DIR)))) and manager['argv'].count('--monitor-only') == 1,
        'New fixed monitor-only four-GPU manager required')
    if initial is not None:
        require(same_identity(manager, initial[0]['manager']), 'Final manager PID/start/argv differs')
    return rows


def worker(path, digest, launch_path):
    require(Path(launch_path).absolute() == LOCAL / 'launch.json', 'Worker cannot use a foreign/old launch')
    launch = read(record(launch_path))
    end = launch['start_monotonic'] + CAP - TAIL
    deadline(end)
    p, bound = verify(path, digest, end)
    _, process, _, _ = helpers()
    require(process.same_process(launch['parent']), 'Approved parent is no longer live')
    ancestors, pid = [], os.getppid()
    while pid > 1:
        a = process.identity(pid)
        ancestors.append(a)
        pid = a['ppid']
    while not (LOCAL / 'supervised.json').is_file():
        deadline(end)
        require(process.same_process(launch['parent']), 'Parent vanished before owned-session publication')
        time.sleep(.01)
    supervised = read(record(LOCAL / 'supervised.json', end), end)
    verify_worker_ancestry(launch, p, bound, ancestors, supervised)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(p['gpu_uuids'])
        and os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1',
        'Exactly four UUIDs and native offline loaders required')
    fresh = boundary(p, idle=False, initial=launch['initial_boundary'])
    allowed = {a['pid'] for a in ancestors} | {os.getpid()}
    require(all(set(r['query']['pids']) <= allowed for r in fresh),
            'External GPU owner appeared before CUDA; do not overlap or kill it')
    stage_row = record(LOCAL / 'stage.json', end)
    staged = read(stage_row, end)
    require(staged['plan'] == bound and staged['GPU_used'] is False and len(staged['records']) == 32,
            'Actual CPU source-frame stage proof required')
    error, records, groups, load_seconds, mapping = None, [], None, None, None
    stop_reason, case_seconds, raw = None, [], None
    started = time.monotonic()
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        deadline(end)
        require(torch.cuda.device_count() == 4, 'Exactly four visible approved CUDA devices required')
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(MODEL_ROOT), local_files_only=True,
            torch_dtype=torch.bfloat16, device_map='balanced', max_memory={i: '34GiB' for i in range(4)},
            attn_implementation='sdpa')
        validate_device_map(model.hf_device_map)
        mapping = model.hf_device_map
        used = set()
        for _, parameter in model.named_parameters():
            require(parameter.device.type == 'cuda' and parameter.device.index in range(4)
                    and parameter.dtype == torch.bfloat16, 'Actual model weights cannot offload or quantize')
            used.add(parameter.device.index)
        require(used == {0, 1, 2, 3}, 'Actual loaded weights must occupy all four approved CUDA devices')
        processor = AutoProcessor.from_pretrained(str(MODEL_ROOT), local_files_only=True)
        load_seconds = time.monotonic() - started
        device = model.get_input_embeddings().weight.device
        previous_max = 0.0
        for job, stage_item in zip(p['jobs'], staged['records']):
            deadline(end)
            if not end-time.monotonic() > previous_max * (len(p['jobs'])-job['index']) + 10:
                # Worst observed case time cannot fit all remaining fixed cases: stop, no retry.
                stop_reason = 'projected_budget_insufficient'
                break
            began, frames, stage_code, tokens = time.monotonic(), None, 'local_copy_verify', {}
            try:
                require(stage_item['index'] == job['index'], 'Ordered real source-frame stage identity differs')
                local = record(job['local_video'], end)
                require(local == stage_item['local_video'] and local['sha256'] == job['video']['sha256'],
                        'Actual local-copy/source-video SHA differs')
                stage_code = 'frame_decode'
                images, frames = sample_images(job['local_video'], p['runtime']['tools'], end)
                require(frames == stage_item['frames'], 'Actual decoded source frame indices differ')
                stage_code, trace = 'facts_generate', {}
                raw = generate_text(model, processor, images, device, 1024, trace=trace)
                tokens['facts'] = trace['new_tokens']
                stage_code = 'facts_parse'
                facts = parse_facts(raw)
                deadline(end)
                stage_code, trace = 'caption_generate', {}
                raw = generate_text(model, processor, images, device, 384, facts, trace=trace)
                tokens['caption'] = trace['new_tokens']
                stage_code = 'caption_validate'
                text = validate_caption(raw)
                deadline(end)
                item = dict(record_base(job, p, frames), facts_json=facts, caption=text,
                            duration_seconds=time.monotonic()-began, new_tokens=tokens)
            except Exception as exc:
                if stage_code in ('facts_generate', 'caption_generate') and 'new_tokens' in trace:
                    tokens['facts' if stage_code == 'facts_generate' else 'caption'] = trace['new_tokens']
                item = dict(record_base(job, p, frames), **case_failure(exc, stage_code, time.monotonic()-began, tokens))
            raw = None
            write_new(LOCAL / f"label{job['index']}.json", item)
            case_seconds.append(item['duration_seconds'])
            previous_max = max(previous_max, time.monotonic()-began)
        else:
            stop_reason = 'completed'
        if stop_reason != 'completed':
            complete_error_records(p, ValueError(), stop_reason)
        del model, processor
        records, groups = aggregate_records(p)
        require(all(g['errors'] == 0 for g in groups.values()), 'At least one case failed; no automatic retry')
    except BaseException as exc:
        error = bounded_error(exc)
        if stop_reason is None:
            stop_reason = ('model_load_failed' if load_seconds is None else
                'allocated_window_exhausted' if isinstance(exc, TimeoutError) else 'worker_exception')
        complete_error_records(p, exc, stop_reason if stop_reason != 'completed' else 'worker_exception')
        records, groups = aggregate_records(p)
    result = dict(schema='teacher_qwen3vl32b_recovery_worker_v1', plan=bound, launch=record(launch_path),
        worker_identity=process.identity(os.getpid()), stage=stage_row, records=records, groups=groups,
        load_seconds=load_seconds, hf_device_map=mapping, error=error, success=error is None,
        stop_reason=stop_reason, cases_attempted=len(case_seconds), case_seconds=case_seconds,
        semantic_review_pass=False, independent_blind_review_required=True, formal_training_allowed=False)
    row = write_new(LOCAL / 'worker.json', result)
    require(error is None, 'Worker failed; safe 32-case evidence retained; no retry')
    return row


def release_queries_valid(queries, uuids, earliest):
    import run_teacher_single_gpu_admission as single
    require(type(queries) is list and len(queries) == 8, 'Exactly two sequential actual queries per GPU required')
    previous = None
    for i, u in enumerate(uuids):
        pair = queries[i*2:i*2+2]
        require(previous is None or previous < pair[0]['start_monotonic'], 'All raw release queries must be distinct and ordered')
        single.release_queries_valid(pair, u, earliest)
        previous = pair[-1]['end_monotonic']
    return True


def verify_release_boundary(final, initial, queries, uuids, start):
    import run_teacher_single_gpu_admission as single
    require(len(final) == len(initial) == 4 and len(queries) == 8, 'Actual all-four release evidence required')
    previous = queries[-1]['end_monotonic']
    for i, (row, uuid) in enumerate(zip(final, uuids)):
        q = row['query']
        require(same_identity(row['manager'], initial[0]['manager'])
            and row['gpu_uuid'] == uuid and previous < q['start_monotonic'],
            'Third fresh ordered query and same manager required')
        single.release_queries_valid([queries[i*2+1], q], uuid, start)
        previous = q['end_monotonic']
    return True


def allocation_cost(start, released=False):
    elapsed = time.monotonic() - start
    require(math.isfinite(elapsed) and elapsed >= 0, 'Finite hostlocal allocated clock required')
    charged = max(CAP, elapsed)
    return dict(elapsed_seconds=elapsed, charged_upper_seconds=charged, allocated_GPUh=CARDS*charged/3600,
        actual_allocated_GPUh=CARDS*elapsed/3600 if released else None, allocation_GPUh_cap=ALLOCATION_GPUH,
        prior_charged_label_GPUh=PRIOR_GPUH, cumulative_label_upper_GPUh=PRIOR_GPUH+CARDS*charged/3600,
        allocated_cards=CARDS, cap_seconds=CAP, released_observed=released,
        budget_exceeded=charged > CAP, clock_host=os.uname().nodename,
        scope='first_prelaunch_hash_through_load_outputs_cleanup_and_terminal_save_return')


def verify_cost_closure(path, digest):
    row = pinned(path, digest)
    root = Path(row['path']).parent
    require(root == LOCAL and Path(path).name == 'cost.closure.json' and not (root / 'cost.failure.json').exists(),
            'New same-root successful terminal closure required')
    c = read(row)
    require(c.get('schema') == 'teacher_qwen3vl32b_recovery_cost_closure_v1' and c.get('phase') == PHASE
        and c.get('success') is True and c.get('released') is True and c.get('terminal') is True
        and c.get('clock_host') == os.uname().nodename and c.get('cap_seconds') == CAP
        and c.get('allocated_cards') == CARDS and c.get('charged_upper_seconds') == CAP
        and c.get('allocation_GPUh_upper') == ALLOCATION_GPUH and c.get('prior_charged_label_GPUh') == PRIOR_GPUH
        and c.get('publication_exit_allowance_seconds') == 1 and c.get('semantic_review_pass') is False,
        'Exact same-host 6GPUh upper-bound terminal closure required')
    for k in ('started_monotonic', 'observed_monotonic', 'actual_elapsed_seconds'):
        require(type(c.get(k)) in (int, float) and math.isfinite(c[k]), 'Actual finite terminal clocks required')
    require(c['actual_elapsed_seconds'] == c['observed_monotonic']-c['started_monotonic']
        and 0 <= c['actual_elapsed_seconds']+1 <= CAP, 'Terminal allocated wall and publication allowance exceed cap')
    p = read(c['plan'])
    parent, cost = read(c['parent']), read(c['cost'])
    require(p['schema'] == SCHEMA and p['phase'] == PHASE and p['output_root'] == str(LOCAL)
        and c['parent']['path'] == str(LOCAL / 'parent.json') and c['cost']['path'] == str(LOCAL / 'cost.json')
        and parent['plan'] == cost['plan'] == c['plan'] and parent['started_monotonic'] == c['started_monotonic']
        and parent['execution_success'] is True and parent['success'] is False and parent['error'] is None
        and parent['released'] is True and parent['interrupted_signals'] == []
        and cost['provisional'] is True and cost['success'] is False and cost['charged_upper_seconds'] >= CAP
        and cost['parent_receipt'] == c['parent'], 'Original plan/parent/provisional cost bytes must bind closure')
    require(parent['child']['pid'] == parent['child']['session'] and parent['tree_released'] is True,
            'Actual owned session release evidence required')
    release_queries_valid(parent['release_queries'], p['gpu_uuids'], c['started_monotonic'])
    verify_release_boundary(parent['final_boundary'], parent['initial_boundary'],
        parent['release_queries'], p['gpu_uuids'], c['started_monotonic'])
    worker_result = read(parent['worker'])
    records, groups = aggregate_records(p)
    require(worker_result['plan'] == c['plan'] and worker_result['success'] is True and worker_result['error'] is None
        and worker_result['records'] == records and worker_result['groups'] == groups
        and all(g['captions'] == 16 and g['errors'] == 0 and g['semantic_review_pass'] is False for g in groups.values()),
        'Actual 32 captions with separate diagnostic/blind nonsemantic coverage required')
    require(not (root / 'cost.failure.json').exists(), 'Terminal failure invalidates an earlier provisional closure')
    return c


def run(path, digest, approval_path, approval_digest, gpu_authorized=False):
    require(gpu_authorized is True and threading.current_thread() is threading.main_thread(),
            'Only explicit authorized main-thread parent may allocate GPUs')
    start, epoch = time.monotonic(), time.time()
    p, bound, approval, process, parent, child, owner = None, None, None, None, None, None, None
    initial, final, queries, tracked, cleanup_signals = [], [], [], [], []
    interrupted, old_handlers, error = [], {}, None
    released, tree_released, attempt_owned, verified, terminal_success = False, False, False, False, False
    worker_row, final_row, lock = None, None, None
    def stop(sig, _frame):
        if sig not in interrupted:
            interrupted.append(sig)
    def alarm(sig, frame):
        stop(sig, frame)
        raise TimeoutError('Allocation cap reached; unclosed evidence cannot pass')
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            old_handlers[sig] = signal.signal(sig, alarm if sig == signal.SIGALRM else stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, start+CAP-TAIL-time.monotonic()))
        require(LOCAL.resolve() == LOCAL and LOCAL.is_dir() and (LOCAL / 'stage.json').is_file(), 'Prior CPU-only new media stage required')
        # Shared cross-node claim, then local claim, before the first hash: one recovery only.
        require(RECOVERY_EVIDENCE.resolve() == RECOVERY_EVIDENCE and RECOVERY_EVIDENCE.is_dir(), 'Shared recovery evidence root required')
        claim = (RECOVERY_EVIDENCE / 'attempt.claim').open('xb')
        claim.write(encoded(dict(hostname=os.uname().nodename, node_ip=os.environ.get('NODE_IP'),
            pid=os.getpid(), started_epoch=epoch, local_root=str(LOCAL))))
        claim.close()
        lock = (LOCAL / 'attempt.lock').open('xb')
        attempt_owned = True
        bound = pinned(path, digest, start+CAP-TAIL)
        p = read(bound, start+CAP-TAIL)
        require(p.get('schema') == SCHEMA and p.get('phase') == PHASE and p.get('hostname') == os.uname().nodename
            and p.get('output_root') == str(LOCAL), 'New same-host label allocation required')
        p, bound = verify(path, digest, start+CAP-TAIL)
        verified = True
        staged = read(record(LOCAL / 'stage.json', start+CAP-TAIL), start+CAP-TAIL)
        require(staged['plan'] == bound and staged['GPU_used'] is False and len(staged['records']) == 32,
                'Actual complete CPU frame stage required')
        require(not any((LOCAL / name).exists() for name in ('launch.json', 'supervised.json', 'parent.json', 'worker.json', 'cost.json', 'cost.failure.json'))
            and not list(LOCAL.glob('label*.json')), 'Never reuse prior allocation or label evidence')
        _, process, _, _ = helpers()
        parent = process.identity(os.getpid())
        approval = verify_approval(approval_path, approval_digest, bound, p, start+CAP-TAIL)
        initial = boundary(p)
        version = subprocess.run([p['runtime']['tools']['timeout'], '--version'], capture_output=True, timeout=10)
        require(version.returncode == 0 and b'GNU coreutils' in version.stdout, 'Actual GNU timeout required')
        seconds = int(start+CAP-TAIL-time.monotonic())
        require(0 < seconds <= CAP-TAIL and not interrupted, 'No remaining bounded worker window')
        launch_row = write_new(LOCAL / 'launch.json', dict(schema='teacher_qwen3vl32b_recovery_launch_v1', plan=bound,
            parent=parent, approval=approval, hostname=p['hostname'], gpu_uuids=p['gpu_uuids'], state_dir=p['state_dir'],
            start_monotonic=start, start_epoch=epoch, cap_seconds=CAP, workload_seconds=seconds, initial_boundary=initial))
        command = [p['runtime']['python'], str(ROOT / 'scripts/gpu_hold.py'), '--action', 'exec',
            '--gpus', '0,1,2,3', '--state-dir', p['state_dir'], '--', p['runtime']['tools']['timeout'],
            '--signal=TERM', '--kill-after=20s', f'{seconds}s', p['runtime']['python'], str(Path(__file__).resolve()),
            'worker', '--plan', bound['path'], '--approved-plan-sha256', digest, '--launch', launch_row['path']]
        if SUPPLEMENTAL:
            command.insert(command.index('worker'), '--supplemental')
        if PROMPT_V2:
            command.insert(command.index('worker'), '--prompt-v2')
        if CONSTRAINED:
            command.insert(command.index('worker'), '--constrained')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(p['gpu_uuids']), HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1')
        with (LOCAL / 'worker.log').open('xb') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            owner = process.identity(child.pid)
            require(owner['pid'] == owner['session'] == child.pid and owner['ppid'] == parent['pid'], 'Actual fresh owned child session required')
            write_new(LOCAL / 'supervised.json', dict(launch=launch_row, parent=parent, child=owner,
                observed_monotonic=time.monotonic(), command=command))
            while child.poll() is None and not interrupted and time.monotonic() < start+CAP-TAIL:
                process.owned_members(owner, tracked)
                time.sleep(.05)
        require(not interrupted and child.poll() == 0, 'Worker failed or timed out; one attempt only')
        worker_row = record(LOCAL / 'worker.json', start+CAP-TAIL)
        result = read(worker_row, start+CAP-TAIL)
        records, groups = aggregate_records(p)
        require(result['plan'] == bound and result['success'] is True and result['error'] is None
            and result['records'] == records and result['groups'] == groups
            and all(g['captions'] == 16 and g['errors'] == 0 for g in groups.values()), 'Full ordered 32-caption coverage required')
    except BaseException as exc:
        error = bounded_error(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, max(.001, start+CAP-time.monotonic()))
        try:
            if child is not None:
                try:
                    if owner is None:
                        if child.poll() is None:
                            try:
                                candidate = process.identity(child.pid)
                                require(candidate['pid'] == candidate['session'] == child.pid
                                    and candidate['ppid'] == os.getpid(), 'Cannot recover actual owned child identity')
                                owner = candidate
                            except (OSError, ValueError, KeyError):
                                child.terminate()
                                try:
                                    child.wait(timeout=min(20, max(.01, start+CAP-90-time.monotonic())))
                                except subprocess.TimeoutExpired:
                                    child.kill()
                                    child.wait(timeout=2)
                        require(owner is not None, 'Unreaped Popen child cleaned, but original tree identity unknown; cap retained')
                    cleanup_signals = process.cleanup_owned(child, owner, start+CAP-90, tracked)
                    tree_released = not process.owned_members(owner, tracked)
                    for u in p['gpu_uuids']:
                        queries.extend([process.query_release(u), process.query_release(u)])
                    release_queries_valid(queries, p['gpu_uuids'], start)
                    final = boundary(p, idle=False, initial=initial)
                    verify_release_boundary(final, initial, queries, p['gpu_uuids'], start)
                    require(all(r['query']['pids'] == [] for r in final) and tree_released,
                            'Actual third query/owned-tree release failed; never kill foreign owners')
                    released = True
                except BaseException as exc:
                    error = bounded_error(exc)
                    released = False
            # No spawn proves no GPU work, but never reduces conservative failure charges.
            if verified and attempt_owned:
                complete_error_records(p, ValueError())
            if attempt_owned:
                receipt = dict(schema='teacher_qwen3vl32b_recovery_parent_v1', phase=PHASE, plan=bound, approval=approval,
                    parent=parent, child=owner, worker=worker_row, initial_boundary=initial, final_boundary=final,
                    release_queries=queries, cleanup_signals=cleanup_signals, tree_released=tree_released,
                    released=released, error=error, interrupted_signals=interrupted, no_child_spawned=child is None,
                    started_monotonic=start, started_epoch=epoch, observed_monotonic=time.monotonic(),
                    execution_success=error is None and released and worker_row is not None and not interrupted,
                    success=False, provisional=True, completion_requires='cost.closure.json',
                    semantic_review_pass=False, formal_training_allowed=False)
                parent_row, cost_row = None, None
                try:
                    parent_row = write_new(LOCAL / 'parent.json', receipt)
                    provisional = allocation_cost(start)
                    provisional.update(plan=bound, parent_receipt=parent_row, provisional=True, success=False)
                    cost_row = write_new(LOCAL / 'cost.json', provisional)
                    require(receipt['execution_success'] and not interrupted, 'Failed allocation remains full-cap reserved')
                    observed = time.monotonic()
                    require(observed-start+1 <= CAP, 'No bounded terminal publication tail remains')
                    closure = dict(schema='teacher_qwen3vl32b_recovery_cost_closure_v1', phase=PHASE, plan=bound,
                        parent=parent_row, cost=cost_row, terminal=True, success=True, released=True,
                        clock_host=os.uname().nodename, started_monotonic=start, observed_monotonic=observed,
                        actual_elapsed_seconds=observed-start, charged_upper_seconds=CAP, allocated_cards=CARDS,
                        allocation_GPUh_upper=ALLOCATION_GPUH, prior_charged_label_GPUh=PRIOR_GPUH,
                        publication_exit_allowance_seconds=1, cap_seconds=CAP, semantic_review_pass=False)
                    closure_row = write_new(LOCAL / 'cost.closure.json', closure)
                    verify_cost_closure(closure_row['path'], closure_row['sha256'])
                    final_row = record(closure_row['path'])
                    finished = time.monotonic()
                    require(finished-observed <= 1 and finished-start <= CAP and not interrupted,
                            'Actual terminal save/verify/return hash exceeded one-second tail or total cap')
                    terminal_success = True
                except BaseException as exc:
                    error = bounded_error(exc)
                    failure = allocation_cost(start, released)
                    failure.update(schema='teacher_qwen3vl32b_recovery_cost_failure_v1', phase=PHASE, plan=bound,
                        parent_receipt=parent_row, cost=cost_row, terminal=True, success=False, error=error)
                    write_new(LOCAL / 'cost.failure.json', failure)
            else:
                failure = allocation_cost(start)
                failure.update(schema='teacher_qwen3vl32b_recovery_cost_failure_v1', phase=PHASE, plan=bound,
                    terminal=True, success=False, error=error, no_child_spawned=child is None)
                write_new(RECOVERY_EVIDENCE / ('qwen3vl32b.allocation-failure.' + uuid.uuid4().hex + '.json'), failure)
        except BaseException as exc:
            # Identity, error-fill, directory and terminal publication failures also retain the cap.
            terminal_success = False
            signal.setitimer(signal.ITIMER_REAL, 0)
            failure = allocation_cost(start, released)
            failure.update(schema='teacher_qwen3vl32b_recovery_cost_failure_v1', phase=PHASE, plan=bound,
                terminal=True, success=False, error=bounded_error(exc), no_child_spawned=child is None)
            destination = LOCAL / 'cost.failure.json' if attempt_owned else RECOVERY_EVIDENCE / (
                'qwen3vl32b.allocation-failure.' + uuid.uuid4().hex + '.json')
            if destination.exists():
                destination = RECOVERY_EVIDENCE / ('qwen3vl32b.terminal-failure.' + uuid.uuid4().hex + '.json')
            write_new(destination, failure)
        finally:
            if lock is not None:
                lock.close()
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    require(terminal_success, 'Recovery failed; conservative four-card costs retained; no retry')
    return final_row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--supplemental', action='store_true', help='Separately authorized second allocation; original claim and costs preserved')
    modes.add_argument('--prompt-v2', action='store_true', help='New positive-output prompt experiment; original validation gates unchanged')
    modes.add_argument('--constrained', action='store_true', help='Prospective vocabulary/EOS decoder constraints, not posthoc label repair')
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--model-proof', required=True)
    prep.add_argument('--output', required=True)
    prep.add_argument('--gpu-uuids', required=True, help='Four explicit GPU0..3 UUIDs in order')
    proof = commands.add_parser('model-proof')
    proof.add_argument('--output', required=True)
    for name in ('stage', 'verify', 'run', 'worker'):
        sub = commands.add_parser(name)
        sub.add_argument('--plan', required=True)
        sub.add_argument('--approved-plan-sha256', required=True)
        if name == 'run':
            sub.add_argument('--approval', required=True)
            sub.add_argument('--approved-approval-sha256', required=True)
            sub.add_argument('--gpu-authorized', action='store_true')
        elif name == 'worker':
            sub.add_argument('--launch', required=True)
    closure = commands.add_parser('verify-closure')
    closure.add_argument('--closure', required=True)
    closure.add_argument('--approved-closure-sha256', required=True)
    args = parser.parse_args(argv)
    if args.supplemental:
        configure_supplemental()
    if args.prompt_v2:
        configure_prompt_v2()
    if args.constrained:
        configure_constrained()
    if args.command == 'run':
        canonical_launch([sys.executable, sys.argv[0], *(sys.argv[1:] if argv is None else argv)])
    if args.command == 'model-proof':
        result = model_proof(args.output)
    elif args.command == 'prepare':
        result = prepare(args.model_proof, args.output, args.gpu_uuids.split(','))
    elif args.command == 'verify':
        p, row = verify(args.plan, args.approved_plan_sha256)
        result = dict(plan=row, CPU_only=True, semantic_review_pass=False, GPU_authorized=False)
    elif args.command == 'stage':
        result = stage(args.plan, args.approved_plan_sha256)
    elif args.command == 'worker':
        result = worker(args.plan, args.approved_plan_sha256, args.launch)
    elif args.command == 'run':
        result = run(args.plan, args.approved_plan_sha256, args.approval, args.approved_approval_sha256,
                     args.gpu_authorized)
    else:
        result = verify_cost_closure(args.closure, args.approved_closure_sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps(bounded_error(exc), sort_keys=True), file=sys.stderr)
        sys.exit(1)
