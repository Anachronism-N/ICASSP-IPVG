"""CPU protocol/admission tests for the one new-cluster recovery; not model/GPU validation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_teacher_qwen3vl32b_recovery as pilot

FACTS = dict(scene='A plain indoor room with a table.', rendered_or_screen=False,
    entities=[dict(id='E1', description='A person wearing a dark shirt', held_object=None),
              dict(id='E2', description='A person wearing a light shirt', held_object='A small dark object')],
    events=[dict(entity_id='E1', frame_start=0, frame_end=15, action='Turns the head toward the table'),
            dict(entity_id='E2', frame_start=2, frame_end=13, action='Raises the visible right arm with the dark object')],
    start_end_difference='The first person ends facing the table while the second person ends with the arm raised.')
CAPTION = ('A person in a dark shirt stands beside a table in a plain indoor room, with another person '
    'wearing a light shirt nearby. The first person turns their head toward the table while the second '
    'raises a visible arm with a small dark object. The table and background remain in the same positions.')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope='session')
def synthetic_evidence(tmp_path_factory):
    # Protocol-only fixtures: never read or publish private authorizations, captions or media.
    root = tmp_path_factory.mktemp('qwen32b_synthetic_evidence')
    original_path, new_path, scope_path = [root / n for n in (
        'synthetic_original.json', 'synthetic_new.json', 'synthetic_scope.json')]
    original = dict(schema='teacher_training_admission_authorization_v1',
        synthetic_test_only=True, stage_caps_GPUh={'label_rebuild_and_cache': 24,
        'single_gpu_training_resume': 2, 'multi_node_technical_validation': 6},
        total_cap_GPUh=32, failed_allocations_charged=True,
        formal_candidate_training_budget_approved=False, phase_borrowing_allowed=False,
        AR_allowed=False, student_or_distillation_allowed=False,
        previous_experiment_budgets_reused=False, training_started=False)
    new = dict(schema='teacher_new_captioner_pilot_authorization_v1', synthetic_test_only=True,
        user_choice='换标注器做新pilot', original_stage_cap_GPUh=24,
        prior_label_GPUh=0.1658982108534676, phase_borrowing_allowed=False,
        automatic_retry_allowed=False, full_rebuild_allowed=False, formal_training_allowed=False,
        failed_evidence_preserved=True, pilot_must_freeze_source_budget_and_samples_before_GPU=True,
        semantic_review_required=True)
    scope = dict(schema='teacher_stronger_captioner_pilot_scope_v1',
        authorized_user_choice='换标注器做新pilot', original_authorization=str(new_path),
        stage='label_rebuild_and_cache', stage_cap_GPUh=24, prior_charged_label_GPUh=0.1658982108534676,
        allocation_GPUh_cap=2, allocated_cards=4, wall_cap_seconds=1800, node_rank=1,
        gpu_indices=[0, 1, 2, 3], model='Qwen3-VL-32B-Instruct', model_root=pilot.OLD_MODEL_ROOT,
        dtype='bfloat16', weight_quantization=False, model_shards=14, frames_per_case=16,
        frame_resize=[640, 352], decoding='greedy,do_sample=false', cases=32,
        case_selection='16previous_diagnostic+16newblind_no_benchmark_scores', selection_seed=20261102,
        max_new_tokens_facts=1024, max_new_tokens_caption=384,
        stages=['grounded_entity_ID_and_visible_events', 'caption_with_same_images_and_facts'],
        full_model_CPU_or_disk_offload_allowed=False, automatic_retry_allowed=False,
        all_failure_costs_charged=True, new_output_root_required=True,
        actual_GPU_entry_requires_source_publication_and_asset_lease_checks=True,
        independent_semantic_review_required=True, full_rebuild_allowed=False,
        formal_training_allowed=False, AR_allowed=False, student_distillation_allowed=False)
    for path, value in zip((original_path, new_path, scope_path), (original, new, scope)):
        path.write_bytes(pilot.encoded(value))
    rows = []
    for i in range(1856):
        video = root / f'fixture_video{i}.mp4'
        video.write_bytes(f'Synthetic metadata-only media bytes {i}\n'.encode())
        prompt = f'A plain object rests on a synthetic table in case {i}.'
        rows.append(dict(sample_id=f'fixture{i}', video=str(video), captions=dict(
            orig=prompt, fixed=prompt if i < 1815 else 'private metadata fixture')))
    manifest = root / 'synthetic_manifest.jsonl'
    manifest.write_bytes(b''.join(pilot.encoded(row) for row in rows))
    import random
    diagnostic = random.Random(20261101).sample(rows[:1815], 16)
    old_path = root / 'synthetic_old_plan.json'
    old_path.write_bytes(pilot.encoded(dict(schema='teacher_label_pilot_execution_v2',
        selection_seed=20261101, manifest=pilot.record(manifest), jobs=[dict(
            index=i, rank=i//4, sample_id=row['sample_id'], video=pilot.record(row['video']),
            original_prompt_sha256=sha(row['captions']['orig'].encode()))
            for i, row in enumerate(diagnostic)])))
    prep, _, _, _ = pilot.helpers()
    _, parsed = prep.manifest_rows(manifest)
    chosen = pilot.selected_rows(parsed, pilot.read(pilot.record(old_path)))
    first_path = root / 'synthetic_first_plan.json'
    first_path.write_bytes(pilot.encoded(dict(schema='teacher_qwen3vl32b_pilot_plan_v1',
        config=dict(facts_prompt_sha256=pilot.fixed_config()['facts_prompt_sha256'],
                    caption_prompt_sha256=pilot.fixed_config()['caption_prompt_sha256']),
        jobs=[dict(index=i, rank=1, group='diagnostic' if i < 16 else 'blind', sample_id=row['sample_id'],
            video=pilot.record(row['video']), original_prompt_sha256=sha(row['captions']['orig'].encode()),
            local_video='/old/video%d.mp4' % i) for i, row in enumerate(chosen)])))
    recovery_path = root / 'synthetic_recovery.json'
    recovery_path.write_bytes(pilot.encoded(dict(schema='teacher_qwen3vl32b_one_recovery_authorization_v1',
        user_choice='批准一次6 GPUh恢复', original_scope_authorization=str(scope_path),
        stage='label_rebuild_and_cache', stage_cap_GPUh=24, new_allocation_GPUh_cap=6, allocated_cards=4,
        wall_cap_seconds=5400, node_rank=1, gpu_indices=[0, 1, 2, 3], frames_per_case=16, cases=32,
        original_fixed_cases_preserved=True, old_pilot_charged_upper_GPUh=2,
        all_previous_label_charged_upper_GPUh=2.1658982108534675,
        cumulative_label_upper_GPUh_cap=8.165898210853468, one_recovery_attempt=True,
        automatic_retry_allowed=False, new_output_root_required=True,
        old_raw_logs_labels_and_costs_preserved=True, safe_failure_stage_and_case_duration_required=True,
        new_source_runtime_publication_gate_required=True, full_rebuild_allowed=False,
        formal_training_allowed=False, AR_allowed=False, student_distillation_allowed=False)))
    cluster_path = root / 'synthetic_cluster.json'
    cluster_path.write_bytes(pilot.encoded(dict(schema='teacher_qwen3vl32b_new_cluster_recovery_authorization_v1',
        synthetic_test_only=True,
        original_recovery_authorization=dict(path=str(recovery_path), sha256=sha(recovery_path.read_bytes())),
        first_pilot_plan=dict(path=str(first_path), sha256=sha(first_path.read_bytes())),
        old_host_gpu_paths_are_execution_credentials=False, recovery_attempts_before_migration=0,
        recovery_count_reset_by_migration=False, one_recovery_attempt=True, nodes=1, allocated_cards=4,
        dtype='bfloat16', cases=32, fixed_groups=[16, 16], frames_per_case=16, wall_cap_seconds=5400,
        new_allocation_GPUh_cap=6, stage_cap_GPUh=24, all_previous_label_charged_upper_GPUh=2.1658982108534675,
        cumulative_label_upper_GPUh_cap=8.165898210853468, automatic_retry_allowed=False,
        sample_substitution_allowed=False, frame_reduction_allowed=False,
        safe_failure_stage_case_duration_and_stop_reason_required=True, new_output_root_required=True,
        new_host_runtime_model_source_freeze_required=True, publication_gate_required=True,
        foreign_gpu_workloads_may_be_killed=False, full_rebuild_allowed=False, formal_training_allowed=False,
        AR_allowed=False, student_distillation_allowed=False)))
    return dict(AUTH=original_path, AUTH_SHA=sha(original_path.read_bytes()),
        NEW_AUTH=new_path, NEW_AUTH_SHA=sha(new_path.read_bytes()),
        SCOPE_AUTH=scope_path, SCOPE_AUTH_SHA=sha(scope_path.read_bytes()),
        MANIFEST=manifest, OLD_PLAN=old_path, OLD_PLAN_SHA=sha(old_path.read_bytes()),
        RECOVERY_AUTH=recovery_path, RECOVERY_AUTH_SHA=sha(recovery_path.read_bytes()),
        CLUSTER_AUTH=cluster_path, CLUSTER_AUTH_SHA=sha(cluster_path.read_bytes()),
        FIRST_PLAN=first_path, FIRST_PLAN_SHA=sha(first_path.read_bytes()))


@pytest.fixture(autouse=True)
def isolated_protocol_evidence(synthetic_evidence, monkeypatch):
    # Patch only fixture paths and their actual raw pins; validation functions stay real.
    for name, value in synthetic_evidence.items():
        monkeypatch.setattr(pilot, name, value)


def test_top_level_standard_library_only_and_help():
    r = subprocess.run([sys.executable, '-c',
        'import sys;sys.path.insert(0,' + repr(str(pilot.ROOT / 'scripts')) + ');'
        'import run_teacher_qwen3vl32b_recovery;assert "torch" not in sys.modules;'
        'assert "transformers" not in sys.modules;assert "numpy" not in sys.modules'], capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    for command in ('model-proof', 'prepare', 'stage', 'verify', 'run', 'worker', 'verify-closure'):
        r = subprocess.run([sys.executable, str(Path(pilot.__file__)), command, '--help'], capture_output=True)
        assert r.returncode == 0


def test_synthetic_authorization_contract_and_actual_raw_hashes():
    rows = pilot.authorization_rows()
    assert [r['sha256'] for r in rows] == [pilot.AUTH_SHA, pilot.NEW_AUTH_SHA, pilot.SCOPE_AUTH_SHA,
        pilot.RECOVERY_AUTH_SHA, pilot.CLUSTER_AUTH_SHA]
    assert pilot.PRIOR_GPUH + pilot.ALLOCATION_GPUH == pytest.approx(pilot.CUMULATIVE_CAP_GPUH, abs=1e-12)
    assert all(sha(Path(r['path']).read_bytes()) == r['sha256'] for r in rows)
    assert pilot.PRIOR_GPUH + pilot.CARDS * pilot.CAP / 3600 < 24


@pytest.mark.parametrize('which,key,bad', [(2, 'allocated_cards', 1), (2, 'allocation_GPUh_cap', 8),
    (2, 'stage', 'single_gpu_training_resume'), (2, 'wall_cap_seconds', 7200),
    (2, 'prior_charged_label_GPUh', 0), (2, 'node_rank', True), (2, 'gpu_indices', [0, 1, 2, 4]),
    (2, 'frames_per_case', 8), (2, 'automatic_retry_allowed', True), (2, 'cases', 16),
    (1, 'phase_borrowing_allowed', True), (0, 'total_cap_GPUh', 32.0),
    (3, 'wall_cap_seconds', 1800), (3, 'new_allocation_GPUh_cap', 8), (3, 'one_recovery_attempt', False),
    (3, 'all_previous_label_charged_upper_GPUh', 0.1658982108534676), (3, 'automatic_retry_allowed', True),
    (4, 'recovery_count_reset_by_migration', True), (4, 'recovery_attempts_before_migration', 1),
    (4, 'frame_reduction_allowed', True), (4, 'sample_substitution_allowed', True),
    (4, 'foreign_gpu_workloads_may_be_killed', True), (4, 'wall_cap_seconds', 7200),
    (4, 'first_pilot_plan', {'path': 'other', 'sha256': '0'*64})])
def test_authorization_scope_drift_rejected(which, key, bad):
    values = [pilot.read(r) for r in pilot.authorization_rows()]
    values[which][key] = bad
    with patch.object(pilot, 'read', side_effect=values), pytest.raises(ValueError):
        pilot.authorization_rows()


def test_original_selection_blind_fixed_no_score_or_previous_output():
    prep, _, _, _ = pilot.helpers()
    _, rows = prep.manifest_rows(pilot.MANIFEST)
    old = pilot.read(pilot.pinned(pilot.OLD_PLAN, pilot.OLD_PLAN_SHA))
    chosen = pilot.selected_rows(rows, old)
    assert len(chosen) == len({r['sample_id'] for r in chosen}) == 32
    assert [r['sample_id'] for r in chosen[:16]] == [j['sample_id'] for j in old['jobs']]
    assert not set(r['sample_id'] for r in chosen[:16]) & set(r['sample_id'] for r in chosen[16:])
    assert chosen == pilot.selected_rows(rows, old)
    altered = copy.deepcopy(old)
    altered['jobs'][0]['original_prompt_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        pilot.selected_rows(rows, altered)
    with patch.object(pilot, 'SEED', 20261103):
        assert pilot.selected_rows(rows, old)[16:] != chosen[16:]
    # The existing raw manifest parser refuses benchmark/output side channels.
    row = copy.deepcopy(rows[0])
    row['benchmark_score'] = 1
    with patch.object(prep, 'bound', return_value=({}, b'\n'.join(pilot.encoded(row).strip() for _ in range(1856)))):
        with pytest.raises(ValueError):
            prep.manifest_rows(pilot.MANIFEST)


def test_exact_source_uniform_indices():
    assert pilot.source_indices(121) == [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]
    for n in (8, 15, True, 121.0):
        with pytest.raises(ValueError):
            pilot.source_indices(n)


def test_valid_typed_facts_and_caption():
    assert pilot.validate_facts(copy.deepcopy(FACTS)) == FACTS
    assert pilot.parse_facts('\n' + json.dumps(FACTS) + '\n') == FACTS
    assert pilot.validate_caption(CAPTION) == CAPTION


@pytest.mark.parametrize('change', [
    lambda f: f.update(global_held_object='A dark object'),
    lambda f: f.update(rendered_or_screen='false'),
    lambda f: f['entities'][1].update(id='E1'),
    lambda f: f['entities'][0].update(id='A person'),
    lambda f: f['entities'][0].update(description='hands'),
    lambda f: f['entities'][0].update(held_object=3),
    lambda f: f['events'][0].update(entity_id='E3'),
    lambda f: f['events'][0].update(frame_start=True),
    lambda f: f['events'][0].update(frame_start=1.0),
    lambda f: f['events'][0].update(frame_start=-1),
    lambda f: f['events'][0].update(frame_end=16),
    lambda f: f['events'][0].update(frame_start=15, frame_end=0),
    lambda f: f['events'][0].update(action=''),
    lambda f: f['events'][0].update(action='Moves another person hands across the table'),
    lambda f: f['entities'][0].update(description='A person named Sample'),
    lambda f: f['events'][0].update(action='Wants to move the dark object'),
    lambda f: f['entities'][0].update(description='A person with invisible hands'),
])
def test_invalid_schema_entity_references_grounding(change):
    value = copy.deepcopy(FACTS)
    change(value)
    with pytest.raises(ValueError):
        pilot.validate_facts(value)


@pytest.mark.parametrize('text', ['A person stands beside the table.', ' '.join(['person']*121),
    CAPTION + ' The final frame is shown.', CAPTION + ' Sampling occurs at sixteen points.',
    CAPTION + ' The logo identifies a brand.', CAPTION + ' The person probably intends to leave.'])
def test_caption_short_metadata_guess_rejected(text):
    with pytest.raises(ValueError):
        pilot.validate_caption(text)


def test_sensitive_input_output_error_is_bounded_without_leak():
    prep, _, _, _ = pilot.helpers()
    blocked = next(x for x in ('private passport', 'weapon') if prep.UNSAFE.search(x))
    value = copy.deepcopy(FACTS)
    value['scene'] = blocked
    with pytest.raises(ValueError):
        pilot.parse_facts(json.dumps(value))
    with pytest.raises(ValueError):
        pilot.validate_caption(CAPTION + blocked)
    error = pilot.bounded_error(ValueError(blocked + '\n' + 'x'*10000))
    assert blocked not in json.dumps(error)
    assert len(error['error_message']) < 200


def test_stage_messages_actual_pil_images_no_global_hands():
    from PIL import Image
    images = [Image.new('RGB', (640, 352), (i, 0, 0)) for i in range(16)]
    first = pilot.stage_messages(images)
    second = pilot.stage_messages(images, FACTS)
    assert [v['image'] for v in first[0]['content'][:16]] == images
    assert [v['image'] for v in second[0]['content'][:16]] == images
    assert all(type(v['image']) is Image.Image for v in second[0]['content'][:16])
    assert 'entity_id' in second[0]['content'][-1]['text']
    assert len(first[0]['content']) == len(second[0]['content']) == 17


def test_actual_ffmpeg_all_source_frames_and_endpoint_selection(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('Actual CPU ffmpeg/ffprobe unavailable')
    path = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
        'testsrc2=size=1280x704:rate=24', '-frames:v', '121', '-threads', '1', '-c:v', 'mpeg4',
        '-pix_fmt', 'yuv420p', str(path)], check=True, timeout=60)
    tools = {n: shutil.which(n) for n in ('ffmpeg', 'ffprobe')}
    images, info = pilot.sample_images(path, tools)
    assert len(images) == 16 and all(i.size == (640, 352) for i in images)
    assert info['source_frames'] == 121 and info['source_indices'][-1] == 120
    assert info['displayed_indices'] == list(range(16))
    # Independent single source-index decode, not the old eight-frame picker.
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(path),
        '-vf', 'select=eq(n\\,120)', '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
        capture_output=True, check=True, timeout=30).stdout
    from PIL import Image
    expected = Image.frombytes('RGB', (1280, 704), raw).resize((640, 352), Image.Resampling.BILINEAR)
    assert images[-1].tobytes() == expected.tobytes()


def test_actual_cpu_native_autoprocessor_typed_16_images_no_cuda(tmp_path):
    if not pilot.MODEL_ROOT.is_dir() or not Path(pilot.PYTHON).is_file():
        pytest.skip('Run this CPU-only compatibility test on actual rank1 local model host')
    path = tmp_path/'actual_source.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
        'testsrc2=size=1280x704:rate=24', '-frames:v', '121', '-threads', '1', '-c:v', 'mpeg4',
        '-pix_fmt', 'yuv420p', str(path)], check=True, timeout=60)
    tools = {n: shutil.which(n) for n in ('ffmpeg', 'ffprobe')}
    program = ('import os,sys;sys.path.insert(0,' + repr(str(pilot.ROOT / 'scripts')) + ');'
        'import torch;from PIL import Image;from transformers import AutoProcessor;'
        'import run_teacher_qwen3vl32b_recovery as p;assert not torch.cuda.is_initialized();'
        'processor=AutoProcessor.from_pretrained(str(p.MODEL_ROOT),local_files_only=True);'
        'images,info=p.sample_images(' + repr(str(path)) + ',' + repr(tools) + ');'
        'assert info["source_indices"]==p.source_indices(121);'
        'messages=p.stage_messages(images);'
        'encoded=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt");'
        'assert encoded["image_grid_thw"].shape[0]==16;assert encoded["input_ids"].device.type=="cpu";'
        'facts=' + repr(dict(FACTS, events=[], start_end_difference='No visible change between the first and last images')) + ';second=p.stage_messages(images,facts);'
        'encoded2=processor.apply_chat_template(second,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt");'
        'assert encoded2["image_grid_thw"].shape[0]==16;assert encoded2["input_ids"].device.type=="cpu";'
        'assert not torch.cuda.is_initialized();print(encoded["input_ids"].shape[-1],encoded2["input_ids"].shape[-1])')
    r = subprocess.run([pilot.PYTHON, '-c', program], env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'), capture_output=True, timeout=120)
    assert r.returncode == 0, r.stderr.decode()[-1000:]
    token_counts = [int(n) for n in r.stdout.decode().strip().split()]
    assert len(token_counts) == 2 and all(n > 3000 for n in token_counts)


@pytest.mark.parametrize('mapping', [{'': 'cpu'}, {'': 'disk'}, {'': 'cuda:4'}, {'': True},
    {'': 0}, {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 'cpu'}])
def test_actual_device_map_no_cpu_disk_extra_gpu(mapping):
    with pytest.raises(ValueError):
        pilot.validate_device_map(mapping)


def test_device_map_all_four():
    assert pilot.validate_device_map({'a': 0, 'b': 'cuda:1', 'c': 2, 'd': 'cuda:3'})


def model_fixture(root):
    names = sorted(set(pilot.WEIGHTS) | pilot.REQUIRED_CONFIGS | {'generation_config.json', 'chat_template.json'})
    rows = [dict(relative=n, path=str(root/n), bytes=1, sha256=sha(n.encode())) for n in names]
    rows[0]['bytes'] += pilot.WEIGHT_BYTES - 14 if rows[0]['relative'] in pilot.WEIGHTS else 0
    # Assign an exact synthetic byte tuple; this is schema validation, not a model measurement.
    for row in rows:
        if row['relative'] == pilot.WEIGHTS[0]:
            row['bytes'] = pilot.WEIGHT_BYTES - 13
    proof = dict(schema='teacher_qwen3vl32b_model_content_v1', rank=pilot.RANK, hostname=os.uname().nodename,
        GPU_used=False, model_type='qwen3_vl', architectures=['Qwen3VLForConditionalGeneration'],
        weight_shards=14, weight_bytes=pilot.WEIGHT_BYTES, model_root=str(root), models=rows,
        content_sha256=sha(pilot.encoded(pilot.model_identity(rows))))
    return proof


@pytest.mark.parametrize('key,bad', [('rank', 1), ('rank', True), ('model_type', 'qwen2_5_vl'),
    ('weight_shards', 5), ('weight_bytes', 1), ('GPU_used', True), ('model_root', '/dockerdata/models')])
def test_model_proof_scope_14_shards_root(key, bad, tmp_path):
    proof = model_fixture(tmp_path)
    with patch.object(pilot, 'MODEL_ROOT', tmp_path):
        pilot.validate_model_proof(proof, current=False)
        proof[key] = bad
        with pytest.raises(ValueError):
            pilot.validate_model_proof(proof, current=False)


def test_model_proof_missing_shard_digest_path_rejected(tmp_path):
    proof = model_fixture(tmp_path)
    with patch.object(pilot, 'MODEL_ROOT', tmp_path):
        for edit in ('shard', 'path', 'hash'):
            p = copy.deepcopy(proof)
            if edit == 'shard':
                p['models'] = [r for r in p['models'] if r['relative'] != pilot.WEIGHTS[-1]]
            elif edit == 'path':
                p['models'][0]['path'] = '/different/root/config.json'
            else:
                p['models'][0]['sha256'] = '0'*64
            with pytest.raises(ValueError):
                pilot.validate_model_proof(p, current=False)


def test_atomic_evidence_no_overwrite_symlink(tmp_path):
    path = tmp_path/'evidence.json'
    row = pilot.write_new(path, {'CPU_only': True})
    with pytest.raises(ValueError):
        pilot.write_new(path, {'CPU_only': False})
    assert pilot.read(row) == {'CPU_only': True}
    link = tmp_path/'link.json'
    link.symlink_to(path)
    with pytest.raises(ValueError):
        pilot.record(link)


def harmless_tree(tmp_path, ignore_term=False):
    marker = tmp_path/'grandchild.json'
    script = ('import subprocess,sys,time,os,json,signal;'
        + ('signal.signal(signal.SIGTERM,signal.SIG_IGN);' if ignore_term else '')
        + 'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);'
        + '\nwith open(' + repr(str(marker) + '.tmp') + ',"w") as f: f.write(json.dumps({"pid":p.pid}))\n'
        + 'os.replace(' + repr(str(marker) + '.tmp') + ',' + repr(str(marker)) + ');time.sleep(60)')
    child = subprocess.Popen([sys.executable, '-c', script], start_new_session=True)
    _, process, _, _ = pilot.helpers()
    owner = process.identity(child.pid)
    until = time.monotonic()+5
    while not marker.is_file() and time.monotonic() < until:
        time.sleep(.01)
    assert marker.is_file()
    grandchild = process.identity(json.loads(marker.read_text())['pid'])
    return child, owner, grandchild, process


def test_actual_cpu_owned_child_grandchild_and_repeat_cleanup(tmp_path):
    child, owner, grandchild, process = harmless_tree(tmp_path)
    tracked = []
    assert grandchild['pid'] in {r['pid'] for r in process.owned_members(owner, tracked)}
    process.cleanup_owned(child, owner, time.monotonic()+5, tracked)
    assert not process.same_process(grandchild)
    assert not process.owned_members(owner, tracked)
    assert process.cleanup_owned(child, owner, time.monotonic()+1, tracked) == []


def test_actual_term_then_kill_cpu_tree_no_foreign_signals(tmp_path):
    child, owner, grandchild, process = harmless_tree(tmp_path, ignore_term=True)
    foreign = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
    foreign_identity = process.identity(foreign.pid)
    try:
        with patch.object(process, 'KILL_SECONDS', .15):
            signals = process.cleanup_owned(child, owner, time.monotonic()+3, [])
        assert process.same_process(foreign_identity)
        assert any(r['signal'] == signal.SIGKILL for r in signals)
        assert all(r['identity']['pid'] != foreign.pid for r in signals)
        assert not process.same_process(grandchild)
    finally:
        foreign.terminate()
        foreign.wait(timeout=2)


def raw_query(uuid, earliest, external=None):
    _, process, _, _ = pilot.helpers()
    begin = max(earliest, time.monotonic())
    time.sleep(.00001)
    return dict(command=process.QUERY_COMMAND, gpu_uuid=uuid, returncode=0, stderr='',
        stdout='' if external is None else uuid+', '+str(external)+'\n', pids=[] if external is None else [external],
        start_monotonic=begin, end_monotonic=time.monotonic())


def test_four_gpu_release_eight_distinct_queries_and_foreign_block():
    uuids = [f'GPU-{i}' for i in range(4)]
    earliest = time.monotonic()
    rows = [raw_query(u, earliest) for u in uuids for _ in range(2)]
    assert pilot.release_queries_valid(rows, uuids, earliest)
    duplicate = copy.deepcopy(rows)
    duplicate[1] = duplicate[0]
    with pytest.raises(ValueError):
        pilot.release_queries_valid(duplicate, uuids, earliest)
    foreign = copy.deepcopy(rows)
    foreign[0]['pids'] = [123]
    foreign[0]['stdout'] = uuids[0]+', 123\n'
    with pytest.raises(ValueError):
        pilot.release_queries_valid(foreign, uuids, earliest)


def test_early_failure_cost_full_four_card_cap_not_zero():
    start = time.monotonic()
    value = pilot.allocation_cost(start, released=True)
    assert value['charged_upper_seconds'] == 5400
    assert value['allocated_GPUh'] == 6
    assert value['prior_charged_label_GPUh'] == pilot.PRIOR_GPUH == 2.1658982108534675
    assert value['cumulative_label_upper_GPUh'] == pytest.approx(pilot.CUMULATIVE_CAP_GPUH, abs=1e-12)
    with patch.object(pilot.time, 'monotonic', return_value=start+5401):
        late = pilot.allocation_cost(start)
        assert late['allocated_GPUh'] > 6 and late['budget_exceeded']


def test_exact_lease_flags_must_not_split_across_ancestors():
    flags = (('--action', 'exec'), ('--gpus', '0,1,2,3'), ('--state-dir', str(pilot.STATE_DIR)))
    assert pilot.exact_flags(['--action', 'exec', '--gpus', '0,1,2,3', '--state-dir', str(pilot.STATE_DIR)], flags)
    assert not pilot.exact_flags(['--action', 'exec', '--gpus', '0', '--state-dir', str(pilot.STATE_DIR)], flags)
    assert not pilot.exact_flags(['--action', 'exec', '--gpus', '0,1,2,3', '--gpus', '0,1,2,3',
        '--state-dir', str(pilot.STATE_DIR)], flags)


def tiny_plan(local):
    plan = dict(schema=pilot.SCHEMA, private=True, phase=pilot.PHASE, node_rank=pilot.RANK,
        node_ip=os.environ.setdefault('NODE_IP', '127.0.0.1'),
        hostname=os.uname().nodename, output_root=str(local), state_dir=str(pilot.STATE_DIR),
        gpu_uuids=[f'GPU-{i}' for i in range(4)], source={}, runtime={'python': sys.executable,
        'tools': {'timeout': shutil.which('timeout')}}, model={'content_sha256': 'a'*64})
    plan['jobs'] = [dict(index=i, group='diagnostic' if i<16 else 'blind', sample_id='case'+str(i),
        video={'path': str(local/f'video{i}.mp4'), 'bytes': 1, 'sha256': 'b'*64},
        original_prompt_sha256='c'*64, local_video=str(local/f'video{i}.mp4')) for i in range(32)]
    return plan


def test_error_records_cover32_without_caption_leak(tmp_path):
    p = tiny_plan(tmp_path)
    with patch.object(pilot, 'LOCAL', tmp_path):
        pilot.complete_error_records(p, ValueError('raw private content'))
        records, groups = pilot.aggregate_records(p)
        assert len(records) == 32 and all(g['errors'] == g['not_attempted'] == 16 for g in groups.values())
        assert {pilot.read(r)['failure_reason'] for r in records} == {'worker_exception'}
        assert all(g['semantic_review_pass'] is False for g in groups.values())
        assert 'private content' not in ''.join(Path(r['path']).read_text() for r in records)
        before = records[:]
        pilot.complete_error_records(p, RuntimeError())
        assert pilot.aggregate_records(p)[0] == before


def test_gpu_requires_explicit_authorization_no_spawn(tmp_path):
    with patch.object(pilot.subprocess, 'Popen') as popen, pytest.raises(ValueError):
        pilot.run('missing', '0'*64, 'missing', '0'*64, False)
    popen.assert_not_called()


@pytest.fixture
def cpu_supervisor(tmp_path, monkeypatch):
    local = tmp_path/'fresh'
    local.mkdir()
    evidence = tmp_path/'failures'
    evidence.mkdir()
    monkeypatch.setattr(pilot, 'LOCAL', local)
    monkeypatch.setattr(pilot, 'EVIDENCE', evidence)
    monkeypatch.setattr(pilot, 'RECOVERY_EVIDENCE', evidence.resolve())
    p = tiny_plan(local)
    plan = pilot.write_new(tmp_path/'plan.json', p)
    stage = pilot.write_new(local/'stage.json', dict(plan=plan, GPU_used=False, records=[{}]*32))
    approval = pilot.write_new(tmp_path/'approval.json', {'fixture': 'CPU-only protocol admission boundary'})
    monkeypatch.setattr(pilot, 'verify', lambda path, digest, end=None: (p, plan))
    monkeypatch.setattr(pilot, 'verify_approval', lambda *args: approval)
    _, process, _, _ = pilot.helpers()
    def boundaries(*args, **kwargs):
        manager = process.identity(os.getpid())
        return [dict(manager=manager, query=raw_query(u, 0), gpu_uuid=u) for u in p['gpu_uuids']]
    monkeypatch.setattr(pilot, 'boundary', boundaries)
    monkeypatch.setattr(process, 'query_release', lambda u: raw_query(u, 0))
    real_popen = subprocess.Popen
    calls = []
    program = ('import sys;sys.path.insert(0,' + repr(str(pilot.ROOT/'scripts')) + ');'
        'import run_teacher_qwen3vl32b_recovery as r;from pathlib import Path;'
        'r.LOCAL=Path(' + repr(str(local)) + ');'
        'p=' + repr(p) + ';bound=' + repr(plan) + ';facts=' + repr(FACTS) + ';caption=' + repr(CAPTION) + ';'
        'frames=dict(source_frames=121,displayed_indices=list(range(16)),source_indices=r.source_indices(121));'
        '\nfor j in p["jobs"]: r.write_new(r.LOCAL/("label"+str(j["index"])+".json"),dict(r.record_base(j,p,frames),facts_json=facts,caption=caption,duration_seconds=1.0,new_tokens=dict(facts=9,caption=9)))'
        '\nrecords,groups=r.aggregate_records(p);'
        'r.write_new(r.LOCAL/"worker.json",dict(plan=bound,records=records,groups=groups,success=True,error=None));')
    def harmless_popen(command, **kwargs):
        # Leave subprocess.run(timeout --version) real; replace only requested GPU dispatch.
        if str(pilot.ROOT/'scripts/gpu_hold.py') in command:
            calls.append(command)
            return real_popen([sys.executable, '-c', program], **kwargs)
        return real_popen(command, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', harmless_popen)
    return p, plan, approval, calls, process, local, evidence


def call_cpu_supervisor(fixture):
    _, plan, approval, *_ = fixture
    return pilot.run(plan['path'], plan['sha256'], approval['path'], approval['sha256'], True)


def test_actual_harmless_supervisor_parent_child_terminal_closure(cpu_supervisor):
    row = call_cpu_supervisor(cpu_supervisor)
    p, plan, approval, calls, process, local, evidence = cpu_supervisor
    assert len(calls) == 1
    assert '--gpus' in calls[0] and calls[0][calls[0].index('--gpus')+1] == '0,1,2,3'
    closure = pilot.verify_cost_closure(row['path'], row['sha256'])
    parent = pilot.read(closure['parent'])
    provisional = pilot.read(closure['cost'])
    assert parent['execution_success'] and not parent['success']
    assert not provisional['success'] and provisional['charged_upper_seconds'] >= pilot.CAP
    assert parent['released'] and parent['tree_released'] and not process.same_process(parent['child'])
    assert closure['allocation_GPUh_upper'] == 6 and not closure['semantic_review_pass']
    with pytest.raises(ValueError):
        call_cpu_supervisor(cpu_supervisor)
    assert len(calls) == 1 and list(evidence.glob('qwen3vl32b.allocation-failure.*.json'))


def test_actual_late_terminal_save_stall_cannot_publish_pass(cpu_supervisor, monkeypatch):
    real = pilot.write_new
    def stalled(path, value):
        if Path(path).name == 'cost.closure.json':
            time.sleep(1.05)
        return real(path, value)
    monkeypatch.setattr(pilot, 'write_new', stalled)
    with pytest.raises(ValueError):
        call_cpu_supervisor(cpu_supervisor)
    local = cpu_supervisor[-2]
    failure = pilot.read(pilot.record(local/'cost.failure.json'))
    assert failure['allocated_GPUh'] >= 2 and failure['success'] is False
    with pytest.raises(ValueError):
        row = pilot.record(local/'cost.closure.json')
        pilot.verify_cost_closure(row['path'], row['sha256'])


def test_early_parent_identity_failure_keeps_cost_and32errors(cpu_supervisor, monkeypatch):
    process = cpu_supervisor[4]
    real = process.identity
    def failed(pid):
        if pid == os.getpid():
            raise OSError('CPU identity fixture failure')
        return real(pid)
    monkeypatch.setattr(process, 'identity', failed)
    with pytest.raises(ValueError):
        call_cpu_supervisor(cpu_supervisor)
    local = cpu_supervisor[-2]
    assert len(list(local.glob('label*.json'))) == 32
    failure = pilot.read(pilot.record(local/'cost.failure.json'))
    assert failure['allocated_GPUh'] == 6 and failure['terminal'] and not cpu_supervisor[3]


def test_actual_popen_child_identity_failure_cleaned_single_attempt(cpu_supervisor, monkeypatch):
    process = cpu_supervisor[4]
    real = process.identity
    failed = []
    def identity(pid):
        if pid != os.getpid() and not failed:
            failed.append(pid)
            raise OSError('One actual newly spawned Popen identity observation fails')
        return real(pid)
    monkeypatch.setattr(process, 'identity', identity)
    with pytest.raises(ValueError):
        call_cpu_supervisor(cpu_supervisor)
    assert len(cpu_supervisor[3]) == 1 and len(failed) == 1
    local = cpu_supervisor[-2]
    parent = pilot.read(pilot.record(local/'parent.json'))
    assert not parent['success'] and parent['tree_released']
    assert not process.same_process(parent['child'])
    assert pilot.read(pilot.record(local/'cost.failure.json'))['allocated_GPUh'] >= 2


def test_foreign_gpu_admission_block_no_spawn_no_kill(cpu_supervisor, monkeypatch):
    def blocked(*args, **kwargs):
        raise ValueError('External compute ownership blocks the fresh GPU boundary')
    monkeypatch.setattr(pilot, 'boundary', blocked)
    with patch.object(os, 'kill') as kill, pytest.raises(ValueError):
        call_cpu_supervisor(cpu_supervisor)
    kill.assert_not_called()
    assert len(cpu_supervisor[3]) == 0
    local = cpu_supervisor[-2]
    assert pilot.read(pilot.record(local/'cost.failure.json'))['allocated_GPUh'] == 6


def test_failure_during_safe_error_publication_still_charged(cpu_supervisor, monkeypatch):
    def damaged(*args):
        raise OSError('CPU-only error-file publication fixture')
    monkeypatch.setattr(pilot, 'complete_error_records', damaged)
    with pytest.raises(ValueError):
        call_cpu_supervisor(cpu_supervisor)
    local = cpu_supervisor[-2]
    assert pilot.read(pilot.record(local/'cost.failure.json'))['allocated_GPUh'] >= 2


def test_wrong_sha_early_failure_receipt_full_cap_no_dispatch(cpu_supervisor):
    _, plan, approval, calls, process, local, evidence = cpu_supervisor
    with pytest.raises(ValueError):
        pilot.run(plan['path'], '0'*64, approval['path'], approval['sha256'], True)
    assert not calls
    receipt = pilot.read(pilot.record(local/'cost.failure.json'))
    assert receipt['allocated_GPUh'] == 6 and (local/'attempt.lock').is_file()
    with pytest.raises(ValueError):
        call_cpu_supervisor(cpu_supervisor)
    assert not calls


@pytest.fixture
def frozen_plan_gate(tmp_path, monkeypatch):
    local = tmp_path/'never-used'
    monkeypatch.setattr(pilot, 'LOCAL', local)
    p = tiny_plan(local)
    p.update(private=True, config=pilot.fixed_config(), gpu_indices=[0, 1, 2, 3],
        cap_seconds=pilot.CAP, tail_seconds=pilot.TAIL, allocated_cards=pilot.CARDS, allocation_GPUh_cap=pilot.ALLOCATION_GPUH,
        prior_charged_label_GPUh=pilot.PRIOR_GPUH, stage_cap_GPUh=24, selection_seed=pilot.SEED,
        metadata_eligible_rows=1815, metadata_rejected_rows=41, fixed_groups=[16, 16],
        single_attempt=True, automatic_retry_allowed=False, full_rebuild_allowed=False,
        formal_training_allowed=False, AR_allowed=False, student_distillation_allowed=False,
        GPU_execution_started=False, semantic_review_pass=False,
        structural_validation_is_semantic_approval=False, selection_uses_benchmark_scores=False,
        authorizations=pilot.authorization_rows(), old_plan=pilot.pinned(pilot.OLD_PLAN, pilot.OLD_PLAN_SHA),
        first_pilot_plan=pilot.pinned(pilot.FIRST_PLAN, pilot.FIRST_PLAN_SHA))
    prep, _, _, _ = pilot.helpers()
    p['manifest'], rows = prep.manifest_rows(pilot.MANIFEST)
    old = pilot.read(p['old_plan'])
    p['jobs'] = pilot.jobs_for(rows, old)
    p['model'] = model_fixture(pilot.MODEL_ROOT)
    p['model_proof'] = pilot.write_new(tmp_path/'model.json', p['model'])
    real_model_check = pilot.validate_model_proof
    monkeypatch.setattr(pilot, 'validate_model_proof', lambda proof, end=None, current=True:
        real_model_check(proof, end, current=False))
    source = tmp_path/'source.py'
    source.write_text('CPU-only fixed gate fixture\n')
    monkeypatch.setattr(pilot, 'ROOT', tmp_path)
    monkeypatch.setattr(pilot, 'SOURCE_NAMES', ('source.py',))
    p['source'] = {'source.py': pilot.record(source)}
    p['runtime'] = {'fixture': 'actual selected-runtime equality boundary'}
    monkeypatch.setattr(pilot, 'runtime', lambda end=None: {'fixture': 'actual selected-runtime equality boundary'})
    return p, source


def test_fixed_plan_gate_accepts_exact_cpu_fixture_only(frozen_plan_gate, tmp_path):
    p, _ = frozen_plan_gate
    row = pilot.write_new(tmp_path/'plan.json', p)
    observed, bound = pilot.verify(row['path'], row['sha256'])
    assert observed == p and bound == row


@pytest.mark.parametrize('key,bad', [('phase', 'single_gpu_training_resume'), ('node_rank', True),
    ('allocated_cards', 1), ('allocation_GPUh_cap', 8), ('prior_charged_label_GPUh', 0.0),
    ('cap_seconds', 7200), ('tail_seconds', 0), ('automatic_retry_allowed', True),
    ('semantic_review_pass', True), ('selection_seed', 20261101), ('fixed_groups', [32, 0]),
    ('gpu_indices', [False, 1, 2, 3]), ('gpu_uuids', ['GPU-0']*4),
    ('output_root', '/dockerdata/wm_tmp/teacher_label_pilot_20261102'),
    ('state_dir', '/dockerdata/wm_logs/gpu_hold'), ('source', {}),
    ('runtime', {'fixture': 'different actual runtime'}), ('old_plan', {'path': 'private', 'bytes': 1, 'sha256': '0'*64})])
def test_full_plan_drift_rejected(frozen_plan_gate, tmp_path, key, bad):
    p, _ = frozen_plan_gate
    value = copy.deepcopy(p)
    value[key] = bad
    row = pilot.write_new(tmp_path/'changed.json', value)
    with pytest.raises((ValueError, KeyError)):
        pilot.verify(row['path'], row['sha256'])


def test_plan_current_source_and_config_boolint_drift_rejected(frozen_plan_gate, tmp_path):
    p, source = frozen_plan_gate
    value = copy.deepcopy(p)
    value['config']['do_sample'] = 0
    row = pilot.write_new(tmp_path/'config.json', value)
    with pytest.raises(ValueError):
        pilot.verify(row['path'], row['sha256'])
    original = pilot.write_new(tmp_path/'original.json', p)
    source.write_text('CPU-only byte drift\n')
    with pytest.raises(ValueError):
        pilot.verify(original['path'], original['sha256'])


def test_blind_selection_identity_permutation_rejected(frozen_plan_gate, tmp_path):
    p, _ = frozen_plan_gate
    p['jobs'][16], p['jobs'][17] = p['jobs'][17], p['jobs'][16]
    row = pilot.write_new(tmp_path/'permuted.json', p)
    with pytest.raises(ValueError):
        pilot.verify(row['path'], row['sha256'])


def test_single_training_approval_not_label_approval(tmp_path):
    import run_teacher_single_gpu_admission as single
    row = pilot.write_new(tmp_path/'approval.json', dict(schema='teacher_single_gpu_execution_approval_v1',
        private=True, GPU_authorized=True, plan={}, source_git_root=str(single.GIT_ROOT), source_commit='a'*40,
        publication_manifest={}, remote_receipt={}, phase=single.PHASE, formal_training_allowed=False,
        branch=single.BRANCH, snapshot_prefix=single.SNAPSHOT_PREFIX))
    with patch.object(subprocess, 'run') as run, pytest.raises(ValueError):
        pilot.verify_approval(row['path'], row['sha256'], {}, {}, time.monotonic()+10)
    run.assert_not_called()


def test_explicit_name_claim_still_rejected():
    value = copy.deepcopy(FACTS)
    value['entities'][0]['description'] = 'A person named Exampleperson wears a dark shirt'
    with pytest.raises(ValueError):
        pilot.validate_facts(value)


def test_actual_harmless_parent_lease_timeout_worker_ancestry(tmp_path, monkeypatch):
    local = tmp_path/'own'
    local.mkdir()
    monkeypatch.setattr(pilot, 'LOCAL', local)
    _, process, _, _ = pilot.helpers()
    plan = pilot.write_new(tmp_path/'plan.json', {'CPU_only': True})
    approval = pilot.write_new(tmp_path/'approval.json', {'CPU_only': True})
    marker = tmp_path/'worker.pid'
    timeout = shutil.which('timeout')
    worker_code = 'import os,time;open(' + repr(str(marker)) + ',"w").write(str(os.getpid()));time.sleep(60)'
    lease_code = 'import subprocess,time;subprocess.Popen(' + repr([timeout, '--signal=TERM',
        '--kill-after=20s', '60s', sys.executable, '-c', worker_code]) + ').wait()'
    lease_command = [sys.executable, '-c', lease_code, str(pilot.ROOT/'scripts/gpu_hold.py'),
        '--action', 'exec', '--gpus', '0,1,2,3', '--state-dir', str(pilot.STATE_DIR), '--']
    parent_code = 'import subprocess,time;p=subprocess.Popen(' + repr(lease_command) + ',start_new_session=True);time.sleep(60)'
    parent_args = [sys.executable, '-c', parent_code, str(Path(pilot.__file__).resolve()), 'run',
        '--plan', plan['path'], '--approved-plan-sha256', plan['sha256'], '--approval', approval['path'],
        '--approved-approval-sha256', approval['sha256'], '--gpu-authorized']
    parent_process = subprocess.Popen(parent_args, start_new_session=True)
    parent_owner = process.identity(parent_process.pid)
    tracked = []
    try:
        until = time.monotonic()+5
        while not marker.is_file() and time.monotonic() < until:
            process.owned_members(parent_owner, tracked)
            time.sleep(.01)
        assert marker.is_file()
        pid = process.identity(int(marker.read_text()))['ppid']
        ancestors = []
        while pid > 1:
            row = process.identity(pid)
            ancestors.append(row)
            pid = row['ppid']
        child = next(a for a in ancestors if a['ppid'] == parent_owner['pid'])
        p = dict(hostname=os.uname().nodename, gpu_uuids=[f'GPU-{i}' for i in range(4)],
            runtime={'tools': {'timeout': timeout}})
        launch = dict(plan=plan, cap_seconds=pilot.CAP, hostname=p['hostname'], gpu_uuids=p['gpu_uuids'],
            state_dir=str(pilot.STATE_DIR), start_monotonic=time.monotonic(), start_epoch=time.time(),
            parent=parent_owner, approval=approval, workload_seconds=60)
        launch_row = pilot.write_new(local/'launch.json', launch)
        supervised = dict(launch=launch_row, parent=parent_owner, child=child)
        assert pilot.verify_worker_ancestry(launch, p, plan, ancestors, supervised)
        for key, bad in (('gpu_uuids', ['GPU-0']*4), ('parent', dict(parent_owner, start_ticks=1)),
                         ('state_dir', '/foreign'), ('workload_seconds', 1800)):
            changed = copy.deepcopy(launch)
            changed[key] = bad
            with pytest.raises(ValueError):
                pilot.verify_worker_ancestry(changed, p, plan, ancestors, supervised)
        split = copy.deepcopy(ancestors)
        lease = next(a for a in split if str(pilot.ROOT/'scripts/gpu_hold.py') in a['argv'])
        i = lease['argv'].index('--gpus')
        lease['argv'][i+1] = '0'
        with pytest.raises(ValueError):
            pilot.verify_worker_ancestry(launch, p, plan, split, supervised)
    finally:
        process.cleanup_owned(parent_process, parent_owner, time.monotonic()+5, tracked)



def test_third_raw_release_query_clones_or_foreign_cannot_pass():
    uuids = [f'GPU-{i}' for i in range(4)]
    _, process, _, _ = pilot.helpers()
    manager = process.identity(os.getpid())
    start = time.monotonic()
    queries = [raw_query(u, start) for u in uuids for _ in range(2)]
    initial = [dict(manager=manager) for _ in uuids]
    final = [dict(manager=manager, gpu_uuid=u, query=raw_query(u, start)) for u in uuids]
    assert pilot.verify_release_boundary(final, initial, queries, uuids, start)
    cloned = copy.deepcopy(final)
    cloned[0]['query'] = queries[1]
    with pytest.raises(ValueError):
        pilot.verify_release_boundary(cloned, initial, queries, uuids, start)
    changed = copy.deepcopy(final)
    changed[0]['query']['stdout'] = uuids[0]+', 123\n'
    with pytest.raises(ValueError):
        pilot.verify_release_boundary(changed, initial, queries, uuids, start)
    changed = copy.deepcopy(final)
    changed[0]['manager']['start_ticks'] += 1
    with pytest.raises(ValueError):
        pilot.verify_release_boundary(changed, initial, queries, uuids, start)


@pytest.mark.parametrize('description', [
    'No visible change between the first and last images',
    'Blue fabric lies beside a Red T-shirt.',
    'Decorative Grey panels surround an LED display.',
    'At the edge of the table, a dark object rests in place.',
])
def test_legal_static_capitalization_preserved(description):
    facts = copy.deepcopy(FACTS)
    facts['scene'] = description
    facts['entities'][0]['description'] = 'Blue T-shirt with an LED light nearby'
    facts['events'] = []
    facts['start_end_difference'] = 'No visible change between the first and last images'
    assert pilot.validate_facts(facts) is facts
    assert pilot.safe_text(description) == description
    assert facts['start_end_difference'].startswith('No ')


@pytest.mark.parametrize('endpoints', [
    'No visible change between the first and last images',
    'No net change between first and last; the arm returns to its initial position.',
])
def test_round_trip_motion_same_endpoints_preserves_events(endpoints):
    facts = copy.deepcopy(FACTS)
    facts['events'] = [
        dict(entity_id='E1', frame_start=0, frame_end=7, action='Raises the visible right arm'),
        dict(entity_id='E1', frame_start=8, frame_end=15, action='Lowers the arm to its initial position'),
    ]
    facts['start_end_difference'] = endpoints
    before = copy.deepcopy(facts)
    assert pilot.validate_facts(facts) == before
    assert pilot.parse_facts(json.dumps(facts)) == before
    assert len(facts['events']) == 2


@pytest.mark.parametrize('facts,max_tokens', [(None, 1024), (FACTS, 384)])
@pytest.mark.parametrize('eos,new_tokens,complete', [
    (151645, [100, 151645], True),
    ([151645, 151643], [100, 151643], True),
    ([151645, 151643], [100, 151645], True),
    (151645, [100, 101], False),
    ([151645, 151643], [100, 151645, 101], False),
    (151645, [], False),
    (True, [100, 1], False),
    ([151645, True], [100, 151645], False),
    ([], [100, 151645], False),
])
def test_generate_two_stages_require_actual_final_eos(facts, max_tokens, eos, new_tokens, complete):
    import torch
    from PIL import Image
    from types import SimpleNamespace
    images = [Image.new('RGB', (640, 352)) for _ in range(16)]
    if new_tokens == [100, 101]:
        new_tokens = [100] * max_tokens  # Both stages hit their actual limit without EOS.
    calls, decoded = [], []
    class CPUInputs(dict):
        def to(self, device):
            assert device == 'cpu'
            return self
    class CPUProcessor:
        def apply_chat_template(self, messages, **kwargs):
            assert len(messages[0]['content']) == 17
            assert kwargs == dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
            return CPUInputs(input_ids=torch.tensor([[10, 11]], dtype=torch.long))
        def batch_decode(self, tokens, **kwargs):
            decoded.append(tokens.tolist())
            assert kwargs == dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return [json.dumps(FACTS) if facts is None else CAPTION]
    class CPUModel:
        generation_config = SimpleNamespace(eos_token_id=eos)
        def generate(self, **kwargs):
            calls.append(kwargs)
            assert kwargs['do_sample'] is False and kwargs['max_new_tokens'] == max_tokens
            return torch.tensor([[10, 11, *new_tokens]], dtype=torch.long)
    if complete:
        text = pilot.generate_text(CPUModel(), CPUProcessor(), images, 'cpu', max_tokens, facts)
        assert text == (json.dumps(FACTS) if facts is None else CAPTION)
        assert decoded == [[new_tokens]]
    else:
        with pytest.raises(ValueError):
            pilot.generate_text(CPUModel(), CPUProcessor(), images, 'cpu', max_tokens, facts)
        assert decoded == []
    assert len(calls) == 1  # No retry, not a CUDA/model-weight test.
    assert not torch.cuda.is_initialized()


def test_safe_case_failure_codes_never_store_model_text():
    raw = ValueError('The private person named Example says hello at /secret/path')
    value = pilot.case_failure(raw, 'facts_parse', 1.5, {'facts': 12})
    assert value['failure_stage'] == 'facts_parse' and value['failure_reason'] == 'unlisted_ValueError'
    assert 'Example' not in json.dumps(value) and value['duration_seconds'] == 1.5
    known = pilot.case_failure(ValueError('Caption must contain 40..120 words'), 'caption_validate', 2.0, {})
    assert known['failure_reason'] == 'Caption must contain 40..120 words'
    with pytest.raises(ValueError):
        pilot.case_failure(raw, 'free_text_stage', 1.0, {})
    with pytest.raises(ValueError):
        pilot.complete_error_records(tiny_plan(Path('/nonexistent')), raw, 'invented reason')


@pytest.mark.parametrize('text,token', [('A person seems to wave a visible hand.', 'seems'),
    ('A person named Example waves.', 'named')])
def test_grounding_deny_keeps_only_fixed_token(text, token):
    with pytest.raises(pilot.DenyMatch) as caught:
        pilot.safe_text(text)
    assert caught.value.token == token
    value = pilot.case_failure(caught.value, 'facts_parse', 1.0, {})
    assert value['deny_token'] == token and 'Example' not in json.dumps(value)


def test_caption_frame_metadata_token_normalized():
    text = CAPTION + ' The clip lasts 5 seconds.'
    with pytest.raises(pilot.DenyMatch) as caught:
        pilot.validate_caption(text)
    assert caught.value.token == 'clip length' or caught.value.token == '# seconds'


def test_success_records_require_duration_and_tokens(tmp_path):
    p = tiny_plan(tmp_path)
    frames = dict(source_frames=121, displayed_indices=list(range(16)), source_indices=pilot.source_indices(121))
    with patch.object(pilot, 'LOCAL', tmp_path):
        for j in p['jobs']:
            pilot.write_new(tmp_path / f"label{j['index']}.json", dict(pilot.record_base(j, p, frames),
                facts_json=FACTS, caption=CAPTION, duration_seconds=3.0, new_tokens={'facts': 9, 'caption': 9}))
        _, groups = pilot.aggregate_records(p)
        assert all(g['captions'] == 16 and g['errors'] == 0 for g in groups.values())
        bad = tmp_path / 'label0.json'
        bad.unlink()
        pilot.write_new(bad, dict(pilot.record_base(p['jobs'][0], p, frames), facts_json=FACTS, caption=CAPTION))
        with pytest.raises(ValueError):
            pilot.aggregate_records(p)


def test_generate_trace_counts_tokens_even_without_eos():
    import torch
    from PIL import Image
    from types import SimpleNamespace
    images = [Image.new('RGB', (640, 352)) for _ in range(16)]
    class CPUInputs(dict):
        def to(self, device):
            return self
    class CPUProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return CPUInputs(input_ids=torch.tensor([[10, 11]], dtype=torch.long))
        def batch_decode(self, tokens, **kwargs):
            raise AssertionError('Truncated output must not be decoded')
    class CPUModel:
        generation_config = SimpleNamespace(eos_token_id=151645)
        def generate(self, **kwargs):
            return torch.tensor([[10, 11] + [5] * 7], dtype=torch.long)
    trace = {}
    with pytest.raises(ValueError) as caught:
        pilot.generate_text(CPUModel(), CPUProcessor(), images, 'cpu', 1024, trace=trace)
    assert trace == {'new_tokens': 7} and pilot.reason_code(caught.value) in pilot.SAFE_REASONS


def test_recovery_requires_first_pilot_cases_and_prompts():
    prep, _, _, _ = pilot.helpers()
    _, rows = prep.manifest_rows(pilot.MANIFEST)
    old = pilot.read(pilot.pinned(pilot.OLD_PLAN, pilot.OLD_PLAN_SHA))
    jobs = pilot.jobs_for(rows, old)
    assert pilot.first_pilot_identity(jobs)
    swapped = copy.deepcopy(jobs)
    swapped[16], swapped[17] = swapped[17], swapped[16]
    with pytest.raises(ValueError):
        pilot.first_pilot_identity(swapped)
    fewer = copy.deepcopy(jobs)[:31]
    with pytest.raises(ValueError):
        pilot.first_pilot_identity(fewer)


def test_second_recovery_claim_refused_without_dispatch(cpu_supervisor):
    p, plan, approval, calls, process, local, evidence = cpu_supervisor
    (evidence / 'attempt.claim').write_bytes(b'{}\n')
    with pytest.raises(ValueError):
        call_cpu_supervisor(cpu_supervisor)
    assert not calls and not (local / 'attempt.lock').exists()
    assert list(evidence.glob('qwen3vl32b.allocation-failure.*.json'))


@pytest.mark.parametrize('path', ['scripts/run_teacher_qwen3vl32b_recovery.py', './scripts/run_teacher_qwen3vl32b_recovery.py'])
def test_relative_launch_rejected_before_claim_or_dispatch(path, monkeypatch):
    calls = []
    monkeypatch.setattr(pilot, 'run', lambda *args: calls.append(args))
    monkeypatch.setattr(sys, 'argv', [path])
    with pytest.raises(ValueError, match='absolute canonical runner path'):
        pilot.main(['run', '--plan', 'missing', '--approved-plan-sha256', 'a'*64,
            '--approval', 'missing', '--approved-approval-sha256', 'b'*64, '--gpu-authorized'])
    assert calls == []


def test_canonical_launch_preserves_exact_scope():
    argv = [sys.executable, str(Path(pilot.__file__).resolve()), 'run']
    assert pilot.canonical_launch(argv)
    with pytest.raises(ValueError):
        pilot.canonical_launch(argv + ['--supplemental'])
    with patch.object(pilot, 'SUPPLEMENTAL', True):
        assert pilot.canonical_launch(argv + ['--supplemental'])
        with pytest.raises(ValueError):
            pilot.canonical_launch(argv)


def test_supplemental_configuration_keeps_original_budget_and_changes_claim_root(monkeypatch):
    keys = ('SUPPLEMENTAL', 'LOCAL', 'STATE_DIR', 'RECOVERY_EVIDENCE', 'PRIOR_GPUH', 'CUMULATIVE_CAP_GPUH')
    before = {k: getattr(pilot, k) for k in keys}
    for k, v in before.items():
        monkeypatch.setattr(pilot, k, v)
    pilot.configure_supplemental()
    assert pilot.PRIOR_GPUH == 8.165898210853467
    assert pilot.CUMULATIVE_CAP_GPUH == 14.165898210853467
    assert pilot.ORIGINAL_PRIOR_GPUH == 2.1658982108534675
    assert pilot.ORIGINAL_CUMULATIVE_CAP_GPUH == 8.165898210853468
    assert pilot.LOCAL != before['LOCAL'] and pilot.RECOVERY_EVIDENCE != before['RECOVERY_EVIDENCE']
    assert pilot.RECOVERY_EVIDENCE == pilot.SUPPLEMENTAL_AUTH.parent


@pytest.mark.parametrize('mutation', [None, 'budget', 'refund', 'claim', 'retry'])
def test_supplemental_auth_real_file_bindings_retain_failure(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(pilot, 'ROOT', tmp_path)
    monkeypatch.setattr(pilot, 'PRIOR_GPUH', 8.165898210853467)
    monkeypatch.setattr(pilot, 'CUMULATIVE_CAP_GPUH', 14.165898210853467)
    old = tmp_path / 'eval/teacher_label_recovery_20261103'
    (old / 'run01_raw_snapshot').mkdir(parents=True)
    failure = pilot.write_new(old / 'run01_raw_snapshot/cost.failure.json', dict(terminal=True,
        success=False, released_observed=True, allocated_GPUh=6.0,
        prior_charged_label_GPUh=2.1658982108534675, cumulative_label_upper_GPUh=8.165898210853467))
    claim = pilot.write_new(old / 'attempt.claim', dict(pid=123))
    a = dict(schema='teacher_qwen32b_supplemental_allocation_authorization_v1',
        user_reply='我给你所有权限,请继续进行推进',
        new_allocation_GPUh_cap=6, wall_cap_seconds=5400, allocated_cards=4, fixed_cases=32,
        fixed_groups=[16, 16], frames_per_case=16, dtype='bfloat16',
        prior_charged_upper_GPUh=8.165898210853467, cumulative_label_upper_GPUh_cap=14.165898210853467,
        label_stage_cap_GPUh=24, previous_consumed_recovery_attempts=1, allocation_ordinal=2,
        original_recovery_claim_preserved=True, previous_failure_costs_preserved=True,
        automatic_retry_allowed=False, sample_substitution_allowed=False, frame_reduction_allowed=False,
        precision_gate_relaxation_allowed=False, formal_training_allowed=False, AR_allowed=False,
        student_distillation_allowed=False, prior_failure=failure, prior_claim=claim)
    if mutation == 'budget': a['new_allocation_GPUh_cap'] = 12
    if mutation == 'refund': a['prior_charged_upper_GPUh'] = 2.1658982108534675
    if mutation == 'claim': a['prior_claim'] = dict(claim, path=str(tmp_path / 'different.claim'))
    if mutation == 'retry': a['automatic_retry_allowed'] = True
    row = pilot.write_new(tmp_path / 'supplemental.json', a)
    monkeypatch.setattr(pilot, 'SUPPLEMENTAL_AUTH', Path(row['path']))
    monkeypatch.setattr(pilot, 'SUPPLEMENTAL_AUTH_SHA', row['sha256'])
    if mutation is None:
        assert pilot.supplemental_authorization() == row
    else:
        with pytest.raises(ValueError): pilot.supplemental_authorization()
    assert pilot.record(claim['path']) == claim and pilot.record(failure['path']) == failure



def test_prompt_v2_only_changes_policy_not_precise_validators(monkeypatch):
    keys = ('PROMPT_V2', 'LOCAL', 'STATE_DIR', 'RECOVERY_EVIDENCE', 'PRIOR_GPUH',
            'CUMULATIVE_CAP_GPUH', 'FACTS_PROMPT', 'CAPTION_PROMPT', 'PROMPT_VERSION')
    for k in keys: monkeypatch.setattr(pilot, k, getattr(pilot, k))
    before = pilot.fixed_config()
    facts_validator, caption_validator = pilot.validate_facts, pilot.validate_caption
    pilot.configure_prompt_v2()
    after = pilot.fixed_config()
    changed = {k for k in before if before[k] != after[k]}
    assert changed == {'facts_prompt_sha256', 'caption_prompt_sha256', 'prompt_version'}
    assert pilot.validate_facts is facts_validator and pilot.validate_caption is caption_validator
    assert pilot.PRIOR_GPUH == 14.165898210853467 and pilot.CUMULATIVE_CAP_GPUH == 20.165898210853467
    assert after['frames'] == 16 and after['caption_words'] == [40, 120] and after['do_sample'] is False
    assert '12 to 22 words' in pilot.CAPTION_PROMPT and '48 to 88 words' in pilot.CAPTION_PROMPT
    assert pilot.canonical_launch([sys.executable, str(Path(pilot.__file__).resolve()), '--prompt-v2', 'run'])
    with pytest.raises(ValueError):
        pilot.canonical_launch([sys.executable, str(Path(pilot.__file__).resolve()), '--supplemental', 'run'])
    for bad in (' '.join(['visible']*121), CAPTION + ' The frame shows a table.'):
        with pytest.raises(ValueError): pilot.validate_caption(bad)


def test_prompt_v2_same_cases_still_rejects_swapping(monkeypatch):
    for k in ('PROMPT_V2', 'LOCAL', 'STATE_DIR', 'RECOVERY_EVIDENCE', 'PRIOR_GPUH',
              'CUMULATIVE_CAP_GPUH', 'FACTS_PROMPT', 'CAPTION_PROMPT', 'PROMPT_VERSION'):
        monkeypatch.setattr(pilot, k, getattr(pilot, k))
    prep, _, _, _ = pilot.helpers()
    _, rows = prep.manifest_rows(pilot.MANIFEST)
    old = pilot.read(pilot.pinned(pilot.OLD_PLAN, pilot.OLD_PLAN_SHA))
    pilot.configure_prompt_v2()
    jobs = pilot.jobs_for(rows, old)
    assert pilot.first_pilot_identity(jobs)
    jobs[0], jobs[1] = jobs[1], jobs[0]
    with pytest.raises(ValueError): pilot.first_pilot_identity(jobs)


def test_protocol_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        pilot.main(['--supplemental', '--prompt-v2', 'verify', '--plan', 'missing',
                    '--approved-plan-sha256', 'a'*64])


@pytest.mark.parametrize('words',[39,40,60])
def test_caption_minimum_words_only_masks_eos_before_40(words):
    import torch
    class Processor:
        def batch_decode(self,tokens,**kw):return [' '.join(['visible']*words)]
    callback=pilot.minimum_caption_words(2,Processor(),[2,3])
    scores=torch.ones((1,5));out=callback(torch.tensor([[8,9,10]]),scores)
    assert out[0,0].item()==1 and out[0,1].item()==1 and out[0,4].item()==1
    assert torch.isneginf(out[0,2]).item()==(words<40)
    assert torch.isneginf(out[0,3]).item()==(words<40)
    assert not torch.cuda.is_initialized()


def test_constrained_budget_prompt_invariance_and_old_exact_gates(monkeypatch):
    keys=('CONSTRAINED','PROMPT_V2','LOCAL','STATE_DIR','RECOVERY_EVIDENCE','CAP',
        'ALLOCATION_GPUH','PRIOR_GPUH','CUMULATIVE_CAP_GPUH','FACTS_PROMPT','CAPTION_PROMPT','PROMPT_VERSION')
    for k in keys:monkeypatch.setattr(pilot,k,getattr(pilot,k))
    pilot.configure_prompt_v2();v2=pilot.fixed_config()
    validators=pilot.validate_facts,pilot.validate_caption
    pilot.configure_constrained();v3=pilot.fixed_config()
    assert v3['facts_prompt_sha256']==v2['facts_prompt_sha256'] and v3['caption_prompt_sha256']==v2['caption_prompt_sha256']
    assert (pilot.validate_facts,pilot.validate_caption)==validators
    assert pilot.CAP==2400 and pilot.ALLOCATION_GPUH==4*2400/3600
    assert pilot.CUMULATIVE_CAP_GPUH+0.15<=24 and pilot.PRIOR_GPUH==21.165898210853467
    assert v3['decoder_constraints']['caption_eos_min_words']==40
    assert 'frame' not in pilot.DENY_DECODE_WORDS and 'frame' in pilot.CAPTION_DECODE_WORDS
    assert pilot.canonical_launch([sys.executable,str(Path(pilot.__file__).resolve()),'--constrained','run'])
    for bad in (' '.join(['visible']*121),CAPTION+' The frame shows a table.'):
        with pytest.raises(ValueError):pilot.validate_caption(bad)


@pytest.mark.parametrize('facts,max_tokens',[(None,1024),(FACTS,384)])
def test_constrained_decode_passes_native_tokenizer_ids_and_stage_specific_eos(monkeypatch,facts,max_tokens):
    import torch
    from PIL import Image
    from types import SimpleNamespace
    monkeypatch.setattr(pilot,'CONSTRAINED',True)
    class Inputs(dict):
        def to(self,device):return self
    class Tokenizer:
        def encode(self,text,**kwargs):assert kwargs=={'add_special_tokens':False};return [30,31]
    class Processor:
        tokenizer=Tokenizer()
        def apply_chat_template(self,*a,**kw):return Inputs(input_ids=torch.tensor([[10,11]]))
        def batch_decode(self,*a,**kw):return [json.dumps(FACTS) if facts is None else CAPTION]
    class Model:
        generation_config=SimpleNamespace(eos_token_id=[151645,151643])
        def generate(self,**kw):
            assert kw['do_sample'] is False and kw['max_new_tokens']==max_tokens and kw['bad_words_ids']
            assert ('logits_processor' in kw)==(facts is not None)
            return torch.tensor([[10,11,100,151645]])
    out=pilot.generate_text(Model(),Processor(),[Image.new('RGB',(640,352)) for _ in range(16)],'cpu',max_tokens,facts)
    assert out==(json.dumps(FACTS) if facts is None else CAPTION)



def test_runtime_owner_check_records_intrusion_without_foreign_signals(monkeypatch):
    from types import SimpleNamespace
    process=SimpleNamespace(owned_members=lambda owner,tracked:[{'pid':3}],
        query_release=lambda uuid:dict(gpu_uuid=uuid,pids=[3]))
    monkeypatch.setattr(pilot,'helpers',lambda:(None,process,None,None))
    p=dict(gpu_uuids=['GPU-0','GPU-1'])
    assert pilot.foreign_gpu_observations(p,{'pid':1},{'pid':2},[])==[]
    process.query_release=lambda uuid:dict(gpu_uuid=uuid,pids=[3,77] if uuid=='GPU-1' else [3])
    assert pilot.foreign_gpu_observations(p,{'pid':1},{'pid':2},[])==[{'gpu_uuid':'GPU-1','pids':[3,77]}]
    assert not hasattr(process,'kill') and not hasattr(process,'cleanup_owned')
