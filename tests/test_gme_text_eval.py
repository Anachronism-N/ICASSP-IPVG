import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

np = pytest.importorskip('numpy')
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gme_text_eval', ROOT / 'scripts/gme_text_eval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def model_lock(tmp_path):
    model = tmp_path / 'gme'
    model.mkdir()
    names = ['config.json', 'modeling_gme_qwen2vl.py', 'model.safetensors.index.json',
             'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json',
             'special_tokens_map.json', 'vocab.json', 'merges.txt', 'added_tokens.json',
             'chat_template.json', 'generation_config.json']
    shards = [f'model-{i:05d}-of-00009.safetensors' for i in range(1, 10)]
    for name in names + shards:
        (model / name).write_text('{}')
    (model / 'config.json').write_text(json.dumps({
        'auto_map': {'AutoConfig': 'modeling_gme_qwen2vl.GmeQwen2VLConfig',
                     'AutoModel': 'modeling_gme_qwen2vl.GmeQwen2VL'},
        '_name_or_path': 'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct'}))
    (model / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {f'fixture.tensor.{i}': name for i, name in enumerate(shards)}}))
    lock = tmp_path / 'model.lock.json'
    lock.write_text(json.dumps({'revision': module.MODEL_REVISION,
                               'files': {name: module.sha256_file(model / name) for name in names + shards}}))
    return model, lock


@pytest.fixture
def frozen_manifest(tmp_path):
    video = tmp_path / 'local.mp4'
    video.write_bytes(b'local fixture, never a generated quality result')
    row = {'sample_id': 'case1', 'video_path': str(video), 'video_sha256': module.sha256_file(video),
           'prompt': '  The exact original prompt.  ', 'group': 'fixture'}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'samples': [row]}))
    return path, row


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape
        self.ndim = self.values.ndim

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


@pytest.fixture
def fake_torch(monkeypatch):
    torch = SimpleNamespace(inference_mode=contextlib.nullcontext, float16='fp16', float32='fp32',
                            set_num_threads=lambda _: None, __version__='fixture')
    monkeypatch.setitem(sys.modules, 'torch', torch)
    return torch


@pytest.mark.parametrize('count', [1, 49, 81])
def test_all_frames_retained_under_limit(count):
    assert module.frame_indices(count).tolist() == list(range(count))


def test_frame_selection_uses_floor_not_rounding():
    assert len(module.frame_indices(49)) == 49
    expected = np.linspace(0, 98, 81, dtype=int)
    assert module.frame_indices(99).tolist() == expected.tolist()
    assert not np.array_equal(expected, np.linspace(0, 98, 81).round().astype(int))
    assert expected[0] == 0 and expected[-1] == 98


@pytest.mark.parametrize('count', [0, -1, True, 49.0, None])
def test_invalid_frame_count_fails(count):
    with pytest.raises(ValueError):
        module.frame_indices(count)


def test_aesr_is_raw_dot_arithmetic_mean_with_no_scaling():
    result = module.frame_text_scores([[1, 0], [0, 1], [-1, 0]], [1, 0])
    assert result['frame_scores'] == [1.0, 0.0, -1.0]
    assert result['gme_aesr'] == 0.0
    assert result['finite_frame_count'] == result['unit_norm_frame_count'] == 3
    assert result['embedding_dimension'] == 2
    assert module.frame_text_scores(np.tile([-1, 0], (49, 1)), [1, 0])['gme_aesr'] == -1.0


def test_fp16_unit_norm_tolerance_does_not_renormalize_or_clip():
    assert module.frame_text_scores([[1.005, 0]], [1, 0])['gme_aesr'] == pytest.approx(1.005)


@pytest.mark.parametrize('images,text', [([], [1, 0]), (np.zeros((1, 2)), [1, 0]),
                                        ([[2, 0]], [1, 0]), ([[1, 0]], [0, 0]),
                                        ([[1, 0]], [1]), ([[1, 0]], [np.nan, 0]),
                                        ([[np.inf, 0]], [1, 0]), ([[np.nan, 0]], [1, 0]),
                                        (np.tile([1, 0], (82, 1)), [1, 0])])
def test_invalid_or_nonunit_embeddings_fail(images, text):
    with pytest.raises(ValueError):
        module.frame_text_scores(images, text)


def test_model_lock_covers_complete_local_native_weights_and_metadata(model_lock):
    model, lock = model_lock
    resolved, hashes = module.validate_model_lock(model, lock)
    assert resolved == model.resolve()
    assert len([name for name in hashes if name.endswith('.safetensors')]) == 9
    assert hashes['modeling_gme_qwen2vl.py'] == module.sha256_file(model / 'modeling_gme_qwen2vl.py')
    cache = model / '.cache/huggingface'
    cache.mkdir(parents=True)
    (cache / 'download.metadata').write_text('ignored download metadata')
    assert module.validate_model_lock(model, lock)[1] == hashes


@pytest.mark.parametrize('change', ['revision', 'missing_code', 'missing_weight_record', 'changed_weight',
                                   'extra_file', 'path_escape', 'absolute_path', 'symlink_escape',
                                   'bad_hash', 'remote_auto_map', 'incomplete_index'])
def test_changed_missing_unlocked_or_remote_model_fails(model_lock, tmp_path, change):
    model, lock = model_lock
    value = module.load_json(lock)
    if change == 'revision':
        value['revision'] = '0' * 40
    elif change == 'missing_code':
        value['files'].pop('modeling_gme_qwen2vl.py')
    elif change == 'missing_weight_record':
        value['files'].pop('model-00009-of-00009.safetensors')
    elif change == 'changed_weight':
        (model / 'model-00001-of-00009.safetensors').write_bytes(b'changed fixture')
    elif change == 'extra_file':
        (model / 'unlocked.py').write_text('unlocked code')
    elif change == 'path_escape':
        value['files']['../outside'] = '0' * 64
    elif change == 'absolute_path':
        value['files'][str(tmp_path / 'outside')] = '0' * 64
    elif change == 'symlink_escape':
        outside = tmp_path / 'outside'
        outside.write_text('outside')
        (model / 'escaped.py').symlink_to(outside)
        value['files']['escaped.py'] = module.sha256_file(outside)
    elif change == 'bad_hash':
        value['files']['config.json'] = 'z' * 64
    elif change == 'remote_auto_map':
        config = model / 'config.json'
        config.write_text(json.dumps({'auto_map': {'AutoModel': 'unapproved/repo--modeling.Model'}}))
        value['files']['config.json'] = module.sha256_file(config)
    elif change == 'incomplete_index':
        index = model / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'one': 'model-00001-of-00009.safetensors'}}))
        value['files'][index.name] = module.sha256_file(index)
    lock.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.validate_model_lock(model, lock)


def test_manifest_keeps_every_group_and_exact_prompt_without_dev_filter(frozen_manifest):
    path, row = frozen_manifest
    second = dict(row, sample_id='holdout_fixture', split='holdout', group='other')
    path.write_text(json.dumps({'samples': [row, second]}))
    assert [item['sample_id'] for item in module.validate_manifest(path)] == ['case1', 'holdout_fixture']
    assert module.validate_manifest(path)[0]['prompt'] == row['prompt']


@pytest.mark.parametrize('change', ['empty', 'duplicate', 'missing_prompt', 'blank_prompt', 'nonstring_prompt',
                                   'bad_hash', 'missing_hash', 'remote_path', 'missing_file', 'missing_id'])
def test_invalid_frozen_manifest_fails(frozen_manifest, change):
    path, row = frozen_manifest
    rows = [dict(row)]
    if change == 'empty':
        rows = []
    elif change == 'duplicate':
        rows.append(dict(row))
    elif change == 'missing_prompt':
        rows[0].pop('prompt')
    elif change == 'blank_prompt':
        rows[0]['prompt'] = '  '
    elif change == 'nonstring_prompt':
        rows[0]['prompt'] = 3
    elif change == 'bad_hash':
        rows[0]['video_sha256'] = '0' * 64
    elif change == 'missing_hash':
        rows[0].pop('video_sha256')
    elif change == 'remote_path':
        rows[0]['video_path'] = 'https://fixture.invalid/video.mp4'
    elif change == 'missing_file':
        rows[0]['video_path'] += '.missing'
    elif change == 'missing_id':
        rows[0].pop('sample_id')
    path.write_text(json.dumps({'samples': rows}))
    with pytest.raises(ValueError):
        module.validate_manifest(path)


def test_relative_video_path_is_local_to_manifest(frozen_manifest):
    path, row = frozen_manifest
    row['video_path'] = Path(row['video_path']).name
    path.write_text(json.dumps({'samples': [row]}))
    assert module.validate_manifest(path)[0]['video_path'] == str(path.parent / row['video_path'])


@pytest.mark.parametrize('device,expected_dtype', [('cuda', 'fp16'), ('cpu', 'fp32')])
def test_native_loader_is_local_and_rebinds_constructor_processor_path(monkeypatch, fake_torch, tmp_path, device, expected_dtype):
    calls = []
    config = SimpleNamespace(_name_or_path='remote/model')
    native = SimpleNamespace()
    native.eval = lambda: native
    native.requires_grad_ = lambda enabled: native if enabled is False else None

    def load_config(path, **kwargs):
        calls.append(('config', path, kwargs))
        return config

    processor_calls = []
    processor = SimpleNamespace(from_pretrained=lambda path, **kwargs: processor_calls.append((path, kwargs)))
    original_processor = processor.from_pretrained

    def load_model(path, **kwargs):
        assert kwargs['config']._name_or_path == str(tmp_path)
        processor.from_pretrained(kwargs['config']._name_or_path, min_pixels=256 * 28 * 28)
        with pytest.raises(ValueError, match='locked local model'):
            processor.from_pretrained('unapproved/model')
        calls.append(('model', path, kwargs))
        return native

    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=load_config), AutoModel=SimpleNamespace(from_pretrained=load_model),
        AutoProcessor=processor))
    assert module.load_native_model(tmp_path, device) is native
    assert all(call[1] == str(tmp_path) and call[2]['local_files_only'] for call in calls)
    assert calls[1][2]['torch_dtype'] == expected_dtype
    assert calls[1][2]['device_map'] == device
    assert calls[1][2]['attn_implementation'] == 'sdpa'
    assert processor.from_pretrained is original_processor
    assert processor_calls == [(str(tmp_path), {'min_pixels': 256 * 28 * 28, 'local_files_only': True})]


def test_native_embed_uses_exact_text_and_local_image_semantics(monkeypatch, fake_torch, frozen_manifest):
    _, row = frozen_manifest
    frames = [object() for _ in range(49)]
    monkeypatch.setattr(module, 'decode_video', lambda _: (frames, {
        'decoded_frame_count': 49, 'selected_frame_indices': list(range(49)),
        'scored_frame_count': 49, 'dimensions': [16, 12], 'decode_coverage': 1.0}))
    calls = []

    def embed(**kwargs):
        calls.append(kwargs)
        if kwargs['images'] == [None]:
            assert kwargs['texts'] == [row['prompt']]
            assert kwargs['instruction'] == module.TEXT_INSTRUCTION
            return Tensor([[1, 0]])
        assert kwargs['texts'] == [None] * len(kwargs['images'])
        assert kwargs['is_query'] is False
        assert all(image in frames for image in kwargs['images'])
        return Tensor(np.tile([1, 0], (len(kwargs['images']), 1)))

    result = module.score_sample(SimpleNamespace(embed=embed), row, 4)
    assert result['scores'] == {'gme_aesr': 1.0}
    assert len(result['frame_scores']) == result['scored_frame_count'] == 49
    assert result['prompt_sha256'] == hashlib.sha256(row['prompt'].encode()).hexdigest()
    assert 'prompt' not in result and 'video_path' not in result
    assert [len(call['images']) for call in calls[1:]] == [4] * 12 + [1]


def test_video_change_during_embedding_fails(monkeypatch, fake_torch, frozen_manifest):
    _, row = frozen_manifest
    monkeypatch.setattr(module, 'decode_video', lambda _: ([object()], {'scored_frame_count': 1}))

    def embed(**kwargs):
        Path(row['video_path']).write_bytes(b'changed during evaluation')
        return Tensor([[1, 0]])

    with pytest.raises(ValueError, match='changed during'):
        module.score_sample(SimpleNamespace(embed=embed), row, 1)


@pytest.mark.parametrize('count,failed_frame,extra', [(49, None, False), (100, None, False),
                                                     (49, 7, False), (100, 7, False), (49, None, True)])
def test_decode_checks_all_frames_and_preserves_floor_selection(monkeypatch, count, failed_frame, extra):
    reads = []
    released = []
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)

    class Capture:
        def isOpened(self):
            return True

        def get(self, _):
            return count

        def read(self):
            index = len(reads)
            reads.append(index)
            if index == failed_frame or index >= count + int(extra):
                return False, None
            return True, rgb

        def release(self):
            released.append(True)

    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(VideoCapture=lambda _: Capture(), CAP_PROP_FRAME_COUNT=7,
                                                          COLOR_BGR2RGB=1, cvtColor=lambda image, _: image))
    monkeypatch.setitem(sys.modules, 'PIL', SimpleNamespace(Image=SimpleNamespace(fromarray=lambda image: image)))
    if failed_frame is not None or extra:
        with pytest.raises(ValueError, match='decode failure|frame count differs'):
            module.decode_video('local-fixture.mp4')
    else:
        frames, coverage = module.decode_video('local-fixture.mp4')
        assert coverage['decoded_frame_count'] == count
        assert coverage['selected_frame_indices'] == module.frame_indices(count).tolist()
        assert len(frames) == min(count, 81)
        assert len(reads) == count + 1
        assert coverage['dimensions'] == [16, 12]
    assert released == [True]


def test_cli_refuses_existing_output_before_loading_model(monkeypatch, tmp_path):
    output = tmp_path / 'existing.json'
    output.write_text('never overwrite')
    monkeypatch.setattr(sys, 'argv', ['gme', '--manifest', 'missing', '--model', 'missing',
                                     '--model-lock', 'missing', '--output', str(output)])
    with pytest.raises(FileExistsError):
        module.main()
    assert output.read_text() == 'never overwrite'


def test_cli_cpu_orchestration_offline_guard_and_redacted_metadata(monkeypatch, fake_torch, model_lock, frozen_manifest, tmp_path):
    model, lock = model_lock
    manifest, row = frozen_manifest
    output = tmp_path / 'output.json'
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    import numpy.testing
    class LazyRuntime:
        __version__ = 'fixture'
        def __getattr__(self, name):
            if name not in ('PreTrainedModel', 'Qwen2VLForConditionalGeneration'):
                raise AttributeError(name)
            assert subprocess.Popen is original_popen
            return object()
    original_popen = subprocess.Popen
    monkeypatch.setitem(sys.modules, 'transformers', LazyRuntime())
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(__version__='fixture'))
    monkeypatch.setitem(sys.modules, 'PIL', SimpleNamespace(__version__='fixture'))
    monkeypatch.setattr(sys, 'argv', ['gme', '--manifest', str(manifest), '--model', str(model),
                                     '--model-lock', str(lock), '--output', str(output), '--device', 'cpu'])
    original = socket.create_connection, subprocess.Popen

    def load_native(path, device):
        assert path == model and device == 'cpu'
        assert module.os.environ['HF_HUB_OFFLINE'] == module.os.environ['TRANSFORMERS_OFFLINE'] == '1'
        for function, args in [(socket.create_connection, (('127.0.0.1', 9),)), (subprocess.Popen, (['wget'],))]:
            with pytest.raises(RuntimeError, match='forbidden'):
                function(*args)
        return SimpleNamespace(max_length=1800, config=SimpleNamespace(min_image_tokens=256, max_image_tokens=1280))

    monkeypatch.setattr(module, 'load_native_model', load_native)
    monkeypatch.setattr(module, 'score_sample', lambda *args: {'sample_id': row['sample_id'], 'scores': {'gme_aesr': 0.25}})
    module.main()
    result = module.load_json(output)
    assert (socket.create_connection, subprocess.Popen) == original
    assert result['model_revision'] == module.MODEL_REVISION
    assert result['model_file_sha256'] == module.load_json(lock)['files']
    assert result['model_lock_sha256'] == module.sha256_file(lock)
    assert result['manifest_sha256'] == module.sha256_file(manifest)
    assert result['execution']['device_mode'] == 'cpu_validation_not_cuda_equivalence'
    assert result['execution']['batch_size'] == 1
    assert result['protocol']['image_default_instruction'] == 'You are a helpful assistant.'
    assert 'truncation=True' in result['protocol']['native_text_processing']
    assert result['execution']['native_max_length'] == 1800
    assert 'without evaluator renormalization' in result['protocol']['normalization']
    assert result['resource']['gpus'] == result['resource']['allocated_gpu_hours'] == 0
    assert result['resource']['elapsed_seconds'] >= 0
    assert result['official_score'] is None
    assert row['prompt'] not in output.read_text()
    assert str(model) not in output.read_text()
