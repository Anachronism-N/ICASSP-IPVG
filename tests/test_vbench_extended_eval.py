import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from contextlib import nullcontext
import urllib.request

import pytest

spec = importlib.util.spec_from_file_location(
    'vbench_extended_eval', Path(__file__).resolve().parents[1] / 'scripts/vbench_extended_eval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def good_probe(path):
    return {'ok': True, 'width': 64, 'height': 32, 'decoded_frames': 8,
            'fps': 16.0, 'duration_s': .5, 'codec': 'h264'}


def sample(tmp_path, sid='a', prompt='The original action, not the refined generation prompt.'):
    path = tmp_path / (sid + '.mp4')
    path.write_bytes(b'CPU placeholder, not a generated quality example')
    return {'sample_id': sid, 'video_path': str(path), 'video_sha256': module.sha256_file(path),
            'prompt': prompt, 'group': 'control'}


def test_manifest_preserves_original_and_hashes(tmp_path):
    row = sample(tmp_path)
    actual = module.validate_manifest({'samples': [row]}, good_probe)[0]
    assert actual['prompt'] == row['prompt']
    assert actual['raft_interval'] == 2
    assert actual['raft_sampled_frames'] == 4
    assert row.get('media') is None


@pytest.mark.parametrize('field,value', [
    ('sample_id', '../unsafe'), ('sample_id', None), ('prompt', None), ('prompt', ''),
    ('prompt', ['wrong']), ('group', ''), ('video_path', 'relative.mp4'),
    ('video_sha256', '0' * 64), ('video_sha256', 'bad'),
])
def test_invalid_manifest_fields_fail(tmp_path, field, value):
    row = sample(tmp_path)
    row[field] = value
    with pytest.raises(ValueError):
        module.validate_manifest({'samples': [row]}, good_probe)


@pytest.mark.parametrize('rows', [None, [], 'wrong', [None]])
def test_nonempty_object_manifest_required(rows):
    with pytest.raises(ValueError):
        module.validate_manifest({'samples': rows}, good_probe)


def test_duplicate_ids_and_duplicate_resolved_paths_fail(tmp_path):
    row = sample(tmp_path)
    with pytest.raises(ValueError, match='sample_id'):
        module.validate_manifest({'samples': [row, row]}, good_probe)
    alias = dict(row, sample_id='alias')
    with pytest.raises(ValueError, match='paths must be unique'):
        module.validate_manifest({'samples': [row, alias]}, good_probe)


@pytest.mark.parametrize('changes', [
    {'ok': False}, {'width': 31}, {'height': 31}, {'decoded_frames': 1},
    {'fps': float('nan')}, {'fps': 0}, {'fps': True}, {'decoded_frames': 2, 'fps': 120.0},
])
def test_media_and_raft_sampling_gotchas(tmp_path, changes):
    row = sample(tmp_path)
    with pytest.raises(ValueError):
        module.validate_manifest({'samples': [row]}, lambda p: dict(good_probe(p), **changes))


def test_unique_private_scratch_uses_original_prompt_and_is_removed(tmp_path):
    row = sample(tmp_path)
    row['refined_prompt'] = 'must never be evaluated'
    output = tmp_path / 'out.json'
    with module.dimension_info([row], output) as first:
        assert first.parent.parent == tmp_path
        assert first.stat().st_mode & 0o777 == 0o600
        data = json.loads(first.read_text())
        assert data == [{'prompt_en': row['prompt'], 'dimension': list(module.DIMENSIONS),
                         'video_list': [row['video_path']]}]
        with module.dimension_info([row], output) as second:
            assert first != second
            assert second.exists()
        assert not second.exists()
    assert not first.parent.exists()
    assert not output.exists()


def test_scratch_removed_on_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with module.dimension_info([sample(tmp_path)], tmp_path / 'out.json') as info:
            raise RuntimeError('expected')
    assert not info.parent.exists()


def test_native_values_no_renormalization_and_equal_video_weight(tmp_path):
    rows = [sample(tmp_path, 'a'), sample(tmp_path, 'b')]
    details = [{'video_path': rows[0]['video_path'], 'video_results': .2},
               {'video_path': rows[1]['video_path'], 'video_results': .8}]
    scores, summary = module.collect_dimension(rows, 'subject_consistency', .75, details)
    assert list(scores.values()) == [.2, .8]
    assert summary['mean'] == .5
    assert summary['native_aggregate'] == .75
    assert summary['coverage'] == {'expected': 2, 'scored': 2, 'fraction': 1.0}
    assert module.finite_score(-.3, 'overall_consistency') == -.3
    assert module.finite_score(1.2, 'aesthetic_quality') == 1.2
    assert module.finite_score(True, 'dynamic_degree') == 1.0
    assert module.finite_score(False, 'dynamic_degree') == 0.0


@pytest.mark.parametrize('value', [None, '0.1', float('nan'), float('inf'), -float('inf'), True])
def test_nonfinite_and_non_numeric_scores_rejected(value):
    with pytest.raises(ValueError):
        module.finite_score(value, 'overall_consistency')


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'unknown'])
def test_no_hidden_per_dimension_skips(tmp_path, kind):
    rows = [sample(tmp_path)]
    detail = [{'video_path': rows[0]['video_path'], 'video_results': .5}]
    if kind == 'missing':
        detail = []
    elif kind == 'duplicate':
        detail *= 2
    else:
        detail[0]['video_path'] = '/unknown.mp4'
    with pytest.raises(ValueError):
        module.collect_dimension(rows, 'aesthetic_quality', .5, detail)


def make_lock(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    cache.mkdir()
    files = {}
    for name in module.ASSETS.values():
        path = cache / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = module.sha256_file(path)
    repo = tmp_path / 'dino'
    repo.mkdir()
    source = {}
    for name in ('hubconf.py', 'vision_transformer.py', 'utils.py'):
        path = repo / name
        path.write_text('# CPU fixture\n')
        source[name] = module.sha256_file(path)
    lock = {'vbench_revision': module.VBENCH_REVISION, 'files': files,
            'dino_repo': str(repo), 'dino_source': source}
    lock_path = tmp_path / 'lock.json'
    lock_path.write_text(json.dumps(lock))
    calls = []

    def fake_git(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=module.VBENCH_REVISION + '\n' if command[-2:] == ['rev-parse', 'HEAD'] else '')

    monkeypatch.setattr(module.subprocess, 'run', fake_git)
    return cache, repo, lock, lock_path, calls


def test_reviewed_lock_complete_and_local_symlinks_allowed(tmp_path, monkeypatch):
    cache, repo, lock, path, calls = make_lock(tmp_path, monkeypatch)
    source = cache / module.ASSETS['dino']
    shared = tmp_path / 'approved_shared.pth'
    source.rename(shared)
    source.symlink_to(shared)
    validated, dino = module.validate_local(tmp_path / 'vbench', cache, path)
    assert validated == lock
    assert dino == repo
    assert any('diff' in call for call in calls)
    assert any('ls-files' in call for call in calls)


@pytest.mark.parametrize('failure', ['revision', 'missing', 'hash', 'vocab', 'traversal', 'source', 'extra_source'])
def test_incomplete_or_changed_asset_lock_fails_before_import(tmp_path, monkeypatch, failure):
    cache, repo, lock, path, _ = make_lock(tmp_path, monkeypatch)
    if failure == 'revision':
        lock['vbench_revision'] = 'wrong'
    elif failure == 'missing':
        (cache / module.ASSETS['raft']).unlink()
    elif failure == 'hash':
        lock['files'][module.ASSETS['dino']] = '0' * 64
    elif failure == 'vocab':
        del lock['files'][module.ASSETS['vocab']]
    elif failure == 'traversal':
        lock['files']['../escape'] = '0' * 64
    elif failure == 'source':
        lock['dino_source']['hubconf.py'] = '0' * 64
    else:
        (repo / 'new.py').write_text('# unreviewed')
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        module.validate_local(tmp_path / 'vbench', cache, path)


def test_wrong_actual_revision_rejected(tmp_path, monkeypatch):
    cache, _, _, path, _ = make_lock(tmp_path, monkeypatch)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='wrong'))
    with pytest.raises(ValueError, match='revision'):
        module.validate_local(tmp_path / 'vbench', cache, path)


def test_dirty_upstream_rejected(tmp_path, monkeypatch):
    cache, _, _, path, _ = make_lock(tmp_path, monkeypatch)

    def dirty_git(command, **kwargs):
        if 'diff' in command:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout=module.VBENCH_REVISION + '\n')

    monkeypatch.setattr(module.subprocess, 'run', dirty_git)
    with pytest.raises(subprocess.CalledProcessError):
        module.validate_local(tmp_path / 'vbench', cache, path)


def test_dino_explicit_local_state_never_hub_url_default(tmp_path):
    calls = []
    model = SimpleNamespace()
    model.load_state_dict = lambda state, **kw: calls.append(('state', state, kw))
    model.float = lambda: model
    model.to = lambda device: model
    model.eval = lambda: model
    torch = SimpleNamespace(
        hub=SimpleNamespace(load=lambda *a, **kw: calls.append(('hub', a, kw)) or model),
        load=lambda *a, **kw: calls.append(('load', a, kw)) or {'local': 'weights'})
    assert module.load_dino(torch, tmp_path, tmp_path / 'weight.pth', 'cpu') is model
    assert calls[0] == ('hub', (str(tmp_path), 'dino_vitb16'), {'source': 'local', 'pretrained': False})
    assert calls[1][2] == {'map_location': 'cpu', 'weights_only': True}
    assert calls[2] == ('state', {'local': 'weights'}, {'strict': True})


def test_offline_guard_blocks_network_subprocess_and_cached_url_loaders():
    torch = pytest.importorskip('torch')
    import numpy.testing  # Preinitialize local CPU capability probing.
    from torch.utils import model_zoo
    original = (torch.hub.load_state_dict_from_url, model_zoo.load_url, subprocess.Popen)
    with module.model_offline(torch):
        for fn, args in [
            (socket.create_connection, (('127.0.0.1', 9),)),
            (urllib.request.urlopen, ('https://example.invalid',)),
            (subprocess.Popen, (['wget'],)),
            (torch.hub.load_state_dict_from_url, ('https://example.invalid/weights.pth',)),
            (model_zoo.load_url, ('https://example.invalid/weights.pth',)),
        ]:
            with pytest.raises(RuntimeError, match='forbidden'):
                fn(*args)
    assert (torch.hub.load_state_dict_from_url, model_zoo.load_url, subprocess.Popen) == original


def test_native_token_overflow_reported_not_refined(tmp_path):
    rows = [sample(tmp_path, prompt='a' * 31)]
    result = module.prompt_tokenization(rows, SimpleNamespace(encode=lambda text: list(text)))
    assert result[rows[0]['video_path']] == {'context_length': 32, 'tokens_with_special': 33,
                                           'truncated': True, 'truncate': True}
    assert 'prompt' not in result[rows[0]['video_path']]


def test_native_model_wiring_never_uses_download_initializers(tmp_path, monkeypatch):
    rows = module.validate_manifest({'samples': [sample(tmp_path)]}, good_probe)
    calls = []

    class Model:
        dtype = 'float16'

        def parameters(self):
            return iter([SimpleNamespace(dtype='float32'), SimpleNamespace(dtype='float16')])

        def float(self):
            calls.append('float')
            return self

        def to(self, device):
            return self

        def eval(self):
            calls.append('eval')
            return self

        def load_state_dict(self, state, **kwargs):
            assert kwargs == {'strict': True}

    def native(dimension, *args, **kwargs):
        calls.append((dimension, args, kwargs))
        value = True if dimension == 'dynamic_degree' else -.2 if dimension == 'overall_consistency' else .6
        return float(value), [{'video_path': rows[0]['video_path'], 'video_results': value}]

    def load_clip(path, **kwargs):
        assert Path(path).suffix == '.pt'
        assert kwargs == {'device': 'cpu', 'jit': False}
        calls.append(('clip', path))
        return Model(), 'preprocess'

    clip = ModuleType('clip')
    clip.load = load_clip
    monkeypatch.setitem(sys.modules, 'clip', clip)
    utils = ModuleType('vbench.utils')
    utils.load_dimension_info = lambda path, **kw: ([], [
        {'prompt': json.loads(Path(path).read_text())[0]['prompt_en'], 'video_list': [rows[0]['video_path']]}])
    monkeypatch.setitem(sys.modules, 'vbench.utils', utils)
    modules = {}
    for dimension in module.DIMENSIONS:
        fake = ModuleType('vbench.' + dimension)
        modules[fake.__name__] = fake
        setattr(fake, dimension if dimension != 'aesthetic_quality' else 'laion_aesthetic',
                lambda *a, dimension=dimension, **kw: native(dimension, *a, **kw))
    dynamic = modules['vbench.dynamic_degree']
    dynamic.edict = dict

    def raft(args, device):
        assert args['mixed_precision'] is False
        assert args['small'] is False
        assert args['model'] == str(tmp_path / module.ASSETS['raft'])
        return SimpleNamespace(model=Model())

    dynamic.DynamicDegree = raft
    overall = modules['vbench.overall_consistency']
    overall.SimpleTokenizer = lambda path: SimpleNamespace(encode=lambda text: list(text))
    overall.ViCLIP = lambda **kw: calls.append(('viclip', kw)) or Model()

    def text_features(model, prompt, tokenizer, cache={'stale': 'must clear'}):
        return cache

    overall.get_text_features = text_features
    monkeypatch.setattr(module.importlib, 'import_module', lambda name: modules[name])
    monkeypatch.setattr(module, 'load_dino', lambda *a: Model())
    torch = SimpleNamespace(inference_mode=nullcontext, nn=SimpleNamespace(Linear=lambda *a: Model()),
                            load=lambda *a, **kw: {}, cuda=SimpleNamespace(empty_cache=lambda: None))
    with module.dimension_info(rows, tmp_path / 'scores.json') as info:
        result, precision, tokens = module.evaluate_native(info, rows, tmp_path, tmp_path, torch, device='cpu')
    assert set(result) == set(module.DIMENSIONS)
    assert result['dynamic_degree'][0][rows[0]['video_path']] == 1.0
    assert result['overall_consistency'][0][rows[0]['video_path']] == -.2
    native_calls = [call for call in calls if isinstance(call, tuple) and call[0] in module.DIMENSIONS]
    assert len(native_calls) == 5
    assert next(c for c in native_calls if c[0] == 'subject_consistency')[1][-2:] == ('cpu', False)
    assert next(c for c in native_calls if c[0] == 'background_consistency')[1][-2:] == ('cpu', False)
    assert next(c for c in native_calls if c[0] == 'overall_consistency')[2] == {'sample': 'middle'}
    assert next(c for c in native_calls if c[0] == 'overall_consistency')[1][1][0]['prompt'] == rows[0]['prompt']
    assert precision['background_consistency']['parameter_dtypes'] == ['float16', 'float32']
    assert precision['background_consistency']['encode_image_dtype'] == 'float16'
    assert text_features.__defaults__[0] == {}
    assert tokens[rows[0]['video_path']]['truncated'] is True


def test_cli_offline_mocked_output_has_all_dimensions_and_no_prompt_or_path(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    row = sample(tmp_path)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'samples': [row]}))
    lock_path = tmp_path / 'lock.json'
    lock_path.write_text('{}')
    output = tmp_path / 'scores.json'
    lock = {'files': {'local.pt': '0' * 64}, 'dino_source': {'hubconf.py': '1' * 64}}
    monkeypatch.setattr(module, 'validate_local', lambda *a: (lock, tmp_path))
    original_validation = module.validate_manifest
    monkeypatch.setattr(module, 'validate_manifest', lambda request: original_validation(request, good_probe))
    for name, value in [('is_available', True), ('device_count', 1), ('max_memory_allocated', 128),
                        ('max_memory_reserved', 256)]:
        monkeypatch.setattr(torch.cuda, name, lambda value=value: value)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)

    def fake_evaluate(info, rows, cache, repo, torch):
        assert json.loads(info.read_text())[0]['prompt_en'] == row['prompt']
        assert module.os.environ['VBENCH_CACHE_DIR'] == str(cache.resolve())
        dims = {d: module.collect_dimension(rows, d, .5,
                [{'video_path': row['video_path'], 'video_results': .5}]) for d in module.DIMENSIONS}
        return dims, {d: 'torch.float32' for d in module.DIMENSIONS}, module.prompt_tokenization(
            rows, SimpleNamespace(encode=lambda text: [1]))

    monkeypatch.setattr(module, 'evaluate_native', fake_evaluate)
    module.main(['--manifest', str(manifest), '--vbench', str(tmp_path / 'vbench'),
                 '--cache', str(tmp_path / 'cache'), '--model-lock', str(lock_path), '--output', str(output)])
    actual = json.loads(output.read_text())
    assert set(actual['records'][0]['scores']) == set(module.DIMENSIONS)
    assert set(actual['dimensions']) == set(module.DIMENSIONS)
    assert actual['official_score'] is None
    assert actual['protocol']['official_final_score'] is None
    assert actual['resource']['allocated_gpu_hours'] >= 0
    assert actual['records'][0]['video_sha256'] == row['video_sha256']
    assert row['prompt'] not in output.read_text()
    assert row['video_path'] not in output.read_text()
    assert not list(tmp_path.glob('.vbench-info-*'))
    with pytest.raises(FileExistsError):
        module.main(['--manifest', str(manifest), '--vbench', str(tmp_path), '--cache', str(tmp_path),
                     '--model-lock', str(lock_path), '--output', str(output)])
