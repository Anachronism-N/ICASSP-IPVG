import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip('numpy')
cv2 = pytest.importorskip('cv2')

spec = importlib.util.spec_from_file_location('curricularface_eval_test',
    Path(__file__).resolve().parents[1] / 'scripts/curricularface_eval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def face(axis=0, sign=1., count=1, padded=False):
    vector = np.zeros(512, dtype=np.float32)
    vector[axis] = sign
    return {'cur_embedding': vector, 'arc_embedding': vector.copy(),
            'face_count': count, 'padded': padded}


class Detector:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.images = []

    def detect(self, image, max_num, metric):
        assert max_num == 0 and metric == 'default'
        self.images.append(image.copy())
        return next(self.replies)


def test_floor_sampling_endpoints_and_short_video_repeats():
    expected = np.linspace(0, 48, 16).astype(int).tolist()
    assert module.frame_indices(49) == expected
    assert expected[1] == 3 and expected[3] == 9
    assert len(expected) == 16 and expected[-1] == 48
    assert module.frame_indices(1) == [0] * 16
    with pytest.raises(ValueError):
        module.frame_indices(0)
    with pytest.raises(ValueError):
        module.frame_indices(49, samples=8)


@pytest.mark.parametrize('empty', [(np.empty((0, 5)), np.empty((0, 5, 2))), ([], []), (None, None)])
def test_empty_detector_retries_once_with_gray_quarter_padding(empty):
    detector = Detector([empty, empty])
    result, reason = module.largest_face(detector, np.zeros((80, 120, 3), dtype=np.uint8))
    assert result is None and reason == 'no_face_after_padding'
    assert len(detector.images) == 2 and detector.images[1].shape == (120, 180, 3)
    assert np.all(detector.images[1][:20] == 128)
    assert np.all(detector.images[1][20:100, 30:150] == 0)


def test_largest_bbox_selected_without_padding_or_identity_matching():
    boxes = np.array([[2, 2, 12, 12, .99], [10, 10, 60, 50, .51]], dtype=np.float32)
    points = np.arange(20, dtype=np.float32).reshape(2, 5, 2)
    detector = Detector([(boxes, points)])
    result, reason = module.largest_face(detector, np.zeros((80, 80, 3), dtype=np.uint8))
    assert reason is None and result['face_count'] == 2 and not result['padded']
    np.testing.assert_array_equal(result['bbox'], boxes[1, :4])
    np.testing.assert_array_equal(result['kps'], points[1])
    assert len(detector.images) == 1


def test_padding_coordinates_return_to_original_anchor_frame():
    points = np.tile([35., 25.], (1, 5, 1))
    detector = Detector([([], []), (np.array([[30, 20, 60, 60, .9]]), points)])
    result, reason = module.largest_face(detector, np.zeros((80, 120, 3), dtype=np.uint8))
    assert reason is None and result['padded']
    np.testing.assert_array_equal(result['bbox'], [0, 0, 30, 40])
    np.testing.assert_array_equal(result['kps'], np.tile([5., 5.], (5, 1)))


def test_equal_areas_use_native_last_detection():
    boxes = np.array([[1, 2, 11, 12], [20, 30, 30, 40]])
    points = np.zeros((2, 5, 2), dtype=np.float32)
    result, _ = module.largest_face(Detector([(boxes, points)]), np.zeros((80, 80, 3), dtype=np.uint8))
    np.testing.assert_array_equal(result['bbox'], boxes[1])


@pytest.mark.parametrize('boxes,points', [
    (np.array([[0, 0, 0, 5]]), np.zeros((1, 5, 2))),
    (np.array([[0, 0, 10, 5]]), np.zeros((1, 4, 2))),
    (np.array([[0, 0, float('nan'), 5]]), np.zeros((1, 5, 2))),
])
def test_invalid_detection_is_hard_error(boxes, points):
    with pytest.raises(ValueError, match='Detector'):
        module.largest_face(Detector([(boxes, points)]), np.zeros((80, 80, 3), dtype=np.uint8))


def test_curricular_preprocess_rgb_shape_dtype_and_scale():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[:] = [0, 128, 255]
    result = module.curricular_input(image)
    assert result.shape == (3, 112, 112) and result.dtype == np.float32
    assert result.flags.c_contiguous
    np.testing.assert_allclose(result[:, 0, 0], [1., 128 / 127.5 - 1., -1.], atol=1e-7)
    with pytest.raises(ValueError):
        module.curricular_input(np.zeros((112, 112, 3), dtype=np.uint8))


@pytest.mark.parametrize('value', [np.zeros(512, dtype=np.float32), np.ones(256, dtype=np.float32),
    np.full(512, np.nan, dtype=np.float32), np.full(512, np.inf, dtype=np.float32),
    np.ones(512, dtype=np.int64), np.ones((2, 512), dtype=np.float32)])
def test_invalid_embedding_is_hard_error(value):
    with pytest.raises(ValueError, match='embedding'):
        module.normalize_embedding(value)


def test_embedding_is_unit_float32_and_batch_dimension_supported():
    result = module.normalize_embedding(np.ones((1, 512), dtype=np.float64))
    assert result.shape == (512,) and result.dtype == np.float32
    assert np.linalg.norm(result) == pytest.approx(1.)
    assert float(result @ result) == pytest.approx(1.)


def test_author_missing_video_zero_distinct_from_strict_missing():
    author = module.score_frames(face(), [None] * 16)
    strict = module.score_frames(face(), [None] * 16, 'omit_missing_video')
    assert author['cur_score'] == author['arc_score'] == 0.
    assert strict['cur_score'] is None and strict['arc_score'] is None
    assert author['cur_frame_cosines'] == strict['cur_frame_cosines'] == [None] * 16
    assert author['face_detection_coverage'] == 0
    assert author['missing_policy'] != strict['missing_policy']


@pytest.mark.parametrize('frames', [[None] * 16, [face()] * 16])
def test_missing_reference_is_never_imputed_zero(frames):
    result = module.score_frames(None, frames)
    assert not result['anchor_available']
    assert result['cur_score'] is None and result['arc_score'] is None
    assert result['cur_frame_cosines'] == result['arc_frame_cosines'] == [None] * 16
    assert result['cur_scoring_coverage'] == 0
    assert result['face_detection_coverage'] == sum(item is not None for item in frames) / 16


def test_valid_mean_clamps_negative_cosine_without_missing_frame_imputation():
    result = module.score_frames(face(), [face(), face(sign=-1), face(axis=1, count=2, padded=True)] + [None] * 13)
    assert result['cur_frame_cosines'][:3] == [1., 0., 0.]
    assert result['arc_score'] == pytest.approx(1 / 3)
    assert result['cur_score'] == pytest.approx(1 / 3)
    assert result['face_detection_coverage'] == 3 / 16
    assert result['multi_face_frames'] == result['padding_recovered_frames'] == 1
    with pytest.raises(ValueError, match='sixteen'):
        module.score_frames(face(), [None] * 8)


class Capture:
    def __init__(self, opened=True, ok=True):
        self.opened, self.ok, self.released = opened, ok, False
        self.indices = []

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return 49

    def set(self, prop, index):
        self.indices.append(index)

    def read(self):
        return self.ok, np.zeros((80, 80, 3), dtype=np.uint8) if self.ok else None

    def release(self):
        self.released = True


@pytest.mark.parametrize('opened,ok', [(False, True), (True, False)])
def test_video_open_and_decode_failure_abort_and_release(monkeypatch, opened, ok):
    capture = Capture(opened, ok)
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda _: capture)
    encoder = SimpleNamespace(process_image=lambda _: (face(), None))
    with pytest.raises(ValueError, match='Video|decoded'):
        module.video_features('unused.mp4', encoder)
    assert capture.released


def test_video_features_retain_missing_frames_and_floor_positions(monkeypatch):
    capture = Capture()
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda _: capture)
    encoder = SimpleNamespace(process_image=lambda _: (None, 'no_face_after_padding'))
    faces, reasons, indices = module.video_features('unused.mp4', encoder)
    assert faces == [None] * 16 and reasons == ['no_face_after_padding'] * 16
    assert indices == capture.indices == module.frame_indices(49)
    assert capture.released


def frozen_sample(tmp_path):
    video, reference = tmp_path / 'video.mp4', tmp_path / 'reference.png'
    video.write_bytes(b'video fixture')
    reference.write_bytes(b'reference fixture')
    return {'sample_id': 'fixture', 'group': 'source', 'prompt': 'fixture prompt',
            'video_path': str(video), 'video_sha256': module.sha256_file(video),
            'reference_image': str(reference), 'reference_sha256': module.sha256_file(reference)}


def test_frozen_manifest_requires_hashes_unique_ids_original_reference(tmp_path):
    row = frozen_sample(tmp_path)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'samples': [row]}))
    assert module.read_frozen_manifest(manifest) == [row]
    for rows in ([row, row], [dict(row, reference_crop='crop.png')],
                 [dict(row, video_sha256='wrong')], [dict(row, identity_id=None)]):
        manifest.write_text(json.dumps({'samples': rows}))
        with pytest.raises(ValueError):
            module.read_frozen_manifest(manifest)


def test_media_hash_checked_before_encoder_and_after_video(monkeypatch, tmp_path):
    row = frozen_sample(tmp_path)
    encoder = SimpleNamespace(process_image=lambda _: (face(), None))
    monkeypatch.setattr(module.cv2, 'imread', lambda _: np.zeros((80, 80, 3), dtype=np.uint8))
    def mutate_video(path, _):
        Path(path).write_bytes(b'changed')
        return [face()] * 16, [None] * 16, module.frame_indices(49)
    monkeypatch.setattr(module, 'video_features', mutate_video)
    with pytest.raises(ValueError, match='hash changed'):
        module.evaluate([row], encoder)
    calls = []
    encoder.process_image = lambda _: calls.append(True)
    with pytest.raises(ValueError, match='hash changed'):
        module.evaluate([row], encoder)
    assert calls == []


def test_reference_hash_checked_after_video(monkeypatch, tmp_path):
    row = frozen_sample(tmp_path)
    monkeypatch.setattr(module.cv2, 'imread', lambda _: np.zeros((80, 80, 3), dtype=np.uint8))
    def mutate_reference(path, _):
        Path(row['reference_image']).write_bytes(b'changed')
        return [face()] * 16, [None] * 16, module.frame_indices(49)
    monkeypatch.setattr(module, 'video_features', mutate_reference)
    with pytest.raises(ValueError, match='reference_image'):
        module.evaluate([row], SimpleNamespace(process_image=lambda _: (face(), None)))


def test_report_missing_anchor_cause_and_no_embeddings_or_media_paths(monkeypatch, tmp_path):
    row = frozen_sample(tmp_path)
    monkeypatch.setattr(module.cv2, 'imread', lambda _: np.zeros((80, 80, 3), dtype=np.uint8))
    monkeypatch.setattr(module, 'video_features', lambda *_: ([face()] * 16, [None] * 16, module.frame_indices(49)))
    records = module.evaluate([row], SimpleNamespace(process_image=lambda _: (None, 'no_face_after_padding')))
    record = records[0]
    assert record['cur_score'] is None and record['arc_score'] is None
    assert record['anchor_missing_reason'] == 'no_face_after_padding'
    assert record['official_score'] is None and record['face_detection_coverage'] == 1.
    serialized = json.dumps(records, allow_nan=False)
    assert 'embedding' not in serialized and str(tmp_path) not in serialized


def locked_fixture(tmp_path, monkeypatch):
    upstream = tmp_path / 'upstream'
    source = upstream / module.SOURCE_FILE
    source.parent.mkdir(parents=True)
    source.write_text('from .common import identity\ndef IR_101(size):\n    return identity(size)\n')
    common = upstream / module.COMMON_FILE
    common.write_text('def identity(value):\n    return value\n')
    (source.parent / '__init__.py').write_text('raise AssertionError("unused native package init executed")\n')
    face_dir = tmp_path / 'face'
    face_dir.mkdir()
    for name in module.FACE_HASHES:
        (face_dir / name).write_bytes(b'face fixture')
    weights = tmp_path / module.WEIGHT_NAME
    weights.write_bytes(b'cur fixture')
    lock = {'upstream_commit': module.UPSTREAM_COMMIT,
            'source_files': {module.SOURCE_FILE: module.sha256_file(source),
                             module.COMMON_FILE: module.sha256_file(common)},
            'files': dict(module.FACE_HASHES, **{module.WEIGHT_NAME: module.WEIGHT_SHA256})}
    monkeypatch.setattr(module, 'SOURCE_HASHES', dict(lock['source_files']))
    lock_path = tmp_path / 'model-lock.json'
    lock_path.write_text(json.dumps(lock))
    actual_hash = module.sha256_file
    expected = dict(module.FACE_HASHES, **{module.WEIGHT_NAME: module.WEIGHT_SHA256})
    monkeypatch.setattr(module, 'sha256_file', lambda p: expected.get(Path(p).name, actual_hash(p)))
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=module.UPSTREAM_COMMIT + '\n'))
    return upstream, weights, face_dir, lock_path, lock


def test_lock_validated_before_native_import_and_model_load(monkeypatch, tmp_path):
    args = locked_fixture(tmp_path, monkeypatch)
    verified = module.verify_model_lock(*args[:4])
    assert verified['files'][module.WEIGHT_NAME] == module.WEIGHT_SHA256
    assert verified['upstream_commit'] == module.UPSTREAM_COMMIT
    assert module.load_native_backbone(args[0]) == [112, 112]
    (args[0] / module.SOURCE_FILE).write_text('raise AssertionError("untrusted code executed")\n')
    with pytest.raises(ValueError, match='source hash mismatch'):
        module.OfflineCurEncoder(*args[:4], device='cpu')
    with pytest.raises(ValueError, match='source changed before import'):
        module.load_native_backbone(args[0])


def test_source_missing_wrong_git_pin_and_source_escape_fail(monkeypatch, tmp_path):
    upstream, weights, face_dir, lock_path, lock = locked_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='0' * 40))
    with pytest.raises(ValueError, match='revision'):
        module.verify_model_lock(upstream, weights, face_dir, lock_path)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=module.UPSTREAM_COMMIT))
    lock['source_files']['../escape.py'] = '0' * 64
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match='local source path'):
        module.verify_model_lock(upstream, weights, face_dir, lock_path)
    lock['source_files'] = {}
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match='source SHA256'):
        module.verify_model_lock(upstream, weights, face_dir, lock_path)


def test_weights_lock_mismatch_and_missing_models_never_download(monkeypatch, tmp_path):
    upstream, weights, face_dir, lock_path, lock = locked_fixture(tmp_path, monkeypatch)
    lock['files'][module.WEIGHT_NAME] = '0' * 64
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match='unpinned'):
        module.verify_model_lock(upstream, weights, face_dir, lock_path)
    lock['files'][module.WEIGHT_NAME] = module.WEIGHT_SHA256
    lock_path.write_text(json.dumps(lock))
    weights.unlink()
    with pytest.raises(FileNotFoundError, match='downloads disabled'):
        module.verify_model_lock(upstream, weights, face_dir, lock_path)


class FakeTensor:
    def __init__(self, shape=(2,), dtype='float32', finite=True):
        self.shape, self.dtype, self.finite = shape, dtype, finite

    def numel(self):
        return int(np.prod(self.shape))


@pytest.mark.parametrize('kind', ['keys', 'shape', 'dtype', 'finite', 'valid'])
def test_checkpoint_weights_only_strict_finite_shape_dtype_without_weight_io(kind):
    expected = {'weight': FakeTensor()}
    state = {'weight': FakeTensor()}
    if kind == 'keys':
        state = {}
    elif kind == 'shape':
        state['weight'].shape = (3,)
    elif kind == 'dtype':
        state['weight'].dtype = 'float16'
    elif kind == 'finite':
        state['weight'].finite = False
    calls = []
    def load(path, **kwargs):
        assert kwargs == {'map_location': 'cpu', 'weights_only': True}
        calls.append(path)
        return state
    torch = SimpleNamespace(load=load, is_tensor=lambda t: isinstance(t, FakeTensor),
        isfinite=lambda t: SimpleNamespace(all=lambda: t.finite))
    loaded = []
    model = SimpleNamespace(state_dict=lambda: expected,
                            load_state_dict=lambda s, strict: loaded.append((s, strict)))
    if kind == 'valid':
        result = module.load_checkpoint(model, 'never-read.bin', torch)
        assert result['tensor_count'] == 1 and result['parameter_elements'] == 2
        assert loaded == [(state, True)]
    else:
        with pytest.raises(ValueError, match='checkpoint'):
            module.load_checkpoint(model, 'never-read.bin', torch)
        assert not loaded
    assert calls == ['never-read.bin']


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_encoder_initialization_explicit_cpu_ort_320_four_threads_and_single_torch_device(monkeypatch, device):
    import sys
    calls = []
    class Model:
        def eval(self):
            calls.append(('eval',))
            return self
        def requires_grad_(self, value):
            calls.append(('requires_grad', value))
            return self
        def to(self, **kwargs):
            calls.append(('to', kwargs))
            return self
    class OrtOptions:
        pass
    def session(path, sess_options, providers):
        calls.append(('session', Path(path).name, sess_options.intra_op_num_threads,
                      sess_options.inter_op_num_threads, providers))
        return object()
    class DetectorModel:
        def __init__(self, model_file, session):
            assert session is not None
        def prepare(self, **kwargs):
            calls.append(('detector_prepare', kwargs))
    class RecognitionModel:
        def __init__(self, model_file, session):
            assert session is not None
        def prepare(self, **kwargs):
            calls.append(('recognition_prepare', kwargs))
    monkeypatch.setattr(module, 'verify_model_lock', lambda *_: {'files': {}})
    monkeypatch.setattr(module, 'load_native_backbone', lambda _: Model())
    monkeypatch.setattr(module, 'load_checkpoint', lambda *_: {'tensor_count': 1})
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(set_num_threads=lambda n: calls.append(('threads', n)),
                        float32='float32', cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(SessionOptions=OrtOptions, InferenceSession=session))
    monkeypatch.setitem(sys.modules, 'insightface.model_zoo.scrfd', SimpleNamespace(SCRFD=DetectorModel))
    monkeypatch.setitem(sys.modules, 'insightface.model_zoo.arcface_onnx', SimpleNamespace(ArcFaceONNX=RecognitionModel))
    monkeypatch.setitem(sys.modules, 'insightface.utils.face_align', SimpleNamespace(norm_crop=lambda *a, **k: None))
    encoder = module.OfflineCurEncoder('upstream', 'weights', 'face-dir', 'lock', device=device)
    assert encoder.device == ('cuda:0' if device == 'cuda' else 'cpu')
    assert ('threads', 4) in calls
    assert ('to', {'device': encoder.device, 'dtype': 'float32'}) in calls
    assert ('detector_prepare', {'ctx_id': 0, 'input_size': (320, 320), 'det_thresh': .5}) in calls
    sessions = [entry for entry in calls if entry[0] == 'session']
    assert len(sessions) == 2 and all(entry[2:] == (4, 1, ['CPUExecutionProvider']) for entry in sessions)


@pytest.mark.parametrize('padded', [False, True])
def test_process_image_native_cur_original_arc_detection_image_alignment_without_gpu(padded):
    from contextlib import nullcontext
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    points = np.tile([35., 25.], (1, 5, 1))
    detection = (np.array([[30, 20, 60, 60, .9]]), points)
    encoder = module.OfflineCurEncoder.__new__(module.OfflineCurEncoder)
    encoder.detector = Detector([([], []), detection] if padded else [detection])
    align_calls = []
    def align(array, landmark, image_size):
        align_calls.append((array.shape, landmark.copy(), image_size))
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)
    class Tensor:
        def __init__(self, value):
            self.value = value
        def to(self, device):
            assert device == 'cpu'
            return self
        def float(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self.value
    encoder.align, encoder.device = align, 'cpu'
    encoder.torch = SimpleNamespace(from_numpy=Tensor, inference_mode=nullcontext)
    def cur_model(tensor):
        assert tensor.value.shape == (1, 3, 112, 112)
        assert tensor.value.dtype == np.float32
        return Tensor(np.ones((1, 512), dtype=np.float32))
    encoder.cur_model = cur_model
    encoder.recognition = SimpleNamespace(get_feat=lambda crop: np.ones((1, 512), dtype=np.float32))
    result, reason = encoder.process_image(image)
    assert reason is None and result['padded'] == padded
    assert align_calls[0][0] == image.shape and align_calls[0][2] == 224
    assert align_calls[1][2] == 112
    assert align_calls[1][0] == ((120, 180, 3) if padded else image.shape)
    np.testing.assert_array_equal(align_calls[1][1], points[0])
    np.testing.assert_array_equal(align_calls[0][1], points[0] - ([30, 20] if padded else [0, 0]))
    for name in ('cur_embedding', 'arc_embedding'):
        assert result[name].shape == (512,) and np.linalg.norm(result[name]) == pytest.approx(1.)


@pytest.mark.parametrize('failed', [False, True])
def test_cli_cpu_report_protocol_provenance_resources_and_hard_failure_without_models(monkeypatch, tmp_path, failed):
    import sys
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'samples': [frozen_sample(tmp_path)]}))
    output = tmp_path / 'output.json'
    argv = ['cur-eval', '--manifest', str(manifest), '--upstream', 'upstream', '--weights', 'weights',
            '--face-model-dir', 'face', '--model-lock', 'lock', '--output', str(output), '--device', 'cpu']
    monkeypatch.setattr(sys, 'argv', argv)
    encoder = SimpleNamespace(device='cpu', model_source={'files': {}, 'source_files': {}})
    monkeypatch.setattr(module, 'OfflineCurEncoder', lambda *a: encoder)
    def evaluate(*args):
        if failed:
            raise ValueError('Sampled video frame cannot be decoded')
        return [dict(module.score_frames(face(), [None] * 16), sample_id='fixture')]
    monkeypatch.setattr(module, 'evaluate', evaluate)
    if failed:
        with pytest.raises(ValueError, match='decoded'):
            module.main()
        assert not output.exists()
    else:
        module.main()
        report = json.loads(output.read_text())
        assert report['device'] == 'cpu' and report['official_score'] is None
        assert report['resources']['gpus'] == 0 and report['resources']['estimated_gpu_hours'] == 0
        assert report['protocol']['frames'] == 16
        assert report['protocol']['missing_policy'] == 'author_all_missing_video_zero'
        assert report['manifest_sha256'] == module.sha256_file(manifest)
        with pytest.raises(FileExistsError):
            module.main()
