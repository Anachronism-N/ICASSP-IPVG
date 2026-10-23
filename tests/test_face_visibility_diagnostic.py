import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip('numpy')
cv2 = pytest.importorskip('cv2')
spec = importlib.util.spec_from_file_location('face_visibility_diagnostic_test',
    Path(__file__).resolve().parents[1] / 'scripts/face_visibility_diagnostic.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Detector:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.images = []

    def detect(self, image, max_num, metric):
        assert max_num == 0 and metric == 'default'
        self.images.append(image.copy())
        return next(self.replies), None


def budget(**kwargs):
    return module.Budget(clock=lambda: 0., **kwargs)


@pytest.mark.parametrize('empty', [None, [], np.empty((0, 5))])
def test_no_faces_retries_exactly_once_with_original_gray_quarter_padding(empty):
    detector = Detector([empty, empty])
    limits = budget()
    result = module.detect_frame(detector, np.zeros((80, 120, 3), np.uint8), limits, 320)
    assert result['face_count'] == 0 and result['category'] == 'no_face'
    assert result['padding_attempted'] and not result['padding_recovered']
    assert result['largest_bbox_normalized'] is None
    assert len(detector.images) == limits.detect_calls == 2
    padded = detector.images[1]
    assert padded.shape == (120, 180, 3)
    assert np.all(padded[:20] == 128) and np.all(padded[:, :30] == 128)
    assert np.all(padded[20:100, 30:150] == 0)
    assert limits.calls_by_kind == {'video_320_direct': 1, 'video_320_padding': 1}


def test_largest_multi_face_normalized_box_and_native_last_tie_without_retry():
    boxes = np.array([[0, 0, 10, 10, .99], [30, 20, 60, 60, .51], [60, 40, 90, 80, .6]])
    detector = Detector([boxes])
    result = module.detect_frame(detector, np.zeros((80, 120, 3), np.uint8), budget(), 640)
    assert result['face_count'] == 3 and result['category'] == 'multi_face'
    assert not result['padding_attempted'] and not result['padding_recovered']
    np.testing.assert_allclose(result['largest_bbox_normalized'], [.5, .5, .75, 1.])
    assert result['largest_bbox_width_normalized'] == .25
    assert result['largest_bbox_height_normalized'] == .5
    assert result['largest_bbox_area_normalized'] == .125
    assert len(detector.images) == 1


def test_padding_box_subtracts_original_offsets_not_clipped():
    detector = Detector([[], np.array([[20., 10., 60., 60., .9]])])
    result = module.detect_frame(detector, np.zeros((80, 120, 3), np.uint8), budget(), 320)
    assert result['category'] == 'single_face' and result['padding_recovered']
    np.testing.assert_allclose(result['largest_bbox_normalized'], [-10 / 120, -10 / 80, 30 / 120, 40 / 80])
    assert result['largest_bbox_area_normalized'] == pytest.approx((40 / 120) * (50 / 80))


@pytest.mark.parametrize('boxes', [np.array([[0, 0, 0, 10]]), np.array([[0, 0, np.nan, 10]]),
                                   np.array([[0, 0, 10]]), np.array([0, 0, 10, 10])])
def test_invalid_detector_boxes_abort(boxes):
    with pytest.raises(ValueError, match='Detector'):
        module.detect_frame(Detector([boxes]), np.zeros((80, 120, 3), np.uint8), budget(), 320)


@pytest.mark.parametrize('image', [np.zeros((80, 120), np.uint8), np.zeros((80, 120, 3), np.float32),
                                   np.zeros((0, 120, 3), np.uint8)])
def test_invalid_input_is_rejected_before_detection(image):
    detector = Detector([])
    with pytest.raises(ValueError, match='input'):
        module.detect_frame(detector, image, budget(), 320)
    assert not detector.images


def detection(found=True, multi=False, recovered=False):
    return {'face_count': (2 if multi else 1) if found else 0,
            'category': ('multi_face' if multi else 'single_face') if found else 'no_face',
            'padding_attempted': not found or recovered, 'padding_recovered': recovered}


def frame_rows(found_320=(), found_640=()):
    return [{'frame_index': index, 'scales': {
        '320': detection(index in found_320), '640': detection(index in found_640)}}
        for index in range(49)]


def test_floor_native_indices_and_full_vs_original_sample_recovery_are_distinct():
    expected = np.linspace(0, 48, 16, dtype=int).tolist()
    assert module.frame_indices() == expected
    assert expected[1] == 3 and expected[3] == 9 and expected[-1] == 48
    assert 1 not in expected
    summary = module.summarize_video(frame_rows(found_640=[1]))
    assert summary['scales']['320']['all_missing_video']
    counts = summary['scales']['640']
    assert counts['detected_frames'] == 1 and counts['full_frame_coverage'] == 1 / 49
    assert counts['original_16_all_missing_video']
    assert counts['original_16_missing_but_full_recovered_video']
    assert summary['scale_320_all_missing_to_640_recovered_video']
    assert summary['scale_320_missing_to_640_recovered_frames'] == 1
    with pytest.raises(ValueError, match='49'):
        module.summarize_video(frame_rows()[:-1])
    rows = frame_rows()
    rows[0]['frame_index'] = 1
    with pytest.raises(ValueError, match='sequential'):
        module.summarize_video(rows)


class Capture:
    def __init__(self, *, opened=True, fail_at=None, extra=False, metadata=None):
        self.opened, self.fail_at, self.extra = opened, fail_at, extra
        self.metadata = dict(frames=49, width=480, height=352, fps=16)
        self.metadata.update(metadata or {})
        self.reads, self.released = 0, False
        self.frame = np.zeros((352, 480, 3), np.uint8)

    def isOpened(self):
        return self.opened

    def get(self, prop):
        names = {cv2.CAP_PROP_FRAME_COUNT: 'frames', cv2.CAP_PROP_FRAME_WIDTH: 'width',
                 cv2.CAP_PROP_FRAME_HEIGHT: 'height', cv2.CAP_PROP_FPS: 'fps'}
        return self.metadata[names[prop]]

    def read(self):
        index = self.reads
        self.reads += 1
        if index == self.fail_at or index >= (50 if self.extra else 49):
            return False, None
        return True, self.frame

    def set(self, *args):
        raise AssertionError('Sequential decoding must not seek')

    def release(self):
        self.released = True


def test_all_49_frames_sequential_both_scales_and_eof_checked(monkeypatch):
    capture = Capture()
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda _: capture)
    detectors = {scale: Detector([np.array([[0, 0, 10, 10, .9]])] * 49) for scale in module.SCALES}
    limits, rows = budget(), []
    assert module.diagnose_video('unused.mp4', detectors, limits, rows) == capture.metadata
    assert capture.reads == limits.decode_calls == 50 and capture.released
    assert limits.decoded_frames == 49 and limits.detect_calls == 98
    assert [row['frame_index'] for row in rows] == list(range(49))
    assert all(set(row['scales']) == {'320', '640'} for row in rows)


@pytest.mark.parametrize('kwargs,message', [({'opened': False}, 'opened'),
    ({'fail_at': 7}, 'frame 7'), ({'extra': True}, 'more than 49'),
    ({'metadata': {'frames': 48}}, 'metadata'), ({'metadata': {'width': 352}}, 'metadata'),
    ({'metadata': {'height': 480}}, 'metadata'), ({'metadata': {'fps': 24}}, 'metadata')])
def test_decode_and_metadata_errors_abort_release_and_preserve_partial(monkeypatch, kwargs, message):
    capture = Capture(**kwargs)
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda _: capture)
    detectors = {scale: Detector([np.array([[0, 0, 10, 10]])] * 49) for scale in module.SCALES}
    rows = []
    with pytest.raises(ValueError, match=message):
        module.diagnose_video('unused.mp4', detectors, budget(), rows)
    assert capture.released
    if kwargs.get('fail_at') == 7:
        assert len(rows) == 7 and capture.reads == 8


def test_timeout_checked_before_each_detection_and_after_inflight_call():
    now = [0.]
    limits = module.Budget(timeout_s=1, clock=lambda: now[0])
    detector = Detector([[], []])
    now[0] = 1.
    with pytest.raises(module.DiagnosticBoundExceeded, match='wall-time'):
        module.detect_frame(detector, np.zeros((80, 120, 3), np.uint8), limits, 320)
    assert limits.detect_calls == 0 and not detector.images
    now[0] = 0.
    def slow_detect(*args, **kwargs):
        now[0] = 1.
        return [], None
    detector.detect = slow_detect
    with pytest.raises(module.DiagnosticBoundExceeded, match='wall-time'):
        module.detect_frame(detector, np.zeros((80, 120, 3), np.uint8), limits, 320)
    assert limits.detect_calls == 1


def test_timeout_before_decode_opens_no_capture(monkeypatch):
    now = [0.]
    limits = module.Budget(timeout_s=1, clock=lambda: now[0])
    now[0] = 1.
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda _: pytest.fail('Must not open capture after timeout'))
    with pytest.raises(module.DiagnosticBoundExceeded):
        module.diagnose_video('unused.mp4', {}, limits, [])


def test_call_limit_stops_padding_without_extra_detect_and_exact_campaign_bound():
    limits = budget(max_calls=1)
    detector = Detector([[], []])
    with pytest.raises(module.DiagnosticBoundExceeded, match='detection-call'):
        module.detect_frame(detector, np.zeros((80, 120, 3), np.uint8), limits, 320)
    assert limits.detect_calls == len(detector.images) == 1
    assert module.MAX_DETECT_CALLS == 48 * 49 * 2 * 2 + 8 * 2 * 2 == 9440
    limits = budget()
    limits.detect_calls = 9439
    limits.detect(Detector([[]]), np.zeros((1, 1, 3), np.uint8), 640, 'reference')
    assert limits.detect_calls == 9440
    with pytest.raises(module.DiagnosticBoundExceeded):
        limits.detect(Detector([]), np.zeros((1, 1, 3), np.uint8), 640, 'reference')


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan')])
def test_timeout_must_be_finite_positive(timeout):
    with pytest.raises(ValueError, match='Timeout'):
        module.Budget(timeout_s=timeout)


def fixture_rows(tmp_path, samples=2, references=1):
    anchors = []
    for index in range(references):
        reference = tmp_path / f'reference_{index}.png'
        reference.write_bytes(f'reference fixture {index}'.encode())
        anchors.append(reference)
    rows = []
    for index in range(samples):
        video = tmp_path / f'video_{index}.mp4'
        video.write_bytes(f'video fixture {index}'.encode())
        reference = anchors[index % references]
        rows.append({'sample_id': f'fixture_{index}', 'group': 'fixture_group',
                     'video_path': str(video), 'video_sha256': module.sha256_file(video),
                     'reference_image': str(reference), 'reference_sha256': module.sha256_file(reference)})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'samples': rows}))
    return rows, manifest


def test_manifest_fixture_count_parameter_and_strict_default_campaign_counts(tmp_path):
    rows, manifest = fixture_rows(tmp_path)
    assert module.read_manifest(manifest, expected_samples=2, expected_references=1) == rows
    with pytest.raises(ValueError, match='48'):
        module.read_manifest(manifest)
    with pytest.raises(ValueError, match='reference'):
        module.read_manifest(manifest, expected_samples=2, expected_references=8)


@pytest.mark.parametrize('mutation', ['id', 'path', 'hash', 'reference', 'crop', 'group'])
def test_manifest_duplicate_ids_video_paths_hashes_and_original_references(tmp_path, mutation):
    rows, manifest = fixture_rows(tmp_path)
    if mutation == 'id':
        rows[1]['sample_id'] = rows[0]['sample_id']
    elif mutation == 'path':
        rows[1]['video_path'] = rows[0]['video_path']
    elif mutation == 'hash':
        rows[0]['video_sha256'] = 'F' * 64
    elif mutation == 'reference':
        del rows[0]['reference_sha256']
    elif mutation == 'crop':
        rows[0]['reference_crop'] = 'wrong.png'
    else:
        rows[0]['group'] = ''
    manifest.write_text(json.dumps({'samples': rows}))
    with pytest.raises(ValueError):
        module.read_manifest(manifest, expected_samples=2, expected_references=1)


def test_conflicting_input_path_hashes_rejected_and_dedup_all_paths(tmp_path):
    rows, manifest = fixture_rows(tmp_path)
    model = tmp_path / module.MODEL_NAME
    model.write_bytes(b'model fixture')
    files = module.expected_files(rows, manifest, module.sha256_file(manifest), model)
    assert len(files) == 5
    rows[1]['reference_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Conflicting'):
        module.expected_files(rows, manifest, module.sha256_file(manifest), model)


def test_reference_detection_dedup_by_sha_even_different_path_and_partial_progress(monkeypatch, tmp_path):
    rows, _ = fixture_rows(tmp_path, samples=3, references=2)
    alias = tmp_path / 'reference_alias.png'
    alias.write_bytes(Path(rows[0]['reference_image']).read_bytes())
    rows[2]['reference_image'] = str(alias)
    imread_calls = []
    monkeypatch.setattr(module.cv2, 'imread', lambda path: imread_calls.append(path) or np.zeros((80, 120, 3), np.uint8))
    detectors = {scale: Detector([np.array([[0, 0, 10, 10]])] * 2) for scale in module.SCALES}
    report = {'local_references': [], 'local_videos': []}
    saves = []
    def video_error(*args):
        raise ValueError('fixture decode failure')
    monkeypatch.setattr(module, 'diagnose_video', video_error)
    limits = budget()
    with pytest.raises(ValueError, match='fixture decode'):
        module.evaluate(rows, detectors, limits, report, lambda: saves.append(True))
    assert len(imread_calls) == len(report['local_references']) == len(saves) == 2
    assert limits.detect_calls == 4
    assert len(report['local_videos']) == 1 and not report['local_videos'][0]['complete']


def test_aggregate_has_only_groups_counts_no_local_ids_hashes_paths_or_boxes():
    rows = frame_rows(found_640=[1])
    videos = [{'group': 'a', 'sample_id': 'private_id', 'video_path': '/private/path',
               'complete': True, 'frame_rows': rows, 'summary': module.summarize_video(rows)},
              {'group': 'a', 'complete': False, 'frame_rows': rows[:2]},
              {'group': 'b', 'complete': True, 'frame_rows': frame_rows(range(49), range(49)),
               'summary': module.summarize_video(frame_rows(range(49), range(49)))}]
    aggregate = module.aggregate_records(videos, [], 48)
    a = aggregate['groups']['a']
    assert a['completed_videos'] == 1 and a['started_videos'] == 2
    assert a['partial_fully_diagnosed_frames'] == 2
    assert a['scales']['320']['all_missing_videos'] == 1
    assert a['scales']['640']['original_16_missing_but_full_recovered_videos'] == 1
    assert a['scales']['640']['full_frame_coverage'] == 1 / 49
    assert a['scale_320_all_missing_to_640_recovered_videos'] == 1
    assert aggregate['completed_videos'] == 2
    serialized = json.dumps(aggregate)
    assert all(text not in serialized for text in ('private_id', '/private/path', 'bbox', 'sha256'))


def test_loader_direct_scrfd_two_explicit_cpu_sessions_threads_scales_no_recognition(monkeypatch, tmp_path):
    model_path = tmp_path / module.MODEL_NAME
    model_path.write_bytes(b'model fixture')
    monkeypatch.setattr(module, 'MODEL_SHA256', module.sha256_file(model_path))
    calls = []
    class Options:
        pass
    def session(path, sess_options, providers):
        calls.append(('session', path, sess_options.intra_op_num_threads, sess_options.inter_op_num_threads, providers))
        return SimpleNamespace(get_providers=lambda: providers)
    class SCRFD:
        def __init__(self, model_file, session):
            assert model_file == str(model_path) and session is not None
        def prepare(self, **kwargs):
            calls.append(('prepare', kwargs))
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(SessionOptions=Options, InferenceSession=session))
    monkeypatch.setitem(sys.modules, 'insightface.model_zoo.scrfd', SimpleNamespace(SCRFD=SCRFD))
    detectors = module.load_detectors(model_path)
    assert set(detectors) == {320, 640}
    assert len([row for row in calls if row[0] == 'session']) == 2
    assert all(row[2:] == (4, 1, ['CPUExecutionProvider']) for row in calls if row[0] == 'session')
    assert [row[1] for row in calls if row[0] == 'prepare'] == [
        {'ctx_id': -1, 'input_size': (320, 320), 'det_thresh': .5},
        {'ctx_id': -1, 'input_size': (640, 640), 'det_thresh': .5}]


def test_loader_missing_or_unpinned_model_rejected_before_ort_import(tmp_path):
    model_path = tmp_path / module.MODEL_NAME
    with pytest.raises(FileNotFoundError, match='downloads disabled'):
        module.load_detectors(model_path)
    model_path.write_bytes(b'not audited model')
    with pytest.raises(ValueError, match='SHA256'):
        module.load_detectors(model_path)


def cli_fixture(monkeypatch, tmp_path):
    rows, manifest = fixture_rows(tmp_path)
    model = tmp_path / module.MODEL_NAME
    model.write_bytes(b'model fixture')
    monkeypatch.setattr(module, 'MODEL_SHA256', module.sha256_file(model))
    native_read = module.read_manifest
    monkeypatch.setattr(module, 'read_manifest', lambda p: native_read(p, expected_samples=2, expected_references=1))
    monkeypatch.setattr(module, 'load_detectors', lambda p: {})
    output = tmp_path / 'diagnostic.json'
    argv = ['--manifest', str(manifest), '--model-dir', str(tmp_path), '--output', str(output)]
    return rows, manifest, model, output, argv


def test_cli_fresh_complete_report_cpu_hashes_and_existing_output_never_overwritten(monkeypatch, tmp_path):
    rows, manifest, model, output, argv = cli_fixture(monkeypatch, tmp_path)
    def evaluate(samples, detectors, limits, report, progress):
        for sample in samples:
            record = dict(sample, complete=True, frame_rows=frame_rows())
            record['summary'] = module.summarize_video(record['frame_rows'])
            report['local_videos'].append(record)
            progress()
            assert json.loads(output.read_text())['status'] == 'incomplete'
    monkeypatch.setattr(module, 'evaluate', evaluate)
    module.main(argv)
    report = json.loads(output.read_text())
    assert report['complete'] and report['status'] == 'complete'
    assert report['hash_verification'] == {'pre': True, 'post': True}
    assert report['diagnostic_only'] and report['official_score'] is None
    assert report['resources']['gpus'] == report['resources']['gpu_hours'] == 0
    assert report['resources']['max_detect_calls'] == 9440
    assert report['aggregate']['completed_videos'] == 2
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.main(argv)
    assert output.read_bytes() == original


@pytest.mark.parametrize('failure', ['decode', 'call_bound', 'timeout'])
def test_cli_failure_writes_incomplete_partial_counts_and_reraises(monkeypatch, tmp_path, failure):
    rows, manifest, model, output, argv = cli_fixture(monkeypatch, tmp_path)
    def evaluate(samples, detectors, limits, report, progress):
        record = dict(samples[0], complete=False, frame_rows=frame_rows()[:3])
        report['local_videos'].append(record)
        limits.detect_calls = 12
        limits.decoded_frames = 3
        if failure == 'decode':
            raise ValueError('fixture decode failed')
        raise module.DiagnosticBoundExceeded('fixture ' + failure)
    monkeypatch.setattr(module, 'evaluate', evaluate)
    with pytest.raises((ValueError, module.DiagnosticBoundExceeded)):
        module.main(argv)
    report = json.loads(output.read_text())
    assert report['status'] == 'incomplete' and not report['complete']
    assert report['resources']['detect_calls'] == 12 and report['resources']['decoded_frames'] == 3
    assert report['aggregate']['completed_videos'] == 0
    assert report['aggregate']['groups']['fixture_group']['partial_fully_diagnosed_frames'] == 3
    assert report['hash_verification'] == {'pre': True, 'post': True}
    assert report['failure']['type'] == ('ValueError' if failure == 'decode' else 'DiagnosticBoundExceeded')


@pytest.mark.parametrize('target', ['video', 'reference', 'manifest', 'model'])
def test_cli_post_hash_drift_rejected_for_all_inputs_and_weights(monkeypatch, tmp_path, target):
    rows, manifest, model, output, argv = cli_fixture(monkeypatch, tmp_path)
    paths = {'video': Path(rows[0]['video_path']), 'reference': Path(rows[0]['reference_image']),
             'manifest': manifest, 'model': model}
    monkeypatch.setattr(module, 'evaluate', lambda *args: paths[target].write_bytes(b'changed'))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        module.main(argv)
    report = json.loads(output.read_text())
    assert not report['complete'] and report['hash_verification'] == {'pre': True, 'post': False}
    assert report['post_verification_failure']['type'] == 'ValueError'


def test_cli_pre_hash_drift_prevents_model_loading_and_detection(monkeypatch, tmp_path):
    rows, manifest, model, output, argv = cli_fixture(monkeypatch, tmp_path)
    Path(rows[0]['video_path']).write_bytes(b'changed')
    monkeypatch.setattr(module, 'load_detectors', lambda _: pytest.fail('Unverified inputs must not load detector'))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        module.main(argv)
    report = json.loads(output.read_text())
    assert report['status'] == 'incomplete'
    assert report['resources']['detect_calls'] == 0
    assert report['hash_verification'] == {'pre': False, 'post': False}


def test_main_native_manifest_count_enforced_and_invalid_manifest_saved_incomplete(monkeypatch, tmp_path):
    _, manifest = fixture_rows(tmp_path)
    output = tmp_path / 'diagnostic.json'
    with pytest.raises(ValueError, match='48'):
        module.main(['--manifest', str(manifest), '--model-dir', str(tmp_path), '--output', str(output)])
    report = json.loads(output.read_text())
    assert report['status'] == 'incomplete' and report['resources']['detect_calls'] == 0
    assert not report['hash_verification']['pre']
