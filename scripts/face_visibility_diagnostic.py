"""Bounded offline CPU face-detection diagnostics; never identity or quality scores.

The full output contains restricted per-frame rows and must stay local. Only its
aggregate section is suitable for a separate publication review.
"""
import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np

MODEL_NAME = 'det_10g.onnx'
MODEL_SHA256 = '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91'
SCALES = (320, 640)
FRAME_COUNT = 49
WIDTH, HEIGHT, FPS = 480, 352, 16
MAX_DETECT_CALLS = 9440
PROTOCOL = {
    'version': 'face_visibility_diagnostic_cpu_v1',
    'kind': 'detection_diagnostic_not_identity_score',
    'scales': list(SCALES), 'det_threshold': .5,
    'expected_video': {'frames': FRAME_COUNT, 'width': WIDTH, 'height': HEIGHT, 'fps': FPS},
    'frame_sampling': 'all 49 sequentially decoded; original floor np.linspace(0,48,16,dtype=int)',
    'padding_retry': 'only after zero detections; original image with 25% per-edge gray 128 padding',
    'bbox_policy': 'largest area; native last-detection tie; subtract padding, normalize to original image; no clipping',
    'reference_policy': 'original reference_image; detect once per distinct SHA256 at each scale',
    'provider': 'CPUExecutionProvider', 'ort_intra_op_threads': 4, 'ort_inter_op_threads': 1,
    'identity_assignment': False, 'existing_scores_modified': False,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def frame_indices():
    return np.linspace(0, FRAME_COUNT - 1, 16, dtype=int).tolist()


def read_manifest(path, expected_samples=48, expected_references=8):
    with Path(path).open(encoding='utf-8') as handle:
        document = json.load(handle)
    rows = document.get('samples') if isinstance(document, dict) else None
    if not isinstance(rows, list) or len(rows) != expected_samples:
        raise ValueError(f'Manifest requires exactly {expected_samples} samples')
    required = ('sample_id', 'group', 'video_path', 'video_sha256', 'reference_image', 'reference_sha256')
    ids, videos, references = set(), set(), set()
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key] for key in required):
            raise ValueError('Manifest sample fields must be nonempty strings')
        if any(not re.fullmatch('[0-9a-f]{64}', row[key]) for key in ('video_sha256', 'reference_sha256')):
            raise ValueError('Manifest SHA256 must be lowercase hexadecimal')
        if any(key in row for key in ('reference_crop', 'reference_images', 'original_reference_images')):
            raise ValueError('Use original reference_image, not conditioning crops')
        video = str(Path(row['video_path']).resolve())
        if row['sample_id'] in ids or video in videos:
            raise ValueError('Manifest requires unique sample IDs and unique video paths')
        ids.add(row['sample_id'])
        videos.add(video)
        references.add(row['reference_sha256'])
    if len(references) != expected_references:
        raise ValueError(f'Manifest requires exactly {expected_references} unique reference SHA256 values')
    return rows


def expected_files(rows, manifest, manifest_sha256, model_path):
    files = {str(Path(manifest).resolve()): manifest_sha256,
             str(Path(model_path).resolve()): MODEL_SHA256}
    for row in rows:
        for path_key, hash_key in (('video_path', 'video_sha256'), ('reference_image', 'reference_sha256')):
            path = str(Path(row[path_key]).resolve())
            if path in files and files[path] != row[hash_key]:
                raise ValueError('Conflicting SHA256 values for one input path')
            files[path] = row[hash_key]
    return files


def verify_files(files, budget=None):
    for path, expected in files.items():
        if budget is not None:
            budget.check()
        if sha256_file(path) != expected:
            raise ValueError('Input or detector SHA256 mismatch: ' + path)
        if budget is not None:
            budget.check()


class DiagnosticBoundExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, timeout_s=1800, max_calls=MAX_DETECT_CALLS, clock=None):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('Timeout must be finite and positive')
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
            raise ValueError('Detection call bound must be a positive integer')
        self.clock = time.monotonic if clock is None else clock
        self.started = self.clock()
        self.timeout_s, self.max_calls = timeout_s, max_calls
        self.detect_calls = 0
        self.decode_calls = 0
        self.decoded_frames = 0
        self.calls_by_kind = {}

    def check(self):
        if self.clock() - self.started >= self.timeout_s:
            raise DiagnosticBoundExceeded('Diagnostic wall-time bound exceeded')

    def detect(self, detector, image, scale, kind, padded=False):
        self.check()
        if self.detect_calls >= self.max_calls:
            raise DiagnosticBoundExceeded('Diagnostic detection-call bound exceeded')
        self.detect_calls += 1
        key = f'{kind}_{scale}_' + ('padding' if padded else 'direct')
        self.calls_by_kind[key] = self.calls_by_kind.get(key, 0) + 1
        result = detector.detect(image, max_num=0, metric='default')
        self.check()
        return result

    def resources(self):
        return {'wall_seconds': self.clock() - self.started, 'timeout_s': self.timeout_s,
                'detect_calls': self.detect_calls, 'max_detect_calls': self.max_calls,
                'calls_by_kind': dict(self.calls_by_kind), 'decode_calls': self.decode_calls,
                'decoded_frames': self.decoded_frames, 'gpus': 0, 'gpu_hours': 0,
                'ort_provider': 'CPUExecutionProvider',
                'ort_intra_op_threads': 4, 'ort_inter_op_threads': 1}


def load_detectors(model_path):
    if not Path(model_path).is_file():
        raise FileNotFoundError('Existing explicit det_10g.onnx required; downloads disabled')
    if sha256_file(model_path) != MODEL_SHA256:
        raise ValueError('Detector SHA256 mismatch')
    import onnxruntime as ort
    from insightface.model_zoo.scrfd import SCRFD
    detectors = {}
    for scale in SCALES:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        if session.get_providers() != ['CPUExecutionProvider']:
            raise RuntimeError('Detector must use CPUExecutionProvider only')
        detector = SCRFD(model_file=str(model_path), session=session)
        detector.prepare(ctx_id=-1, input_size=(scale, scale), det_thresh=.5)
        detectors[scale] = detector
    return detectors


def detect_frame(detector, image, budget, scale, kind='video'):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Detection input must be a uint8 BGR image')
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError('Detection input dimensions must be positive')
    boxes, _ = budget.detect(detector, image, scale, kind)
    padded, offset = False, np.zeros(4, dtype=np.float64)
    if boxes is None or len(boxes) == 0:
        pad_x, pad_y = int(width * .25), int(height * .25)
        expanded = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x,
                                     cv2.BORDER_CONSTANT, value=(128, 128, 128))
        boxes, _ = budget.detect(detector, expanded, scale, kind, padded=True)
        padded, offset = True, np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float64)
    result = {'face_count': 0, 'category': 'no_face', 'padding_attempted': padded,
              'padding_recovered': False, 'largest_bbox_normalized': None,
              'largest_bbox_area_normalized': None, 'largest_bbox_width_normalized': None,
              'largest_bbox_height_normalized': None}
    if boxes is None or len(boxes) == 0:
        return result
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[1] < 4 or not np.isfinite(boxes).all():
        raise ValueError('Detector returned invalid boxes')
    sizes = boxes[:, 2:4] - boxes[:, :2]
    if (sizes <= 0).any():
        raise ValueError('Detector returned nonpositive boxes')
    areas = np.prod(sizes, axis=1)
    largest = int(np.flatnonzero(areas == areas.max())[-1])
    bbox = boxes[largest, :4].astype(np.float64) - offset
    normalized = bbox / np.array([width, height, width, height], dtype=np.float64)
    box_width, box_height = float(normalized[2] - normalized[0]), float(normalized[3] - normalized[1])
    result.update(face_count=len(boxes), category='single_face' if len(boxes) == 1 else 'multi_face',
                  padding_recovered=padded, largest_bbox_normalized=normalized.tolist(),
                  largest_bbox_area_normalized=box_width * box_height,
                  largest_bbox_width_normalized=box_width, largest_bbox_height_normalized=box_height)
    return result


def summarize_video(frame_rows):
    if len(frame_rows) != FRAME_COUNT or [row['frame_index'] for row in frame_rows] != list(range(FRAME_COUNT)):
        raise ValueError('Summary requires all 49 sequential frame rows')
    indices = frame_indices()
    scales = {}
    for scale in SCALES:
        rows = [frame['scales'][str(scale)] for frame in frame_rows]
        found = [row['face_count'] > 0 for row in rows]
        detected, sampled = sum(found), sum(found[index] for index in indices)
        scales[str(scale)] = {
            'decoded_frames': FRAME_COUNT, 'detected_frames': detected,
            'full_frame_coverage': detected / FRAME_COUNT, 'all_missing_video': detected == 0,
            'all_frames_detected_video': detected == FRAME_COUNT,
            'original_16_detected_frames': sampled, 'original_16_coverage': sampled / 16,
            'original_16_all_missing_video': sampled == 0,
            'original_16_missing_but_full_recovered_video': sampled == 0 and detected > 0,
            'no_face_frames': FRAME_COUNT - detected,
            'single_face_frames': sum(row['category'] == 'single_face' for row in rows),
            'multi_face_frames': sum(row['category'] == 'multi_face' for row in rows),
            'padding_attempted_frames': sum(row['padding_attempted'] for row in rows),
            'padding_recovered_frames': sum(row['padding_recovered'] for row in rows),
        }
    return {'scales': scales,
            'scale_320_all_missing_to_640_recovered_video':
                scales['320']['all_missing_video'] and not scales['640']['all_missing_video'],
            'scale_320_missing_to_640_recovered_frames': sum(
                frame['scales']['320']['face_count'] == 0 and frame['scales']['640']['face_count'] > 0
                for frame in frame_rows)}


def diagnose_video(path, detectors, budget, frame_rows):
    budget.check()
    capture = cv2.VideoCapture(str(path))
    try:
        budget.check()
        if not capture.isOpened():
            raise ValueError('Video cannot be opened')
        metadata = {name: capture.get(prop) for name, prop in (
            ('frames', cv2.CAP_PROP_FRAME_COUNT), ('width', cv2.CAP_PROP_FRAME_WIDTH),
            ('height', cv2.CAP_PROP_FRAME_HEIGHT), ('fps', cv2.CAP_PROP_FPS))}
        if metadata != {'frames': FRAME_COUNT, 'width': WIDTH, 'height': HEIGHT, 'fps': FPS}:
            raise ValueError('Video metadata must be 49 frames, 480x352, 16fps')
        for index in range(FRAME_COUNT):
            budget.check()
            budget.decode_calls += 1
            ok, image = capture.read()
            budget.check()
            if not ok or image is None:
                raise ValueError(f'Sequential video decode failed at frame {index}')
            if image.shape != (HEIGHT, WIDTH, 3) or image.dtype != np.uint8:
                raise ValueError('Decoded video frame must be 480x352 uint8 BGR')
            budget.decoded_frames += 1
            row = {'frame_index': index, 'scales': {}}
            frame_rows.append(row)
            for scale in SCALES:
                row['scales'][str(scale)] = detect_frame(detectors[scale], image, budget, scale)
        budget.check()
        budget.decode_calls += 1
        ok, image = capture.read()
        budget.check()
        if ok or image is not None:
            raise ValueError('Video contains more than 49 decoded frames')
        return metadata
    finally:
        capture.release()


def aggregate_records(videos, references, expected_samples):
    groups = {}
    for video in videos:
        group = groups.setdefault(video['group'], {'started_videos': 0, 'completed_videos': 0,
                                                   'partial_fully_diagnosed_frames': 0, 'scales': {},
                                                   'scale_320_all_missing_to_640_recovered_videos': 0,
                                                   'scale_320_missing_to_640_recovered_frames': 0})
        group['started_videos'] += 1
        if not video['complete']:
            group['partial_fully_diagnosed_frames'] += sum(len(row['scales']) == len(SCALES)
                                                          for row in video['frame_rows'])
            continue
        group['completed_videos'] += 1
        summary = video['summary']
        group['scale_320_all_missing_to_640_recovered_videos'] += int(
            summary['scale_320_all_missing_to_640_recovered_video'])
        group['scale_320_missing_to_640_recovered_frames'] += summary['scale_320_missing_to_640_recovered_frames']
        for scale, counts in summary['scales'].items():
            total = group['scales'].setdefault(scale, {})
            for key in ('decoded_frames', 'detected_frames', 'original_16_detected_frames',
                        'no_face_frames', 'single_face_frames', 'multi_face_frames',
                        'padding_attempted_frames', 'padding_recovered_frames'):
                total[key] = total.get(key, 0) + counts[key]
            for key in ('all_missing_video', 'all_frames_detected_video', 'original_16_all_missing_video',
                        'original_16_missing_but_full_recovered_video'):
                plural = key + 's'
                total[plural] = total.get(plural, 0) + int(counts[key])
    for group in groups.values():
        for counts in group['scales'].values():
            counts['full_frame_coverage'] = counts['detected_frames'] / counts['decoded_frames']
            counts['original_16_coverage'] = counts['original_16_detected_frames'] / (16 * group['completed_videos'])
    reference_counts = {}
    for scale in SCALES:
        rows = [reference['scales'][str(scale)] for reference in references if str(scale) in reference['scales']]
        reference_counts[str(scale)] = {
            'diagnosed_unique_references': len(rows),
            'no_face_references': sum(row['category'] == 'no_face' for row in rows),
            'single_face_references': sum(row['category'] == 'single_face' for row in rows),
            'multi_face_references': sum(row['category'] == 'multi_face' for row in rows),
            'padding_recovered_references': sum(row['padding_recovered'] for row in rows),
        }
    return {'expected_videos': expected_samples, 'started_videos': len(videos),
            'completed_videos': sum(video['complete'] for video in videos),
            'groups': groups, 'references': reference_counts,
            'denominator_policy': 'video coverage uses only fully decoded/diagnosed videos; partial counts separate',
            'interpretation': 'Detection is not identity matching or a face-visibility ground truth; no winner or score update.'}


def evaluate(rows, detectors, budget, report, progress):
    seen_references = set()
    for sample in rows:
        digest = sample['reference_sha256']
        if digest in seen_references:
            continue
        seen_references.add(digest)
        reference = {'reference_sha256': digest, 'reference_image': sample['reference_image'], 'scales': {}}
        report['local_references'].append(reference)
        budget.check()
        image = cv2.imread(sample['reference_image'])
        budget.check()
        if image is None:
            raise ValueError('Original reference image cannot be decoded')
        for scale in SCALES:
            reference['scales'][str(scale)] = detect_frame(detectors[scale], image, budget, scale, kind='reference')
        progress()
    for sample in rows:
        record = {key: sample[key] for key in ('sample_id', 'group', 'video_path', 'video_sha256', 'reference_sha256')}
        record.update(complete=False, frame_rows=[], original_16_indices=frame_indices())
        report['local_videos'].append(record)
        record['metadata'] = diagnose_video(sample['video_path'], detectors, budget, record['frame_rows'])
        record['summary'] = summarize_video(record['frame_rows'])
        record['complete'] = True
        progress()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest', 'model-dir', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--timeout-s', type=float, default=1800)
    args = parser.parse_args(argv)
    budget = Budget(timeout_s=args.timeout_s)
    if args.output.exists():
        raise FileExistsError('Choose a fresh diagnostic output; prior reports cannot be overwritten')
    if not args.output.parent.is_dir():
        raise FileNotFoundError('Output parent directory must already exist')
    model_path = args.model_dir / MODEL_NAME
    report = {'status': 'incomplete', 'complete': False, 'diagnostic_only': True,
              'official_score': None, 'protocol': PROTOCOL,
              'privacy': 'Restricted local per-frame/face-bbox rows and media paths; publish only reviewed aggregates.',
              'model_sha256': MODEL_SHA256, 'hash_verification': {'pre': False, 'post': False},
              'local_references': [], 'local_videos': []}
    files, error = {}, None
    with args.output.open('x', encoding='utf-8') as handle:
        def progress():
            report['resources'] = budget.resources()
            report['aggregate'] = aggregate_records(report['local_videos'], report['local_references'], 48)
            content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
            handle.seek(0)
            handle.write(content)
            handle.truncate()
            handle.flush()
        progress()
        try:
            budget.check()
            manifest_sha256 = sha256_file(args.manifest)
            report['manifest_sha256'] = manifest_sha256
            rows = read_manifest(args.manifest)
            files = expected_files(rows, args.manifest, manifest_sha256, model_path)
            verify_files(files, budget)
            report['hash_verification']['pre'] = True
            detectors = load_detectors(model_path)
            budget.check()
            evaluate(rows, detectors, budget, report, progress)
            budget.check()
        except Exception as exc:
            error = exc
            report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        try:
            if files:
                verify_files(files)
                report['hash_verification']['post'] = True
            if error is None:
                budget.check()
        except Exception as exc:
            report['post_verification_failure'] = {'type': type(exc).__name__, 'message': str(exc)}
            if error is None:
                error = exc
                report['failure'] = dict(report['post_verification_failure'])
        if error is None:
            report.update(status='complete', complete=True)
        progress()
        if error is not None:
            raise error


if __name__ == '__main__':
    main()
