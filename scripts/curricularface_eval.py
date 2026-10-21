"""Offline author-style CurricularFace/ArcFace proxies; never official scores."""
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np

from ipvg.common import load_json, sha256_file

UPSTREAM_COMMIT = '60429e4b0d24d36ce8df7fb58f7ab42b86e2b70e'
SOURCE_FILE = 'util/on_going_module/curricularface/model_irse.py'
COMMON_FILE = 'util/on_going_module/curricularface/common.py'
SOURCE_HASHES = {
    SOURCE_FILE: '2be54709dae2612bc298ead4ae5b5d3332c99682025c3ad1454dd701c1832b6a',
    COMMON_FILE: '2c4c5d0bbd66aa7e4193945b8d863a8ac812436070c78cdce5356a267d750806',
}
WEIGHT_NAME = 'glint360k_curricular_face_r101_backbone.bin'
WEIGHT_SHA256 = 'aa54ae6dc9f7ac6e262809f1a81d6108e221952bf836aee4cb677ef19246c6ec'
FACE_HASHES = {
    'det_10g.onnx': '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91',
    'w600k_r50.onnx': '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43',
}
MISSING_POLICIES = ('author_all_missing_video_zero', 'omit_missing_video')
PROTOCOL = {
    'version': 'consisid_cur_arc_author_style_cpu_detector_v1',
    'frames': 16,
    'frame_sampling': 'floor linspace indices including endpoints; repeat indices for short videos',
    'det_size': [320, 320],
    'det_threshold': .5,
    'multi_face_policy': 'largest bbox area; last detection breaks ties as native stable sort; never identity matching',
    'padding_retry': 'only if no faces; 0.25 of each dimension per edge, gray 128; subtract keypoint offsets',
    'identity_anchor': 'original manifest reference_image, never candidate conditioning crop',
    'cur_alignment': 'InsightFace norm_crop 224 BGR; RGB; OpenCV linear resize 112; CHW float32 /127.5-1',
    'arc_alignment': 'InsightFace norm_crop 112 BGR; native ArcFaceONNX.get_feat; padded detection image on retry',
    'aggregation': 'mean max(0, cosine) over valid frames; no per-frame zero imputation',
    'missing_reference': 'scores null with explicit cause, including all-missing videos',
    'detector_provider': 'CPUExecutionProvider; author-style provider equivalent, not an official scorer',
    'ort_intra_op_threads': 4,
    'ort_inter_op_threads': 1,
    'torch_threads': 4,
    'cur_model': 'native ConsisID IR_101([112,112]), Backbone num_layers=100',
    'arc_model': 'buffalo_l/w600k_r50',
    'score_kind': 'local_proxy',
    'official_score': None,
}


def frame_indices(count, samples=16):
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('Video must have a positive decoded frame count')
    if samples != 16:
        raise ValueError('Author-style protocol requires exactly sixteen samples')
    return np.linspace(0, count - 1, samples).astype(int).tolist()


def normalize_embedding(value):
    vector = np.asarray(value)
    if vector.shape not in ((512,), (1, 512)) or not np.issubdtype(vector.dtype, np.floating):
        raise ValueError('Encoder must return a floating 512-dimensional embedding')
    vector = vector.reshape(512).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError('Encoder embedding must be finite and have a positive norm')
    return vector / norm


def curricular_input(aligned_bgr):
    if aligned_bgr.shape != (224, 224, 3) or aligned_bgr.dtype != np.uint8:
        raise ValueError('CurricularFace alignment must return a 224x224 uint8 BGR crop')
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
    tensor = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
    tensor /= 255.
    tensor -= .5
    tensor /= .5
    return tensor


def largest_face(detector, image_bgr):
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError('Face input must be a uint8 BGR image')
    offset = np.zeros(2, dtype=np.float32)
    boxes, keypoints = detector.detect(image_bgr, max_num=0, metric='default')
    padded = False
    if boxes is None or len(boxes) == 0:
        height, width = image_bgr.shape[:2]
        pad_x, pad_y = int(width * .25), int(height * .25)
        expanded = cv2.copyMakeBorder(image_bgr, pad_y, pad_y, pad_x, pad_x,
                                      cv2.BORDER_CONSTANT, value=(128, 128, 128))
        boxes, keypoints = detector.detect(expanded, max_num=0, metric='default')
        padded = True
        offset[:] = (pad_x, pad_y)
    if boxes is None or len(boxes) == 0:
        return None, 'no_face_after_padding'
    boxes, keypoints = np.asarray(boxes), np.asarray(keypoints)
    if (boxes.ndim != 2 or boxes.shape[1] < 4 or keypoints.shape != (len(boxes), 5, 2)
            or not np.isfinite(boxes).all() or not np.isfinite(keypoints).all()):
        raise ValueError('Detector returned invalid boxes or five-point landmarks')
    sizes = boxes[:, 2:4] - boxes[:, :2]
    if (sizes <= 0).any():
        raise ValueError('Detector returned nonpositive face boxes')
    areas = np.prod(sizes, axis=1)
    index = int(np.flatnonzero(areas == areas.max())[-1])
    bbox = boxes[index, :4].astype(np.float32).copy() - np.tile(offset, 2)
    return {'bbox': bbox, 'kps': keypoints[index].astype(np.float32) - offset,
            'face_count': len(boxes), 'padded': padded}, None


def verify_model_lock(upstream, weights, face_model_dir, model_lock):
    upstream, weights, face_model_dir = Path(upstream).resolve(), Path(weights).resolve(), Path(face_model_dir).resolve()
    lock = load_json(model_lock)
    if lock.get('upstream_commit') != UPSTREAM_COMMIT:
        raise ValueError('Model lock must pin the audited ConsisID upstream commit')
    revision = subprocess.run(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], check=True,
                              capture_output=True, text=True).stdout.strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError('Upstream git revision differs from the audited pin')
    source_hashes = lock.get('source_files', {})
    if any(source_hashes.get(path) != digest for path, digest in SOURCE_HASHES.items()):
        raise ValueError('Model lock must include pinned native model_irse.py and common.py source SHA256')
    observed_sources = {}
    for relative, digest in source_hashes.items():
        path = (upstream / relative).resolve()
        if not path.is_relative_to(upstream) or not path.is_file() or not re.fullmatch('[0-9a-f]{64}', str(digest)):
            raise ValueError('Invalid audited local source path or SHA256')
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError('Upstream source hash mismatch: ' + relative)
        observed_sources[relative] = actual
    files = lock.get('files', {})
    expected = dict(FACE_HASHES, **{WEIGHT_NAME: WEIGHT_SHA256})
    paths = {name: face_model_dir / name for name in FACE_HASHES}
    paths[WEIGHT_NAME] = weights
    for name, digest in expected.items():
        if files.get(name) != digest:
            raise ValueError('Model lock has an unpinned checkpoint SHA256: ' + name)
        if not paths[name].is_file():
            raise FileNotFoundError('Explicit local models required; automatic downloads disabled: ' + name)
        if sha256_file(paths[name]) != digest:
            raise ValueError('Checkpoint hash mismatch: ' + name)
    return {'upstream_commit': revision, 'source_files': observed_sources, 'files': expected,
            'model_lock_sha256': sha256_file(model_lock)}


def load_native_backbone(upstream):
    root = Path(upstream).resolve()
    package_name = '_ipvg_audited_cur_' + sha256_file(root / SOURCE_FILE)[:16]
    package = ModuleType(package_name)
    # Load the audited pair only, without running the upstream package's extra imports.
    package.__path__ = []
    sys.modules[package_name] = package
    try:
        for name, relative in (('common', COMMON_FILE), ('model_irse', SOURCE_FILE)):
            module_name = package_name + '.' + name
            spec = importlib.util.spec_from_file_location(module_name, root / relative)
            native = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = native
            source = (root / relative).read_bytes()
            if hashlib.sha256(source).hexdigest() != SOURCE_HASHES[relative]:
                raise ValueError('Audited native source changed before import: ' + relative)
            exec(compile(source, str(root / relative), 'exec'), native.__dict__)
        return native.IR_101([112, 112])
    finally:
        for name in (package_name + '.model_irse', package_name + '.common', package_name):
            sys.modules.pop(name, None)


def load_checkpoint(model, weights, torch):
    state = torch.load(str(weights), map_location='cpu', weights_only=True)
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('Native checkpoint keys must exactly match IR_101 state_dict')
    for name, tensor in state.items():
        if (not torch.is_tensor(tensor) or tensor.shape != expected[name].shape
                or tensor.dtype != expected[name].dtype):
            raise ValueError('Native checkpoint tensor shape/dtype mismatch: ' + name)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('Native checkpoint contains nonfinite tensor: ' + name)
    model.load_state_dict(state, strict=True)
    return {'tensor_count': len(state), 'parameter_elements': sum(t.numel() for t in state.values()),
            'load': 'torch.load weights_only=True CPU; exact keys/shapes/dtypes; finite; strict=True'}


class OfflineCurEncoder:
    def __init__(self, upstream, weights, face_model_dir, model_lock, device='cuda'):
        if device not in ('cpu', 'cuda'):
            raise ValueError('Device must be cpu or single-GPU cuda')
        self.model_source = verify_model_lock(upstream, weights, face_model_dir, model_lock)
        import torch
        import onnxruntime as ort
        from insightface.model_zoo.scrfd import SCRFD
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        from insightface.utils.face_align import norm_crop
        torch.set_num_threads(4)
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
        self.device = 'cuda:0' if device == 'cuda' else 'cpu'
        self.torch, self.align = torch, norm_crop
        self.cur_model = load_native_backbone(upstream)
        self.model_source['checkpoint_validation'] = load_checkpoint(self.cur_model, weights, torch)
        self.cur_model.eval().requires_grad_(False).to(device=self.device, dtype=torch.float32)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        paths = [Path(face_model_dir) / name for name in FACE_HASHES]
        sessions = [ort.InferenceSession(str(path), sess_options=options,
                    providers=['CPUExecutionProvider']) for path in paths]
        self.detector = SCRFD(model_file=str(paths[0]), session=sessions[0])
        self.recognition = ArcFaceONNX(model_file=str(paths[1]), session=sessions[1])
        self.detector.prepare(ctx_id=0, input_size=(320, 320), det_thresh=.5)
        self.recognition.prepare(ctx_id=0)
        self.weight_hashes = self.model_source['files']

    def process_image(self, image_bgr):
        face, reason = largest_face(self.detector, image_bgr)
        if face is None:
            return None, reason
        cur_crop = self.align(image_bgr, landmark=face['kps'], image_size=224)
        arc_image, arc_kps = image_bgr, face['kps']
        if face['padded']:
            height, width = image_bgr.shape[:2]
            pad_x, pad_y = int(width * .25), int(height * .25)
            arc_image = cv2.copyMakeBorder(image_bgr, pad_y, pad_y, pad_x, pad_x,
                                           cv2.BORDER_CONSTANT, value=(128, 128, 128))
            arc_kps = face['kps'] + np.array([pad_x, pad_y], dtype=np.float32)
        arc_crop = self.align(arc_image, landmark=arc_kps, image_size=112)
        tensor = self.torch.from_numpy(curricular_input(cur_crop)[None]).to(self.device)
        with self.torch.inference_mode():
            cur = self.cur_model(tensor).float().cpu().numpy()
        face['cur_embedding'] = normalize_embedding(cur)
        face['arc_embedding'] = normalize_embedding(self.recognition.get_feat(arc_crop))
        return face, None


def video_features(path, encoder):
    capture = cv2.VideoCapture(str(path))
    faces, reasons = [], []
    try:
        if not capture.isOpened():
            raise ValueError('Video cannot be opened')
        indices = frame_indices(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError('Sampled video frame cannot be decoded')
            face, reason = encoder.process_image(frame)
            faces.append(face)
            reasons.append(reason)
    finally:
        capture.release()
    return faces, reasons, indices


def score_frames(anchor, faces, missing_policy='author_all_missing_video_zero'):
    if missing_policy not in MISSING_POLICIES:
        raise ValueError('Unknown missing video policy')
    if len(faces) != 16:
        raise ValueError('Require exactly sixteen sampled frame results')
    result = {'anchor_available': anchor is not None,
              'face_detection_coverage': sum(face is not None for face in faces) / 16,
              'multi_face_frames': sum(face is not None and face['face_count'] > 1 for face in faces),
              'padding_recovered_frames': sum(face is not None and face['padded'] for face in faces),
              'missing_policy': missing_policy}
    for name in ('cur', 'arc'):
        values = []
        reference = None if anchor is None else normalize_embedding(anchor[name + '_embedding'])
        for face in faces:
            values.append(None if face is None or reference is None else
                          max(0., float(reference @ normalize_embedding(face[name + '_embedding']))))
        valid = [value for value in values if value is not None]
        score = float(np.mean(valid)) if valid else None
        if anchor is not None and not any(face is not None for face in faces) and missing_policy == MISSING_POLICIES[0]:
            score = 0.
        result[name + '_score'] = score
        result[name + '_frame_cosines'] = values
        result[name + '_scoring_coverage'] = len(valid) / 16
    return result


def read_frozen_manifest(path):
    document = load_json(path)
    rows = document.get('samples') if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows:
        raise ValueError('Frozen JSON manifest requires nonempty samples')
    seen = set()
    required = ('sample_id', 'video_path', 'video_sha256', 'reference_image', 'reference_sha256', 'prompt', 'group')
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key] for key in required):
            raise ValueError('Frozen sample fields must be nonempty strings')
        if row['sample_id'] in seen:
            raise ValueError('Duplicate frozen sample_id')
        seen.add(row['sample_id'])
        if any(not re.fullmatch('[0-9a-f]{64}', row[key]) for key in ('video_sha256', 'reference_sha256')):
            raise ValueError('Frozen media SHA256 must be lowercase hexadecimal')
        if any(key in row for key in ('reference_crop', 'original_reference_images', 'reference_images')):
            raise ValueError('Use the original frozen reference_image, not a conditioning crop manifest')
        if 'identity_id' in row and (not isinstance(row['identity_id'], str) or not row['identity_id']):
            raise ValueError('Optional calibration identity_id must be a nonempty string')
    return rows


def verify_media(row):
    for path_key, hash_key in (('video_path', 'video_sha256'), ('reference_image', 'reference_sha256')):
        if sha256_file(row[path_key]) != row[hash_key]:
            raise ValueError('Frozen original media hash changed: ' + path_key)


def evaluate(rows, encoder, missing_policy='author_all_missing_video_zero'):
    records = []
    for row in rows:
        verify_media(row)
        image = cv2.imread(row['reference_image'])
        if image is None:
            raise ValueError('Original reference image cannot be decoded')
        anchor, anchor_reason = encoder.process_image(image)
        faces, reasons, indices = video_features(row['video_path'], encoder)
        verify_media(row)
        record = {key: row[key] for key in ('sample_id', 'group', 'video_sha256', 'reference_sha256')}
        if 'identity_id' in row:
            record['identity_id'] = row['identity_id']
        record.update(score_frames(anchor, faces, missing_policy))
        record.update(frame_indices=indices, frame_missing_reasons=reasons,
                      anchor_missing_reason=anchor_reason,
                      anchor_face_count=0 if anchor is None else anchor['face_count'],
                      anchor_padding_recovered=False if anchor is None else anchor['padded'],
                      failure_classes=sorted({reason for reason in reasons + [anchor_reason] if reason}),
                      score_kind='local_proxy', official_score=None)
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest', 'upstream', 'weights', 'face-model-dir', 'model-lock', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--missing-policy', choices=MISSING_POLICIES, default=MISSING_POLICIES[0])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh CurricularFace evaluation output')
    if not args.output.parent.is_dir():
        raise FileNotFoundError('Output parent directory must already exist')
    started = time.monotonic()
    manifest_hash = sha256_file(args.manifest)
    rows = read_frozen_manifest(args.manifest)
    encoder = OfflineCurEncoder(args.upstream, args.weights, args.face_model_dir, args.model_lock, args.device)
    records = evaluate(rows, encoder, args.missing_policy)
    if sha256_file(args.manifest) != manifest_hash:
        raise ValueError('Frozen manifest changed during evaluation')
    elapsed = time.monotonic() - started
    report = {'protocol': dict(PROTOCOL, missing_policy=args.missing_policy), 'model_source': encoder.model_source,
              'manifest_sha256': manifest_hash, 'device': encoder.device, 'records': records,
              'resources': {'wall_seconds': elapsed, 'gpus': int(args.device == 'cuda'),
                            'estimated_gpu_hours': elapsed / 3600 if args.device == 'cuda' else 0.,
                            'torch_threads': 4, 'ort_provider': 'CPUExecutionProvider',
                            'ort_intra_op_threads': 4, 'ort_inter_op_threads': 1},
              'score_kind': 'local_proxy', 'official_score': None,
              'limitations': ['Native author-style metric replication, not the official competition scorer.',
                              'Largest-face selection is not identity assignment; coverage is reported separately.',
                              'No biometric embeddings or original media paths persisted in this report.',
                              'Checkpoint, face encoder, source data and personality rights require separate permission.']}
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
