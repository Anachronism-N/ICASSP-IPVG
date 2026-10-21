"""Offline native GME-7B frame/text AESR; not an official challenge final score."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from ipvg.common import load_json, sha256_file

MODEL_REVISION = 'e54cb53a76dba4895a7a2f88fc8021f3679ed4f1'
TEXT_INSTRUCTION = 'Find frames that match the given text description.'
MAX_FRAMES = 81
UNIT_NORM_TOLERANCE = 0.01
PROTOCOL = {
    'version': 'native_gme7b_aesr_v1',
    'model': 'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct',
    'model_revision': MODEL_REVISION,
    'sampling': 'all frames up to 81; otherwise np.linspace(0, count-1, 81, dtype=int)',
    'images': 'native embed with local full-frame RGB PIL images, texts=None, is_query=False',
    'image_default_instruction': 'You are a helpful assistant.',
    'normalization': 'native model.normalize=True; verify finite/unit norm without evaluator renormalization',
    'text': 'exact frozen prompt passed unchanged to native embed with image=None and the fixed instruction',
    'native_text_processing': 'native processor truncation=True, max_length from native config (default 1800)',
    'text_instruction': TEXT_INSTRUCTION,
    'aggregation': 'arithmetic mean of raw unit-embedding dot products; no x100 or logit scale',
    'embedding_unit_norm_tolerance': UNIT_NORM_TOLERANCE,
    'implementation': 'independent published frame/text AESR algorithm, no AESR source reuse',
    'official_final_score': None,
    'limitations': ['Frame similarity alone does not verify action timing, speech or human quality.'],
}


def frame_indices(count):
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('Require a positive decoded frame count')
    return np.arange(count, dtype=int) if count <= MAX_FRAMES else np.linspace(0, count - 1, MAX_FRAMES, dtype=int)


def frame_text_scores(images, text):
    images = np.asarray(images, dtype=np.float32)
    text = np.asarray(text, dtype=np.float32)
    if (images.ndim != 2 or not 1 <= len(images) <= MAX_FRAMES or images.shape[1] < 1
            or text.shape != (images.shape[1],) or not np.isfinite(images).all()
            or not np.isfinite(text).all()):
        raise ValueError('Require finite image embeddings and a matching finite text embedding')
    image_norms = np.linalg.norm(images, axis=1)
    text_norm = float(np.linalg.norm(text))
    if (np.abs(image_norms - 1) > UNIT_NORM_TOLERANCE).any() or abs(text_norm - 1) > UNIT_NORM_TOLERANCE:
        raise ValueError('Native GME embeddings must already have unit norm')
    values = images @ text
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite frame/text similarity')
    return {
        'gme_aesr': float(values.mean()),
        'frame_scores': values.tolist(),
        'embedding_dimension': int(images.shape[1]),
        'finite_frame_count': len(values),
        'unit_norm_frame_count': len(values),
        'max_image_unit_norm_error': float(np.abs(image_norms - 1).max()),
        'text_unit_norm_error': abs(text_norm - 1),
    }


def validate_model_lock(model, lock_path):
    model = Path(model).resolve()
    if not model.is_dir():
        raise ValueError('Require an existing local model directory; downloads forbidden')
    lock = load_json(lock_path)
    if not isinstance(lock, dict) or lock.get('revision') != MODEL_REVISION:
        raise ValueError('GME model revision differs from the fixed revision')
    files = lock.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Model lock requires all local file SHA256 records')
    required = {'config.json', 'modeling_gme_qwen2vl.py', 'model.safetensors.index.json',
                'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json',
                'special_tokens_map.json', 'vocab.json', 'merges.txt', 'added_tokens.json',
                'chat_template.json', 'generation_config.json'}
    if not required.issubset(files):
        raise ValueError('Model lock omits native weights/code/config/tokenizer metadata')
    for name, expected in files.items():
        if (not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts
                or not isinstance(expected, str) or len(expected) != 64
                or any(c not in '0123456789abcdef' for c in expected)):
            raise ValueError('Invalid model lock path or SHA256')
        path = (model / name).resolve()
        if not path.is_relative_to(model) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError('Locked local model file missing, escaped or changed')
    actual = {path.relative_to(model).as_posix() for path in model.rglob('*') if path.is_file()
              and '.cache' not in path.relative_to(model).parts and '__pycache__' not in path.relative_to(model).parts}
    if actual != set(files):
        raise ValueError('Model lock must cover every non-cache local model file')
    index = load_json(model / 'model.safetensors.index.json')
    weight_map = index.get('weight_map')
    shards = {f'model-{i:05d}-of-00009.safetensors' for i in range(1, 10)}
    if not isinstance(weight_map, dict) or not weight_map or set(weight_map.values()) != shards or not shards.issubset(files):
        raise ValueError('Require the complete indexed nine-shard native GME weights')
    config = load_json(model / 'config.json')
    if config.get('auto_map') != {'AutoConfig': 'modeling_gme_qwen2vl.GmeQwen2VLConfig',
                                  'AutoModel': 'modeling_gme_qwen2vl.GmeQwen2VL'}:
        raise ValueError('Require the audited local native GME AutoModel configuration')
    return model, dict(files)


def validate_manifest(manifest):
    request = load_json(manifest)
    rows = request.get('samples') if isinstance(request, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError('Require a nonempty frozen samples manifest')
    ids = set()
    result = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('sample_id'), str) or not row['sample_id'].strip():
            raise ValueError('Require nonempty sample IDs')
        if row['sample_id'] in ids:
            raise ValueError('Require unique sample IDs')
        ids.add(row['sample_id'])
        if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
            raise ValueError('Require the exact nonempty frozen prompt')
        value = row.get('video_path')
        if not isinstance(value, str) or not value or '://' in value:
            raise ValueError('Require a local video path')
        path = Path(value)
        if not path.is_absolute():
            path = Path(manifest).resolve().parent / path
        path = path.resolve()
        if not path.is_file() or sha256_file(path) != row.get('video_sha256'):
            raise ValueError('Frozen video SHA256 missing or changed')
        result.append(dict(row, video_path=str(path)))
    return result


def decode_video(path):
    import cv2
    from PIL import Image
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('Video cannot be opened locally')
        reported_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite(reported_count) or reported_count != int(reported_count):
            raise ValueError('Invalid video frame count')
        count = int(reported_count)
        indices = frame_indices(count)
        selected = set(indices.tolist())
        frames = []
        dimensions = None
        for index in range(count):
            ok, frame = capture.read()
            if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError('Video frame decode failure; no failed frames may be dropped')
            size = [int(frame.shape[1]), int(frame.shape[0])]
            if dimensions is None:
                dimensions = size
            elif size != dimensions:
                raise ValueError('Video frame dimensions changed')
            if index in selected:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if capture.read()[0]:
            raise ValueError('Video decoded frame count differs from the frozen container count')
        return frames, {'decoded_frame_count': count, 'selected_frame_indices': indices.tolist(),
                        'scored_frame_count': len(frames), 'dimensions': dimensions,
                        'decode_coverage': 1.0, 'selected_frame_coverage': len(frames) / count}
    finally:
        capture.release()


def load_native_model(model_path, device):
    import torch
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from unittest.mock import patch
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    # Native GME's constructor builds its processor from config._name_or_path.
    config._name_or_path = str(model_path)
    original_processor = AutoProcessor.from_pretrained

    def local_processor(path, **kwargs):
        if Path(path).resolve() != model_path.resolve():
            raise ValueError('Native processor must resolve to the locked local model')
        kwargs['local_files_only'] = True
        return original_processor(str(model_path), **kwargs)

    with patch.object(AutoProcessor, 'from_pretrained', local_processor):
        model = AutoModel.from_pretrained(str(model_path), config=config, trust_remote_code=True,
                                         local_files_only=True, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                                         device_map=device, attn_implementation='sdpa')
    return model.eval().requires_grad_(False)


def score_sample(model, row, batch_size):
    import torch
    if sha256_file(row['video_path']) != row['video_sha256']:
        raise ValueError('Frozen video changed before evaluation')
    frames, coverage = decode_video(row['video_path'])
    with torch.inference_mode():
        text = model.embed(texts=[row['prompt']], images=[None], instruction=TEXT_INSTRUCTION)
        images = []
        for offset in range(0, len(frames), batch_size):
            batch = frames[offset:offset + batch_size]
            embeddings = model.embed(texts=[None] * len(batch), images=batch, is_query=False)
            if tuple(embeddings.shape[:1]) != (len(batch),):
                raise ValueError('Native image embedding count differs from the frame batch')
            images.append(embeddings.float().cpu().numpy())
    if text.ndim != 2 or len(text) != 1:
        raise ValueError('Native text embedding must contain exactly one prompt vector')
    values = frame_text_scores(np.concatenate(images), text[0].float().cpu().numpy())
    if values['finite_frame_count'] != coverage['scored_frame_count']:
        raise ValueError('Native embedding coverage differs from selected frame count')
    if sha256_file(row['video_path']) != row['video_sha256']:
        raise ValueError('Frozen video changed during evaluation')
    return {'sample_id': row['sample_id'], 'group': row.get('group'),
            'video_sha256': row['video_sha256'],
            'prompt_sha256': hashlib.sha256(row['prompt'].encode('utf-8')).hexdigest(),
            'scores': {'gme_aesr': values.pop('gme_aesr')}, **values, **coverage,
            'finite_frame_coverage': 1.0, 'unit_norm_frame_coverage': 1.0,
            'score_kind': 'paper_related_dimension'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('Choose a fresh GME score output')
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive')
    if not args.output.parent.is_dir():
        raise ValueError('Output parent directory must exist')
    started = time.monotonic()
    manifest_hash = sha256_file(args.manifest)
    lock_hash = sha256_file(args.model_lock)
    rows = validate_manifest(args.manifest)
    model_path, file_hashes = validate_model_lock(args.model, args.model_lock)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    import numpy.testing  # Finish the local CPU capability probe before blocking subprocesses.
    import torch
    import transformers
    # Resolve lazy model imports and their local CPU-capability probes before Popen lockdown.
    from transformers import PreTrainedModel, Qwen2VLForConditionalGeneration
    import cv2
    import PIL
    from mm2026_quality_eval import offline_only
    torch.set_num_threads(4)
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA evaluation requires a visible CUDA device')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with offline_only():
        model = load_native_model(model_path, args.device)
        records = [score_sample(model, row, args.batch_size) for row in rows]
    if sha256_file(args.manifest) != manifest_hash or sha256_file(args.model_lock) != lock_hash:
        raise ValueError('Frozen manifest or model lock changed during evaluation')
    if args.device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    result = {
        'protocol': PROTOCOL, 'manifest_sha256': manifest_hash, 'model_lock_sha256': lock_hash,
        'model_revision': MODEL_REVISION, 'model_file_sha256': file_hashes,
        'evaluator_sha256': sha256_file(__file__),
        'offline_guard_sha256': sha256_file(Path(__file__).with_name('mm2026_quality_eval.py')),
        'records': records, 'status': 'completed', 'official_score': None,
        'execution': {'device': args.device, 'dtype': 'float16' if args.device == 'cuda' else 'float32',
                      'device_mode': 'native_cuda' if args.device == 'cuda' else 'cpu_validation_not_cuda_equivalence',
                      'batch_size': args.batch_size, 'threads': 4, 'attention_implementation': 'sdpa',
                      'native_max_length': model.max_length,
                      'native_min_image_tokens': model.config.min_image_tokens,
                      'native_max_image_tokens': model.config.max_image_tokens,
                      'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
                      'numpy_version': np.__version__, 'opencv_version': cv2.__version__, 'pillow_version': PIL.__version__},
        'resource': {'gpus': int(args.device == 'cuda'), 'elapsed_seconds': elapsed,
                     'allocated_gpu_hours': elapsed / 3600 if args.device == 'cuda' else 0.0,
                     'peak_allocated_vram_gb': torch.cuda.max_memory_allocated() / 1e9 if args.device == 'cuda' else 0.0,
                     'peak_rss_gb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9},
    }
    # Exclusive creation also refuses an output created by another process during scoring.
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print('Offline native GME AESR scored', len(records), 'videos')


if __name__ == '__main__':
    main()
