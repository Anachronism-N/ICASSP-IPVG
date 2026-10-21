"""Local caption/KaLM story similarity proxy; never the Gemini-based author StoryScore."""
import argparse
import hashlib
import os
from pathlib import Path
import time
from ipvg.common import load_json, sha256_file, write_json

CAPTION_PROMPT = ('Give a detailed video caption of the input video. Describe characters, environments, dynamics, '
                  'camera movement, and fine-grained element interactions. Output150-250words, only visibly supported '
                  'observations in beginning-middle-end order. Do not infer names, audio, dialogue or hidden intent.')
PROTOCOL = {'version': 'local_story_similarity_v1', 'caption_model': 'Qwen2.5-VL-7B-Instruct',
            'caption_frames': 8, 'frame_sampling': 'rounded linspace including endpoints',
            'caption_instruction': CAPTION_PROMPT, 'caption_max_new_tokens': 512, 'caption_do_sample': False,
            'embedding_model': 'HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2',
            'embedding_revision': '348e90d4084c1e2fbaac61f16a2d8c636192352a', 'embedding_max_tokens': 512,
            'score': 'dot product of normalized exact prompt and local video-caption embeddings',
            'official_story_score': None, 'official_final_score': None,
            'warning': 'Local captioner substitution is not MSVBench/AESR Gemini StoryScore or human action truth.'}


def checked_files(folder, expected):
    if not expected: raise ValueError('Explicit nonempty model file lock required')
    for relative, digest in expected.items():
        p = Path(relative)
        if p.is_absolute() or '..' in p.parts: raise ValueError('Unsafe model-lock path')
        if sha256_file(folder / p) != digest: raise ValueError('Local model hash changed: ' + relative)


def frame_indices(count):
    import numpy as np
    if isinstance(count, bool) or not isinstance(count, int) or count < 8:
        raise ValueError('Require at least8decoded frames')
    return np.linspace(0, count - 1, 8).round().astype(int).tolist()


def normalized_similarity(left, right):
    import numpy as np
    a, b = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Require finite matching vectors')
    if not np.isclose(np.linalg.norm(a), 1, atol=.01) or not np.isclose(np.linalg.norm(b), 1, atol=.01):
        raise ValueError('Embeddings not normalized')
    return float(a @ b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'caption-model', 'embedding-model', 'model-lock', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('Choose fresh score output')
    lock = load_json(args.model_lock)
    checked_files(args.caption_model, lock['caption_model_files'])
    checked_files(args.embedding_model, lock['embedding_model_files'])
    request = load_json(args.manifest); rows = request['samples']
    if not rows or len({r['sample_id'] for r in rows}) != len(rows): raise ValueError('Require unique inputs')
    for r in rows:
        if not isinstance(r['prompt'], str) or not r['prompt'].strip(): raise ValueError('Nonempty prompt required')
        if sha256_file(r['video_path']) != r['video_sha256']: raise ValueError('Video hash drift')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import numpy as np
    import numpy.testing
    import torch
    import cv2
    from PIL import Image
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from sentence_transformers import SentenceTransformer
    from mm2026_quality_eval import offline_only
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    if torch.cuda.device_count() != 1: raise ValueError('Exactly one visible CUDAdevice required')
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start = time.monotonic()
    captions = []
    with offline_only():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.caption_model), local_files_only=True,
                    torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa').eval().requires_grad_(False)
        processor = AutoProcessor.from_pretrained(str(args.caption_model), local_files_only=True,
                                                  min_pixels=224*224, max_pixels=320*352)
        for row in rows:
            cap = cv2.VideoCapture(row['video_path']); frames = []
            try:
                expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                while True:
                    ok, frame = cap.read()
                    if not ok: break
                    frames.append(frame)
                if len(frames) != expected: raise ValueError('Full video decode failure')
            finally: cap.release()
            indices = frame_indices(len(frames))
            images = [Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)) for i in indices]
            message = [{'role': 'user', 'content': [{'type': 'image', 'image': image} for image in images]
                          + [{'type': 'text', 'text': CAPTION_PROMPT}]}]
            text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors='pt', padding=True).to('cuda')
            with torch.inference_mode(): output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            caption = processor.batch_decode(output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0].strip()
            if not caption: raise ValueError('Caption missing; cannot impute story score')
            captions.append({'sample_id': row['sample_id'], 'group': row.get('group'),
                             'video_sha256': row['video_sha256'], 'decoded_frames': len(frames),
                             'caption_indices': indices, 'caption': caption, 'caption_words': len(caption.split()),
                             'caption_sha256': hashlib.sha256(caption.encode()).hexdigest(),
                             'caption_manual_verification': False})
            print('Captioned', len(captions), 'of', len(rows), flush=True)
        del inputs, output, model, processor
        torch.cuda.empty_cache()
        embedding = SentenceTransformer(str(args.embedding_model), trust_remote_code=True, device='cuda',
                    local_files_only=True, model_kwargs={'torch_dtype': torch.bfloat16, 'attn_implementation': 'sdpa'})
        embedding.max_seq_length = 512
        for row, record in zip(rows, captions):
            counts = [len(embedding.tokenizer.encode(t, add_special_tokens=True)) for t in (row['prompt'], record['caption'])]
            if max(counts) > 512: raise ValueError('Story text would truncate; preserve exact inputs')
            features = embedding.encode([row['prompt'], record['caption']], normalize_embeddings=True,
                                         batch_size=2, convert_to_numpy=True, show_progress_bar=False)
            record['scores'] = {'local_story_similarity': normalized_similarity(features[0], features[1])}
            record['text_token_counts'] = counts
            record['prompt_sha256'] = hashlib.sha256(row['prompt'].encode()).hexdigest()
            if sha256_file(row['video_path']) != row['video_sha256']: raise ValueError('Video changed while scoring')
    torch.cuda.synchronize(); elapsed = time.monotonic() - start
    write_json(args.output, {'status': 'completed', 'protocol': PROTOCOL, 'model_lock_sha256': sha256_file(args.model_lock),
                            'manifest_sha256': sha256_file(args.manifest), 'records': captions,
                            'resource': {'gpus': 1, 'elapsed_seconds': elapsed, 'allocated_gpu_hours': elapsed/3600,
                                         'peak_allocated_vram_gb': torch.cuda.max_memory_allocated()/1e9},
                            'official_story_score': None, 'official_final_score': None})
    print('Scored local story proxy', len(captions), 'videos; captions remain local')


if __name__ == '__main__': main()
