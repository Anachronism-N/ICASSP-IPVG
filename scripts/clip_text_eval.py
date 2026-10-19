"""Offline full-frame CLIP text proxy; not action recognition or official scoring."""
import argparse
from pathlib import Path
import numpy as np

from ipvg.common import sha256_file,write_json
from ipvg.data import read_manifest

WEIGHT_SHA256='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
PROTOCOL={'version':'clip_b32_frame_text_v1','model':'OpenAI CLIP ViT-B/32',
          'weights_sha256':WEIGHT_SHA256,'frames':8,'sampling':'rounded linspace including endpoints',
          'device':'cpu','dtype':'float32','threads':4,'image':'official preprocess full frame, no face selection',
          'text':'exact manifest prompt, context length 77, no truncation',
          'aggregation':'mean cosine of unit image/text features, no logit_scale',
          'kind':'local_proxy','limitations':['Single frames cannot verify speech, action occurrence or ordering.',
          'Full-frame composition and backgrounds confound similarity; not a perceptual or human-quality score.']}


def mean_cosine(images,text):
    images=np.asarray(images,dtype=np.float32)
    text=np.asarray(text,dtype=np.float32)
    if images.ndim!=2 or text.shape!=(images.shape[1],) or len(images)!=8 or not np.isfinite(images).all() or not np.isfinite(text).all():
        raise ValueError('Require eight finite image vectors and matching finite text')
    norms=np.linalg.norm(images,axis=1)
    norm=np.linalg.norm(text)
    if (norms<=0).any() or norm<=0:
        raise ValueError('Embedding norms must be positive')
    values=(images/norms[:,None])@(text/norm)
    return {'clip_frame_text_cosine_mean':float(values.mean()),'clip_frame_text_cosine_min':float(values.min())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--outputs',type=Path)
    parser.add_argument('--source-videos',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Choose fresh scoring output')
    if args.source_videos==bool(args.outputs): raise ValueError('Choose exactly source videos or outputs directory')
    if not args.weights.is_file() or sha256_file(args.weights)!=WEIGHT_SHA256:
        raise ValueError('Require the exact local official B/32 weight; downloads forbidden')
    import torch
    import clip
    import cv2
    from PIL import Image
    torch.set_num_threads(4)
    model,preprocess=clip.load(str(args.weights.resolve()),device='cpu',jit=False)
    model.eval().requires_grad_(False)
    rows=[r for r in read_manifest(args.manifest) if r.get('split')=='dev']
    if not rows: raise ValueError('Require nonempty dev scoring cohort')
    prompts=sorted({r['prompt'] for r in rows})
    tokens=clip.tokenize(prompts,truncate=False)
    with torch.inference_mode():
        contexts=model.encode_text(tokens).float().cpu().numpy()
    text_vectors=dict(zip(prompts,contexts))
    records=[]
    for row in rows:
        path=Path(row['video_path']) if args.source_videos else args.outputs/(row['sample_id']+'.mp4')
        capture=cv2.VideoCapture(str(path))
        try:
            count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if count<8: raise ValueError('Insufficient frames')
            batch=[]
            for index in np.linspace(0,count-1,8).round().astype(int):
                capture.set(cv2.CAP_PROP_POS_FRAMES,int(index))
                ok,frame=capture.read()
                if not ok: raise ValueError('Frame decode failure')
                batch.append(preprocess(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))))
        finally:
            capture.release()
        with torch.inference_mode():
            features=model.encode_image(torch.stack(batch)).float().cpu().numpy()
        scores=mean_cosine(features,text_vectors[row['prompt']])
        if len(prompts)==2:
            other=next(p for p in prompts if p!=row['prompt'])
            scores['clip_prompt_contrast']=scores['clip_frame_text_cosine_mean']-mean_cosine(features,text_vectors[other])['clip_frame_text_cosine_mean']
        records.append({'sample_id':row['sample_id'],'identity_id':row['identity_id'],'scores':scores,
                        'video_sha256':sha256_file(path),'score_kind':'local_proxy'})
    write_json(args.output,{'protocol':PROTOCOL,'manifest_sha256':sha256_file(args.manifest),
                            'source_videos':args.source_videos,'records':records,'official_score':None})
    print('Scored',len(records),'videos with offline full-frame text proxy')


if __name__=='__main__':
    main()
