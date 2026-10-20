"""Offline train-only resampling and dense captions; never consume dev/holdout target media."""
import argparse
import copy
from pathlib import Path
import re
import os
from ipvg.common import sha256_file,write_json
from ipvg.data import read_manifest,validate_manifest,write_manifest

CAPTION_PROMPT=("These eight images show successive moments in one short recording. Write one factual English paragraph of 100-180 words for training a reference-conditioned video generator. "
    "Describe visible setting and lighting, the subject's visible appearance and clothing, facial expression and mouth/head changes in beginning-middle-end order, interactions if visible, and camera framing. "
    "For subtle movement describe only observable blinking, lips and head motion. Do not infer names, identities, dialogue, audio, hidden intent or objects, or force a happy expression. "
    "Do not describe absent objects or begin with 'the video shows'. Do not mention these instructions, images or timestamps. Output only the paragraph.")


def sampling_indices(count,frames=49):
    if isinstance(count,bool) or not isinstance(count,int) or count<frames or frames not in (17,49):
        raise ValueError('Require at least requested17/49 decoded frames')
    import numpy as np
    return np.linspace(0,count-1,frames).round().astype(int).tolist()


def validate_caption(text):
    count=len(text.split())
    if not 100<=count<=180 or re.search(r"as an ai|i cannot|i can.t view|i.m sorry",text,re.I):
        raise ValueError('Dense caption length or refusal validation failed')
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--num-frames',type=int,choices=[17,49],default=49)
    parser.add_argument('--reuse-partial',type=Path)
    args=parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():raise FileExistsError('Choose fresh data directory')
    if not args.output_dir.parent.is_dir():raise FileNotFoundError('Output parent must exist')
    original_hash=sha256_file(args.manifest)
    all_rows=read_manifest(args.manifest)
    split=validate_manifest(all_rows)
    if not split['leakage']['safe']:raise ValueError('Original split leaks')
    rows=sorted([r for r in all_rows if r.get('split')=='train'],key=lambda r:r['sample_id'])
    if len(rows)!=32 or len({r['identity_id'] for r in rows})!=16:raise ValueError('Require frozen32train/16actors')
    if not args.model.is_dir():raise FileNotFoundError('Explicit local VLM required; no downloads')
    import cv2
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    args.output_dir.mkdir()
    (args.output_dir/'videos').mkdir()
    model_hashes={p.name:sha256_file(p) for p in sorted(args.model.glob('*.safetensors'))}
    if len(model_hashes)!=5:raise ValueError('Expect original five-shard Qwen2.5VL7B')
    write_json(args.output_dir/'model_provenance.json',{'model':'Qwen2.5-VL-7B-Instruct','sha256':model_hashes,
                      'config_sha256':sha256_file(args.model/'config.json'),'caption_prompt':CAPTION_PROMPT,'word_range':[100,180]})
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.model),local_files_only=True,
                     torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(str(args.model),local_files_only=True,min_pixels=224*224,max_pixels=320*352)
    prepared=[]
    if args.reuse_partial:
        previous=read_manifest(args.reuse_partial/'partial_manifest.jsonl')
        provenance=__import__('json').loads((args.reuse_partial/'model_provenance.json').read_text())
        if provenance['sha256']!=model_hashes or provenance['caption_prompt']!=CAPTION_PROMPT:
            raise ValueError('Partial caption model or prompt differs')
        original_by_id={r['sample_id']:r for r in rows}
        import shutil
        for item in previous:
            anchor=original_by_id.get(item['sample_id'])
            if anchor is None or item.get('split')!='train' or item['training_source']['sha256']!=anchor['media_sha256'] or item['reference_sha256']!=anchor['reference_sha256']:
                raise ValueError('Partial caption provenance differs')
            validate_caption(item['prompt'])
            if sha256_file(item['video_path'])!=item['media_sha256']:
                raise ValueError('Partial training media changed')
            candidate=copy.deepcopy(item)
            path=args.output_dir/'videos'/(item['sample_id']+'.mp4')
            shutil.copyfile(item['video_path'],path)
            candidate['video_path']=str(path.resolve())
            prepared.append(candidate)
    completed_ids={r['sample_id'] for r in prepared}
    current=None
    try:
        for row in rows:
            if row['sample_id'] in completed_ids:continue
            current=row['sample_id']
            if sha256_file(row['video_path'])!=row['media_sha256'] or sha256_file(row['reference_images'][0])!=row['reference_sha256']:
                raise ValueError('Original source/reference hash drift')
            cap=cv2.VideoCapture(row['video_path'])
            frames=[]
            fps=float(cap.get(cv2.CAP_PROP_FPS))
            try:
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    if frame.shape!=(360,480,3):raise ValueError('Unexpected original source dimensions')
                    frames.append(cv2.cvtColor(frame[4:356],cv2.COLOR_BGR2RGB))
            finally:cap.release()
            if fps<=0 or not np.isfinite(fps):raise ValueError('Source fps invalid')
            indices=sampling_indices(len(frames),args.num_frames)
            chosen=[frames[i] for i in indices]
            duration=len(frames)/fps
            path=args.output_dir/'videos'/(row['sample_id']+'.mp4')
            with imageio.get_writer(str(path),fps=args.num_frames/duration,codec='libx264',quality=7,macro_block_size=1) as writer:
                for frame in chosen:writer.append_data(frame)
            # Caption actual decoded training pixels, not pre-encoding frames.
            capture=cv2.VideoCapture(str(path));actual=[]
            try:
                while True:
                    ok,f=capture.read()
                    if not ok:break
                    actual.append(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)))
            finally:capture.release()
            if len(actual)!=args.num_frames:raise ValueError('Resampled source decode count mismatch')
            ids=np.linspace(0,args.num_frames-1,8).round().astype(int)
            images=[actual[int(i)] for i in ids]
            messages=[{'role':'user','content':[{'type':'image','image':image} for image in images]+[{'type':'text','text':CAPTION_PROMPT}]}]
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[prompt],images=images,return_tensors='pt',padding=True).to('cuda')
            with torch.inference_mode():outputs=model.generate(**inputs,max_new_tokens=512,do_sample=False)
            text=processor.batch_decode(outputs[:,inputs.input_ids.shape[-1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
            attempts=[text]
            for repair in range(2):
                try:
                    words=validate_caption(text)
                    break
                except ValueError:
                    instruction=('Rewrite the following draft into one factual English paragraph of 120-150 words. '
                                 'Preserve only details supported by the same chronological images. '
                                 'Do not add any new action or fact to pad the length. Output only the paragraph. Draft: '+text)
                    revision=[{'role':'user','content':[{'type':'image','image':image} for image in images]+[{'type':'text','text':instruction}]}]
                    formatted=processor.apply_chat_template(revision,tokenize=False,add_generation_prompt=True)
                    revision_inputs=processor(text=[formatted],images=images,return_tensors='pt',padding=True).to('cuda')
                    with torch.inference_mode():revised=model.generate(**revision_inputs,max_new_tokens=512,do_sample=False)
                    text=processor.batch_decode(revised[:,revision_inputs.input_ids.shape[-1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
                    attempts.append(text)
            write_json(args.output_dir/(row['sample_id']+'_caption_attempts.json'),{'attempts':attempts,'bounded_length_repairs':len(attempts)-1})
            words=validate_caption(text)
            candidate=copy.deepcopy(row)
            candidate.update(video_path=str(path.resolve()),media_sha256=sha256_file(path),prompt=text,
                       prompt_source='local Qwen2.5-VL7B dense visible temporal caption, not manually verified',
                       training_source={'original_video_path':row['video_path'],'sha256':row['media_sha256'],'decoded_frames':len(frames),
                       'original_fps':fps,'original_duration':duration,'selected_indices':indices,'prepared_fps':args.num_frames/duration},
                       caption_words=words,caption_model='Qwen2.5-VL-7B-Instruct')
            prepared.append(candidate)
            write_json(args.output_dir/'progress.json',{'status':'running','completed':len(prepared),'total':32})
            write_manifest(args.output_dir/'partial_manifest.jsonl',prepared)
            print('Prepared dense caption',len(prepared),'of32',words,'words',flush=True)
        if sha256_file(args.manifest)!=original_hash:raise ValueError('Original manifest changed')
        validate_manifest(prepared,check_paths=True)
        write_manifest(args.output_dir/'train.jsonl',prepared)
        write_manifest(args.output_dir/'single_pilot.jsonl',prepared[:2])
        write_json(args.output_dir/'preparation_report.json',{'status':'passed','train_videos':32,'train_actors':16,
                    'source_manifest_sha256':original_hash,'train_manifest_sha256':sha256_file(args.output_dir/'train.jsonl'),
                    'frames':args.num_frames,'width':480,'height':352,'word_min':min(r['caption_words'] for r in prepared),
                    'word_max':max(r['caption_words'] for r in prepared),'dev_or_holdout_media_consumed':False,
                    'manual_caption_verification':False,'model_hashes':model_hashes,
                    'reuse_partial':str(args.reuse_partial) if args.reuse_partial else None,
                    'max_length_repairs_per_new_caption':2})
    except Exception as exc:
        write_json(args.output_dir/'failure.json',{'status':'failed','sample_id':current,'error_class':type(exc).__name__,
                                                  'error':str(exc),'completed':len(prepared)})
        raise


if __name__=='__main__':main()
