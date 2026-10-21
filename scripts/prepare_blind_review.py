"""Prepare a local neutral-name review pack; no human scores are generated."""
import argparse
import hashlib
import html
import random
from pathlib import Path
import shutil
from ipvg.common import load_json,write_json,sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError('Choose a fresh private review directory')
    if not args.output_dir.parent.is_dir():raise FileNotFoundError('Review parent must exist')
    rows=load_json(args.manifest)['samples']
    main_rows=[r for r in rows if not r['group'].startswith('archive_')]
    pairs={}
    for r in main_rows:pairs.setdefault(r['base_sample_id'],{})[r['group']]=r
    if len(pairs)!=16:raise ValueError('Expected16main development inputs')
    args.output_dir.mkdir();(args.output_dir/'media').mkdir()
    rng=random.Random(42); reviews=[]; key=[]; parts=[]
    for base_id,arms in sorted(pairs.items()):
        for candidate in ('lr1','lr5','alpha05','alpha025','structure0'):
            left,right='baseline',candidate
            if rng.getrandbits(1):left,right=right,left
            rid=hashlib.sha256((base_id+':'+candidate).encode()).hexdigest()[:16]
            item={'review_id':rid,'prompt':arms['baseline']['prompt'],'reference':'media/'+rid+'_ref.png',
                  'left':'media/'+rid+'_left.mp4','right':'media/'+rid+'_right.mp4',
                  'human_scores':None,'axes':['identity','action_completion','natural_motion','visible_artifacts','overall_preference'],
                  'reviewer_confidence':None}
            ref=Path(arms['baseline']['reference_image'])
            if sha256_file(ref)!=arms['baseline']['reference_sha256']:raise ValueError('Reference hash changed')
            shutil.copyfile(ref,args.output_dir/item['reference'])
            for side,arm in (('left',left),('right',right)):
                row=arms[arm];file=Path(row['video_path'])
                if sha256_file(file)!=row['video_sha256']:raise ValueError('Review video hash changed')
                shutil.copyfile(file,args.output_dir/item[side])
            reviews.append(item);key.append({'review_id':rid,'base_sample_id':base_id,'left_method':left,'right_method':right})
            parts.append('<section><h2>'+rid+'</h2><p>'+html.escape(item['prompt'])+'</p><img width="240" src="'+item['reference']+'"><div><video width="480" controls loop src="'+item['left']+'"></video><video width="480" controls loop src="'+item['right']+'"></video></div></section>')
    write_json(args.output_dir/'review_form.json',{'protocol':'local_blind_review_v1','human_completed':False,'reviews':reviews,
       'instructions':'Independent reviewer fills ratings, confidence and reasons. Source language alone does not establish actual speech or smile truth. Not official challenge human scoring.'})
    write_json(args.output_dir/'private_method_key.json',{'key':key,'keep_separate_from_reviewer':True})
    (args.output_dir/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>本地视频盲评</title><body><h1>本地视频盲评</h1><p>仅在获准环境查看。请独立填写review_form.json；身份、动作完成、自然运动、伪影和总体偏好分列，无法判断要标记缺失。没有预填人工分数。</p>'+''.join(parts)+'</body></html>',encoding='utf-8')
    print('Prepared',len(reviews),'local neutral-name blinded comparisons; all human scores remain missing')


if __name__=='__main__':main()
