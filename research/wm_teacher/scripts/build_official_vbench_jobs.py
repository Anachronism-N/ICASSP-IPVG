#!/usr/bin/env python3
"""Build the 5590 official VBench-I2V jobs for official_wan22_generate.py (po prompts, 81 frames, 16 fps)."""
import glob,json,hashlib,sys,os
E='/srv/wm_teacher/eval/teacher_progress_campaign_20261104'; O='/srv/wm_teacher/eval/official_reeval_20261122'; ROOT='/srv/wm_teacher'
model=sys.argv[1]; ckpt=sys.argv[2]
po={}
for f in glob.glob(O+'/vbench_po_prompts/shard*.jsonl'):
    for l in open(f):
        r=json.loads(l); po[(os.path.basename(r['image']),r['raw_prompt'])]=r
recs=json.load(open(E+'/vbench_official_complete_coverage.preflight.json'))['coverage_records']
missing={(r['image_key'],r['raw_prompt']) for r in recs}-set(po)
assert not missing, ('missing extensions',len(missing))
jobs=[]
for r in recs:
    p=po[(r['image_key'],r['raw_prompt'])]
    jobs.append(dict(id=r['filename'][:-4],image=ROOT+'/eval/vbench_i2v/images_16-9/'+r['image_key'],prompt=p['prompt'],
        out=ROOT+'/eval/vbench_i2v/videos_official5/'+model+'/'+r['filename'],seed=r['proposed_seed'],frame_num=81,fps=16,ckpt_dir=ckpt))
assert len(jobs)==5590 and len({j['out'] for j in jobs})==5590
# interleave core and camera so every node makes balanced progress on both
core=[j for j,r in zip(jobs,recs) if r['group']=='core']; cam=[j for j,r in zip(jobs,recs) if r['group']=='camera']
p=O+'/vbench_official/jobs_'+model+'.jsonl'; os.makedirs(os.path.dirname(p),exist_ok=True)
open(p,'x').write(''.join(json.dumps(j,sort_keys=True,ensure_ascii=False)+'\n' for j in core+cam))
print(p,len(jobs),hashlib.sha256(open(p,'rb').read()).hexdigest(),'fallbacks',sum(v['fallback'] for v in po.values()))
