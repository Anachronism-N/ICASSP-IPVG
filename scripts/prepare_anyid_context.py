"""Freeze paired conditioning-crop manifests without changing original anchors or prompts."""
import argparse
import copy
import hashlib
from collections import Counter
from pathlib import Path

from ipvg.common import sha256_file, write_json
from ipvg.data import read_manifest, validate_manifest, write_manifest


def select_actors(original, legacy, extra=8):
    dev=[r for r in original if r.get('split')=='dev']
    counts=Counter(r['identity_id'] for r in dev)
    if not counts or any(n!=2 for n in counts.values()):
        raise ValueError('Require two dev prompts per actor')
    known={r['identity_id'] for r in legacy}
    anchors={r['sample_id']:r for r in dev}
    if len(known)!=4 or len(legacy)!=8 or len({r['sample_id'] for r in legacy})!=8:
        raise ValueError('Require the frozen four-actor, eight-input legacy cohort')
    for row in legacy:
        anchor=anchors.get(row['sample_id'])
        if anchor is None or row['identity_id']!=anchor['identity_id'] or row['prompt']!=anchor['prompt']:
            raise ValueError('Legacy cohort does not match original dev anchors')
    remaining=sorted(set(counts)-known,key=lambda a:hashlib.sha256(('anyid-context-v1:'+a).encode()).hexdigest())
    if len(remaining)<extra:
        raise ValueError('Insufficient additional dev actors')
    return sorted(known),remaining[:extra]


def paired_manifests(original, narrow, wide, legacy, pilot_sample):
    known,new=select_actors(original,legacy)
    actors=sorted(known+new)
    selected=sorted([r for r in original if r.get('split')=='dev' and r['identity_id'] in actors],key=lambda r:r['sample_id'])
    anchors={r['sample_id']:r for r in selected}
    conditions={}
    for name,rows,margin in [('narrow',narrow,.3),('wide',wide,.6)]:
        by_id={r['sample_id']:r for r in rows}
        if len(by_id)!=len(rows):
            raise ValueError('Duplicate conditioning sample IDs')
        for sid,anchor in anchors.items():
            row=by_id.get(sid)
            if row is None or row['prompt']!=anchor['prompt'] or row['identity_id']!=anchor['identity_id'] or row.get('split')!='dev':
                raise ValueError('Conditioning rows must preserve original dev prompts and actors')
            if row.get('original_reference_images')!=anchor['reference_images'] or row.get('reference_sha256')!=anchor['reference_sha256']:
                raise ValueError('Conditioning must preserve original identity anchors')
            if row.get('reference_crop',{}).get('margin')!=margin:
                raise ValueError('Incorrect conditioning margin')
            for field in ('source_video_id','reference_source_video_id'):
                if row.get(field)!=anchor.get(field):
                    raise ValueError('Conditioning source provenance changed')
        conditions[name]=by_id
    pilot=pilot_sample
    if pilot not in anchors:
        raise ValueError('Frozen diagnostic pilot is absent')
    manifests={}
    for name,by_id in conditions.items():
        manifests[name+'_pilot']=[dict(copy.deepcopy(by_id[pilot]),generation_seed=42)]
        for seed in (42,43):
            for shard in (0,1):
                cohort=set(actors[shard::2])
                rows=[dict(copy.deepcopy(by_id[r['sample_id']]),generation_seed=seed) for r in selected
                      if r['identity_id'] in cohort and not (seed==42 and r['sample_id']==pilot)]
                manifests[f'{name}_s{seed}_shard{shard}']=rows
    return manifests,{'legacy_actors':known,'additional_dev_actors':new,'seeds':[42,43],
                      'margins':{'narrow':.3,'wide':.6},'output_count':sum(len(r) for r in manifests.values()),
                      'pilot_sample':pilot,'identity_anchor':'original manifest','holdout_used':False,
                      'notice':'Legacy cohort is diagnostic; additional actors are dev expansion, not independent holdout.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','narrow','wide','legacy','output_dir'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--pilot-sample',required=True)
    args=parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError('Choose a fresh campaign directory')
    if not args.output_dir.parent.is_dir():
        raise FileNotFoundError('Output parent must exist')
    original=read_manifest(args.original)
    report=validate_manifest(original,check_paths=False)
    if len(original)!=128 or report['counts']!={'train':32,'dev':64,'holdout':32,'unassigned':0} or not report['leakage']['safe']:
        raise ValueError('Require the frozen 128-record isolated CREMA-D manifest')
    inputs={n:getattr(args,n) for n in ('original','narrow','wide','legacy')}
    hashes={n:sha256_file(p) for n,p in inputs.items()}
    manifests,protocol=paired_manifests(original,read_manifest(args.narrow),read_manifest(args.wide),read_manifest(args.legacy),args.pilot_sample)
    for rows in manifests.values():
        validate_manifest(rows,check_paths=True)
        for row in rows:
            if sha256_file(row['reference_images'][0])!=row['reference_crop']['sha256'] or sha256_file(row['original_reference_images'][0])!=row['reference_sha256']:
                raise ValueError('Conditioning or original reference hash changed')
    if hashes!={n:sha256_file(p) for n,p in inputs.items()}:
        raise ValueError('Input manifests changed during preparation')
    args.output_dir.mkdir()
    paths={}
    for name,rows in manifests.items():
        path=args.output_dir/(name+'.jsonl')
        write_manifest(path,rows)
        paths[name]={'path':str(path.resolve()),'count':len(rows),'sha256':sha256_file(path)}
    write_json(args.output_dir/'protocol.json',dict(protocol,input_sha256=hashes,manifests=paths))
    print('Frozen 96 paired outputs: 12 dev actors, two prompts, two seeds, two margins')


if __name__=='__main__':
    main()
