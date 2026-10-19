"""Actor-paired context-crop aggregation, retaining missing scores and seed strata."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from ipvg.common import load_json,write_json


def summarize_pairs(narrow,wide,actors,seed=None):
    def index(records):
        result={}
        for row in records:
            if isinstance(row['seed_offset'],bool) or not isinstance(row['seed_offset'],int) or row['seed_offset'] not in (42,43):
                raise ValueError('Unexpected seed offset')
            key=(row['sample_id'],row['seed_offset'])
            if key in result: raise ValueError('Duplicate sample/seed')
            values=row['frame_cosines']
            if len(values)!=8 or any(v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v)) for v in values):
                raise ValueError('Require eight finite cosines or missing entries')
            coverage=sum(v is not None for v in values)/8
            if row['scores']['identity_scoring_coverage']!=coverage:
                raise ValueError('Coverage inconsistent with valid identity frames')
            result[key]=row
        return result
    left,right=index(narrow),index(wide)
    if not left or set(left)!=set(right): raise ValueError('Nonempty paired sample/seed sets must match')
    if any(left[k]['identity_id']!=right[k]['identity_id'] for k in left): raise ValueError('Paired actors differ')
    keys=sorted(k for k,r in left.items() if r['identity_id'] in actors and (seed is None or k[1]==seed))
    if not keys: raise ValueError('Empty selected cohort')
    result={}
    for arm,items in [('narrow',left),('wide',right)]:
        rows=[items[k] for k in keys]
        means=[float(np.mean([v for v in r['frame_cosines'] if v is not None])) for r in rows if any(v is not None for v in r['frame_cosines'])]
        motion=[r['scores']['sampled_gray_motion_mean_abs'] for r in rows]
        if not all(not isinstance(v,bool) and isinstance(v,(int,float)) and np.isfinite(v) and v>=0 for v in motion):
            raise ValueError('Invalid motion observations')
        result[arm]={'videos':len(rows),'scoreable_videos':len(means),
                     'videos_at_least_seven_frames':sum(r['scores']['identity_scoring_coverage']>=.875 for r in rows),
                     'identity_coverage':float(np.mean([r['scores']['identity_scoring_coverage'] for r in rows])),
                     'valid_video_identity_mean':float(np.mean(means)) if means else None,
                     'motion_mean':float(np.mean(motion)),'static_fraction':float(np.mean(np.array(motion)<.5))}
    deltas,p10=defaultdict(list),defaultdict(list)
    recovered=lost=0
    for key in keys:
        a,b=left[key],right[key]
        av,bv=sum(v is not None for v in a['frame_cosines']),sum(v is not None for v in b['frame_cosines'])
        recovered+=av<7<=bv
        lost+=bv<7<=av
        common=[(x,y) for x,y in zip(a['frame_cosines'],b['frame_cosines']) if x is not None and y is not None]
        if len(common)>=7:
            deltas[a['identity_id']].append(float(np.mean([y-x for x,y in common])))
            p10[a['identity_id']].append(float(np.quantile([y for x,y in common],.1)-np.quantile([x for x,y in common],.1)))
    values=np.array([np.mean(deltas[a]) for a in sorted(deltas)])
    rng=np.random.default_rng(42)
    boot=np.mean(rng.choice(values,(2000,len(values)),replace=True),axis=1) if len(values) else []
    result.update(paired_videos=sum(map(len,deltas.values())),paired_actors=len(deltas),
                  identity_actor_mean_delta=float(np.mean(values)) if len(values) else None,
                  identity_actor_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist() if len(boot) else [None,None],
                  identity_p10_actor_delta=float(np.mean([np.mean(v) for v in p10.values()])) if p10 else None,
                  coverage_delta=result['wide']['identity_coverage']-result['narrow']['identity_coverage'],
                  motion_ratio=result['wide']['motion_mean']/max(result['narrow']['motion_mean'],1e-9),
                  static_fraction_delta=result['wide']['static_fraction']-result['narrow']['static_fraction'],
                  recovered_seven_frame_videos=recovered,lost_seven_frame_videos=lost)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores-dir',type=Path,required=True)
    parser.add_argument('--cohorts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Choose fresh aggregate output')
    protocol=load_json(args.cohorts)
    records={'narrow':[],'wide':[]}
    frozen=None
    for arm in records:
        for stage in ('pilot','s42_shard0','s42_shard1','s43_shard0','s43_shard1'):
            report=load_json(args.scores_dir/(arm+'_'+stage+'.json'))
            current=(report['protocol'],report['weight_hashes'])
            if frozen is not None and current!=frozen: raise ValueError('Metric protocol/weights differ')
            frozen=current
            seed=43 if stage.startswith('s43') else 42
            records[arm].extend(dict(r,seed_offset=seed) for r in report['records'])
    known,new=protocol['legacy_actors'],protocol['additional_dev_actors']
    if len(known)!=4 or len(new)!=8 or set(known)&set(new): raise ValueError('Require distinct four/eight actor cohorts')
    if any(len(r)!=48 for r in records.values()): raise ValueError('Require 48 outputs per condition')
    actors=known+new
    result={name:summarize_pairs(records['narrow'],records['wide'],cohort,seed) for name,cohort,seed in
            [('all',actors,None),('legacy_diagnostic',known,None),('additional_dev',new,None),('seed42',actors,42),('seed43',actors,43)]}
    write_json(args.output,{'cohorts':result,'identity_protocol':frozen[0],'weight_hashes':frozen[1],
                            'notice':'Local development proxy, descriptive actor bootstrap; not official or holdout evaluation.'})
    print('Aggregated 48 paired videos over 12 actors and two seeds')


if __name__=='__main__':
    main()
