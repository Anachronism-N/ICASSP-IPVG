"""Independent sparse-reconstruction audit, not identity editing or official action scoring."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from ipvg.common import load_json,write_json,sha256_file
from ipvg.data import read_manifest


def error_summary(source,candidate,keys,common_nonkeys):
    a,b=np.asarray(source,dtype=np.float32),np.asarray(candidate,dtype=np.float32)
    if a.shape!=b.shape or a.shape!=(33,192,320,3):raise ValueError('Probe media shapes differ')
    if not common_nonkeys:raise ValueError('Empty common nonkey set')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite probe frames')
    if len(keys)!=5 or keys!=sorted(set(keys)) or not {0,16,32}<=set(keys):raise ValueError('Invalid anchor schedule')
    if len(common_nonkeys)!=len(set(common_nonkeys)) or any(isinstance(i,bool) or not isinstance(i,int) or i<0 or i>=33 or i in keys for i in common_nonkeys):
        raise ValueError('Common nonkey frame set is invalid')
    differences=np.abs(a-b).mean(axis=(1,2,3))
    nonkeys=[i for i in range(33) if i not in keys]
    return {'fullframe_mae':float(differences.mean()),'nonkey_mae':float(differences[nonkeys].mean()),
            'common_nonkey_mae':float(differences[common_nonkeys].mean()),
            'adjacent_motion_mae':float(np.abs(np.diff(a,axis=0)-np.diff(b,axis=0)).mean()),
            'maximum_anchor_gap':max(y-x for x,y in zip(keys[:-1],keys[1:])),
            'sampled_motion_mean':float(np.abs(np.diff(b,axis=0)).mean())}


def main():
    from keyframe_event_ablation import decode
    from evaluate_window_probe import color_centroid,event_score
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--runs-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot-run',default='mm26_keyframes_pilot_20261020_r002')
    parser.add_argument('--batch-run',default='mm26_keyframes_batch_20261020_r001')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose fresh independent audit output')
    rows=read_manifest(args.manifest)
    metrics={}
    for run_id in (args.pilot_run,args.batch_run):
        run=args.runs_dir/run_id
        result=load_json(run/'result.json')
        backend=load_json(run/'backend_metrics.json')
        if result['status']!='completed' or backend['status']!='completed':raise ValueError('Incomplete reconstruction')
        for record in backend['samples']:
            key=(record['group'],record['base_sample_id'],record['mode'])
            if key in metrics:raise ValueError('Duplicate reconstructed input')
            path=run/'outputs'/(record['sample_id']+'.mp4')
            if record['model_calls']!=28 or sha256_file(path)!=record['output_sha256'] or record['output_sha256']!=result['output_sha256'][path.name]:
                raise ValueError('Reconstruction calls or hash mismatch')
            metrics[key]=(record,path)
    input_rows={}
    for row in rows:
        key=(row['group'],row['base_sample_id'])
        if key in input_rows:
            previous=input_rows[key]
            if any(row[k]!=previous[k] for k in ('input_video_sha256','identity_id','timeline')):
                raise ValueError('Paired source or timeline differs')
        input_rows[key]=row
    if len(metrics)!=2*len(input_rows):raise ValueError('Reconstruction input coverage differs')
    details=[]
    for key,row in sorted(input_rows.items()):
        if sha256_file(row['input_video_path'])!=row['input_video_sha256']:raise ValueError('Source changed')
        source=decode(row['input_video_path'])
        positions=[color_centroid(cv2.cvtColor(f,cv2.COLOR_RGB2BGR),row['identity_id']) for f in source]
        source_events=[event_score(positions,8,e,320,192) for e in row['timeline']]
        common=[i for i in range(33) if all(i not in metrics[key+(mode,)][0]['keyframe_indices'] for mode in ('uniform','event_error'))]
        for mode in ('uniform','event_error'):
            record,path=metrics[key+(mode,)]
            frames=decode(path)
            keys=record['keyframe_indices']
            if len(keys)!=5 or not all(np.array_equal(frames[i],source[i]) for i in keys):raise ValueError('Encoded anchors changed')
            positions=[color_centroid(cv2.cvtColor(f,cv2.COLOR_RGB2BGR),row['identity_id']) for f in frames]
            events=[event_score(positions,8,e,320,192) for e in row['timeline']]
            scores=error_summary(source,frames,keys,common)
            details.append({'group':key[0],'base_sample_id':key[1],'mode':mode,'scores':scores,
                            'event_pass_count':sum(e['direction_pass'] for e in events),'source_event_pass_count':sum(e['direction_pass'] for e in source_events),
                            'events':events,'both_events_pass':all(e['direction_pass'] for e in events),
                            'color_detection_coverage':sum(p is not None for p in positions)/33,
                            'keyframe_indices':keys,'input_sha256':row['input_video_sha256'],'output_sha256':record['output_sha256']})
    groups={}
    for group in sorted({r['group'] for r in details}):
        stats={}
        for mode in ('uniform','event_error'):
            subset=[r for r in details if r['group']==group and r['mode']==mode]
            stats[mode]={name:float(np.mean([r['scores'][name] for r in subset])) for name in subset[0]['scores']}
            stats[mode].update(videos=len(subset),event_pass_count=sum(r['event_pass_count'] for r in subset),
                               both_events_pass_count=sum(r['both_events_pass'] for r in subset),
                               color_coverage=float(np.mean([r['color_detection_coverage'] for r in subset])))
        stats['common_nonkey_mae_ratio']=stats['event_error']['common_nonkey_mae']/max(stats['uniform']['common_nonkey_mae'],1e-9)
        stats['source_event_pass_count']=sum(r['source_event_pass_count'] for r in details if r['group']==group and r['mode']=='uniform')
        groups[group]=stats
    write_json(args.output,{'status':'completed','groups':groups,'records':details,
                            'limitations':['Raw source is reconstruction target, not action ground truth.',
                            'Color centroid is restricted to self-generated nonhuman probes.',
                            'Cannot restore actions absent from the draft; no identity injection or complete KeyID reproduction.']})
    print({k:v for k,v in groups.items()})


if __name__=='__main__':main()
