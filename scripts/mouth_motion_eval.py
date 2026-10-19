"""Offline aligned-mouth motion diagnostic; never a speech or expression classifier."""
import argparse
import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ipvg.common import load_json, sha256_file, write_json
from ipvg.data import read_manifest

PROTOCOL={'version':'local_mouth_motion_v1','frames':17,'detector':'SCRFD det_10g, threshold .5, size640',
          'alignment':'two-eye similarity to [38.2946,51.6963],[73.5318,51.5014],112square',
          'mouth_roi_xyxy':[28,80,84,106],'reference_roi_xyxy':[28,58,84,78],
          'flow':'Farneback pyr_scale .5, levels3, winsize15, iterations3, poly_n5, poly_sigma1.2, flags0',
          'score':'mean adjacent sampled-pair mouth magnitude after subtracting reference ROI median flow vector',
          'missing':'only adjacent pairs both single-face valid; never bridge missing frames',
          'min_pair_coverage':.75,'threshold':'max calibration rigid-control video score + 1e-6',
          'source_label':'acted utterance source, not independently annotated mouth-motion positives',
          'notice':'Local nonrigid mouth-region activity diagnostic, not speech, smile, lip sync or naturalness.'}


def eye_transform(points):
    eyes=np.asarray(points,dtype=np.float64)
    if eyes.shape!=(2,2) or not np.isfinite(eyes).all(): raise ValueError('Require two finite eye landmarks')
    target=np.array([[38.2946,51.6963],[73.5318,51.5014]])
    delta=eyes[1]-eyes[0]
    denominator=float(delta@delta)
    if denominator<1e-8: raise ValueError('Coincident eyes')
    goal=target[1]-target[0]
    a=float(delta@goal)/denominator
    b=float(delta[0]*goal[1]-delta[1]*goal[0])/denominator
    rotation=np.array([[a,-b],[b,a]])
    return np.column_stack((rotation,target[0]-rotation@eyes[0])).astype(np.float32)


def aligned_gray(frame,encoder):
    boxes,kps=encoder.detector.detect(frame,max_num=0,metric='default')
    if len(boxes)!=1: return None,'no_face' if not len(boxes) else 'multiple_faces'
    try: transform=eye_transform(kps[0,:2])
    except ValueError: return None,'invalid_eyes'
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    aligned=cv2.warpAffine(gray,transform,(112,112),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    return aligned,None


def residual_flow_score(flow):
    flow=np.asarray(flow)
    if flow.shape!=(112,112,2) or not np.isfinite(flow).all(): raise ValueError('Require finite 112x112x2 flow')
    drift=np.median(flow[58:78,28:84].reshape(-1,2),axis=0)
    mouth=flow[80:106,28:84]-drift
    return float(np.linalg.norm(mouth,axis=-1).mean())


def sequence_scores(frames):
    if len(frames)!=17: raise ValueError('Require exactly seventeen observations')
    values=[]
    gray_changes=[]
    for a,b in zip(frames[:-1],frames[1:]):
        if a is None or b is None:
            values.append(None)
            continue
        if a.shape!=(112,112) or b.shape!=(112,112): raise ValueError('Require aligned112 gray frames')
        flow=cv2.calcOpticalFlowFarneback(a,b,None,.5,3,15,3,5,1.2,0)
        values.append(residual_flow_score(flow))
        gray_changes.append(float(np.abs(b[80:106,28:84].astype(np.float32)-a[80:106,28:84]).mean()))
    valid=[v for v in values if v is not None]
    result={'frame_coverage':sum(f is not None for f in frames)/17,'pair_coverage':len(valid)/16,'pair_scores':values}
    if len(valid)>=12:
        result['mouth_residual_flow_mean']=float(np.mean(valid))
        result['mouth_gray_difference_mean']=float(np.mean(gray_changes))
    return result


def video_observation(path,encoder,controls=False):
    cap=cv2.VideoCapture(str(path))
    observations=[]
    reasons=[]
    try:
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count<17: raise ValueError('Insufficient video frames')
        ids=np.linspace(0,count-1,17).round().astype(int)
        initial=None
        for index in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(index))
            ok,frame=cap.read()
            if not ok: raise ValueError('Video decode failure')
            if initial is None: initial=frame.copy()
            aligned,reason=aligned_gray(frame,encoder)
            observations.append(aligned)
            reasons.append(reason)
    finally: cap.release()
    result={'source':sequence_scores(observations),'failure_counts':dict(Counter(r for r in reasons if r)),
            'video_sha256':sha256_file(path)}
    if controls:
        fixed,reason=aligned_gray(initial,encoder)
        result['freeze']=sequence_scores([fixed]*17)
        moved=[]
        for step in range(17):
            dx=4*np.sin(step*np.pi/4)
            dy=3*np.cos(step*np.pi/4)
            frame=cv2.warpAffine(initial,np.float32([[1,0,dx],[0,1,dy]]),initial.shape[1::-1],
                                 flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            aligned,_=aligned_gray(frame,encoder)
            moved.append(aligned)
        result['rigid']=sequence_scores(moved)
    return result


def calibrate(rows,encoder):
    actors=sorted({r['identity_id'] for r in rows},key=lambda a:hashlib.sha256(('mouth-cal-v1:'+a).encode()).hexdigest())
    if len(actors)!=32 or any(r.get('split')!='dev' for r in rows): raise ValueError('Require32dev actors only')
    records=[]
    for row in rows:
        records.append(dict(video_observation(row['video_path'],encoder,True),identity_id=row['identity_id'],sample_id=row['sample_id']))
    cal=[r for r in records if r['identity_id'] in actors[:16]]
    val=[r for r in records if r['identity_id'] in actors[16:]]
    values=[r['rigid']['mouth_residual_flow_mean'] for r in cal if 'mouth_residual_flow_mean' in r['rigid']]
    threshold=max(values)+1e-6 if values else None
    groups=[]
    for items in (cal,val):
        summary={}
        for kind in ('source','freeze','rigid'):
            valid=[r for r in items if 'mouth_residual_flow_mean' in r[kind]]
            scores=[r[kind]['mouth_residual_flow_mean'] for r in valid]
            summary[kind]={'scoreable_videos':len(valid),'scoreable_actors':len({r['identity_id'] for r in valid}),
                           'mean':float(np.mean(scores)) if scores else None,
                           'activation_fraction':float(np.mean(np.array(scores)>threshold)) if scores and threshold is not None else None}
        groups.append(summary)
    passed=bool(threshold is not None and all(g[k]['scoreable_videos']>=28 and g[k]['scoreable_actors']>=14
                                               for g in groups for k in ('source','freeze','rigid')))
    passed=bool(passed and all(g['freeze']['activation_fraction']==0 for g in groups)
                and groups[1]['rigid']['activation_fraction']<=.10 and groups[1]['source']['activation_fraction']>=.75)
    return {'status':'passed_local_activity_controls' if passed else 'motion_proxy_not_validated',
            'protocol':PROTOCOL,'weight_hashes':encoder.weight_hashes,'threshold':threshold,
            'calibration':groups[0],'validation':groups[1],'records':records,
            'speech_or_expression_validation_passed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['calibrate','evaluate'])
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--outputs',type=Path)
    parser.add_argument('--calibration',type=Path)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Choose fresh evaluation output')
    from identity_eval import OfflineFaceEncoder
    cv2.setNumThreads(4)
    encoder=OfflineFaceEncoder(args.model_dir)
    rows=[r for r in read_manifest(args.manifest) if r.get('split')=='dev']
    if args.mode=='calibrate':
        report=calibrate(rows,encoder)
    else:
        if not args.calibration or not args.outputs: raise ValueError('Require frozen calibration and outputs')
        calibration=load_json(args.calibration)
        if calibration['protocol']!=PROTOCOL or calibration['weight_hashes']!=encoder.weight_hashes:
            raise ValueError('Frozen motion protocol or weights changed')
        records=[]
        for row in rows:
            data=video_observation(args.outputs/(row['sample_id']+'.mp4'),encoder)
            metrics=data['source']
            if 'mouth_residual_flow_mean' in metrics and calibration['threshold'] is not None:
                metrics['local_activity_above_control']=int(metrics['mouth_residual_flow_mean']>calibration['threshold'])
            records.append(dict(data,sample_id=row['sample_id'],identity_id=row['identity_id']))
        report={'protocol':PROTOCOL,'weight_hashes':encoder.weight_hashes,'calibration_status':calibration['status'],
                'threshold':calibration['threshold'],'records':records,'speech_or_expression_validation_passed':False}
    report['manifest_sha256']=sha256_file(args.manifest)
    write_json(args.output,report)
    print(report['status'] if 'status' in report else 'Scored local mouth activity',len(rows),'dev videos')


if __name__=='__main__':
    main()
