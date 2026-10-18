"""Offline local identity proxy, source-video calibration and paired analysis."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from ipvg.common import load_json, sha256_file, write_json
from ipvg.data import read_manifest

PROTOCOL = {'version': 'arcface_cremad_v1', 'frames': 8, 'frame_sampling': 'linspace indices including endpoints',
            'det_threshold': .5, 'det_size': [640,640], 'multi_face_policy': 'exactly one face else missing',
            'identity_anchor': 'original reference, never candidate conditioning crop',
            'percentile_method': 'linear', 'missing': 'omit identity score; report coverage separately',
            'provider': 'CPUExecutionProvider', 'recognition': 'buffalo_l/w600k_r50', 'alignment': 'InsightFace norm_crop 5-point 112 pixels',
            'calibration': 'real source targets only; hash-split 16 actors calibration and 16 validation',
            'calibration_gate': {'min_positive':28,'min_actors':14,'min_valid_frames':7,'auc_ci_lower':.90,'max_eer':.10,'far':.01,'min_validation_tpr':.80}}


def frame_indices(count, samples=8):
    if count < samples:
        raise ValueError('Video has fewer frames than the fixed sampling count')
    return np.linspace(0,count-1,samples).round().astype(int).tolist()


def scores(values, total=8, threshold=None):
    valid=[float(v) for v in values if v is not None and np.isfinite(v)]
    result={'face_detection_coverage':len(valid)/total}
    if valid:
        result.update(face_identity_cosine_mean=float(np.mean(valid)),
                      face_identity_cosine_p10=float(np.quantile(valid,.1,method='linear')),
                      face_identity_cosine_min=min(valid))
        if threshold is not None and len(valid)>=7:
            result['face_identity_video_below_threshold']=int(np.mean(valid)<threshold)
    return result


class OfflineFaceEncoder:
    def __init__(self, model_dir):
        import onnxruntime as ort
        from insightface.model_zoo.scrfd import SCRFD
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        self.paths=[Path(model_dir)/n for n in ('det_10g.onnx','w600k_r50.onnx')]
        if not all(p.is_file() for p in self.paths):
            raise FileNotFoundError('Explicit local detector and recognition ONNX files are required; automatic downloads disabled')
        options=ort.SessionOptions()
        options.intra_op_num_threads=4
        options.inter_op_num_threads=1
        detector_session=ort.InferenceSession(str(self.paths[0]),sess_options=options,providers=['CPUExecutionProvider'])
        recognition_session=ort.InferenceSession(str(self.paths[1]),sess_options=options,providers=['CPUExecutionProvider'])
        self.detector=SCRFD(model_file=str(self.paths[0]),session=detector_session)
        self.recognition=ArcFaceONNX(model_file=str(self.paths[1]),session=recognition_session)
        self.detector.prepare(ctx_id=0,input_size=(640,640),det_thresh=.5)
        self.recognition.prepare(ctx_id=0)
        self.weight_hashes={p.name:sha256_file(p) for p in self.paths}

    def face(self, image_bgr):
        from insightface.utils.face_align import norm_crop
        boxes,kps=self.detector.detect(image_bgr,max_num=0,metric='default')
        if len(boxes)!=1:
            return None,'no_face' if not len(boxes) else 'multiple_faces'
        crop=norm_crop(image_bgr,landmark=kps[0],image_size=112)
        feature=self.recognition.get_feat(crop).reshape(-1).astype(np.float32)
        norm=float(np.linalg.norm(feature))
        if not np.isfinite(feature).all() or norm<=0:
            return None,'invalid_embedding'
        return {'embedding':feature/norm,'bbox':boxes[0,:4],'kps':kps[0]},None


def video_features(path, encoder):
    capture=cv2.VideoCapture(str(path))
    vectors=[]
    reasons=[]
    gray=[]
    try:
        ids=frame_indices(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        for index in ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=capture.read()
            if not ok:
                raise ValueError('Sampled video frame cannot be decoded')
            face,reason=encoder.face(frame)
            vectors.append(None if face is None else face['embedding'])
            reasons.append(reason)
            gray.append(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(120,120)).astype(np.float32))
    finally:
        capture.release()
    motion=float(np.abs(np.diff(np.stack(gray),axis=0)).mean())
    return vectors,reasons,motion


def references(rows, encoder):
    result={}
    for row in rows:
        actor=row['identity_id']
        path=row['reference_images'][0]
        if actor not in result:
            image=cv2.imread(path)
            if image is None: raise ValueError('Reference decode failure')
            face,reason=encoder.face(image)
            result[actor]={'face':face,'reason':reason,'path':path}
        elif result[actor]['path']!=path:
            raise ValueError('Actor reference anchor differs across samples')
    return result


def roc_stats(positive, negative):
    from sklearn.metrics import roc_auc_score,roc_curve
    labels=np.array([1]*len(positive)+[0]*len(negative))
    values=np.array(positive+negative)
    fpr,tpr,thresholds=roc_curve(labels,values)
    difference=fpr+tpr-1
    upper=int(np.flatnonzero(difference>=0)[0])
    if upper==0:
        eer=float(fpr[0])
    else:
        lower=upper-1
        fraction=float(-difference[lower]/(difference[upper]-difference[lower]))
        eer=float(fpr[lower]+fraction*(fpr[upper]-fpr[lower]))
    return {'auc':float(roc_auc_score(labels,values)),'eer':eer}


def split_pairs(actors, observations, refs):
    positive=[]
    negative=[]
    positive_by_actor={}
    negative_by_actor={}
    for actor in actors:
        own=refs[actor]['face']
        pos=[]
        neg=[]
        if own is not None:
            for item in observations:
                if item['actor']!=actor or len(item['valid'])<7: continue
                pos.append(float(np.mean([own['embedding']@v for v in item['valid']])))
                for other in actors:
                    if other!=actor and refs[other]['face'] is not None:
                        neg.append(float(np.mean([refs[other]['face']['embedding']@v for v in item['valid']])))
        positive_by_actor[actor]=pos
        negative_by_actor[actor]=neg
        positive+=pos
        negative+=neg
    return positive,negative,positive_by_actor,negative_by_actor


def bootstrap_auc(actors, positive, negative, seed=42, count=1000):
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(count):
        selected=rng.choice(actors,len(actors),replace=True)
        p=[v for a in selected for v in positive[a]]
        n=[v for a in selected for v in negative[a]]
        if p and n: values.append(roc_stats(p,n)['auc'])
    return np.quantile(values,[.025,.975]).tolist() if values else [None,None]


def calibrate(rows, encoder, output):
    rows=[r for r in rows if r.get('split')=='dev']
    actors=sorted({r['identity_id'] for r in rows},key=lambda a:hashlib.sha256(('identity-cal-v1:'+a).encode()).hexdigest())
    if len(actors)!=32: raise ValueError('Calibration expects exactly the frozen 32 dev actors')
    refs=references(rows,encoder)
    observations=[]
    for row in rows:
        vectors,reasons,motion=video_features(row['video_path'],encoder)
        observations.append({'actor':row['identity_id'],'valid':[v for v in vectors if v is not None]})
    groups=[]
    for group in (actors[:16],actors[16:]):
        p,n,pa,na=split_pairs(group,observations,refs)
        stats=roc_stats(p,n) if p and n else {'auc':None,'eer':None}
        stats.update(positive_pairs=len(p),negative_pairs=len(n),valid_actors=sum(bool(pa[a]) for a in group),
                     auc_actor_bootstrap_ci95=bootstrap_auc(group,pa,na))
        groups.append((stats,p,n))
    cal,validation=groups[0][0],groups[1][0]
    negatives=groups[0][2]
    threshold=float(np.quantile(negatives,.99,method='higher'))+1e-7 if negatives else None
    if threshold is not None:
        cal.update(threshold=threshold,empirical_far=float(np.mean(np.array(negatives)>=threshold)))
        validation.update(frozen_threshold=threshold,empirical_far=float(np.mean(np.array(groups[1][2])>=threshold)),
                          tpr=float(np.mean(np.array(groups[1][1])>=threshold)))
    passed=all(s['positive_pairs']>=28 and s['valid_actors']>=14 and s['auc_ci_lower']>.90 and s['eer']<=.10
               for s in [dict(g[0],auc_ci_lower=g[0]['auc_actor_bootstrap_ci95'][0] or 0) for g in groups])
    passed=bool(passed and threshold is not None and validation['tpr']>=.80 and validation['empirical_far']<=.01)
    report={'status':'passed' if passed else 'failed','protocol':PROTOCOL,'weight_hashes':encoder.weight_hashes,
            'manifest_sha256':None,'calibration':cal,'validation':validation,'threshold':threshold,
            'self_similarity_max_error':max([abs(float(r['face']['embedding']@r['face']['embedding'])-1) for r in refs.values() if r['face'] is not None],default=None),
            'notice':'Real same-actor cross-source calibration; empirical small-set FAR is not a certification; no generated score used to select threshold.'}
    write_json(output,report)
    return report


def evaluate(rows, encoder, outputs, output, threshold=None):
    rows=[r for r in rows if r.get('split')=='dev']
    refs=references(rows,encoder)
    details=[]
    proxy=[]
    for row in rows:
        anchor=refs[row['identity_id']]
        if 'reference_crop' in row or row.get('original_reference_images'):
            raise ValueError('Identity evaluation must consume the original anchor manifest, not crop conditioning')
        if row.get('reference_sha256') and sha256_file(anchor['path'])!=row['reference_sha256']:
            raise ValueError('Original reference anchor hash changed')
        path=Path(outputs)/(row['sample_id']+'.mp4')
        vectors,reasons,motion=video_features(path,encoder)
        values=[None if vector is None or anchor['face'] is None else float(anchor['face']['embedding']@vector) for vector in vectors]
        metrics=scores(values,threshold=threshold)
        metrics['face_detection_coverage']=sum(v is not None for v in vectors)/8
        metrics['identity_scoring_coverage']=sum(v is not None for v in values)/8
        metrics['sampled_gray_motion_mean_abs']=motion
        failures=sorted({r for r in reasons+[anchor['reason']] if r})
        details.append({'sample_id':row['sample_id'],'identity_id':row['identity_id'],'scores':metrics,
                        'frame_cosines':values,'failure_classes':failures,'output_sha256':sha256_file(path)})
        proxy.append({'sample_id':row['sample_id'],'score_kind':'local_proxy','scores':metrics,'failure_classes':failures})
    output=Path(output)
    if output.exists(): raise FileExistsError('Choose a fresh identity evaluation output')
    write_json(output,{'protocol':PROTOCOL,'weight_hashes':encoder.weight_hashes,'records':details,'identity_score_kind':'local_proxy','official_score':None})
    with output.with_suffix('.jsonl').open('x') as handle:
        for item in proxy: handle.write(json.dumps(item,allow_nan=False)+'\n')
    return details


def paired_report(base, candidate, output):
    base_report,candidate_report=load_json(base),load_json(candidate)
    if base_report['protocol']!=candidate_report['protocol'] or base_report['weight_hashes']!=candidate_report['weight_hashes']:
        raise ValueError('Paired evaluator protocol or weights differ')
    left={r['sample_id']:r for r in base_report['records']}
    right={r['sample_id']:r for r in candidate_report['records']}
    if len(left)!=len(base_report['records']) or len(right)!=len(candidate_report['records']):
        raise ValueError('Duplicate paired sample IDs')
    if set(left)!=set(right): raise ValueError('Paired samples differ')
    if not left: raise ValueError('Paired reports contain no samples')
    differences={}
    motion=[]
    coverage=[]
    p10=[]
    for sample in left:
        a,b=left[sample],right[sample]
        coverage.append(b['scores']['face_detection_coverage']-a['scores']['face_detection_coverage'])
        if a['identity_id']!=b['identity_id']:
            raise ValueError('Paired actor IDs differ')
        if (a['scores'].get('identity_scoring_coverage',a['scores']['face_detection_coverage'])>=.875
                and b['scores'].get('identity_scoring_coverage',b['scores']['face_detection_coverage'])>=.875):
            common=[(x,y) for x,y in zip(a['frame_cosines'],b['frame_cosines']) if x is not None and y is not None]
            if len(common)>=7:
                differences.setdefault(a['identity_id'],[]).append(float(np.mean([y-x for x,y in common])))
                p10.append(float(np.quantile([y for x,y in common],.1)-np.quantile([x for x,y in common],.1)))
        motion.append((a['scores']['sampled_gray_motion_mean_abs'],b['scores']['sampled_gray_motion_mean_abs']))
    rng=np.random.default_rng(42)
    actor_means=np.array([np.mean(v) for v in differences.values()])
    boot=np.mean(rng.choice(actor_means,(2000,len(actor_means)),replace=True),axis=1) if len(actor_means) else np.array([])
    ci=np.quantile(boot,[.025,.975]).tolist() if len(boot) else [None,None]
    motion_ratio=float(np.mean([b for a,b in motion])/max(np.mean([a for a,b in motion]),1e-9))
    static_delta=float(np.mean([b<.5 for a,b in motion])-np.mean([a<.5 for a,b in motion]))
    passed=bool(sum(map(len,differences.values()))>=56 and len(differences)>=28 and np.mean(coverage)>=-.05
                and ci[0] is not None and ci[0]>0 and np.mean(p10)>=0 and motion_ratio>=.8 and static_delta<=.10)
    base_means=[r['scores']['face_identity_cosine_mean'] for r in left.values() if 'face_identity_cosine_mean' in r['scores']]
    candidate_means=[r['scores']['face_identity_cosine_mean'] for r in right.values() if 'face_identity_cosine_mean' in r['scores']]
    report={'status':'supports_h1_on_this_dev_set' if passed else 'h1_not_promoted',
            'paired_samples':sum(map(len,differences.values())),'actors':len(differences),
            'identity_mean_delta':float(np.mean(actor_means)) if len(actor_means) else None,'actor_bootstrap_ci95':ci,
            'identity_p10_delta':float(np.mean(p10)) if p10 else None,'coverage_delta':float(np.mean(coverage)),
            'sampled_gray_motion_ratio':motion_ratio,'static_fraction_delta':static_delta,'static_threshold_gray_difference':.5,'base_identity_mean':float(np.mean(base_means)) if base_means else None,
            'candidate_identity_mean':float(np.mean(candidate_means)) if candidate_means else None,
            'limitations':['Single seed, acted talking-head domain, local proxy not official score.',
                           'Gray motion proxy is not text/action correctness or motion quality.']}
    write_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['calibrate','evaluate','compare'])
    parser.add_argument('--manifest')
    parser.add_argument('--model-dir')
    parser.add_argument('--outputs')
    parser.add_argument('--output',required=True)
    parser.add_argument('--calibration')
    parser.add_argument('--baseline')
    parser.add_argument('--candidate')
    args=parser.parse_args()
    if args.mode=='compare':
        print(json.dumps(paired_report(args.baseline,args.candidate,args.output),indent=2))
        return
    encoder=OfflineFaceEncoder(args.model_dir)
    rows=read_manifest(args.manifest,check_paths=False)
    if args.mode=='calibrate':
        report=calibrate(rows,encoder,args.output)
        report['manifest_sha256']=sha256_file(args.manifest)
        write_json(args.output,report)
        print(json.dumps(report,indent=2))
    else:
        if not args.calibration:
            raise ValueError('A frozen successful calibration is required')
        calibration=load_json(args.calibration)
        if (calibration['status']!='passed' or calibration['weight_hashes']!=encoder.weight_hashes
                or calibration['protocol']!=PROTOCOL):
            raise ValueError('Identity calibration failed or protocol/weights changed')
        evaluate(rows,encoder,args.outputs,args.output,calibration['threshold'])
        print('Scored',sum(r.get('split')=='dev' for r in rows),'samples with original reference anchors')


if __name__=='__main__':
    main()
