"""Local color-centroid timeline proxy for programmatic nonhuman probes, not a general action metric."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from ipvg.common import sha256_file,write_json
from ipvg.data import read_manifest
from ipvg.evaluate import probe_video


def color_centroid(image_bgr, subject, minimum_area=30):
    blue,green,red=image_bgr.astype(np.float32).transpose(2,0,1)
    if subject=='red_ball':
        mask=(red>80)&(red>1.5*green)&(red>1.5*blue)
    elif subject=='blue_square':
        mask=(blue>80)&(blue>1.5*green)&(blue>1.5*red)
    else:
        raise ValueError('Probe subject must be red_ball or blue_square')
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count<=1: return None
    component=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    if stats[component,cv2.CC_STAT_AREA]<minimum_area: return None
    return centroids[component].tolist()


def event_direction(action):
    for name,axis,sign in [('right',0,1),('left',0,-1),('upward',1,-1),('downward',1,1)]:
        if name in action: return axis,sign
    raise ValueError('Unsupported controlled-probe direction')


def event_score(positions,fps,event,width,height):
    axis,sign=event_direction(event['action'])
    indices=[i for i,p in enumerate(positions) if event['start_s']<=i/fps<event['end_s']]
    valid=[i for i in indices if positions[i] is not None]
    coverage=len(valid)/len(indices) if indices else 0
    if len(valid)<2: return {'coverage':coverage,'signed_normalized_displacement':None,'direction_pass':False}
    delta=sign*(positions[valid[-1]][axis]-positions[valid[0]][axis])/(width if axis==0 else height)
    return {'coverage':coverage,'signed_normalized_displacement':float(delta),'direction_pass':bool(coverage>=.8 and delta>=.02)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--outputs',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--swapped',action='store_true')
    args=parser.parse_args()
    rows=read_manifest(args.manifest)
    records=[]
    for row in rows:
        path=Path(args.outputs)/(row['sample_id']+'.mp4')
        media=probe_video(path)
        if not media['ok'] or media['decoded_frames']!=33 or media['width']!=320 or media['height']!=192 or abs(media['fps']-8)>1e-6:
            raise RuntimeError('Probe output media mismatch')
        capture=cv2.VideoCapture(str(path))
        positions=[]
        try:
            while True:
                ok,frame=capture.read()
                if not ok: break
                positions.append(color_centroid(frame,row['identity_id']))
        finally:
            capture.release()
        events=[dict(e) for e in row['timeline']]
        if args.swapped:
            events[0]['action'],events[1]['action']=events[1]['action'],events[0]['action']
        scores=[event_score(positions,8,e,320,192) for e in events]
        records.append({'sample_id':row['sample_id'],'seed':row['seed'],'color_detection_coverage':sum(p is not None for p in positions)/len(positions),
                        'events':scores,'both_events_pass':all(e['direction_pass'] for e in scores),'output_sha256':sha256_file(path)})
    report={'status':'evaluated','metric_kind':'controlled_color_centroid_proxy_only','swapped_expected_windows':args.swapped,
            'samples':len(records),'event_pass_count':sum(e['direction_pass'] for r in records for e in r['events']),
            'total_events':2*len(records),'both_events_pass_count':sum(r['both_events_pass'] for r in records),
            'mean_color_detection_coverage':float(np.mean([r['color_detection_coverage'] for r in records])),
            'protocol':{'minimum_component_pixels':30,'min_event_coverage':.8,'min_normalized_direction_displacement':.02,
                        'fps':8,'frames':33,'windows':'half-open seconds at frame_index/fps'},'records':records,
            'notice':'No face identity score or general human/animal action accuracy; no generated pose/control video was supplied.'}
    write_json(args.output,report)
    print({k:v for k,v in report.items() if k!='records'})


if __name__=='__main__':
    main()
