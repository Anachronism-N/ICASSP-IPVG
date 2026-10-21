"""Source-only CurricularFace calibration; never tune thresholds on generated videos."""
import argparse
import hashlib
from pathlib import Path
import time
import cv2
import numpy as np
from ipvg.common import load_json, sha256_file, write_json


def actor_split(actors):
    ordered = sorted(actors, key=lambda a: hashlib.sha256(('cur-cal-v1:' + a).encode()).hexdigest())
    if len(ordered) != 32: raise ValueError('Require frozen32source actors')
    return ordered[:16], ordered[16:]


def pairs(group, references, observations):
    positive, negative, by_positive, by_negative = [], [], {}, {}
    for actor in group:
        ps, ns = [], []
        anchor = references[actor]
        if anchor is not None:
            for item in observations:
                if item['actor'] != actor or len(item['vectors']) < 14: continue
                ps.append(float(np.mean([anchor @ v for v in item['vectors']])))
                for other in group:
                    if other != actor and references[other] is not None:
                        ns.append(float(np.mean([references[other] @ v for v in item['vectors']])))
        positive.extend(ps); negative.extend(ns); by_positive[actor] = ps; by_negative[actor] = ns
    return positive, negative, by_positive, by_negative


def main():
    from curricularface_eval import OfflineCurEncoder, video_features, PROTOCOL
    from identity_eval import roc_stats, bootstrap_auc
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'upstream', 'weights', 'face-model-dir', 'model-lock', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('Choose fresh source-calibration output')
    rows = load_json(args.manifest)['samples']
    if len(rows) != 64 or any(r['split'] != 'dev' or r['group'] != 'source_cur_calibration' for r in rows):
        raise ValueError('Only frozen64original source videos allowed')
    actors = {r['identity_id'] for r in rows}; cal_group, val_group = actor_split(actors)
    import torch
    import numpy.testing
    from mm2026_quality_eval import offline_only
    torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
    references, observations, self_errors, repeat_errors = {}, [], [], []
    # Constructor validates the native git revision and explicit local model paths.
    encoder = OfflineCurEncoder(args.upstream, args.weights, args.face_model_dir, args.model_lock, device='cuda')
    with offline_only():
        for row in rows:
            actor = row['identity_id']
            if sha256_file(row['reference_image']) != row['reference_sha256'] or sha256_file(row['video_path']) != row['video_sha256']:
                raise ValueError('Frozen source/reference hash changed')
            if actor not in references:
                image = cv2.imread(row['reference_image'])
                face, reason = encoder.process_image(image)
                reference = None if face is None else face['cur_embedding']
                references[actor] = reference
                if face is not None:
                    self_errors.append(abs(float(reference @ reference) - 1))
                    repeated, _ = encoder.process_image(image)
                    if repeated is None: raise ValueError('Repeated reference unexpectedly missing')
                    repeat_errors.append(float(np.max(np.abs(repeated['cur_embedding'] - reference))))
            faces, reasons, indices = video_features(row['video_path'], encoder)
            vectors = [r['cur_embedding'] for r in faces if r is not None]
            observations.append({'actor': actor, 'vectors': vectors})
            print('Cur source observations', len(observations), 'of64', flush=True)
    stats = []
    for group in (cal_group, val_group):
        pos, neg, pa, na = pairs(group, references, observations)
        if not pos or not neg: raise ValueError('No reliable source calibration pairs')
        summary = roc_stats(pos, neg)
        summary.update(positive_pairs=len(pos), negative_pairs=len(neg), actors=len(group),
                       valid_actors=sum(bool(pa[a]) for a in group),
                       auc_actor_bootstrap_ci95=bootstrap_auc(group, pa, na))
        stats.append((summary, pos, neg))
    threshold = float(np.quantile(stats[0][2], .99, method='higher')) + 1e-7
    calibration, validation = stats[0][0], stats[1][0]
    calibration.update(threshold=threshold, empirical_far=float(np.mean(np.array(stats[0][2]) >= threshold)))
    validation.update(frozen_threshold=threshold, empirical_far=float(np.mean(np.array(stats[1][2]) >= threshold)),
                      tpr=float(np.mean(np.array(stats[1][1]) >= threshold)))
    checks = {'source_pairs_complete': all(s[0]['positive_pairs'] == 32 and s[0]['negative_pairs'] == 480 for s in stats),
              'calibration_discrimination': calibration['auc'] >= .90 and calibration['eer'] <= .10,
              'validation_discrimination': validation['auc'] >= .90 and validation['eer'] <= .10,
              'validation_far_tpr': validation['empirical_far'] <= .01 and validation['tpr'] >= .8,
              'self_similarity': max(self_errors, default=1) <= 1e-5,
              'repeat_embedding': max(repeat_errors, default=1) <= 1e-5}
    torch.cuda.synchronize(); elapsed = time.monotonic() - start
    write_json(args.output, {'status': 'passed_source_identity_controls' if all(checks.values()) else 'failed_source_identity_controls',
        'protocol': PROTOCOL, 'manifest_sha256': sha256_file(args.manifest), 'model_source': encoder.model_source,
        'calibration': calibration, 'validation': validation, 'checks': checks,
        'self_similarity_max_error': max(self_errors, default=None), 'repeat_embedding_max_abs_error': max(repeat_errors, default=None),
        'resource': {'gpus': 1, 'elapsed_seconds': elapsed, 'allocated_gpu_hours': elapsed/3600,
                     'peak_allocated_vram_gb': torch.cuda.max_memory_allocated()/1e9},
        'notice': 'Source-only32actors64videos; clipped author scores separately reported. Empirical small-set FAR not official authentication; no generated threshold tuning.'})
    print('Cur source calibration:', 'passed' if all(checks.values()) else 'failed')


if __name__ == '__main__': main()
