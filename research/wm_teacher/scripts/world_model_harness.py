"""CPU evaluation coverage and byte audit for inference-only World Model experiments.

Interactive state and backend dispatch live in interactive_world_model.py.
A verified declared set is not an official benchmark or a SOTA/quality approval.
"""
import hashlib
import json
import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def validate_evaluation_contract(contract):
    require(contract.get('schema') == 'world_model_evaluation_contract_v1', 'Explicit contract required')
    p = contract['protocol']
    require(set(p) == {'prompt_variant', 'input_mode', 'samples_per_pair', 'selection',
                       'checkpoint_sha256', 'scorer_sha256', 'inputs_sha256', 'media', 'sampler'},
            'Complete frozen protocol required')
    sampler = p['sampler']
    require(set(sampler) == {'steps', 'cfg', 'shift'} and type(sampler['steps']) is int
            and sampler['steps'] > 0 and all(type(sampler[k]) in (int, float)
            and math.isfinite(sampler[k]) and sampler[k] > 0 for k in ('cfg', 'shift')),
            'Explicit finite sampler required')
    n = p['samples_per_pair']
    media = p['media']
    require(set(media) == {'width', 'height', 'frames', 'fps'} and all(type(v) is int and v > 0 for v in media.values()), 'Exact media specification required')
    require(p['prompt_variant'] in ('raw', 'po') and p['input_mode'] in ('I2V', 'V2V'), 'Separate input protocols required')
    require(type(n) is int and n > 0 and p['selection'] in ('single', 'all_samples', 'BoN'), 'Sampling protocol required')
    require(p['selection'] != 'single' or n == 1, 'Single sample cannot hide BoN')
    for k in ('checkpoint_sha256', 'scorer_sha256', 'inputs_sha256'):
        v = p[k];require(type(v) is str and len(v) == 64 and all(c in '0123456789abcdef' for c in v), 'Protocol digest required')
    jobs, expected = contract['jobs'], contract['expected_scores']
    require(type(jobs) is list and jobs and type(expected) is dict and expected, 'Jobs and dimensions required')
    ids = [j['id'] for j in jobs];require(all(type(i) is str and i for i in ids) and len(ids) == len(set(ids)), 'Duplicate or invalid IDs')
    pairs = {}
    for j in jobs:
        require(type(j['sample']) is int and 0 <= j['sample'] < n and type(j['pair']) is str and j['pair'], 'Pair/sample required')
        require(type(j['seed']) is int and 0 <= j['seed'] < 2**63, 'Seed required')
        pairs.setdefault(j['pair'], []).append(j['sample'])
        from pathlib import Path
        require(type(j['output']) is str and Path(j['output']).name == j['output'] and j['output'].endswith('.mp4'), 'Output basename required')
    require(all(sorted(v) == list(range(n)) for v in pairs.values()), 'Complete samples per pair required')
    require(len({j['output'] for j in jobs}) == len(jobs), 'Duplicate output filenames')
    for dimension, members in expected.items():
        require(type(dimension) is str and dimension, 'Dimension name required')
        require(type(members) is list and members and len(members) == len(set(members)) and set(members) <= set(ids), 'Per-dimension coverage differs')
    require(set().union(*(set(v) for v in expected.values())) == set(ids), 'Unscored jobs')
    # Bind jobs/seeds and expected dimensions, not just shared sampler settings.
    return digest(dict(protocol=p, jobs=jobs, expected_scores=expected))

def comparable_protocols(left, right):
    validate_evaluation_contract(left);validate_evaluation_contract(right)
    a, b = dict(left['protocol']), dict(right['protocol'])
    a.pop('checkpoint_sha256');b.pop('checkpoint_sha256')
    return a == b and left['jobs'] == right['jobs'] and left['expected_scores'] == right['expected_scores']


def file_record(path):
    from pathlib import Path
    p = Path(path);require(p.is_absolute() and p.resolve() == p and p.is_file(), 'Canonical regular file required')
    before = p.stat();h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):h.update(block)
    after = p.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), 'Input changed during hash')
    return dict(path=str(p), bytes=after.st_size, sha256=h.hexdigest())

def audit_evaluation(contract, videos, scores):
    from pathlib import Path
    import math
    sha = validate_evaluation_contract(contract)
    jobs, expected = contract['jobs'], contract['expected_scores']
    require(type(videos) is list and len(videos) == len(jobs), 'Complete video evidence required')
    by_id = {r['id']:r for r in videos}
    require(len(by_id) == len(videos) and set(by_id) == {j['id'] for j in jobs}, 'Missing or duplicate video')
    for j in jobs:
        r = by_id[j['id']]
        require(r['protocol_sha256'] == sha and r['full_decode_pass'] is True, 'Unbound or undecoded video')
        require(Path(r['file']['path']).name == j['output'] and file_record(r['file']['path']) == r['file'], 'Video bytes changed')
        validate_video(r['file']['path'], contract['protocol']['media'])
        require(file_record(r['file']['path']) == r['file'], 'Video changed during decode')
    require(len({r['file']['path'] for r in videos}) == len(videos), 'Distinct video paths required')
    video_bytes_sha = digest({i:r['file']['sha256'] for i,r in sorted(by_id.items())})
    require(type(scores) is dict and set(scores) == set(expected), 'All dimensions required')
    for dim, members in expected.items():
        r = scores[dim];require(r['protocol_sha256'] == sha, 'Scorer protocol differs')
        require(file_record(r['file']['path']) == r['file'], 'Score file changed')
        payload = json.loads(Path(r['file']['path']).read_bytes())
        require(payload['protocol_sha256'] == sha and payload['dimension'] == dim and payload['video_bytes_sha256'] == video_bytes_sha, 'Score file protocol or video bytes differ')
        require(file_record(r['file']['path']) == r['file'], 'Score changed while reading')
        rows = payload['rows']
        require(len(rows) == len(members) and {r['id'] for r in rows} == set(members), 'Score coverage differs')
        require(all(type(r['value']) in (int, float) and math.isfinite(r['value']) for r in rows), 'Nonfinite score')
    return dict(protocol_sha256=sha, videos_verified=len(videos), dimensions_verified=len(scores), declared_coverage_verified=True, official_coverage_proven=False, metric_correctness_proven=False, model_quality_measured=False, sota_proven=False)


def validate_video(path, spec):
    import subprocess
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(path), '-map', '0:v:0', '-f', 'null', '-'], check=True, capture_output=True, timeout=120)
    result = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(path)], check=True, capture_output=True, timeout=120)
    streams = json.loads(result.stdout)['streams']
    require(len(streams) == 1 and streams[0]['codec_type'] == 'video', 'Exactly one video stream required')
    stream = streams[0]
    require(stream['width'] == spec['width'] and stream['height'] == spec['height'] and int(stream['nb_read_frames']) == spec['frames'], 'Media dimensions or frame count differ')
    for key in ('avg_frame_rate', 'r_frame_rate'):
        numerator, denominator = map(int, stream[key].split('/'))
        require(denominator > 0 and numerator == denominator * spec['fps'], 'Frame rate differs')

def main(argv=None):
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate', 'audit'])
    parser.add_argument('--contract', required=True);parser.add_argument('--videos');parser.add_argument('--scores')
    args = parser.parse_args(argv);contract = json.loads(Path(args.contract).read_bytes())
    if args.command == 'validate':
        result = dict(protocol_sha256=validate_evaluation_contract(contract), GPU_used=False, execution_authorized=False)
    else:
        require(args.videos and args.scores, 'Video and score receipts required')
        result = audit_evaluation(contract, json.loads(Path(args.videos).read_bytes()), json.loads(Path(args.scores).read_bytes()))
    print(json.dumps(result, sort_keys=True))

if __name__ == '__main__':
    main()
