#!/usr/bin/env python3
"""
Score generated videos with VBench-I2V; legacy core or isolated official five-sample nine-dimension mode.

  suite=vbench : eval/vbench_i2v/videos/<model>/<prompt>-0.mp4, official full_info (16:9 crops)
  suite=wm     : eval/wm_bench/videos/<model>/<id>.mp4 ; a custom full_info is written with the GT
                 first frame (1280x704 png) as the conditioning image, so the same i2v_subject /
                 i2v_background / quality dims apply to the in-domain benchmark.
                 model "gt" scores the real clips themselves (reference ceiling for real footage).

Run with /dockerdata/venv_vbench/bin/python from /dockerdata/wm_eval/VBench (relative crop paths).
  python vbench_eval.py --suite vbench --model round2
"""
import argparse
import fcntl
import hashlib
import json
import math
import importlib
from pathlib import Path
import os
import shutil
import sys

import torch

ROOT = "/srv/wm_teacher"
EV = f"{ROOT}/eval"
VBREPO = "/dockerdata/wm_eval/VBench"
WORK = "/dockerdata/wm_eval/work"
DIMS = ["i2v_subject", "i2v_background", "subject_consistency", "background_consistency",
        "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality"]

COTRACKER2_WEIGHT_SHA256 = '362f5274376d610dc987b6daf2c2fefe63e06e1835f4ec1a10d0a15c5a4eef4f'

CAMERA_TYPES = {
    'camera pans left': 'pan_left', 'camera pans right': 'pan_right',
    'camera tilts up': 'tilt_up', 'camera tilts down': 'tilt_down',
    'camera zooms in': 'zoom_in', 'camera zooms out': 'zoom_out', 'camera static': 'static',
}


def official_expectations(info):
    expected = {dim: set() for dim in DIMS + ['camera_motion']}
    filenames = set()
    for case in info:
        applicable = set(case['dimension']) & set(expected)
        if not applicable:
            continue
        for sample in range(5):
            name = case['prompt_en'].replace('/', '_') + f'-{sample}.mp4'
            if name in filenames or os.path.basename(name) != name:
                raise ValueError('Duplicate or invalid official video filename')
            filenames.add(name)
            for dim in applicable:
                expected[dim].add(name)
    if not filenames or any(not names for names in expected.values()):
        raise ValueError('Official metadata must cover all nine dimensions')
    return filenames, expected


def valid_official_result(result, expected, dimension):
    camera = dimension == 'camera_motion'
    try:
        if len(result) != (3 if camera else 2) or not expected:
            return False
        def finite(value):
            return type(value) in (int, float) and math.isfinite(value)
        if not finite(result[0]):
            return False
        rows = result[2] if camera else result[1]
        names = [os.path.basename(row['video_path']) for row in rows]
        if len(names) != len(expected) or set(names) != expected:
            return False
        if not all(finite(row['video_results']) or (dimension == 'dynamic_degree' and type(row['video_results']) is bool) for row in rows):
            return False
        if camera:
            per_type = result[1]
            if set(per_type) != set(CAMERA_TYPES.values()) or not all(finite(v) for v in per_type.values()):
                return False
            groups = {key: [] for key in CAMERA_TYPES.values()}
            for row, name in zip(rows, names):
                targets = [target for text, target in CAMERA_TYPES.items() if text in name]
                if len(targets) != 1 or row['prompt_type'] != targets[0] or row['video_results'] not in (0, 1):
                    return False
                predictions = row['predict_type']
                if type(predictions) is not list or any(type(item) is not str for item in predictions):
                    return False
                if row['video_results'] != int(targets[0] in predictions):
                    return False
                groups[targets[0]].append(row['video_results'])
            if any(not values or abs(per_type[key] - sum(values) / len(values)) > 1e-12 for key, values in groups.items()):
                return False
            if abs(result[0] - sum(row['video_results'] for row in rows) / len(rows)) > 1e-12:
                return False
        return True
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return False


def install_frozen_camera_backend(proof_path):
    proof = json.loads(Path(proof_path).read_bytes())
    if proof.get('schema') != 'vbench_camera_offline_backend_v1':
        raise ValueError('Pinned offline camera backend proof required')
    if proof['weight']['sha256'] != COTRACKER2_WEIGHT_SHA256:
        raise ValueError('Exact official CoTracker2 weight SHA required')
    files = [proof['weight'], proof['camera_source'], *proof['cotracker_sources']]
    if len({row['path'] for row in files}) != len(files):
        raise ValueError('Duplicate offline camera source records')
    for row in files:
        path = Path(row['path'])
        if not path.is_file() or path.stat().st_size != row['bytes'] or file_sha256(path) != row['sha256']:
            raise ValueError('Offline camera source or weight bytes changed')
    root = Path(proof['cotracker_root']).resolve()
    if {str(path.resolve()) for path in root.rglob('*.py')} != {str(Path(row['path']).resolve()) for row in proof['cotracker_sources']}:
        raise ValueError('Complete pinned CoTracker source tree required')
    expected_camera = Path(VBREPO, 'vbench2_beta_i2v/camera_motion.py').resolve()
    if Path(proof['camera_source']['path']).resolve() != expected_camera:
        raise ValueError('Offline backend does not bind the selected official scorer')
    loaded = sys.modules.get('cotracker')
    if loaded is not None and Path(loaded.__file__).resolve() != root / '__init__.py':
        raise ValueError('Previously imported CoTracker source differs')
    sys.path.insert(0, str(root.parent))
    predictor = importlib.import_module('cotracker.predictor')
    if Path(predictor.__file__).resolve() != root / 'predictor.py':
        raise ValueError('Imported CoTracker predictor differs from the frozen source')
    camera = importlib.import_module('vbench2_beta_i2v.camera_motion')
    if Path(camera.__file__).resolve() != expected_camera:
        raise ValueError('Imported camera scorer differs from frozen source')
    def offline_init(self, device, submodules_list):
        if submodules_list != {'repo': 'facebookresearch/co-tracker', 'model': 'cotracker2'}:
            raise ValueError('Official CoTracker2 camera configuration required')
        self.device = device
        self.grid_size = 10
        self.number_points = 1
        self.model = predictor.CoTrackerPredictor(checkpoint=proof['weight']['path']).to(device)
    camera.CameraPredict.__init__ = offline_init
    return file_sha256(proof_path)


def prepare_wm(model):
    """Custom full_info + local copies (VBench decodes every frame; avoid the fuse mount)."""
    cases = [json.loads(l) for l in open(f"{EV}/wm_bench/cases.jsonl")]
    vdir = f"{WORK}/wm/{model}/videos"
    idir = f"{WORK}/wm/images"
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(idir, exist_ok=True)
    import imageio.v2 as imageio
    from PIL import Image
    info = []
    for c in cases:
        img = f"{idir}/{c['id']}.png"
        if not os.path.exists(img):
            r = imageio.get_reader(c["gt"])
            Image.fromarray(r.get_data(0)).save(img)
            r.close()
        src = c["gt"] if model == "gt" else f"{EV}/wm_bench/videos/{model}/{c['id']}.mp4"
        dst = f"{vdir}/{c['id']}-0.mp4"
        if not os.path.exists(src):
            continue
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        info.append({"prompt_en": c["id"], "dimension": DIMS, "custom_image_path": img})
    fi = f"{WORK}/wm/{model}/full_info.json"
    json.dump(info, open(fi, "w"))
    return vdir, fi, len(info)


def file_sha256(path):
    with open(path, 'rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def bind_official_score_inputs(out, model, vdir, full_info, names, camera_backend=None):
    binding = {
        'protocol': 'vbench_i2v_official_five_v1',
        'metadata_sha256': file_sha256(full_info),
        'scoring_source_sha256': file_sha256(__file__),
        'video_sha256': {name: file_sha256(os.path.join(vdir, name)) for name in sorted(names)},
    }
    if camera_backend is not None:
        binding['camera_backend_sha256'] = camera_backend
    path = os.path.join(out, model + '__score_inputs.json')
    if os.path.exists(path):
        if json.load(open(path)) != binding:
            raise RuntimeError('Official score inputs changed; prior scores retained')
    else:
        if any(name.startswith(model + '__') and name.endswith('_eval_results.json') for name in os.listdir(out)):
            raise RuntimeError('Existing official scores lack input byte binding; retained without reuse')
        with open(path, 'x') as target:
            json.dump(binding, target, sort_keys=True)
    return binding


def prepare_vbench(model, official_five=False):
    work_suite = "vbench_official5" if official_five else "vbench"
    vdir = f"{WORK}/{work_suite}/{model}/videos"
    os.makedirs(vdir, exist_ok=True)
    source_dir = "videos_official5" if official_five else "videos"
    src = f"{EV}/vbench_i2v/{source_dir}/{model}"
    for f in os.listdir(src):
        if not f.endswith(".mp4"):
            continue
        source, target = f"{src}/{f}", f"{vdir}/{f}"
        if not os.path.exists(target):
            shutil.copyfile(source, target)
        if official_five and file_sha256(source) != file_sha256(target):
            raise RuntimeError("Staged official video bytes differ; original files retained")
    return vdir, f"{VBREPO}/vbench2_beta_i2v/vbench2_i2v_full_info.json", len(os.listdir(vdir))


def valid_result(result, expected):
    try:
        names = [os.path.basename(v["video_path"]) for v in result[1]]
        return (len(names) == len(expected) and set(names) == expected
                and math.isfinite(float(result[0]))
                and all(math.isfinite(float(v["video_results"])) for v in result[1]))
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def main():
    global VBREPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["vbench", "wm"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dims", default=None)
    ap.add_argument("--official-five", action="store_true", help="Separate nine-dimension, five-samples-per-pair protocol; no legacy result reuse")
    ap.add_argument("--vbench-repo", default=VBREPO, help="Explicit frozen local VBench source root")
    ap.add_argument("--camera-backend", help="Pinned local camera source and official CoTracker2 weight proof")
    a = ap.parse_args()
    VBREPO = a.vbench_repo
    if a.official_five and a.suite != "vbench":
        ap.error("--official-five applies only to the official vbench suite")
    if a.official_five and (os.path.basename(a.model) != a.model or a.model in (".", "..")):
        ap.error("Official model must be a directory name, not a path")
    selected_dims = DIMS + (["camera_motion"] if a.official_five else [])
    if a.dims is None:
        a.dims = ",".join(selected_dims)
    if a.dims != "none" and any(dim not in selected_dims for dim in a.dims.split(",")):
        ap.error("Requested dimensions do not match the selected protocol")
    os.chdir(VBREPO)
    sys.path.insert(0, VBREPO)
    from vbench2_beta_i2v import VBenchI2V

    suite_dir = f"{EV}/{'wm_bench' if a.suite == 'wm' else 'vbench_i2v'}"
    if a.official_five:
        info = json.load(open(f"{VBREPO}/vbench2_beta_i2v/vbench2_i2v_full_info.json"))
        all_names, expected = official_expectations(info)
        if len(info) != 1118 or len(all_names) != 5590 or len(expected["camera_motion"]) != 3815:
            raise RuntimeError("Official full benchmark metadata coverage differs")
        source = f"{suite_dir}/videos_official5/{a.model}"
        actual = {name for name in os.listdir(source) if name.endswith(".mp4")}
        if actual != all_names or any(os.path.getsize(os.path.join(source, name)) == 0 for name in all_names):
            raise RuntimeError("Official five-sample video coverage is missing, extra or empty")
        vdir, full_info, n = prepare_vbench(a.model, official_five=True)
        if {name for name in os.listdir(vdir) if name.endswith(".mp4")} != all_names:
            raise RuntimeError("Staged official video coverage differs")
    else:
        cases = [json.loads(line) for line in open(f"{suite_dir}/cases.jsonl")]
        missing = []
        for case in cases:
            if a.model == "gt" and a.suite == "wm":
                path = case["gt"]
            else:
                name = case["id"] + ("-0.mp4" if a.suite == "vbench" else ".mp4")
                path = f"{suite_dir}/videos/{a.model}/{name}"
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                missing.append(case["id"])
        if missing:
            raise RuntimeError(f"Incomplete {a.suite}/{a.model}: {len(missing)}/{len(cases)} missing or empty videos")
        expected = {dim: {c["id"] + "-0.mp4" for c in cases if dim in c.get("dims", DIMS)} for dim in DIMS}
        vdir, full_info, n = (prepare_wm if a.suite == "wm" else prepare_vbench)(a.model)
    score_dir = "scores_official5" if a.official_five else "scores"
    out = f"{suite_dir}/{score_dir}"
    os.makedirs(out, exist_ok=True)
    print(f"[{a.suite}/{a.model}] {n} videos, full_info={full_info}", flush=True)
    if a.dims == "none":
        return
    camera_binding = None
    if a.official_five and "camera_motion" in a.dims.split(","):
        if a.camera_backend is None:
            raise RuntimeError("Official camera scoring requires a pinned offline backend; floating torch.hub and disabled TLS are forbidden")
        camera_binding = install_frozen_camera_backend(a.camera_backend)
    if a.official_five:
        bind_official_score_inputs(out, a.model, vdir, full_info, all_names, camera_binding)
    vb = VBenchI2V(torch.device("cuda"), full_info, out)
    for dim in a.dims.split(","):
        result_path = f"{out}/{a.model}__{dim}_eval_results.json"
        with open(f"{os.path.dirname(vdir)}/{dim}.score.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if os.path.exists(result_path):
                result = json.load(open(result_path))[dim]
                if (valid_official_result(result, expected[dim], dim) if a.official_five else valid_result(result, expected[dim])):
                    continue
                if a.official_five:
                    raise RuntimeError(f"Existing invalid official {dim} result retained; no automatic retry")
                print(f"Re-scoring incomplete/invalid {dim}; expected {len(expected[dim])} cases", flush=True)
            vb.evaluate(videos_path=vdir, name=f"{a.model}__{dim}", dimension_list=[dim],
                        resolution="16-9", mode="vbench_standard")
            result = json.load(open(result_path))[dim]
            if not (valid_official_result(result, expected[dim], dim) if a.official_five else valid_result(result, expected[dim])):
                raise RuntimeError(f"Incomplete/invalid {dim} results; expected {len(expected[dim])} cases")
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
