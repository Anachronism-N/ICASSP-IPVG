#!/usr/bin/env python3
"""Resume original WMB GPU judging after an isolated real one-case smoke.

CUDA_VISIBLE_DEVICES is supplied by the caller. Invalid existing/partial results
are preserved for review; this entry point never changes generation or prompts.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from score_merge_cpu import load_jobs, read_json, require, finite, publish, suite_lock

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("m_full1_a05", "m_full1_a03", "m_full1_a07")
KEYS = {"instruction": 1, "physical_laws": 5, "common_sense": 2}
BENCH = Path("/dockerdata/wm_eval/bench/WorldModelBench")


def validate_result(path, expected):
    data = read_json(path)
    require(isinstance(data, dict) and isinstance(data.get("preds"), dict) and
            isinstance(data.get("accs"), dict), f"WMB invalid result structure: {path}")
    require(set(data["preds"]) == expected, f"WMB wrong/missing case stems: {path}")
    for key, per_case in KEYS.items():
        predictions = []
        for row in data["preds"].values():
            require(isinstance(row, dict) and isinstance(row.get(key), list) and
                    len(row[key]) == per_case and
                    all(isinstance(pred, str) and pred.strip() for pred in row[key]),
                    f"WMB wrong per-case {key} predictions: {path}")
            for pred in row[key]:
                if key == "instruction":
                    try:
                        value = float(pred.split(":")[-1].strip(" ."))
                    except ValueError:
                        # The unchanged official evaluator assigns zero to unparseable text.
                        value = 0.0
                    require(finite(value) and 0 <= value <= 3,
                            f"WMB instruction prediction outside [0, 3]: {path}")
                else:
                    value = "no" in pred.lower()
                predictions.append(value)
        values = data["accs"].get(key)
        valid = (lambda v: finite(v) and 0 <= v <= 3) if key == "instruction" else (
            lambda v: isinstance(v, bool) or finite(v) and v in (0, 1))
        require(isinstance(values, list) and len(values) == len(expected) * per_case and
                all(valid(v) for v in values), f"WMB incomplete/non-finite/out-of-range {key}: {path}")
        require(values == predictions, f"WMB {key} aggregates differ from official prediction scoring: {path}")
    return data


def original_stems(bench):
    items = read_json(bench / "worldmodelbench.json")
    require(isinstance(items, list) and len(items) == 350, "WMB requires the original 350-case benchmark")
    names = {Path(item["first_frame"]).stem for item in items}
    require(len(names) == 350, "WMB original benchmark has duplicate basenames")
    return names


def smoke(root, bench, python, judge, work_parent, run=subprocess.run):
    work = Path(tempfile.mkdtemp(prefix="wmb_smoke_", dir=work_parent))
    videos = work / "videos"
    videos.mkdir()
    item = read_json(bench / "worldmodelbench.json")[0]
    name = Path(item["first_frame"]).stem
    source = root / "eval/wmb/videos/m_full1_a05" / (name + ".mp4")
    require(source.is_file() and source.stat().st_size > 0, f"Missing real smoke video: {source}")
    (videos / source.name).symlink_to(source)
    (work / "worldmodelbench.json").write_text(json.dumps([item]) + "\n")
    run([python, str(bench / "evaluation.py"), "--judge", judge, "--video_dir", str(videos),
         "--model_name", "migration_smoke", "--save_name", str(work / "result")],
        cwd=work, check=True)
    result = validate_result(work / "result.json", {name})
    instruction = float(result["preds"][name]["instruction"][0].split(":")[-1].strip(" ."))
    require(finite(instruction) and 0 <= instruction <= 3, "WMB smoke instruction outside [0, 3]")
    print("REAL_WMB_SINGLE_CASE_SMOKE_PASS", work / "result.json", flush=True)


def score_model(root, model, bench, python, judge, run=subprocess.run):
    _, groups, _ = load_jobs(root, model)
    jobs = groups["wmb"]
    expected = {Path(j["out"]).stem for j in jobs}
    require(len(expected) == 350 and expected == original_stems(bench),
            f"{model}: expected the original 350 WMB basenames")
    videos = root / "eval/wmb/videos" / model
    require({p.stem for p in videos.glob("*.mp4")} == expected and
            all(Path(j["out"]).is_file() and Path(j["out"]).stat().st_size > 0 for j in jobs),
            f"{model}: incomplete/wrong original WMB videos")
    destination = root / "eval/wmb/results" / (model + ".json")
    with suite_lock(root, model, "wmb"):
        if destination.exists():
            validate_result(destination, expected)
            print("SKIP_VALID_WMB350", model, flush=True)
            return
        # Unique benchmark-local save prefix preserves failed work and the original cwd/CLI.
        work = Path(tempfile.mkdtemp(prefix="migration_" + model + "_", dir=bench))
        save = work / "result"
        log = root / "dataset/download_logs" / ("score_wmb_" + model + "_20261018.log")
        with log.open("a") as handle:
            run([python, str(bench / "evaluation.py"), "--judge", judge, "--video_dir", str(videos),
                 "--model_name", model, "--save_name", str(save)], cwd=bench,
                stdout=handle, stderr=subprocess.STDOUT, check=True)
        source = save.with_suffix(".json")
        validate_result(source, expected)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent,
                                         prefix="." + model + ".wmb-", delete=False) as handle:
            json.dump(read_json(source), handle, indent=2, allow_nan=False)
            handle.write("\n")
            temporary = Path(handle.name)
        validate_result(temporary, expected)
        publish(temporary, destination)
        print("WMB_COUNTS_FINITE_PASS", model, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--bench", type=Path, default=BENCH)
    parser.add_argument("--python", default="/dockerdata/conda_vila/bin/python")
    parser.add_argument("--judge", default="/dockerdata/wm_eval/bench/judges/vila-ewm-qwen2-1.5b")
    parser.add_argument("--work-dir", type=Path, default=Path("/dockerdata/wm_tmp"))
    args = parser.parse_args()
    os.umask(0o077)
    root, bench = args.root.resolve(), args.bench.resolve()
    smoke(root, bench, args.python, args.judge, args.work_dir)
    for model in args.models:
        score_model(root, model, bench, args.python, args.judge)


if __name__ == "__main__":
    main()
