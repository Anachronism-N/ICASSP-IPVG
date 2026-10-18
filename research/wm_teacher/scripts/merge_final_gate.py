#!/usr/bin/env python3
"""Publish completion only after all merge decoding, fixed120 losses and both manifests pass.

This is an experimental coverage gate, not a Teacher quality/human-review gate.
No Student/FT3 training, downloads or uploads are authorized by completion.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from score_merge_cpu import load_jobs, validate_decode, require
from wmb_merge_score import BENCH as WMB_BENCH, original_stems, validate_result as validate_wmb
from val_merge_score import MODELS, LOSS_DIR, fixed_base, validate_loss

ROOT = Path(__file__).resolve().parents[1]


def summary_api(root):
    spec = importlib.util.spec_from_file_location("merge_summary_gate", root / "scripts/summarize_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.EV = str(root), str(root / "eval")
    return module.validate_jobs


def check_complete(root, validate_jobs=None):
    base = fixed_base(root)
    for model in MODELS:
        jobs, _, _ = load_jobs(root, model)
        validate_decode(root / "eval/diag" / ("decode_" + model + "_20261018.json"), model, jobs)
        validate_loss(root / LOSS_DIR / ("loss_" + model + ".json"), base)
    manifests = [str(root / "eval/jobs_ft2_all.jsonl"), str(root / "eval/jobs_merge_all.jsonl")]
    # validate_jobs accepts any subset; independently forbid empty/subset manifests.
    with Path(manifests[0]).open() as handle:
        ft2 = [json.loads(line) for line in handle]
    ft2_counts = {"ft2": 1011, "full1_po": 903, "ft2_po": 903, "base_po": 548, "round4_po": 548}
    require(len(ft2) == 3913 and len({job["out"] for job in ft2}) == 3913 and
            {model: sum(j["model"] == model for j in ft2) for model in ft2_counts} == ft2_counts,
            "FT2 requires the complete original 3913-job manifest, including independent PO groups")
    with Path(manifests[1]).open() as handle:
        merges = [json.loads(line) for line in handle]
    require(len(merges) == 3033 and {job["model"] for job in merges} == set(MODELS) and
            len({job["out"] for job in merges}) == 3033, "Merge requires the complete 3033-job manifest")
    ft2_suites = {
        "ft2": {"wm_bench": 108, "vbench_i2v": 355, "physiq": 198, "wmb": 350},
        "full1_po": {"vbench_i2v": 355, "physiq": 198, "wmb": 350},
        "ft2_po": {"vbench_i2v": 355, "physiq": 198, "wmb": 350},
        "base_po": {"physiq": 198, "wmb": 350},
        "round4_po": {"physiq": 198, "wmb": 350},
    }
    actual = {}
    wmb_groups = {}
    for job in ft2 + merges:
        parts = Path(job["out"]).relative_to(root / "eval").parts
        suite, model = parts[0], job["model"]
        if model in ft2_suites:
            pair = (model, suite)
            actual[pair] = actual.get(pair, 0) + 1
        if suite == "wmb":
            wmb_groups.setdefault(model, set()).add(Path(job["out"]).stem)
    require(actual == {(model, suite): count for model, suites in ft2_suites.items()
                       for suite, count in suites.items()},
            "FT2 requires the complete original model/suite groups")
    original = original_stems(WMB_BENCH)
    for model, expected in wmb_groups.items():
        require(expected == original, f"{model}: WMB manifest differs from original 350 basenames")
        validate_wmb(root / "eval/wmb/results" / (model + ".json"), expected)
    with contextlib.redirect_stdout(io.StringIO()):
        (validate_jobs or summary_api(root))(manifests)
    return manifests


REPORT_PROGRAM = """
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'scripts'))
import summarize_eval as summary
summary.ROOT, summary.EV = str(root), str(root / 'eval')
sys.argv = ['summarize_eval.py'] + [arg for path in sys.argv[2:] for arg in ('--validate-jobs', path)]
summary.main()
import build_demo as demo
import build_teacher_review as teacher
for module in (demo, teacher):
    module.ROOT = str(root)
    module.DEMO, module.MEDIA = str(root / 'demo'), str(root / 'demo/media')
teacher.EV, teacher.WM = str(root / 'eval'), str(root / 'eval/wm_bench')
teacher.LOSS = str(root / 'models/train/round2_5s_lora_r32/eval')
sys.argv = ['build_teacher_review.py']
teacher.main()
"""


def finish(root, python, run=subprocess.run, validate_jobs=None):
    manifests = check_complete(root, validate_jobs)
    run([python, "-c", REPORT_PROGRAM, str(root), *manifests], cwd=root, check=True)
    # Recheck after reporting: never mark a changed/incomplete result set complete.
    check_complete(root, validate_jobs)
    report = {"status": "COMPLETE", "at": time.strftime("%F %T %Z"), "models": list(MODELS),
              "video_expected": 3033, "fixed_val_n": 120, "benchmark_coverage": "PASS",
              "caveat": "Experimental outputs/coverage only. Teacher quality gate and user review are separate; no Student/FT3 training authorized."}
    destination = root / "eval/diag/merge_eval_complete_20261018.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent,
                                     prefix=".merge_complete-", delete=False) as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    print("MERGE_EVAL_COMPLETE_REPORTS_UPDATED_REQUIRE_TEACHER_REVIEW", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--python", default="/opt/conda/envs/torch-base/bin/python")
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--cap-hours", type=float, default=72)
    parser.add_argument("--once", action="store_true", help="Check now and fail nonzero if incomplete")
    args = parser.parse_args()
    require(args.interval > 0 and args.cap_hours > 0, "Interval/cap must be positive")
    root, started, last = args.root.resolve(), time.monotonic(), None
    while True:
        try:
            check_complete(root)
            break
        except (OSError, RuntimeError, KeyError, ValueError, TypeError) as error:
            status = type(error).__name__ + ": " + str(error)[:700]
            if args.once or time.monotonic() - started >= args.cap_hours * 3600:
                raise RuntimeError("EVAL_GATE_INCOMPLETE " + status) from error
            if status != last:
                print("EVAL_GATE_PENDING", status, flush=True)
                last = status
            time.sleep(min(args.interval, max(0, args.cap_hours * 3600 - (time.monotonic() - started))))
    print("ALL_MERGE_VIDEO_DECODE_LOSS_AND_BENCHMARK_COVERAGE_PASS", flush=True)
    finish(root, args.python)


if __name__ == "__main__":
    main()
