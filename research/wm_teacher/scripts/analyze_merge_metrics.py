#!/usr/bin/env python3
"""Strict paired GTref108/original Physics66 analysis, never a full-evaluation gate.

Use the restored Physics-IQ Python environment; no videos are generated or scored.
Bootstrap units are fixed GTref cases and complete three-view Physics scenarios.
"""
import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path("/dockerdata/wm_eval/bench/physics-IQ-benchmark")
REFERENCES = ("base", "round4", "full1", "ft2")
LABELS = {"base": "base", "round4": "R4", "full1": "FT1", "ft2": "FT2"}
GT_KEYS = ("dreamsim", "dreamsim_last", "motion_ratio", "dur")
GT_METRICS = GT_KEYS[:3]
BOOTSTRAP = 5000
SEED = 20261018
TOLERANCE = 1e-8
CAVEAT = ("Fixed generated seed and paired case/scenario percentile bootstrap only; "
          "no source-cluster, generation-seed, training or judge variability. "
          "Multiple comparisons are unadjusted. This is neither causal training evidence "
          "nor WMB/VBench/validation-loss/full-evaluation/Teacher acceptance.")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result
    try:
        with path.open() as handle:
            return json.load(handle, object_pairs_hook=unique)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Missing/unreadable JSON {path}: {error}") from error


def original_ids(root, bench):
    case_path = root / "eval/wm_bench/cases.jsonl"
    with case_path.open() as handle:
        cases = [json.loads(line)["id"] for line in handle]
    require(len(cases) == len(set(cases)) == 108, "Expected 108 unique original GTref cases")
    description_path = bench / "descriptions/descriptions_original.csv"
    with description_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["generated_video_name"] for row in rows if "take-2" not in row["scenario"]]
    scenarios = [name.rsplit("_", 1)[-1] for name in names]
    require(len(names) == len(set(names)) == 198 and len(set(scenarios)) == 66,
            "Expected original Physics198 views/66 scenarios")
    views = {"perspective-left", "perspective-center", "perspective-right"}
    require(all({name.split("_")[1] for name in names if name.rsplit("_", 1)[-1] == scenario} == views
                and scenarios.count(scenario) == 3 for scenario in set(scenarios)),
            "Each original Physics scenario must have exactly left/center/right views")
    return sorted(cases), sorted(set(scenarios)), (case_path, description_path)


def load_gtref(path, cases):
    result = read_json(path)
    require(isinstance(result, dict) and len(result) == 108 and set(result) == set(cases),
            f"Incomplete/wrong GTref108 cases: {path}")
    require(all(isinstance(row, dict) and all(finite(row.get(key)) for key in GT_KEYS)
                for row in result.values()), f"Missing/non-finite GTref required values: {path}")
    return result


def load_iqtable(bench):
    sys.path.insert(0, str(bench))
    return importlib.import_module("physiq.calculate_iq_score_stable").IQTable


def validate_physics_csv(path, scenarios, iq_class):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)), f"Duplicate Physics CSV headers: {path}")
        rows = list(reader)
    require(len(rows) == 66 and {row.get("scenario") for row in rows} == set(scenarios),
            f"Incomplete/duplicate/wrong Physics66 scenarios: {path}")
    list_columns = iq_class.get_list_columns()
    scalar_columns = [f"{metric}_{view}" for metric in iq_class.get_scalar_keys() for view in iq_class.views]
    require(set(list_columns + scalar_columns) <= set(fields), f"Missing original Physics metric/view columns: {path}")
    lengths = {column: set() for column in list_columns}
    for row in rows:
        for column in list_columns:
            try:
                values = json.loads(row[column])
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"Malformed Physics list {column}: {path}") from error
            require(isinstance(values, list) and values and all(finite(value) for value in values),
                    f"Empty/non-finite Physics list {column}: {path}")
            lengths[column].add(len(values))
        for column in scalar_columns:
            try:
                value = float(row[column])
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"Missing/malformed Physics scalar {column}: {path}") from error
            require(math.isfinite(value), f"Non-finite Physics scalar {column}: {path}")
    require(all(len(values) == 1 for values in lengths.values()),
            f"Inconsistent Physics frame-list lengths: {path}")


def load_physics(csv_path, metrics_path, scenarios, iq_class):
    import numpy as np

    validate_physics_csv(csv_path, scenarios, iq_class)
    metrics = read_json(metrics_path)
    require(isinstance(metrics, dict) and finite(metrics.get("final_score_orig")),
            f"Missing/non-finite original Physics point: {metrics_path}")
    table = iq_class.from_csv(str(csv_path))
    official = float(table.compute_final_score_orig())
    require(finite(official) and abs(official - metrics["final_score_orig"]) <= TOLERANCE,
            f"IQTable point does not match saved final_score_orig within 1e-8: {csv_path}")
    frame = table.df.set_index("scenario").loc[scenarios].reset_index()
    columns = [table.spatiotemporal_iou_key, table.variance_spatiotemporal_iou_key,
               table.spatial_iou_key, table.variance_spatial_key,
               table.weighted_spatial_iou_key, table.variance_weighted_spatial_key,
               table.mse_key, table.variance_mse_key]
    values = frame[columns].to_numpy(dtype=float)
    require(values.shape == (66, 8) and np.isfinite(values).all(),
            f"Non-finite/malformed IQTable aggregates: {csv_path}")
    require((values[:, (1, 3, 5)].mean(axis=0) > 0).all(), f"Zero/negative Physics variance denominator: {csv_path}")
    point = float(original_score(values.mean(axis=0)))
    require(abs(point - official) <= TOLERANCE, f"Original formula differs from IQTable: {csv_path}")
    return values, frame, point


def original_score(means):
    import numpy as np

    denominator = means[..., (1, 3, 5)]
    require(np.isfinite(means).all() and (denominator > 0).all(),
            "Non-finite/zero-denominator original Physics resample")
    score = ((means[..., 0] / means[..., 1] + means[..., 2] / means[..., 3] +
              means[..., 4] / means[..., 5]) / 3 - (means[..., 6] - means[..., 7]))
    return np.clip(score, 0, 1)


def comparisons(models, references):
    return [(model, reference) for model in models for reference in references] + [
        (models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]


def interval(delta, samples):
    import numpy as np

    require(finite(float(delta)) and np.isfinite(samples).all(), "Non-finite paired bootstrap values")
    return {"delta": float(delta), "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}


def analyze(root, models, bench=BENCH, bootstrap=BOOTSTRAP, seed=SEED, iq_class=None, references=REFERENCES):
    import numpy as np

    require(references and len(references) == len(set(references)) and
            all(re.fullmatch(r"[A-Za-z0-9_-]+", reference) for reference in references),
            "Provide unique nonempty reference names")
    require(models and len(models) == len(set(models)) and
            all(re.fullmatch(r"[A-Za-z0-9_-]+", model) and model not in references for model in models),
            "Provide unique merge model names, distinct from references")
    require(isinstance(bootstrap, int) and bootstrap > 0 and isinstance(seed, int) and seed >= 0,
            "Bootstrap must be positive and seed nonnegative")
    cases, scenarios, manifests = original_ids(root, bench)
    iq_class = iq_class or load_iqtable(bench)
    all_models = list(models) + list(references)
    sources = list(manifests)
    gtref, physics = {}, {}
    for model in all_models:
        gt_path = root / "eval/wm_bench" / f"gtref_{model}.json"
        gtref[model] = load_gtref(gt_path, cases)
        base = root / "eval/physiq/results/physics-IQ-benchmark/results" / (model + "-op-run_01")
        csv_path, metrics_path = base.with_suffix(".csv"), base.with_name(base.name + "_metrics.json")
        physics[model] = load_physics(csv_path, metrics_path, scenarios, iq_class)
        sources.extend((gt_path, csv_path, metrics_path))
    pairs = comparisons(models, references)
    gt_indices = np.random.default_rng(seed).integers(0, 108, size=(bootstrap, 108))
    gt_observed, gt_paired = {}, {}
    for metric in GT_METRICS:
        values = {model: np.array([gtref[model][case][metric] for case in cases]) for model in all_models}
        gt_observed[metric] = {model: float(values[model].mean()) for model in all_models}
        gt_paired[metric] = {f"{left} minus {right}": interval(
            (values[left] - values[right]).mean(), (values[left] - values[right])[gt_indices].mean(axis=1))
            for left, right in pairs}
    phys_indices = np.random.default_rng(seed).integers(0, 66, size=(bootstrap, 66))
    phys_samples, phys_observed = {}, {}
    for model in all_models:
        values, frame, point = physics[model]
        samples = original_score(values[phys_indices].mean(axis=1))
        for i in range(min(3, bootstrap)):
            official_sample = float(iq_class(frame.iloc[phys_indices[i]].reset_index(drop=True)).compute_final_score_orig())
            require(finite(official_sample) and abs(float(samples[i]) - official_sample) <= TOLERANCE,
                    f"Original bootstrap formula differs from IQTable within 1e-8: {model}, resample {i}")
        phys_samples[model] = 100 * samples
        phys_observed[model] = 100 * point
    phys_paired = {f"{left} minus {right}": interval(phys_observed[left] - phys_observed[right],
                   phys_samples[left] - phys_samples[right]) for left, right in pairs}
    return {"created_at": time.strftime("%F %T %Z"), "models": list(models),
            "references": {reference: LABELS.get(reference, reference) for reference in references},
            "bootstrap": bootstrap, "seed": seed,
            "confidence_interval": "paired percentile 95%; identical row indices for every model",
            "gtref": {"n_cases": 108, "case_ids": cases, "lower_is_better": list(GT_METRICS),
                      "observed": gt_observed, "paired": gt_paired},
            "physics": {"n_scenarios": 66, "views_per_scenario": 3, "scenario_ids": scenarios,
                        "protocol": "Original raw 120f@24 pad/smear/crf12; complete three-view groups",
                        "units": "100 * unrounded final_score_orig; delta in percentage points; higher is better",
                        "formula_validation": {"iqtable_point_matches_saved_metric_abs_tolerance": TOLERANCE,
                                               "iqtable_resamples_checked_per_model": min(3, bootstrap)},
                        "observed": phys_observed, "paired": phys_paired},
            "source_sha256": {str(path.relative_to(root)) if path.is_relative_to(root) else
                              "Physics-IQ/descriptions/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources}, "caveat": CAVEAT}


def write_atomic(path, report):
    require(not path.exists() and not path.is_symlink(), f"Refusing to overwrite existing analysis: {path}")
    require(path.parent.is_dir(), f"Analysis output directory must already exist: {path.parent}")
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    # A hard link atomically publishes the complete file and cannot replace a racing writer.
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise RuntimeError(f"Output appeared concurrently; existing file preserved, review {temporary}") from error
    temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--models", default="m_full1_a03,m_full1_a05,m_full1_a07", help="Comma-separated merge models")
    parser.add_argument("--references", default=",".join(REFERENCES), help="Comma-separated reference model IDs")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bench", type=Path, default=BENCH)
    parser.add_argument("--out", type=Path, required=True, help="New independent JSON; existing files are never overwritten")
    args = parser.parse_args()
    require(not args.out.exists() and not args.out.is_symlink(), f"Refusing to overwrite existing analysis: {args.out}")
    report = analyze(args.root.resolve(), args.models.split(","), bench=args.bench.resolve(),
                     references=args.references.split(","), bootstrap=args.bootstrap, seed=args.seed)
    write_atomic(args.out, report)
    print(f"PAIRED_METRICS_ANALYSIS_COMPLETE {args.out}", flush=True)


if __name__ == "__main__":
    main()
