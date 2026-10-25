#!/usr/bin/env python3
"""Strict, aggregate-only four-seed E1 replication summary; no new config selection.

CPU usage: python scripts/summarize_e1_seeds.py [--root PROJECT_ROOT]
No output is written unless all eight seed/config groups cover the same 94 cases.
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from score_e1 import (  # noqa: E402
    BOOT_B, BOOT_SEED, DIMS, GT_KEYS, RULE, guard_checks, load_cases,
    load_config_rows, mean_stats, paired,
)

ROOT = SCRIPTS.parent
SEED_DIRS = {
    "seed0": "eval/e1_devset_20261021",
    "seed1": "eval/e1_seed1_20261023",
    "seed2": "eval/e1_seed2_20261023",
    "seed3": "eval/e1_seed3_20261024",
}
CONFIGS = ("s5_c5", "s12_c5")
BASE_CONFIG, CANDIDATE_CONFIG = CONFIGS
N_CASES = 94
METRICS = DIMS + list(GT_KEYS) + ["total"]
OUTPUT = "eval/diag/teacher_e1_four_seed_public_summary_20261025.json"


def cross_seed_stats(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "range": [min(values), max(values)],
        "n_seeds": len(values),
    }


def summarize(roots, bootstrap_samples=BOOT_B):
    if set(roots) != set(SEED_DIRS):
        raise ValueError("Exactly seed0, seed1, seed2 and seed3 roots are required")
    cases_by_seed, reference = {}, None
    for seed in SEED_DIRS:
        cases = load_cases(roots[seed])
        json.dumps(cases, allow_nan=False)  # Reject non-finite case metadata as well as scores.
        if len(cases) != N_CASES:
            raise ValueError(f"{seed}: expected exactly {N_CASES} cases")
        if any(not isinstance(c.get("id"), str) or not c["id"]
               or not isinstance(c.get("dataset"), str) or not c["dataset"] for c in cases):
            raise ValueError(f"{seed}: missing/invalid case id or dataset")
        by_id = {c["id"]: c for c in cases}
        if len(by_id) != N_CASES:
            raise ValueError(f"{seed}: duplicate case ids")
        if reference is None:
            reference = by_id
        elif by_id != reference:
            raise ValueError(f"{seed}: case metadata differ from seed0")
        cases_by_seed[seed] = cases

    # Load every group before computing a complete summary; there is no partial mode.
    rows = {}
    for seed in SEED_DIRS:
        rows[seed] = {}
        for config in CONFIGS:
            try:
                rows[seed][config] = load_config_rows(config, cases_by_seed[seed], roots[seed])
            except RuntimeError as error:
                raise RuntimeError(f"{seed}: {error}") from error
    seeds = {}
    for seed in SEED_DIRS:
        means = {config: mean_stats(rows[seed][config]) for config in CONFIGS}
        checks = guard_checks(means[CANDIDATE_CONFIG], means[BASE_CONFIG])
        seeds[seed] = {
            "means": means,
            "paired_delta": paired(rows[seed][CANDIDATE_CONFIG], rows[seed][BASE_CONFIG],
                                   METRICS, b=bootstrap_samples, seed=BOOT_SEED),
            "guard_checks": checks,
            "all_guards_pass": all(checks.values()),
        }
    summary = {
        "protocol": "E1 fixed s12_c5 vs s5_c5 replication on four seeds; docs/43 section 3 guards",
        "complete": True,
        "n_groups": len(SEED_DIRS) * len(CONFIGS),
        "n_cases_per_group": N_CASES,
        "n_seeds": len(SEED_DIRS),
        "configs": list(CONFIGS),
        "baseline": BASE_CONFIG,
        "candidate": CANDIDATE_CONFIG,
        "delta_definition": "s12_c5 minus s5_c5, paired within each seed on the same cases",
        "metric_units": {k: "native GT-reference units" if k in GT_KEYS else "fraction, not percentage points"
                         for k in METRICS},
        "guard_rule": RULE,
        "seeds": seeds,
        "cross_seed_delta": {
            metric: cross_seed_stats([seeds[seed]["paired_delta"][metric]["mean"] for seed in SEED_DIRS])
            for metric in METRICS
        },
        "methodology": {
            "paired_ci": "95% case-bootstrap interval within each seed; descriptive only",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": BOOT_SEED,
            "sample_std": "standard deviation of seed-level paired means, denominator n_seeds - 1",
            "inference": "Four seeds are not a strict population-significance claim; shared cases are not independent seed-case samples.",
            "interpretation": "A gain on one seed does not establish a stable gain across seeds.",
            "scope": "Fixed two-config replication only; no new selection, tuning or full-benchmark result.",
        },
    }
    json.dumps(summary, allow_nan=False)
    return summary


def main(argv=None, roots=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT, help="project root containing eval/")
    args = ap.parse_args(argv)
    roots = roots if roots is not None else {seed: args.root / rel for seed, rel in SEED_DIRS.items()}
    summary = summarize(roots)
    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    out = args.root / OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, out)
    print(f"Complete: 8 groups x {N_CASES} paired cases; descriptive summary -> {out}")
    return summary


if __name__ == "__main__":
    main()
