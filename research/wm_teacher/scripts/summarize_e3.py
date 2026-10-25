#!/usr/bin/env python3
"""
E3 three-arm dev-set summary and preregistered guard judgment (docs/43 section 3 rule, applied per
docs/47 section 4). Thin wrapper over score_e1.py: no metric is re-implemented.

  - Loads the three E3 arms (armA_orig / armB_fixed / armC_fixed_anchor) from
    eval/e3_devset_eval_20261024 (VBench 8 dims + gtref DreamSim/motion error).
  - Base reference: eval/e1_devset_20261021 s5_c5 seed0, already fully scored (same 94 cases,
    same images/prompts/seed; cases.jsonl sha256 identical).
  - Guards (deltas = arm minus base on dev-set means): motion_smoothness >= base - 0.3pp,
    subject_consistency >= base - 1.0pp, DreamSim <= base + 0.005, motion error <= base + 0.05.
  - Selection: among arms passing all guards, the highest Total*; if none pass, the preregistered
    outcome is a negative result (no arm advances to the full benchmark).

Usage:
  /opt/conda/envs/torch-base/bin/python scripts/summarize_e3.py            # full, fails if incomplete
  /opt/conda/envs/torch-base/bin/python scripts/summarize_e3.py --partial  # report complete arms only
"""
import argparse
import json
import os
import sys

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
from score_e1 import (  # noqa: E402
    DIMS, E3_ARMS, E1, GT_KEYS, RULE, guard_checks, load_cases, load_config_rows, mean_stats, paired,
)

ROOT = os.path.dirname(SCRIPTS)
E3 = f"{ROOT}/eval/e3_devset_eval_20261024"
BASE_CONFIG = "s5_c5"
BASE_ROOT = E1
BOOT_KEYS = DIMS + list(GT_KEYS) + ["total"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="report complete arms only; no selection")
    a = ap.parse_args(argv)

    cases = load_cases(E3)
    base_cases = load_cases(BASE_ROOT)
    if cases != base_cases:
        raise ValueError("E3 case metadata differ from the base dev set")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate E3 case ids")

    rows, missing = {}, {}
    for arm in E3_ARMS:
        try:
            rows[arm] = load_config_rows(arm, cases, E3)
        except RuntimeError as e:
            missing[arm] = str(e)
    rows[BASE_CONFIG] = load_config_rows(BASE_CONFIG, cases, BASE_ROOT)
    if missing and not a.partial:
        raise SystemExit("Incomplete E3 scores:\n" + "\n".join(f"{k}: {v}" for k, v in missing.items()))

    stats = {k: mean_stats(v) for k, v in rows.items()}
    by_ds = {}
    for c in cases:
        by_ds.setdefault(c["dataset"], []).append(c["id"])

    summary = {
        "protocol": "E3 three-arm dev-set eval vs base s5_c5 seed0 (docs/43 s3 rule / docs/47 s4 guards)",
        "n_cases": len(cases), "arms": list(E3_ARMS), "base": BASE_CONFIG, "missing": missing,
        "means": stats,
        "per_dataset": {k: {ds: mean_stats({i: r[i] for i in ds_ids}) for ds, ds_ids in sorted(by_ds.items())}
                        for k, r in rows.items()},
        "paired_vs_base": {arm: paired(rows[arm], rows[BASE_CONFIG], BOOT_KEYS) for arm in rows if arm != BASE_CONFIG},
    }

    if not missing:
        d = stats[BASE_CONFIG]
        checks = {arm: guard_checks(stats[arm], d, RULE) for arm in E3_ARMS}
        passing = [arm for arm in E3_ARMS if all(checks[arm].values())]
        chosen = max(passing, key=lambda arm: stats[arm]["total"]) if passing else None
        summary["judgment"] = {
            "rule": RULE, "checks": checks, "passing_arms": passing, "selected": chosen,
            "negative_result": not passing,
            "next_step": ("selected arm -> full benchmark acceptance run" if chosen
                          else "no arm passes guards; publish negative result, no full-benchmark run"),
        }
        out = f"{E3}/e3_summary.json"
    else:
        out = f"{E3}/e3_summary_partial.json"
    with open(out + ".tmp", "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(out + ".tmp", out)

    print("| arm | Total* | dynamic | aesthetic | imaging | motion_smooth | subject | DreamSim | motion_err |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, s in stats.items():
        print(f"| {k} | {100 * s['total']:.2f} | {100 * s['dynamic_degree']:.2f} | {100 * s['aesthetic_quality']:.2f} | "
              f"{100 * s['imaging_quality']:.2f} | {100 * s['motion_smoothness']:.2f} | "
              f"{100 * s['subject_consistency']:.2f} | {s['dreamsim']:.4f} | {s['motion_ratio']:.3f} |")
    if "judgment" in summary:
        j = summary["judgment"]
        print("guards: " + "; ".join(f"{arm}: {'PASS' if all(c.values()) else 'FAIL ' + ','.join(k for k, v in c.items() if not v)}"
                                      for arm, c in j["checks"].items()))
        print(f"passing: {j['passing_arms']} selected: {j['selected']} negative_result: {j['negative_result']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
