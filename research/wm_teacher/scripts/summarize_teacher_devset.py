#!/usr/bin/env python3
"""Aggregate-only two-seed Teacher qualification. No third seed or E3 change.

All metrics reuse score_e1. Cases keep their two seeds together in bootstrap;
UNKNOWN parent sources are disclosed, never invented as independent parents.
Local blind-review receipts are structural evidence, not proof of human gold.
"""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import random
import time

import score_e1 as e1
import score_teacher_devset as score

BOOT_B = 2000
BOOT_SEED = 20261025
METRICS = e1.DIMS + list(e1.GT_KEYS) + ["total"]
CANDIDATES = ("alg", "dymos")
REVIEW_FLAGS = ("jitter", "flicker", "camera_drift_only", "action_incorrect")
REVIEW_SCHEMA = "teacher_devset_blind_review_protocol_v1"
REVIEW_RECEIPT_SCHEMA = "teacher_devset_blind_review_receipt_v1"
RUBRIC = {
    "scope": "Local private review of every complete generated video paired with its complete native NONE base video",
    "playback": "Watch all 121 frames at native 24fps; slow replay allowed; no sampled-frame-only verdict",
    "jitter": "Reject apparent dynamic gain explained by temporal jitter rather than coherent object/scene motion",
    "flicker": "Reject apparent dynamic gain explained by flashing, texture/color flicker or discontinuous appearance",
    "camera_drift_only": "Reject apparent motion gain explained only by unrequested camera drift without coherent intended action",
    "action_incorrect": "Reject actions inconsistent with original input/reference; schema validity is not semantic correctness",
    "independence": "Two independent local blinded reviewers; all disagreements require a third independent local adjudicator",
    "privacy": "No automatic upload; blind mapping, media, case IDs and original prompts remain private",
}
require = score.require


def parent_groups(cases):
    known, unknown = {}, []
    for case in cases:
        keys = case.get("parent_keys")
        require(isinstance(keys, list) and keys and all(isinstance(k, str) and k for k in keys) and
                len(keys) == len(set(keys)) and type(case.get("parent_resolved")) is bool, "Invalid parent metadata")
        if case["parent_resolved"]:
            require(case["dataset"] != "egoexo4d" and len(keys) == 1, "Resolved parent schema drift")
            known[case["id"]] = (case["dataset"], keys[0])
        else:
            require(case["dataset"] == "egoexo4d", "Unknown parent outside frozen egoexo4d6")
            unknown.append(case["id"])
    require(len(known) == 88 and len(set(known.values())) == 88 and len(unknown) == 6,
            "Frozen 88 distinct resolved sources plus 6 UNKNOWN required")
    # UNKNOWN cases use descriptive case-level fallback; this does not claim source independence.
    units = [[i] for i in sorted(known)] + [[i] for i in sorted(unknown)]
    return units, {
        "resolved_cases": 88, "resolved_distinct_parent_keys": 88, "unknown_cases": 6,
        "unknown_dataset": "egoexo4d", "parent_status": "UNKNOWN for 6 egoexo4d cases",
        "unknown_fallback": "case-level resampling; parent independence unverified; non-egoexo4d88 sensitivity mandatory",
    }


def paired_grouped(left, right, units, bootstrap=BOOT_B, seed=BOOT_SEED):
    require(type(bootstrap) is int and bootstrap > 0, "Invalid bootstrap count")
    require(set(left) == set(score.SEEDS) and set(right) == set(score.SEEDS), "Exactly two seeds, never 188 independent observations")
    ids = [i for group in units for i in group]
    require(ids and len(ids) == len(set(ids)), "Invalid bootstrap source/case units")
    for name in score.SEEDS:
        score.exact(left[name], ids, "Candidate pairing")
        score.exact(right[name], ids, "Base pairing")
    rng = random.Random(seed)
    differences = {k: {i: sum(left[s][i][k] - right[s][i][k] for s in score.SEEDS) / 2 for i in ids}
                   for k in METRICS}
    samples = {k: [] for k in METRICS}
    for _ in range(bootstrap):
        chosen = [i for _ in units for i in units[rng.randrange(len(units))]]
        for key in METRICS:
            samples[key].append(sum(differences[key][i] for i in chosen) / len(chosen))
    result = {}
    for key in METRICS:
        draws = sorted(samples[key])
        result[key] = {"mean": sum(differences[key].values()) / len(ids),
                       "ci95": [draws[int(0.025 * bootstrap)], draws[int(0.975 * bootstrap) - 1]],
                       "n_cases": len(ids), "n_seeds": 2, "n_resampling_units": len(units)}
    return result


def qualify(means, visual_status):
    passing, checks = [], {}
    for method in CANDIDATES:
        per_seed = {}
        for seed in score.SEEDS:
            candidate, base = means[method][seed], means["base"][seed]
            guards = e1.guard_checks(candidate, base)
            delta_dynamic = candidate["dynamic_degree"] - base["dynamic_degree"]
            delta_total = candidate["total"] - base["total"]
            per_seed[seed] = {"four_guards": guards, "all_guards_pass": all(guards.values()),
                              "dynamic_delta": delta_dynamic, "dynamic_strictly_positive": delta_dynamic > 0,
                              "total_delta": delta_total, "total_nonnegative": delta_total >= 0}
        numeric_pass = all(s["all_guards_pass"] and s["dynamic_strictly_positive"] and s["total_nonnegative"]
                           for s in per_seed.values())
        visual = visual_status[method]
        qualified = numeric_pass and visual == "passed"
        if qualified:
            passing.append(method)
        checks[method] = {"seeds": per_seed, "numeric_eligible": numeric_pass, "quality_status": visual,
                          "qualified_for_further_frozen_acceptance": qualified,
                          "two_seed_mean_total": sum(means[method][s]["total"] for s in score.SEEDS) / 2}
    # Exact float equality, not rounded/tolerance ties. DyMoS wins an exact tie only.
    passing.sort(key=lambda m: (-checks[m]["two_seed_mean_total"], m != "dymos"))
    return {"selected": passing[0] if passing else None, "passing": passing, "checks": checks,
            "rule": "Every seed independently: all four guards, dynamic>0, Total*>=0; complete blind review must pass; highest two-seed mean Total*, exact ties DyMoS",
            "guard_rule": e1.RULE, "no_third_seed": True, "e3_unchanged": True,
            "scope": "Qualification to request a separately frozen full acceptance only, not quality/SOTA/fullbenchmark proof"}


def review_media(plan):
    result = []
    for method in score.METHODS:
        for seed in score.SEEDS:
            for row in plan["groups"][method]["seeds"][seed]["media"]:
                result.append({"method": method, "seed": seed, "case_id": row["case_id"], "file": row["file"]})
    return result


def validate_review_protocol(protocol, plan, plan_sha):
    require(protocol.get("schema") == REVIEW_SCHEMA and protocol.get("private") is True and
            protocol.get("plan_sha256") == plan_sha and protocol.get("bundle_sha256") == plan["bundle"]["sha256"] and
            protocol.get("common_execution_sha256") == plan["common_execution_sha256"] and
            protocol.get("rubric") == RUBRIC and protocol.get("rubric_sha256") == score.digest(RUBRIC) and
            protocol.get("local_only") is True and protocol.get("full_video_required") is True and
            protocol.get("independent_reviewers") == 2 and protocol.get("independent_adjudicator") is True,
            "Unfrozen/incomplete blind rubric/protocol")
    require(type(protocol.get("created_epoch")) in (int, float) and math.isfinite(protocol["created_epoch"]) and
            protocol.get("frozen_before_scores") is True, "Review protocol must precede any scoring")
    expected = review_media(plan)
    media = protocol["media"]
    require(len(media) == 564, "Full blind coverage of all 564 generated videos required")
    require([{k: row[k] for k in ("method", "seed", "case_id", "file")} for row in media] == expected,
            "Blind full media order/SHA coverage mismatch")
    blind_ids = [row["blind_id"] for row in media]
    require(all(isinstance(i, str) and i and all(c.isalnum() or c in "_-" for c in i) for i in blind_ids), "Unsafe blind ID")
    score.exact(blind_ids, set(blind_ids), "Unique complete-video blind IDs")
    # The mapping is private; identities must not directly encode method, seed, or case names.
    for row in media:
        require(not any(token in row["blind_id"].lower() for token in (row["method"], row["seed"], row["case_id"].lower())),
                "Unblinded method/seed/case identity in blind ID")
        score.check_record(row["file"])
    cases = score.read_cases(plan["cases"]["path"])
    inputs = [{"case_id": c["id"], "gt": score.file_record(c["gt"]), "image": score.file_record(c["image"])} for c in cases]
    require(protocol["inputs"] == inputs, "Review input GT/image SHA coverage drift")
    reviewers = protocol["reviewers"]
    require(isinstance(reviewers, list) and len(reviewers) == 2 and len(set(reviewers)) == 2 and
            all(isinstance(r, str) and r.strip() for r in reviewers) and
            isinstance(protocol["adjudicator"], str) and protocol["adjudicator"].strip() and
            protocol["adjudicator"] not in reviewers, "Review independence roles missing")
    return protocol


def freeze_review(plan_path, approved_plan_sha, assignments_path, output):
    require(score.sha256(plan_path) == approved_plan_sha, "Unapproved scoring plan")
    plan = score.read_json(plan_path)
    score.validate_plan(plan, verify_generation=False, verify_bundle_content=False)
    for method in score.METHODS:
        for seed in score.SEEDS:
            root = Path(plan["groups"][method]["seeds"][seed]["root"])
            require(not any((root / "scores").glob("*_eval_results.json")) and
                    not any((root / "gtref").glob("*.json")), "Cannot freeze rubric after scores exist")
    assignments = score.read_json(assignments_path)
    protocol = {"schema": REVIEW_SCHEMA, "private": True, "plan_sha256": approved_plan_sha,
                "bundle_sha256": plan["bundle"]["sha256"], "common_execution_sha256": plan["common_execution_sha256"],
                "rubric": RUBRIC, "rubric_sha256": score.digest(RUBRIC), "local_only": True,
                "full_video_required": True, "independent_reviewers": 2, "independent_adjudicator": True,
                "created_epoch": time.time(), "frozen_before_scores": True,
                "media": assignments["media"], "inputs": assignments["inputs"],
                "reviewers": assignments["reviewers"], "adjudicator": assignments["adjudicator"]}
    validate_review_protocol(protocol, plan, approved_plan_sha)
    return score.write_new(output, protocol)


def review_status(protocol, receipt_record=None, protocol_sha=None):
    if receipt_record is None:
        return {method: "pending" for method in CANDIDATES}
    score.check_record(receipt_record)
    receipt = score.read_json(receipt_record["path"])
    require(receipt.get("schema") == REVIEW_RECEIPT_SCHEMA and receipt.get("protocol_sha256") == (protocol_sha or score.digest(protocol)) and
            receipt.get("rubric_sha256") == score.digest(RUBRIC) and receipt.get("local_only") is True and
            receipt.get("complete") is True, "Unbound/incomplete local blind review receipt")
    media = {r["blind_id"]: r for r in protocol["media"]}
    candidate_ids = {i for i, row in media.items() if row["method"] in CANDIDATES}
    base = {(r["seed"], r["case_id"]): i for i, r in media.items() if r["method"] == "base"}
    decisions = {}
    for row in receipt["reviews"]:
        blind, reviewer = row["blind_id"], row["reviewer"]
        require(blind in candidate_ids and reviewer in protocol["reviewers"] and
                (blind, reviewer) not in decisions, "Duplicate/unexpected reviewer decision")
        require(row.get("base_blind_id") == base[(media[blind]["seed"], media[blind]["case_id"])] and
                row.get("all_frames_watched") is True and row.get("base_all_frames_watched") is True and
                row.get("generated_sha256") == media[blind]["file"]["sha256"] and
                row.get("base_sha256") == media[row["base_blind_id"]]["file"]["sha256"], "Review full-video/media binding missing")
        flags = row["flags"]
        require(set(flags) == set(REVIEW_FLAGS) and all(type(v) is bool for v in flags.values()), "Missing human review artifact/action flags")
        decisions[(blind, reviewer)] = flags
    require(set(decisions) == {(i, r) for i in candidate_ids for r in protocol["reviewers"]}, "Incomplete independent review coverage")
    conflicts = {i for i in candidate_ids if decisions[(i, protocol["reviewers"][0])] != decisions[(i, protocol["reviewers"][1])]}
    adjudications = {}
    for row in receipt["adjudications"]:
        blind = row["blind_id"]
        require(blind in conflicts and blind not in adjudications and row.get("adjudicator") == protocol["adjudicator"] and
                row.get("all_frames_watched") is True and row.get("base_all_frames_watched") is True and
                row.get("generated_sha256") == media[blind]["file"]["sha256"] and
                row.get("base_sha256") == media[base[(media[blind]["seed"], media[blind]["case_id"]) ]]["file"]["sha256"],
                "Missing/duplicate/unbound independent adjudication")
        require(set(row["flags"]) == set(REVIEW_FLAGS) and all(type(v) is bool for v in row["flags"].values()), "Invalid adjudicator flags")
        adjudications[blind] = row["flags"]
    require(set(adjudications) == conflicts, "Every disagreement requires independent adjudication")
    failed = set()
    for blind in candidate_ids:
        flags = adjudications[blind] if blind in conflicts else decisions[(blind, protocol["reviewers"][0])]
        if any(flags.values()):
            failed.add(media[blind]["method"])
    return {method: "failed" if method in failed else "passed" for method in CANDIDATES}


def summarize(plan_path, approved_sha, protocol_path, approved_protocol_sha, review_receipt=None):
    require(score.sha256(plan_path) == approved_sha, "Unapproved scoring plan SHA")
    plan = score.read_json(plan_path)
    cases = score.validate_plan(plan)
    units, parents = parent_groups(cases)
    require(score.sha256(protocol_path) == approved_protocol_sha, "Unapproved blind review protocol SHA")
    protocol = validate_review_protocol(score.read_json(protocol_path), plan, approved_sha)
    rows, receipts = {}, []
    for method in score.METHODS:
        rows[method] = {}
        for seed in score.SEEDS:
            entry = plan["groups"][method]["seeds"][seed]
            path = Path(entry["root"]) / "teacher_score.receipt.json"
            rows[method][seed], receipt = score.verify_score_receipt(plan, approved_sha, method, seed, path)
            require(type(receipt.get("started_epoch")) in (int, float) and math.isfinite(receipt["started_epoch"]) and
                    receipt["started_epoch"] >= protocol["created_epoch"], "Scoring began before frozen blind rubric/media protocol")
            receipts.append(receipt)
        require(sum(r["allocated_gpu_seconds"] for r in receipts if r["method"] == method) <= score.SCORE_CAP_SECONDS[method],
                "Two-seed scoring exceeded independent fixed stage cap")
    visual = review_status(protocol, review_receipt, approved_protocol_sha)
    means = {m: {s: e1.mean_stats(rows[m][s]) for s in score.SEEDS} for m in score.METHODS}
    keep = {c["id"] for c in cases if c["dataset"] != "egoexo4d"}
    datasets = defaultdict(list)
    for case in cases:
        datasets[case["dataset"]].append(case["id"])
    sensitivity = {m: {s: {i: row for i, row in rows[m][s].items() if i in keep} for s in score.SEEDS} for m in score.METHODS}
    summary = {
        "protocol": "Teacher docs/51 fixed two-seed qualification; original score_e1 all8/GT formulas",
        "complete_metrics": True, "n_cases": 94, "n_seeds": 2, "n_videos_per_method": 188,
        "means": means, "selection": qualify(means, visual), "parent_sources": parents,
        "paired_two_seed": {m: paired_grouped(rows[m], rows["base"], units) for m in CANDIDATES},
        "per_dataset": {ds: {m: {s: e1.mean_stats({i: rows[m][s][i] for i in ids}) for s in score.SEEDS}
                                   for m in score.METHODS} for ds, ids in sorted(datasets.items())},
        "sensitivity_non_egoexo4d88": {
            "n_cases": 88, "means": {m: {s: e1.mean_stats(sensitivity[m][s]) for s in score.SEEDS} for m in score.METHODS},
            "paired_two_seed": {m: paired_grouped(sensitivity[m], sensitivity["base"], [[i] for i in sorted(keep)]) for m in CANDIDATES},
            "selection_unchanged": True},
        "bootstrap": {"samples": BOOT_B, "seed": BOOT_SEED,
                      "unit": "resolved parent groups; both seeds of every case sampled together; UNKNOWN6 case fallback disclosed",
                      "same_draws_every_metric": True, "not_188_independent": True,
                      "limitation": "Two seeds do not estimate population variance adequately; descriptive intervals only"},
        "quality_status": visual, "review_schema_not_human_gold": True,
        "qualification_pending_without_review": review_receipt is None,
        "release_scope": "Aggregate diagnostics only; no private media/IDs/prompts and no fullbench/SOTA/AR acceptance",
    }
    require(score.sha256(plan_path) == approved_sha and score.sha256(protocol_path) == approved_protocol_sha,
            "Plan/review protocol changed during summary")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze-review")
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--approved-plan-sha256", required=True)
    p.add_argument("--assignments", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("summarize")
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--approved-plan-sha256", required=True)
    p.add_argument("--review-protocol", required=True, type=Path)
    p.add_argument("--approved-review-protocol-sha256", required=True)
    p.add_argument("--review-receipt", type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "freeze-review":
        freeze_review(args.plan, args.approved_plan_sha256, args.assignments, args.out)
    else:
        receipt = score.file_record(args.review_receipt) if args.review_receipt else None
        value = summarize(args.plan, args.approved_plan_sha256, args.review_protocol,
                          args.approved_review_protocol_sha256, receipt)
        score.write_new(args.out, value)


if __name__ == "__main__":
    main()
