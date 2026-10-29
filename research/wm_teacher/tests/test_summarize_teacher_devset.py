"""Synthetic CPU qualification/review tests. Mocked scope is not real approval."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import score_e1 as e1
import score_teacher_devset as score
import summarize_teacher_devset as summary
from test_score_teacher_devset import dump, make_plan, make_scores, small_file, stable_source_inventory


def means_fixture():
    row = {k: 0.5 for k in summary.METRICS}
    row.update(dreamsim=0.1, dreamsim_last=0.1, motion_ratio=0.1, total=0.5)
    result = {m: {s: row.copy() for s in score.SEEDS} for m in score.METHODS}
    for method in summary.CANDIDATES:
        for seed in score.SEEDS:
            result[method][seed]["dynamic_degree"] += 0.1
            result[method][seed]["total"] += 0.01
    return result


def protocol_fixture(plan, plan_sha):
    media = [{**row, "blind_id": "v" + str(i).zfill(6)} for i, row in enumerate(summary.review_media(plan))]
    cases = score.read_cases(plan["cases"]["path"])
    return {"schema": summary.REVIEW_SCHEMA, "private": True, "plan_sha256": plan_sha,
            "bundle_sha256": plan["bundle"]["sha256"], "common_execution_sha256": plan["common_execution_sha256"],
            "rubric": summary.RUBRIC, "rubric_sha256": score.digest(summary.RUBRIC), "local_only": True,
            "full_video_required": True, "independent_reviewers": 2, "independent_adjudicator": True,
            "created_epoch": 10, "frozen_before_scores": True, "media": media,
            "inputs": [{"case_id": c["id"], "gt": score.file_record(c["gt"]), "image": score.file_record(c["image"])} for c in cases],
            "reviewers": ["local-reviewer-one", "local-reviewer-two"], "adjudicator": "local-third"}


def review_fixture(protocol):
    base = {(r["seed"], r["case_id"]): r for r in protocol["media"] if r["method"] == "base"}
    rows = []
    for row in protocol["media"]:
        if row["method"] not in summary.CANDIDATES:
            continue
        reference = base[(row["seed"], row["case_id"])]
        for reviewer in protocol["reviewers"]:
            rows.append({"blind_id": row["blind_id"], "reviewer": reviewer, "base_blind_id": reference["blind_id"],
                         "all_frames_watched": True, "base_all_frames_watched": True,
                         "generated_sha256": row["file"]["sha256"], "base_sha256": reference["file"]["sha256"],
                         "flags": {f: False for f in summary.REVIEW_FLAGS}})
    return {"schema": summary.REVIEW_RECEIPT_SCHEMA, "protocol_sha256": score.digest(protocol),
            "rubric_sha256": score.digest(summary.RUBRIC), "local_only": True, "complete": True,
            "reviews": rows, "adjudications": []}


@pytest.fixture
def plan(tmp_path, monkeypatch):
    value = make_plan(tmp_path)
    monkeypatch.setattr(score, "FROZEN_CASES_SHA256", value["cases"]["sha256"])
    return value


def test_each_seed_guards_do_not_average_mask_failure():
    means = means_fixture()
    means["alg"]["seed0"]["subject_consistency"] -= 0.011
    means["alg"]["seed1"]["subject_consistency"] += 0.1
    result = summary.qualify(means, {"alg": "passed", "dymos": "pending"})
    assert result["selected"] is None
    assert not result["checks"]["alg"]["numeric_eligible"]


@pytest.mark.parametrize("metric,value", [("dynamic_degree", 0.5), ("dynamic_degree", 0.4), ("total", 0.499999999999)])
def test_strict_perseed_dynamic_and_total_threshold(metric, value):
    means = means_fixture()
    means["alg"]["seed1"][metric] = value
    assert not summary.qualify(means, {"alg": "passed", "dymos": "pending"})["checks"]["alg"]["numeric_eligible"]


def test_total_zero_delta_passes_dynamic_positive():
    means = means_fixture()
    means["alg"]["seed0"]["total"] = means["base"]["seed0"]["total"]
    assert summary.qualify(means, {"alg": "passed", "dymos": "pending"})["selected"] == "alg"


def test_four_guard_boundaries_reuse_original_eps():
    means = means_fixture()
    for seed in score.SEEDS:
        means["alg"][seed].update(motion_smoothness=0.497, subject_consistency=0.49, dreamsim=0.105, motion_ratio=0.15)
    assert summary.qualify(means, {"alg": "passed", "dymos": "pending"})["selected"] == "alg"


def test_exact_tie_dymos_no_rounding():
    means = means_fixture()
    assert summary.qualify(means, {"alg": "passed", "dymos": "passed"})["selected"] == "dymos"
    means["alg"]["seed0"]["total"] += 1e-13
    assert summary.qualify(means, {"alg": "passed", "dymos": "passed"})["selected"] == "alg"


@pytest.mark.parametrize("state", ["pending", "failed"])
def test_no_review_or_artifacts_no_qualification(state):
    result = summary.qualify(means_fixture(), {"alg": state, "dymos": state})
    assert result["selected"] is None
    assert result["checks"]["alg"]["numeric_eligible"]
    assert not result["checks"]["alg"]["qualified_for_further_frozen_acceptance"]


def test_parent_known88_unknown6_and_sensitive88(plan):
    units, disclosure = summary.parent_groups(score.read_cases(plan["cases"]["path"]))
    assert len(units) == 94
    assert disclosure["resolved_distinct_parent_keys"] == 88
    assert disclosure["unknown_cases"] == 6
    assert "UNKNOWN" in disclosure["parent_status"]


@pytest.mark.parametrize("kind", ["duplicate_known", "invent_unknown", "missing_key"])
def test_parent_evidence_drift_refused(plan, kind):
    cases = score.read_cases(plan["cases"]["path"])
    if kind == "duplicate_known":
        cases[1]["parent_keys"] = cases[0]["parent_keys"]
    elif kind == "invent_unknown":
        cases[-1]["parent_resolved"] = True
    else:
        cases[0]["parent_keys"] = []
    with pytest.raises(RuntimeError):
        summary.parent_groups(cases)


def test_bootstrap_two_seeds_stay_together_no188_independence():
    base = {s: {str(i): {k: 0 for k in summary.METRICS} for i in range(3)} for s in score.SEEDS}
    candidate = copy.deepcopy(base)
    for i in range(3):
        for key in summary.METRICS:
            candidate["seed0"][str(i)][key] = 100 + i
            candidate["seed1"][str(i)][key] = -100 - i
    paired = summary.paired_grouped(candidate, base, [[str(i)] for i in range(3)])
    assert all(row["mean"] == 0 and row["ci95"] == [0, 0] for row in paired.values())
    assert all(row["n_cases"] == 3 and row["n_seeds"] == 2 and row["n_resampling_units"] == 3 for row in paired.values())


def test_bootstrap_same_draws_all_metrics_and_deterministic():
    base = {s: {str(i): {k: 0 for k in summary.METRICS} for i in range(4)} for s in score.SEEDS}
    candidate = {s: {str(i): {k: i for k in summary.METRICS} for i in range(4)} for s in score.SEEDS}
    units = [[str(i)] for i in range(4)]
    result = summary.paired_grouped(candidate, base, units)
    assert result == summary.paired_grouped(candidate, base, units)
    assert result[summary.METRICS[0]] == result[summary.METRICS[-1]]
    assert summary.BOOT_B == 2000 and summary.BOOT_SEED == 20261025


def test_bootstrap_rejects_seed3():
    left = {s: {} for s in (*score.SEEDS, "seed2")}
    with pytest.raises(RuntimeError, match="Exactly two seeds"):
        summary.paired_grouped(left, left, [["x"]])


def test_review_protocol_full_media_rubric_inputs(plan):
    protocol = protocol_fixture(plan, "2" * 64)
    assert summary.validate_review_protocol(protocol, plan, "2" * 64) == protocol
    assert len(protocol["media"]) == 564


@pytest.mark.parametrize("kind", ["rubric", "missing_video", "duplicate_blind", "unblind", "input_sha", "same_adjudicator", "sampled_frames", "remote"])
def test_review_protocol_drift_refused(plan, kind):
    protocol = copy.deepcopy(protocol_fixture(plan, "2" * 64))
    if kind == "rubric":
        protocol["rubric"]["jitter"] = "always pass"
    elif kind == "missing_video":
        protocol["media"].pop()
    elif kind == "duplicate_blind":
        protocol["media"][1]["blind_id"] = protocol["media"][0]["blind_id"]
    elif kind == "unblind":
        protocol["media"][0]["blind_id"] = "base_seed0_case000"
    elif kind == "input_sha":
        protocol["inputs"][0]["gt"]["sha256"] = "9" * 64
    elif kind == "same_adjudicator":
        protocol["adjudicator"] = protocol["reviewers"][0]
    elif kind == "sampled_frames":
        protocol["full_video_required"] = False
    else:
        protocol["local_only"] = False
    with pytest.raises(RuntimeError):
        summary.validate_review_protocol(protocol, plan, "2" * 64)


def test_no_receipt_quality_pending(plan):
    protocol = protocol_fixture(plan, "2" * 64)
    assert summary.review_status(protocol) == {"alg": "pending", "dymos": "pending"}


def test_complete_synthetic_review_schema_only_not_human_gold(plan, tmp_path):
    protocol = protocol_fixture(plan, "2" * 64)
    receipt = dump(tmp_path / "review.json", review_fixture(protocol))
    assert summary.review_status(protocol, receipt) == {"alg": "passed", "dymos": "passed"}


@pytest.mark.parametrize("kind", ["missing_row", "duplicate", "bad_base", "sampled_frames", "wrong_sha", "missing_flag", "rubric", "remote"])
def test_strict_review_receipt_drift(plan, tmp_path, kind):
    protocol = protocol_fixture(plan, "2" * 64)
    value = review_fixture(protocol)
    if kind == "missing_row":
        value["reviews"].pop()
    elif kind == "duplicate":
        value["reviews"].append(value["reviews"][0])
    elif kind == "bad_base":
        value["reviews"][0]["base_blind_id"] = "v000001"
    elif kind == "sampled_frames":
        value["reviews"][0]["all_frames_watched"] = False
    elif kind == "wrong_sha":
        value["reviews"][0]["generated_sha256"] = "9" * 64
    elif kind == "missing_flag":
        del value["reviews"][0]["flags"]["flicker"]
    elif kind == "rubric":
        value["rubric_sha256"] = "9" * 64
    else:
        value["local_only"] = False
    receipt = dump(tmp_path / "review.json", value)
    with pytest.raises(RuntimeError):
        summary.review_status(protocol, receipt)


def test_independent_conflict_adjudication_required(plan, tmp_path):
    protocol = protocol_fixture(plan, "2" * 64)
    value = review_fixture(protocol)
    value["reviews"][0]["flags"]["jitter"] = True
    receipt = dump(tmp_path / "review.json", value)
    with pytest.raises(RuntimeError, match="Every disagreement"):
        summary.review_status(protocol, receipt)
    first = value["reviews"][0]
    value["adjudications"] = [{"blind_id": first["blind_id"], "adjudicator": protocol["adjudicator"],
                               "all_frames_watched": True, "base_all_frames_watched": True,
                               "generated_sha256": first["generated_sha256"], "base_sha256": first["base_sha256"],
                               "flags": first["flags"]}]
    receipt = dump(tmp_path / "review.json", value)
    assert summary.review_status(protocol, receipt) == {"alg": "failed", "dymos": "passed"}


def test_freeze_review_before_scores_preserves_existing(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(score, "verify_comparison_binding", lambda *args, **kwargs: True)
    monkeypatch.setattr(score, "verify_completion_media", lambda *args, **kwargs: True)
    monkeypatch.setattr(score, "verify_completion", lambda *args, **kwargs: True)
    record = dump(tmp_path / "plan.json", plan)
    protocol = protocol_fixture(plan, record["sha256"])
    assignments = dump(tmp_path / "assignments.json", {k: protocol[k] for k in ("media", "inputs", "reviewers", "adjudicator")})
    target = tmp_path / "review.protocol.json"
    summary.freeze_review(record["path"], record["sha256"], assignments["path"], target)
    with pytest.raises(RuntimeError, match="Preserve"):
        summary.freeze_review(record["path"], record["sha256"], assignments["path"], target)
    make_scores(plan, record["sha256"], "base", "seed0")
    with pytest.raises(RuntimeError, match="after scores"):
        summary.freeze_review(record["path"], record["sha256"], assignments["path"], tmp_path / "late.json")


def complete_fixture(plan, tmp_path, monkeypatch):
    record = dump(tmp_path / "plan.json", plan)
    protocol = protocol_fixture(plan, record["sha256"])
    protocol_record = dump(tmp_path / "review.protocol.json", protocol)
    # Only this synthetic fixture replaces unavailable real execution evidence verifiers.
    monkeypatch.setattr(score, "verify_completion", lambda *args, **kwargs: True)
    monkeypatch.setattr(score, "verify_comparison_binding", lambda *args, **kwargs: True)
    monkeypatch.setattr(score, "verify_completion_media", lambda *args, **kwargs: True)
    monkeypatch.setattr(score, "verify_scoring_execution", lambda *args, **kwargs: True)
    for method in score.METHODS:
        for seed in score.SEEDS:
            make_scores(plan, record["sha256"], method, seed, dynamic=0.5 if method == "base" else 0.6)
    return record, protocol_record, protocol


def test_complete_cpu_summary_pending_not_qualified_no_identifiers(plan, tmp_path, monkeypatch):
    record, protocol_record, _ = complete_fixture(plan, tmp_path, monkeypatch)
    result = summary.summarize(record["path"], record["sha256"], protocol_record["path"], protocol_record["sha256"])
    assert result["complete_metrics"] and result["n_cases"] == 94 and result["n_seeds"] == 2
    assert result["selection"]["selected"] is None
    assert result["qualification_pending_without_review"]
    assert result["sensitivity_non_egoexo4d88"]["n_cases"] == 88
    assert result["review_schema_not_human_gold"] and result["bootstrap"]["not_188_independent"]
    assert "case000" not in str(result) and "parent0" not in str(result) and "synthetic action" not in str(result)


def test_complete_synthetic_summary_review_schema_does_not_qualify(plan, tmp_path, monkeypatch):
    record, protocol_record, protocol = complete_fixture(plan, tmp_path, monkeypatch)
    review_value = review_fixture(protocol)
    review_value["protocol_sha256"] = protocol_record["sha256"]
    review = dump(tmp_path / "review.json", review_value)
    result = summary.summarize(record["path"], record["sha256"], protocol_record["path"], protocol_record["sha256"], review)
    assert result["selection"]["selected"] is None
    assert result["review_schema_status"] == {"alg": "passed", "dymos": "passed"}
    assert result["quality_status"] == {"alg": "pending", "dymos": "pending"}
    assert result["review_execution_authenticated"] is False
    assert result["qualification_pending_without_verified_review"]


def test_total_score_caps_across_bothseeds(plan, tmp_path, monkeypatch):
    record, protocol_record, _ = complete_fixture(plan, tmp_path, monkeypatch)
    for seed in score.SEEDS:
        path = Path(plan["groups"]["base"]["seeds"][seed]["root"]) / "teacher_score.receipt.json"
        value = score.read_json(path)
        value["allocated_gpu_seconds"] = 10000
        dump(path, value)
    with pytest.raises(RuntimeError, match="Two-seed scoring exceeded"):
        summary.summarize(record["path"], record["sha256"], protocol_record["path"], protocol_record["sha256"])


def test_scoring_before_rubric_rejected(plan, tmp_path, monkeypatch):
    record, protocol_record, _ = complete_fixture(plan, tmp_path, monkeypatch)
    path = Path(plan["groups"]["base"]["seeds"]["seed0"]["root"]) / "teacher_score.receipt.json"
    value = score.read_json(path)
    value["started_epoch"] = 9
    dump(path, value)
    with pytest.raises(RuntimeError, match="before frozen"):
        summary.summarize(record["path"], record["sha256"], protocol_record["path"], protocol_record["sha256"])
