#!/usr/bin/env python3
"""CPU-only metadata draft for a fresh v2 pilot; never a GPU authorization."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re

from recaption_e3 import PROMPT_VERSION, validate_facts

SCHEMA = "teacher_label_cpu_draft_v1"
STAGE = "label_rebuild_and_cache24"
SEED = 20261101
MAX_METADATA_BYTES = 8 * 1024 * 1024
BLOCKERS = ["independent_cpu_acceptance", "public_source_freeze_approval",
            "full_model_assets_content_approval", "actual_media_content_verification",
            "independent_pilot_media_safety_review",
            "explicit_plan_and_authorization_sha256", "live_rank_gpu_uuid_selected_one_idle",
            "live_gpu_external_owners_refused",
            "owned_parent_and_tree_gpu_release", "production_worker_not_implemented"]
# Metadata prescreen only, not an assertion about unseen video content.
UNSAFE = re.compile(r"\b(president|prime minister|election|politic\w*|government|parliament|"
                    r"communist|nazi|war|military|terror\w*|weapon\w*|gun|rifle|bomb|"
                    r"shoot\w*|kill\w*|blood|violent|fight\w*|punch\w*|sex\w*|porn\w*|"
                    r"nude|naked|bikini|lingerie|racis\w*|hate|suicide|drug|cocaine|"
                    r"marijuana|private|passport|identity card|credit card)\b|"
                    r"政治|政府|主席|总统|选举|战争|枪|炸弹|色情|裸体|暴力|毒品", re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA256")
    return value


def path_value(value, *, exists=True):
    require(isinstance(value, str) and value and "\0" not in value, "Invalid path")
    p = Path(value)
    require(p.is_absolute() and ".." not in p.parts and p.resolve() == p,
            "Absolute non-symlink non-traversal path required")
    if exists:
        require(p.exists(), "Evidence path missing")
    return p


def strict_json(raw):
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, "Duplicate JSON key")
            obj[key] = value
        return obj
    def invalid(_):
        raise ValueError("Nonfinite JSON")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    def check(item):
        if isinstance(item, str):
            require("\0" not in item, "NUL in metadata")
        elif isinstance(item, dict):
            for key, val in item.items():
                check(key)
                check(val)
        elif isinstance(item, list):
            for val in item:
                check(val)
        elif isinstance(item, float):
            require(math.isfinite(item), "Nonfinite JSON")
    check(value)
    return value


def bound(path):
    p = path_value(str(path))
    before = p.stat()
    require(p.is_file() and before.st_size <= MAX_METADATA_BYTES, "Only bounded CPU metadata")
    raw = p.read_bytes()
    after = p.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), "Evidence changed during read")
    return {"path": str(p), "bytes": len(raw), "sha256": sha(raw)}, raw


def verify_bound(record):
    require(isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"}, "Bad file proof")
    digest(record["sha256"])
    require(type(record["bytes"]) is int and record["bytes"] >= 0, "Bad evidence size")
    observed, raw = bound(record["path"])
    require(observed == record, "Content proof drift")
    return raw


def read_evidence(path):
    proof, raw = bound(path)
    value = strict_json(raw)
    require(isinstance(value, dict), "Evidence root object required")
    return proof, value


def manifest_rows(path):
    proof, raw = bound(path)
    rows = [strict_json(line) for line in raw.splitlines()]
    require(len(rows) == 1856, "Original 1856-row manifest required")
    ids, videos = set(), set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"sample_id", "video", "captions"},
                "Unexpected manifest field (scores forbidden)")
        sid, video, captions = row["sample_id"], row["video"], row["captions"]
        require(isinstance(sid, str) and sid.strip() and sid not in ids, "Duplicate or invalid sample")
        path_value(video)
        require(video not in videos, "Duplicate video")
        require(isinstance(captions, dict) and set(captions) == {"orig", "fixed"} and
                all(isinstance(v, str) and v.strip() for v in captions.values()), "Invalid prompts")
        ids.add(sid)
        videos.add(video)
    return proof, rows


def validate_model(value):
    require(set(value) == {"schema", "name", "revision", "root", "files"} and
            value["schema"] == "teacher_label_model_file_proof_v1", "Model tuple proof required")
    require(value["name"] == "Qwen2.5-VL-7B-Instruct" and
            isinstance(value["revision"], str) and value["revision"].strip(), "Explicit model tuple required")
    root = path_value(value["root"])
    require(root.is_dir() and isinstance(value["files"], list) and value["files"], "Model file proof empty")
    seen = set()
    for item in value["files"]:
        require(isinstance(item, dict) and set(item) == {"relative_path", "bytes", "sha256"}, "Bad model file proof")
        relative = item["relative_path"]
        require(isinstance(relative, str) and relative and "\0" not in relative and
                not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "Unsafe model file path")
        require(relative not in seen, "Duplicate model file")
        seen.add(relative)
        file = path_value(str(root / relative))
        require(file.is_file() and type(item["bytes"]) is int and item["bytes"] > 0 and
                file.stat().st_size == item["bytes"], "Model file size or type drift")
        digest(item["sha256"])
    # Supplied weight digests are claims, not verified full-asset approval.


def prepare(manifest, media_bindings, model_proof, cpu_records, owned_parent):
    manifest_proof, rows = manifest_rows(manifest)
    media_proof, media = read_evidence(media_bindings)
    model_record, model = read_evidence(model_proof)
    cpu_proof, cpu = read_evidence(cpu_records)
    validate_model(model)
    require(set(media) == {"schema", "manifest_sha256", "items"} and
            media["schema"] == "teacher_label_media_binding_v1" and
            media["manifest_sha256"] == manifest_proof["sha256"], "Media binding manifest drift")
    require(isinstance(media["items"], list) and len(media["items"]) == 1856, "Complete media binding required")
    by_id = {}
    for item in media["items"]:
        require(isinstance(item, dict) and set(item) == {"sample_id", "video", "bytes", "sha256"}, "Bad media proof")
        require(isinstance(item["sample_id"], str) and item["sample_id"] not in by_id, "Duplicate media proof")
        require(type(item["bytes"]) is int and item["bytes"] > 0, "Bad media size")
        digest(item["sha256"])
        by_id[item["sample_id"]] = item
    for row in rows:
        item = by_id.get(row["sample_id"])
        require(item is not None and item["video"] == row["video"] and
                Path(row["video"]).stat().st_size == item["bytes"], "Media identity or size drift")
    require(set(cpu) == {"schema", "scope", "source", "runtime", "environment",
                         "model_proof_sha256", "media_binding_sha256", "approval_record"} and
            cpu["schema"] == "teacher_label_cpu_records_v1" and cpu["scope"] == "cpu_metadata_only",
            "Explicit CPU records required, not GPU or fullassets approval")
    require(cpu["model_proof_sha256"] == model_record["sha256"] and
            cpu["media_binding_sha256"] == media_proof["sha256"], "CPU evidence crossbinding drift")
    for kind in ("source", "runtime", "environment"):
        require(isinstance(cpu[kind], list) and cpu[kind], "Source/runtime/environment records missing")
        require(len({r["path"] for r in cpu[kind]}) == len(cpu[kind]), "Duplicate CPU evidence")
        for record in cpu[kind]:
            verify_bound(record)
    approval = strict_json(verify_bound(cpu["approval_record"]))
    require(isinstance(approval, dict) and approval.get("scope") == "cpu_metadata_only",
            "CPU-only approval record required")
    scripts = Path(__file__).resolve().parent
    require({str(scripts / name) for name in ("recaption_e3.py", "prepare_teacher_label_admission.py",
            "run_teacher_label_admission.py")} <= {r["path"] for r in cpu["source"]}, "Actual helper/source freeze missing")
    parent = path_value(owned_parent)
    require(parent.is_dir() and parent.stat().st_uid == os.geteuid() and
            parent.stat().st_mode & 0o077 == 0, "Private current-user draft parent required")
    eligible = [r for r in rows if not UNSAFE.search(" ".join([r["sample_id"],
                 Path(r["video"]).name, *r["captions"].values()]))]
    require(len(eligible) >= 16, "Insufficient metadata-prescreened pilot pool")
    selected = random.Random(SEED).sample(eligible, 16)
    pilot = [{"sample_id": row["sample_id"], "video": row["video"],
              "video_binding": by_id[row["sample_id"]],
              "original_prompt_sha256": sha(row["captions"]["orig"].encode()),
              "fixed_prompt_sha256": sha(row["captions"]["fixed"].encode())} for row in selected]
    ranks = [{"rank": rank, "allocated_gpus": 1, "cap_seconds": 900,
              "pilot_indices": list(range(rank * 4, rank * 4 + 4)),
              "output": str(parent / ("rank%d" % rank) / "v2_labels.jsonl")} for rank in range(4)]
    for row in ranks:
        path_value(row["output"], exists=False)
        require(not Path(row["output"]).parent.exists(), "Existing rank paths cannot be reused or reset")
    return {"schema": SCHEMA, "status": "BLOCKED_CPU_DRAFT", "prompt_version": PROMPT_VERSION,
            "stage": STAGE, "stage_gpu_hours": 24, "global_pilot_reserved_gpu_seconds": 3600,
            "remaining_fixed_stage_gpu_seconds": 23 * 3600, "random_seed": SEED,
            "input_rows": 1856, "metadata_eligible_rows": len(eligible), "pilot": pilot, "ranks": ranks,
            "owned_parent": str(parent), "manifest": manifest_proof, "media_proof": media_proof,
            "model_proof": model_record, "cpu_records": cpu_proof,
            "media_content_verified": False, "model_content_verified": False,
            "semantic_review_pass": False, "semantic_review_status": "independent_real_review_pending",
            "pilot_is_semantic_gold": False, "scoreblind": True, "gpu_execution_allowed": False,
            "blockers": BLOCKERS.copy(), "accounting": "all_allocated_load_compile_wait_save_cleanup_failures",
            "ledger_scope": "node_local_method_owned_not_peer_mutable",
            "worker_contract": "recaption_e3.recaption_one_unchanged_sampling_math_v2",
            "next_gate": "pilot_success_schema_and_independent_caption_fidelity_review_then_new_fixed_remaining_label_cache_allocation"}


def validate_caption_record(value):
    require(isinstance(value, dict) and value.get("prompt_version") == PROMPT_VERSION and
            "error" not in value, "Fresh successful v2 record required")
    validate_facts(value.get("facts_json"))
    require(isinstance(value.get("caption"), str) and value["caption"].strip(), "Nonempty caption required")
    return {"schema_pass": True, "semantic_review_pass": False,
            "semantic_review_status": "independent_real_review_pending"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "media-bindings", "model-proof", "cpu-records", "owned-parent", "out"):
        ap.add_argument("--" + name, required=True)
    args = ap.parse_args(argv)
    try:
        value = prepare(args.manifest, args.media_bindings, args.model_proof, args.cpu_records, args.owned_parent)
        output = path_value(args.out, exists=False)
        require(output.parent == Path(value["owned_parent"]), "Draft output must be directly owned")
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b"\n"
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as sink:
            sink.write(raw)
        print(json.dumps({"status": value["status"], "draft_sha256": sha(raw), "pilot_rows": 16,
                          "semantic_review_pass": False, "gpu_execution_allowed": False}))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
