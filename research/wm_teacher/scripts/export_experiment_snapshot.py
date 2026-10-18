#!/usr/bin/env python3
"""Export an explicitly selected, credential-checked experiment snapshot without Git history.

The source repository contains unrelated/private files. Never copy its .git, cookies,
model weights, videos, environments or operational logs to the publication repository.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "README.md", "PROGRESS.md", "docs/README.md", "docs/WAN22_TRAINING_PLAN.md",
    "docs/24_Wan2.2首轮训练执行计划.md", "docs/25_Wan22_Training_Infra.md",
    "docs/27_数据预处理与训练格式适配.md", "docs/28_数据总量与项目目标评估.md",
    "docs/30_Teacher模型训练实验记录.md", "docs/31_服务器迁移与CodingAgent交接.md",
    "patches/diffsynth_studio_fb337fb.patch", "docs/32_插值Teacher完整评测结论.md",
    "eval/summary.json", "eval/summary.md", "eval/diag/merge_eval_complete_20261018.json",
    "training_metadata/full/ds_zero2_offload.json",
] + ["scripts/" + name for name in (
    "gen_jobs.py", "run_gen.sh", "score_extra.sh", "eval_val_loss.py", "score_merge_cpu.py",
    "train_full_16gpu.sh", "train_round2_16gpu.sh", "launch_round2.sh",
    "monitor_merge.py", "wmb_merge_score.py", "val_merge_score.py", "merge_final_gate.py",
    "analyze_merge_metrics.py", "report_merge_results.py", "export_experiment_snapshot.py", "build_eval_jobs.py", "build_extra_bench_jobs.py",
    "gt_ref_metrics.py", "vbench_eval.py", "score_pool.py", "summarize_eval.py",
    "build_teacher_review.py", "build_demo.py", "serve_demo.py", "merge_ckpt.py",
    "build_round1_mixture.py", "build_round2_mixture.py", "build_round3_mixture.py",
    "build_round4_mixture.py", "build_round5_mixture.py", "build_ft2_mixture.py",
    "score_camera.py", "scan_cuts.py", "rewrite_prompts_vbench.py", "reslice_5s.py",
    "finalize_5s.py", "inventory_5s.py", "slice_new_sources.py", "caption_5s_extra.py",
)] + ["tests/" + name for name in (
    "test_score_merge_cpu.py", "test_monitor_val_orchestration.py", "test_wmb_final_orchestration.py", "test_analyze_merge_metrics.py",
    "test_export_experiment_snapshot.py", "test_report_merge_results.py", "test_score_pool.py",
)] + ["eval/diag/" + name for name in (
    "ft2_paired_bootstrap.json", "merge_integrity.json", "physiq_paired_bootstrap.json",
    "caption_token_lengths.json", "parameter_drift.json", "validation_noise_analysis.json",
    "vbench_target_gap.json", "physiq_prompt_lexical_audit.json", "phase_e_integrity_final.json",
    "wmb_full_raw_po_bootstrap.json", "physiq_merge_a05_paired_bootstrap_20261018.json",
    "gtref_merge_a05_paired_bootstrap_20261018.json", "gtref_merge_a03_paired_bootstrap_20261018.json",
    "merge_a03_a05_analysis_20261018.json", "merge_all_cpu_analysis_20261018.json", "merge_all_analysis_20261018.json",
)]
REQUIRED = {"scripts/gen_jobs.py", "scripts/run_gen.sh", "scripts/score_merge_cpu.py",
            "scripts/analyze_merge_metrics.py", "scripts/monitor_merge.py", "scripts/wmb_merge_score.py",
            "scripts/val_merge_score.py", "scripts/merge_final_gate.py", "docs/30_Teacher模型训练实验记录.md"}
SECRET_PATTERNS = [
    re.compile(r"github_pat_[A-Za-z0-9_]+"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
    re.compile(r"(?i)(?:password|token|api_key|secret)\s*=\s*['\"][A-Za-z0-9_+/=-]{16,}['\"]"),
]


def sanitize(relative, text):
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise RuntimeError(f"Credential-like content in selected file: {relative}; inspect privately")
    # Internal deployment addresses are not necessary to reproduce the algorithms.
    return re.sub(r"(?<![\w.])11(?:\.\d{1,3}){3}(?![\w.])", "PRIVATE_NODE_ADDRESS", text)


def export(root, destination):
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("Destination already exists; preserve it and choose a new snapshot path")
    root, destination = root.resolve(), destination.resolve()
    if destination.exists():
        raise RuntimeError("Destination already exists; preserve it and choose a new snapshot path")
    if root == destination or root in destination.parents:
        raise RuntimeError("Export outside the working repository to avoid publishing private artifacts")
    missing = sorted(path for path in REQUIRED if not (root / path).is_file())
    if missing:
        raise RuntimeError("Required experiment implementation not ready: " + ", ".join(missing))
    selected = []
    for relative in FILES:
        source = root / relative
        if not source.is_file():
            continue
        if source.is_symlink() or root not in source.resolve().parents:
            raise RuntimeError(f"Selected source must be a repository-local regular file: {relative}")
        raw = source.read_bytes()
        if len(raw) > 10 * 1024 * 1024 or b"\x00" in raw:
            raise RuntimeError(f"Not a small text publication asset: {relative}")
        text = sanitize(relative, raw.decode("utf-8"))
        selected.append((relative, text, hashlib.sha256(raw).hexdigest()))
    status_path = root / "eval/diag/merge_monitor_latest.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    if not isinstance(status, dict):
        raise RuntimeError("Invalid monitor snapshot; preserve it and review before publication")
    progress = {key: status.get(key) for key in ("at", "complete_nonempty", "total", "by_model")}
    progress = json.loads(sanitize("publication progress metadata", json.dumps(progress, allow_nan=False)))
    os.umask(0o077)
    destination.mkdir(parents=True, mode=0o700)
    records = []
    for relative, text, original_hash in selected:
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        records.append({"path": relative, "source_sha256": original_hash,
                        "snapshot_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    scope = {"created_at": time.strftime("%F %T %Z"), "files": records,
             "experiment_progress": progress,
             "caveat": "Selected source snapshot, not source Git history or private data. Nonempty output counts are not complete experiment/Teacher acceptance. GitHub push status is separate. Internal deployment addresses are placeholders."}
    (destination / "PUBLICATION_MANIFEST.json").write_text(json.dumps(scope, indent=2, ensure_ascii=False) + "\n")
    (destination / ".gitignore").write_text("*.mp4\n*.safetensors\n*.pth\n*.tar\n*.log\n__pycache__/\ncookies*\n.git/config\n")
    (destination / "PUBLICATION.md").write_text(
        "# Selected Teacher experiment source snapshot\n\n"
        "This directory deliberately excludes the original Git history, cookies, credentials, "
        "data, model weights, videos, environments and private logs. Internal node addresses "
        "are placeholders; configure your own deployment before running scripts.\n\n"
        "Use PUBLICATION_MANIFEST.json for source/snapshot hashes and the measured snapshot time. "
        "Historical planning documents are not proof of current experiment completion. "
        "All benchmark coverage, fixed validation loss and Teacher review requirements still apply.\n")
    return scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scope = export(args.root, args.out)
    print("SELECTED_PUBLICATION_SNAPSHOT_CREATED", args.out, len(scope["files"]), "files; not pushed")


if __name__ == "__main__":
    main()
