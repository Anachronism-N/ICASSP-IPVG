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
    "docs/33_Teacher优化开发消融计划.md", "docs/34_Teacher_CFG开发消融结论.md",
    "eval/diag/teacher_p1_results_20261019.json",
    "docs/35_Teacher机制诊断与随机标签审计计划.md",
    "docs/36_Teacher优化相关工作与可证伪方案.md",
    "docs/38_标签语义校准与C1推理条件实验.md",
    "docs/39_C1原与校订Caption完整结论.md", "eval/diag/C1_caption_public_results_20261020.json",
    "eval/diag/C1_FT1_membership_public_summary_20261020.json",
    "docs/40_四节点受控训练资源规划.md",
    "docs/41_窗口标注链与命名歧义验收.md",
    "docs/42_LoRA有限训练技术Smoke.md",
    "docs/43_四节点追平SOTA实验计划.md",
    "docs/44_委托代审17例caption结论.md",
    "eval/diag/delegated_label_review_17_public_summary_20261021.json",
    "eval/diag/LoRA_native_CPU_microcheck_20261020.json",
    "eval/diag/teacher_training_smoke_public_results_20261020.json",
    "eval/e1_devset_20261021/e1b_summary.json",
    "eval/diag/teacher_e1_devset_public_summary_20261021.json",
    "eval/diag/teacher_e1b_cfgzero_public_summary_20261021.json",
    "eval/diag/teacher_e1_three_seed_public_summary_20261023.json",
    "docs/46_E2完整基准结论.md",
    "docs/47_节点idea全景与SOTA距离.md",
    "docs/48_背景实验全景与阶段总结.md",
    "docs/49_E3三臂开发集结论与完整基准验收.md",
    "docs/50_C臂276条分层子集诊断预注册.md",
    "docs/51_免训练Teacher多维策略有限实验预注册.md",
    "docs/52_四基准SOTA逐维对照与协议核验.md",
    "docs/53_标签解析修复与偏好训练门禁.md",
    "docs/54_C276预算修订与一次收尾授权.md",
    "docs/55_ALG与DyMoS真实机制Smoke结论.md",
    "docs/56_双策略性能复验与双seed开发评测准入.md",
    "docs/57_双策略实测吞吐与有限开发集调度.md",
    "eval/diag/teacher_batch_throughput_public_20261026.json",
    "eval/diag/teacher_devset_generation_plan_public_20261026.json",
    "eval/diag/teacher_inference_smoke_public_results_20261026.json",
    "eval/diag/teacher_C276_generation_public_20261026.json",
    "eval/diag/teacher_C276_public_results_20261026.json",
    "eval/diag/teacher_C276_independent_audit_public_20261026.json",
    "docs/58_C276四基准子集完整结论.md",
    "scripts/prepare_teacher_recovery.py", "tests/test_teacher_recovery.py",
    "scripts/run_teacher_generation.py", "tests/test_teacher_generation.py",
    "scripts/run_teacher_scoring.py", "tests/test_run_teacher_scoring.py",
    "scripts/verify_teacher_history.py", "tests/test_verify_teacher_history.py",
    "eval/diag/teacher_execution_resume_public_20261029.json",
    "scripts/recover_teacher_devset_ledger.py",
    "tests/test_gpu_hold_monitor.py", "scripts/run_teacher_base.py", "tests/test_teacher_base.py",
    "scripts/prepare_teacher_comparison.py", "tests/test_prepare_teacher_comparison.py",
    "scripts/score_teacher_devset.py", "tests/test_score_teacher_devset.py",
    "scripts/summarize_teacher_devset.py", "tests/test_summarize_teacher_devset.py",
    "eval/diag/teacher_cpu_readiness_public_20261028.json",
    "eval/diag/e3_recaption_schema_public_20261025.json",
    "eval/diag/teacher_sota_protocol_public_20261025.json",
    "eval/diag/teacher_e3_subset_preregistered_public_20261025.json",
    "eval/diag/teacher_e3_devset_public_summary_20261025.json",
    "eval/diag/teacher_e3_integrity_public_20261025.json",
    "eval/diag/teacher_e3_seed1_public_summary_20261025.json",
    "eval/diag/teacher_e3_seed1_integrity_public_20261025.json",
    "eval/diag/teacher_e1_four_seed_public_summary_20261025.json",
    "eval/e2_s12c5_20261021/plan.json",
    "training_metadata/teacher_namespace_candidates_20261020_v1/eligibility_summary.json",
    "eval/diag/namespace_inventory_public_summary_20261020.json",
    "eval/diag/window_caption_chain_public_summary_20261020.json",
    "eval/diag/teacher_L1_semantic_pilot_summary_20261020.json",
    "docs/37_Teacher机制诊断M1实测结论.md", "eval/diag/teacher_M1_public_results_20261020.json",
    "docs/37_M1同状态预测分解实验结论.md", "eval/diag/teacher_m1_public_summary_20261020.json",
    "eval/diag/teacher_label_audit_L1_public_summary_20261020.json",
    "eval/summary.json", "eval/summary.md", "eval/diag/merge_eval_complete_20261018.json",
    "training_metadata/full/ds_zero2_offload.json",
    "DiffSynth-Studio/LICENSE", "DiffSynth-Studio/diffsynth/diffusion/flow_match.py",
    "DiffSynth-Studio/diffsynth/diffusion/training_module.py",
    "DiffSynth-Studio/diffsynth/diffusion/base_pipeline.py",
    "DiffSynth-Studio/diffsynth/models/wan_video_dit.py",
    "DiffSynth-Studio/diffsynth/configs/model_configs.py",
    "DiffSynth-Studio/diffsynth/diffusion/loss.py",
    "DiffSynth-Studio/diffsynth/pipelines/wan_video.py",
] + ["scripts/" + name for name in (
    "gen_jobs.py", "run_gen.sh", "score_extra.sh", "eval_val_loss.py", "score_merge_cpu.py",
    "train_full_16gpu.sh", "train_round2_16gpu.sh", "launch_round2.sh",
    "monitor_merge.py", "wmb_merge_score.py", "val_merge_score.py", "merge_final_gate.py",
    "analyze_merge_metrics.py", "report_merge_results.py", "export_experiment_snapshot.py", "build_eval_jobs.py", "build_extra_bench_jobs.py",
    "gt_ref_metrics.py", "vbench_eval.py", "score_pool.py", "summarize_eval.py",
    "run_teacher_probe.py", "score_teacher_probe.py",
    "audit_teacher_labels.py", "run_teacher_mechanism_probe.py", "run_teacher_caption_probe.py",
    "prepare_teacher_eligibility.py", "run_teacher_training_smoke.py", "serve_label_review.py",
    "train_e3_lora.py", "cache_e3_inputs.py", "recaption_e3.py", "build_e3_pool.py", "score_e3_motion.py",
    "merge_e3_lora.py", "prepare_e1_devset.py", "build_e1_jobs.py", "score_e1.py", "summarize_e3.py", "summarize_e1_seeds.py",
    "prepare_e3_subset.py", "run_e3_subset_node.py", "score_e3_subset.py", "gpu_hold.py",
    "run_e3_subset_recovery.py", "run_e3_subset_completion.py", "run_e3_subset_scoring.py",
    "teacher_alg.py", "teacher_dymos.py", "run_teacher_inference_smoke.py",
    "measure_teacher_throughput.py", "run_teacher_devset.py", "run_teacher_devset_node.py",
    "build_teacher_review.py", "build_demo.py", "serve_demo.py", "merge_ckpt.py",
    "build_round1_mixture.py", "build_round2_mixture.py", "build_round3_mixture.py",
    "build_round4_mixture.py", "build_round5_mixture.py", "build_ft2_mixture.py",
    "score_camera.py", "scan_cuts.py", "rewrite_prompts_vbench.py", "reslice_5s.py",
    "finalize_5s.py", "inventory_5s.py", "slice_new_sources.py", "caption_5s_extra.py",
)] + ["tests/" + name for name in (
    "test_score_merge_cpu.py", "test_monitor_val_orchestration.py", "test_wmb_final_orchestration.py", "test_analyze_merge_metrics.py",
    "test_export_experiment_snapshot.py", "test_report_merge_results.py", "test_score_pool.py",
    "test_run_teacher_probe.py", "test_score_teacher_probe.py",
    "test_audit_teacher_labels.py", "test_teacher_mechanism_probe.py", "test_teacher_caption_probe.py",
    "test_reslice_namespace.py", "test_prepare_teacher_eligibility.py", "test_teacher_training_smoke.py", "test_serve_label_review.py",
    "test_e1_devset.py", "test_e1_jobs.py", "test_e1_score.py", "test_summarize_e3.py", "test_e1_seeds.py",
    "test_e3_subset.py", "test_e3_subset_node.py", "test_e3_subset_score.py",
    "test_e3_subset_recovery.py", "test_e3_subset_completion.py", "test_e3_subset_scoring_driver.py",
    "test_teacher_alg.py", "test_teacher_dymos.py", "test_teacher_inference_smoke.py",
    "test_teacher_throughput.py", "test_teacher_devset.py", "test_teacher_devset_node.py",
    "test_e3_train.py", "test_recaption_e3.py", "test_e3_data.py",
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
