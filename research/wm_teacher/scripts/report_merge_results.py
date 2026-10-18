#!/usr/bin/env python3
"""Write a new auditable Chinese report only after the strict merge coverage gate.

Raw and prompt-optimized protocols are never pooled. Report completion is not
Teacher acceptance, human approval, or authorization for Student/FT3 training.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import merge_final_gate
from score_merge_cpu import BENCH, finite, read_json, require, validate_gtref, validate_physics
from summarize_eval import DIMS, LB_COL, label, total
from build_teacher_review import BASELINES, NONDYN, NEED_WINS, PEERS
from val_merge_score import LOSS_DIR, MODELS, fixed_base, validate_loss
from wmb_merge_score import BENCH as WMB_BENCH, original_stems, validate_result

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ("base", "round4", "full1", "ft2")
ANALYSIS = "eval/diag/merge_all_analysis_20261018.json"
MIGRATION = "eval/diag/migration_checksums.json"
CHECKPOINTS = {
    "round4": "models/train/round4_aes_lora_r32/run1/step-1644.safetensors",
    "full1": "models/train/full1_5b/run1/step-2203.safetensors",
    "ft2": "models/train/ft2_5b/run1/step-2158.safetensors",
    **{m: "models/train/merge/" + m.removeprefix("m_") + ".safetensors" for m in MODELS},
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values):
    require(values and all(finite(v) for v in values), "Missing/non-finite numeric values")
    result = math.fsum(values) / len(values)
    require(finite(result), "Non-finite aggregate")
    return result


def vbench_values(result, dim, expected):
    require(isinstance(result, dict) and dim in result, "Missing VBench dimension: " + dim)
    row = result[dim]
    require(isinstance(row, list) and len(row) == 2 and finite(row[0]) and
            isinstance(row[1], list), "Invalid/non-finite VBench dimension: " + dim)
    require(all(isinstance(x, dict) and isinstance(x.get("video_path"), str) and
                finite(x.get("video_results")) for x in row[1]), "Non-finite VBench per-case score")
    actual = [Path(x["video_path"]).name for x in row[1]]
    require(len(actual) == len(expected) and set(actual) == expected,
            "Wrong/incomplete VBench case coverage: " + dim)
    # Preserve the original metric values, especially imaging_quality; do not clamp/rescale.
    return row[0]


def teacher_checks(rows, peers):
    base = rows["base"]
    compared = [m for m in ("base", *BASELINES) if finite(rows[m].get("dreamsim")) and
                finite(rows[m].get("motion_err"))]
    comparisons = [rows[m] for m in compared]
    result = {}
    for model in MODELS:
        row = rows[model]
        better = sum(b < a for a, b in zip(base["clip_losses"], row["clip_losses"]))
        wins = [d for d in NONDYN if row["vbench"][d] >= max(p[d] for p in peers.values())]
        checks = {"loss": better >= 0.9 * 120,
                  "gtref": all(row["dreamsim"] < x["dreamsim"] and
                               row["motion_err"] < x["motion_err"] for x in comparisons),
                  "vbench": len(wins) >= NEED_WINS}
        result[model] = {"checks": checks, "better": better, "wins": wins,
                         "automatic": "PASS" if all(checks.values()) else "FAIL",
                         "human_review": "pending", "compared": compared}
    return result


def pareto(rows):
    directions = {"vbench_total": 1, "dreamsim": -1, "physics": 1, "wmb": 1, "val": -1}
    eligible = {m: [directions[k] * row[k] for k in directions]
                for m, row in rows.items() if all(finite(row.get(k)) for k in directions)}
    return [m for m, point in eligible.items() if not any(
        other != m and all(a >= b for a, b in zip(candidate, point)) and
        any(a > b for a, b in zip(candidate, point))
        for other, candidate in eligible.items())]


def load_report(root, gate=None, physics_bench=BENCH / "physics-IQ-benchmark", wmb_bench=WMB_BENCH):
    (gate or merge_final_gate.check_complete)(root)
    sources, paths = {}, {}

    def track(path, public_name=None):
        name = public_name or path.relative_to(root).as_posix()
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Source names must be safe relative paths")
        before = sha256(path)
        require(name not in sources or sources[name] == before, "Source changed while reading: " + name)
        sources[name], paths[name] = before, path
        return path

    def load(path):
        track(path)
        data = read_json(path)
        require(sha256(path) == sources[path.relative_to(root).as_posix()], "Source changed during read")
        return data

    cases = {}
    for suite, count in (("wm_bench", 108), ("vbench_i2v", 355)):
        path = track(root / "eval" / suite / "cases.jsonl")
        with path.open() as handle:
            cases[suite] = [json.loads(line) for line in handle]
        ids = [c["id"] for c in cases[suite]]
        require(len(ids) == len(set(ids)) == count, "Wrong original case manifest: " + suite)
    description = track(physics_bench / "descriptions/descriptions_original.csv",
                        "Physics-IQ/descriptions/descriptions_original.csv")
    with description.open(newline="") as handle:
        descriptions = list(csv.DictReader(handle))
    names = [r["generated_video_name"] for r in descriptions if "take-2" not in r["scenario"]]
    scenarios = {n.rsplit("_", 1)[-1] for n in names}
    require(len(names) == len(set(names)) == 198 and len(scenarios) == 66 and
            all(sum(n.rsplit("_", 1)[-1] == s for n in names) == 3 and
                {n.split("_")[1] for n in names if n.rsplit("_", 1)[-1] == s} ==
                {"perspective-left", "perspective-center", "perspective-right"} for s in scenarios),
            "Wrong original Physics three-view groups")
    track(wmb_bench / "worldmodelbench.json", "WorldModelBench/worldmodelbench.json")
    stems = original_stems(wmb_bench)
    base = fixed_base(root)
    track(root / "training_metadata/round2/round2_val.csv")
    rows = {}
    for model in (*CONTROLS, *MODELS, *BASELINES):
        row = {"model": model}
        for suite, key in (("wm_bench", "wm"), ("vbench_i2v", "vbench")):
            score_paths = {d: root / "eval" / suite / "scores" /
                           (model + "__" + d + "_eval_results.json") for d in DIMS}
            exists = [p.exists() or p.is_symlink() for p in score_paths.values()]
            required = model in MODELS or (suite == "wm_bench" and model in CONTROLS)
            require(not required or all(exists), "Required raw scores missing: " + suite + "/" + model)
            require(not any(exists) or all(exists), "Partial raw scores: " + suite + "/" + model)
            values = None
            if all(exists):
                values = {d: vbench_values(load(p), d, {c["id"] + "-0.mp4" for c in cases[suite]
                                                       if d in c.get("dims", DIMS)})
                          for d, p in score_paths.items()}
            row[key] = values
            row[key + "_total"] = total(values)[0] if values else None
            require(row[key + "_total"] is None or finite(row[key + "_total"]), "Non-finite Total*")
        gt_path = root / "eval/wm_bench" / ("gtref_" + model + ".json")
        row.update({"dreamsim": None, "dreamsim_last": None, "motion_err": None})
        if model not in BASELINES or gt_path.exists() or gt_path.is_symlink():
            track(gt_path)
            validate_gtref(gt_path, {c["id"] for c in cases["wm_bench"]})
            gt = load(gt_path)
            row.update({key: mean([v[field] for v in gt.values()]) for key, field in
                        (("dreamsim", "dreamsim"), ("dreamsim_last", "dreamsim_last"), ("motion_err", "motion_ratio"))})
        loss_path = root / LOSS_DIR / ("loss_" + model + ".json")
        row["val"], row["clip_losses"] = None, None
        if model not in BASELINES or loss_path.exists() or loss_path.is_symlink():
            track(loss_path)
            loss = validate_loss(loss_path, base)
            row["val"] = loss["mean_loss"]
            row["clip_losses"] = [c["loss"] for c in loss["clips"]]
        pq = root / "eval/physiq/results/physics-IQ-benchmark/results" / (model + "-op-run_01_metrics.json")
        pc = pq.with_name(model + "-op-run_01.csv")
        row["physics"] = None
        if model not in BASELINES or pq.exists() or pc.exists() or pq.is_symlink() or pc.is_symlink():
            track(pq)
            track(pc)
            validate_physics(pq, pc, scenarios)
            metrics = load(pq)
            require(all(finite(v) for v in metrics.values()), "Non-finite raw Physics metrics")
            require(0 <= metrics["final_score_orig"] <= 1, "Original Physics fraction outside [0, 1]")
            row["physics"] = metrics["final_score_orig"]
        wb_path = root / "eval/wmb/results" / (model + ".json")
        row["wmb"], row["wmb_parts"] = None, None
        if model in MODELS or wb_path.exists() or wb_path.is_symlink():
            track(wb_path)
            wb = validate_result(wb_path, stems)
            acc = wb["accs"]
            parts = (mean(acc["instruction"]), 5 * sum(acc["physical_laws"]) / 1750,
                     2 * sum(acc["common_sense"]) / 700)
            row["wmb_parts"], row["wmb"] = parts, sum(parts)
        rows[model] = row
    lb_path = track(root / "eval/vbench_i2v_leaderboard_20261011.csv")
    with lb_path.open(newline="") as handle:
        records = [r for r in csv.DictReader(handle) if r["Model Name (clickable)"] in PEERS]
    require(len(records) == len(PEERS) and {r["Model Name (clickable)"] for r in records} == set(PEERS),
            "All original Teacher PEERS are required")
    peers = {r["Model Name (clickable)"]: {d: float(r[LB_COL[d]].strip("%")) / 100 for d in DIMS}
             for r in records}
    require(all(finite(v) for p in peers.values() for v in p.values()), "Non-finite leaderboard score")
    analysis = load(root / ANALYSIS)
    require(analysis["models"] == list(MODELS) and set(analysis["references"]) == set(CONTROLS) and
            analysis["gtref"]["n_cases"] == 108 and set(analysis["gtref"]["case_ids"]) ==
            {c["id"] for c in cases["wm_bench"]} and analysis["physics"]["n_scenarios"] == 66 and
            set(analysis["physics"]["scenario_ids"]) == scenarios and
            analysis["physics"]["views_per_scenario"] == 3 and analysis["seed"] == 20261018 and
            analysis["bootstrap"] == 5000, "Wrong/incomplete fixed paired analysis")
    require(analysis["physics"]["formula_validation"]["iqtable_point_matches_saved_metric_abs_tolerance"] <= 1e-8 and
            analysis["physics"]["formula_validation"]["iqtable_resamples_checked_per_model"] >= 3,
            "Missing strict original Physics formula validation")
    paired_sources = {"eval/wm_bench/cases.jsonl", "Physics-IQ/descriptions/descriptions_original.csv"}
    for model in (*MODELS, *CONTROLS):
        paired_sources.update(("eval/wm_bench/gtref_" + model + ".json",
                              "eval/physiq/results/physics-IQ-benchmark/results/" + model + "-op-run_01.csv",
                              "eval/physiq/results/physics-IQ-benchmark/results/" + model + "-op-run_01_metrics.json"))
        for metric, key in (("dreamsim", "dreamsim"), ("dreamsim_last", "dreamsim_last"), ("motion_ratio", "motion_err")):
            value = analysis["gtref"]["observed"][metric][model]
            require(finite(value) and math.isclose(value, rows[model][key], abs_tol=1e-10), "Stale paired GT point")
        value = analysis["physics"]["observed"][model]
        require(finite(value) and math.isclose(value, 100 * rows[model]["physics"], abs_tol=1e-8), "Stale paired Physics point")
    require(set(analysis["source_sha256"]) == paired_sources and
            all(analysis["source_sha256"][name] == sources[name] for name in paired_sources), "Stale paired source hashes")
    pairs = [m + " minus " + c for m in MODELS for c in CONTROLS] + [
        MODELS[i] + " minus " + MODELS[j] for i in range(3) for j in range(i + 1, 3)]
    require(set(analysis["gtref"]["paired"]) == {"dreamsim", "dreamsim_last", "motion_ratio"},
            "Missing paired GT metrics")
    comparisons = [(metric, entries, analysis["gtref"]["observed"][metric])
                   for metric, entries in analysis["gtref"]["paired"].items()] + [
                       ("physics", analysis["physics"]["paired"], analysis["physics"]["observed"])]
    for metric, entries, observed in comparisons:
        require(set(entries) == set(pairs), "Incomplete paired comparisons")
        for pair, value in entries.items():
            require(finite(value["delta"]) and isinstance(value["ci95"], list) and len(value["ci95"]) == 2 and
                    all(finite(v) for v in value["ci95"]) and value["ci95"][0] <= value["ci95"][1], "Non-finite paired CI")
            left, right = pair.split(" minus ")
            require(math.isclose(value["delta"], observed[left] - observed[right], abs_tol=1e-8),
                    "Paired delta differs from observed points: " + metric)
    inventory = load(root / MIGRATION)
    checkpoints = {}
    for model, name in CHECKPOINTS.items():
        matches = [r for r in inventory["files"] if r.get("relative_path") == name]
        require(len(matches) == 1 and re.fullmatch("[0-9a-f]{64}", matches[0].get("sha256", "")),
                "Missing original migration checkpoint hash: " + model)
        require((root / name).is_file() and (root / name).stat().st_size == matches[0]["size"],
                "Missing/size-changed migrated checkpoint: " + model)
        checkpoints[name] = matches[0]["sha256"]
    for name in ("scripts/report_merge_results.py", "scripts/merge_final_gate.py", "scripts/score_merge_cpu.py",
                 "scripts/val_merge_score.py", "scripts/wmb_merge_score.py", "scripts/summarize_eval.py",
                 "scripts/build_teacher_review.py", "scripts/analyze_merge_metrics.py",
                 "eval/jobs_merge_all.jsonl", "eval/jobs_ft2_all.jsonl"):
        track(root / name)
    for model in MODELS:
        track(root / "eval/diag" / ("decode_" + model + "_20261018.json"))
    (gate or merge_final_gate.check_complete)(root)
    require(all(sha256(paths[name]) == digest for name, digest in sources.items()), "Source changed before report")
    return {"rows": rows, "peers": peers, "analysis": analysis, "source_sha256": sources,
            "checkpoints": checkpoints, "teacher": teacher_checks(rows, peers),
            "pareto": pareto({m: rows[m] for m in MODELS})}


def number(value, percent=False, places=4):
    require(value is None or finite(value), "Non-finite report value")
    if value is None:
        return "未测"
    scaled = value * (100 if percent else 1)
    require(finite(scaled), "Non-finite scaled report value")
    return f"{scaled:.{places}f}" + ("%" if percent else "")


def table(headers, rows):
    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return ["| " + " | ".join(map(safe, headers)) + " |", "|" + "---|" * len(headers)] + [
        "| " + " | ".join(map(safe, row)) + " |" for row in rows]


def render_report(report):
    rows, teachers, analysis = report["rows"], report["teacher"], report["analysis"]
    auto = "PASS" if any(x["automatic"] == "PASS" for x in teachers.values()) else "FAIL"
    md = ["# 三个权重插值模型：完整评测与可复核结论", "",
          "- 完整覆盖门禁：PASS（严格 merge_final_gate；不是质量门禁）。",
          f"- Teacher 自动条件：{auto}（按每个候选的全部自动条件判定）。",
          "- 人工盲评 / 用户门禁：pending，尚未完成；不得宣称 Teacher 通过或 SOTA。",
          "- 报告完成不授权因果/流式 Student、FT3 训练。", "",
          "## 协议与单位", "",
          "所有本地比较只用原始 raw prompt；不混入 _po。VBench 公共集355条，WM108条，Physics198视角/66完整场景，WMB350条，固定验证120条/121帧/5时间步。",
          "VBench/WM Total* 严格调用原 summarize_eval.total：排除 camera_motion；dynamic 权重0.5，仍计入 Total*；七个 non-dynamic 单项用于榜首操作性条件。不是官方原始 Total。",
          "imaging_quality 原值/归一化保持原脚本处理，不额外 clamp 或缩放；表中 VBench/WM 含该项统一乘100显示，异常幅度不隐藏。Physics 原始分数0–1与100×百分比分开；WMB约0–10，loss/DreamSim/motionErr非百分数。未测不补零、不推断。", ""]
    for key, title in (("vbench", "公开 VBench-I2V raw"), ("wm", "领域内 WM-Bench raw")):
        md += ["## " + title, ""]
        order = ["Total*", "I2V*", "Quality*"] + list(NONDYN) + ["dynamic_degree（不计head wins）"]
        values = []
        for model, row in rows.items():
            dv = row[key]
            totals = total(dv) if dv else (None, None, None)
            values.append([label(model) + " [" + model + "]"] + [number(v, True, 2) for v in totals] +
                          [number(dv[d] if dv else None, True, 2) for d in NONDYN + ["dynamic_degree"]])
        md += table(["模型"] + order, values) + [""]
    md += ["## GT参考、物理基准与验证loss", ""]
    md += table(["模型", "DreamSim↓", "DreamSim末帧↓", "motionErr↓", "Physics raw↑", "Physics %↑",
                 "WMB /10↑", "指令 /3", "物理 /5", "常识 /2", "fixed120 loss↓"], [
        [label(m) + " [" + m + "]", number(r["dreamsim"]), number(r["dreamsim_last"]), number(r["motion_err"]),
         number(r["physics"], places=6), number(r["physics"], True, 2), number(r["wmb"], places=3)] +
        [number(v, places=3) for v in (r["wmb_parts"] or (None, None, None))] + [number(r["val"], places=6)]
        for m, r in rows.items()]) + ["", "motionErr 为原GTref motion_ratio字段（绝对log运动量误差），不解释为运动质量或 dynamic 单项。", ""]
    md += ["## Teacher 操作性自动条件与人工pending", ""]
    md += table(["候选", "loss胜过base条数 /120（≥108）", "GT优于base及已有相关基线", "实际GT对照", "head wins /7（≥3）", "领先维度", "全部自动", "人工"], [
        [m, str(t["better"]), "PASS" if t["checks"]["gtref"] else "FAIL", ", ".join(t["compared"]), str(len(t["wins"])),
         ", ".join(t["wins"]) or "无", t["automatic"], "pending"] for m, t in teachers.items()]) + [""]
    md += table(["non-dynamic维度"] + list(MODELS) + list(PEERS), [
        [d] + [number(rows[m]["vbench"][d], True, 2) for m in MODELS] +
        [number(report["peers"][p][d], True, 2) for p in PEERS] for d in NONDYN]) + ["",
        "head wins仅指候选维度≥上述六个原榜单头部该维度最大值，不代表配对显著性、总体胜出或SOTA。本地固定1 seed与榜单5 seed协议不同；人工盲评始终独立pending。", "",
        "## 描述性权衡与Pareto", "",
        "Teacher GT条件沿用原脚本：比较base及已有相关基线；未测相关基线不扩大为新增评分任务，也不宣称优于未测者。", ""]
    directions = (("vbench_total", True, "VBench Total*"), ("dreamsim", False, "DreamSim"),
                  ("physics", True, "Physics raw"), ("wmb", True, "WMB"), ("val", False, "验证loss"))
    for metric, higher, title in directions:
        available = {m: rows[m][metric] for m in MODELS if finite(rows[m].get(metric))}
        best = (max if higher else min)(available.values())
        winners = [m for m, v in available.items() if v == best]
        md.append(f"- 三插值候选的{title}{'最高' if higher else '最低'}：{', '.join(winners)}；仅当前固定输出描述。")
    excluded = [m for m in MODELS if not all(finite(rows[m].get(k)) for k, _, _ in directions)]
    md += ["- Pareto非支配集合（仅三个完整raw插值候选）：" + ", ".join(report["pareto"]) + "。",
           "- 比较方向：VBench Total*↑、DreamSim↓、Physics↑、WMB↑、val↓；不以dynamic单项作为择优依据。",
           "- 未齐这五项而排除的插值候选：" + (", ".join(excluded) or "无") + "；controls/相关baseline只作表格对照，不参与候选Pareto。",
           "- 不设任意跨基准权重，不据单seed分差宣称通用最优；不同指标存在描述性权衡。", "",
           "## 固定seed配对不确定性（已有分析，不重新估计）", "",
           "固定输出seed；同一案例/三视角完整场景配对 percentile 95% CI；bootstrap=" + str(analysis["bootstrap"]) +
           "，分析RNG seed=" + str(analysis["seed"]) + "。",
           "CI不包含来源cluster、生成seed、训练或judge变异；多重比较未校正，不做因果结论或跨seed外推。Physics差值已是百分比点（pp），不再次乘100。", ""]
    for metric, entries in [(k, v) for k, v in analysis["gtref"]["paired"].items()] + [("Physics（pp）", analysis["physics"]["paired"])]:
        md += ["### " + metric, ""] + table(["候选 minus 对照", "Δ", "95% CI"], [
            [pair, number(v["delta"], places=6), "[" + number(v["ci95"][0], places=6) + ", " +
             number(v["ci95"][1], places=6) + "]"] for pair, v in entries.items()]) + [""]
    md += ["## 可复核来源", "", "只列相对路径和SHA256，不导出prompt、私有日志、节点地址、凭据或人工身份。source_sha256是本次报告输入快照；checkpoint下表是原迁移清单hash，当前仅复查存在/大小，未重新计算10GB checkpoint hash；base外部预训练权重原清单未记录，不能造hash。", "",
           "### source_sha256", ""] + table(["相对来源", "SHA256"], sorted(report["source_sha256"].items()))
    md += ["", "### model checkpoint 原migration hashes", ""] + table(["checkpoint相对路径", "原SHA256"], sorted(report["checkpoints"].items()))
    return "\n".join(md) + "\n"


def write_no_clobber(path, content):
    require(path.suffix.lower() == ".md", "Output must be a new Markdown (.md) path")
    require(not path.exists() and not path.is_symlink(), "Refusing to overwrite existing output")
    require(path.parent.is_dir(), "Output parent directory must already exist")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except FileExistsError as error:
        raise RuntimeError("Output appeared concurrently; existing output preserved") from error
    finally:
        if temporary is not None:
            temporary.unlink()


def generate_report(root, out, gate=None, **kwargs):
    require(not out.exists() and not out.is_symlink(), "Refusing to overwrite existing output")
    report = load_report(root, gate=gate, **kwargs)
    write_no_clobber(out, render_report(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True, help="New Markdown path; no overwrites")
    args = parser.parse_args()
    try:
        generate_report(args.root.resolve(), args.out)
    except (OSError, RuntimeError, KeyError, ValueError, TypeError) as error:
        # Do not expose absolute paths or scorer/private source contents on failure.
        raise SystemExit("FINAL_REPORT_REFUSED: incomplete/invalid gate or sources, or output exists (" +
                         type(error).__name__ + ")") from None
    print("FINAL_REPORT_WRITTEN_TEACHER_HUMAN_PENDING", flush=True)


if __name__ == "__main__":
    main()
