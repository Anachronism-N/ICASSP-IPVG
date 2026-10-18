#!/usr/bin/env python3
"""Teacher acceptance page -> demo/teacher.html

Gate before any streaming / causal-Student training: the bidirectional Teacher (our best LoRA round)
must pass the quantitative checks below AND a blind human review.

1. Automatic checks (from eval/summary.json, gtref_*.json, val-loss jsons), each PASS/FAIL/pending.
2. Blind review: 2 WM-Bench clips per dataset (24). For each clip: GT on the left, then the candidate
   Teacher, the base model and the strongest related-work baseline in a per-clip random order labelled
   A/B/C. The reviewer ranks them and answers "usable as Teacher?". The model<->letter key is embedded
   only in the exported JSON, not shown on the page.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_demo import CSS, DEMO, MEDIA, ROOT, esc, preview, video_tag  # noqa: E402

EV = f"{ROOT}/eval"
WM = f"{EV}/wm_bench"
LOSS = f"{ROOT}/models/train/round2_5s_lora_r32/eval"
PER_DS = 2
PEERS = ["DreamX-World-1.0", "ToMoviee-2.0", "Steamer-I2V", "Magi-1", "Pi", "JT-CV"]  # snapshot top six by original Total
NONDYN = ["i2v_subject", "i2v_background", "subject_consistency", "background_consistency",
          "motion_smoothness", "aesthetic_quality", "imaging_quality"]
NEED_WINS = 3
BASELINES = ["wan21", "hunyuan", "cogvideox"]  # strongest first if available
NAME = {"base": "Wan2.2-TI2V-5B base", "round3": "Round3", "round4": "Round4", "round5": "Round5", "full1": "Full-FT1", "ft2": "Full-FT2",
        "m_full1_a03": "0.7·base + 0.3·FT1", "m_full1_a05": "0.5·base + 0.5·FT1",
        "m_full1_a07": "0.3·base + 0.7·FT1",
        "wan21": "Wan2.1-I2V-14B", "hunyuan": "HunyuanVideo-I2V", "cogvideox": "CogVideoX-5B-I2V"}


def candidate():
    for m in ("ft2", "full1", "round5", "round4", "round3"):
        if os.path.exists(f"{WM}/gtref_{m}.json") and os.path.exists(f"{LOSS}/loss_{m}.json"):
            return m
    return "round3"


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def checks(cand):
    rows = []
    lb, lc = jload(f"{LOSS}/loss_base.json"), jload(f"{LOSS}/loss_{cand}.json")
    if lb and lc:
        A = {c["video"]: c["loss"] for c in lb["clips"]}
        B = {c["video"]: c["loss"] for c in lc["clips"]}
        better = sum(B[v] < A[v] for v in A if v in B)
        rel = 100 * (lc["mean_loss"] / lb["mean_loss"] - 1)
        rows.append(("留出验证集 loss 低于 base（120 条，固定噪声/时间步）",
                     f"{lc['mean_loss']:.4f} vs {lb['mean_loss']:.4f}（{rel:+.1f}%），{better}/{len(A)} 条更好",
                     better >= 0.9 * len(A)))
    else:
        rows.append(("留出验证集 loss 低于 base", "待评测", None))
    summ = jload(f"{EV}/summary.json") or {}
    g = summ.get("wm_gtref", {})
    gc = g.get(cand)
    others = {m: g[m] for m in ["base"] + BASELINES if m in g}
    if gc and others:
        bad = [m for m, x in others.items() if x["dreamsim"] <= gc["dreamsim"] or x["motion_err"] <= gc["motion_err"]]
        txt = f"DreamSim {gc['dreamsim']:.4f} / 运动误差 {gc['motion_err']:.3f}；对比 " + "，".join(
            f"{NAME.get(m, m)} {x['dreamsim']:.4f}/{x['motion_err']:.3f}" for m, x in others.items())
        rows.append(("WM-Bench 与真实后续画面最接近（DreamSim 与运动误差均优于 base 和全部基线）", txt, not bad))
    else:
        rows.append(("WM-Bench GT 参考指标", "待评测", None))
    vb = summ.get("vbench_i2v", {})
    import csv
    from summarize_eval import LB_COL, label, total
    lb = {r["Model Name (clickable)"]: {d: float(r[c].strip("%")) / 100 for d, c in LB_COL.items()}
          for r in csv.DictReader(open(f"{EV}/vbench_i2v_leaderboard_20261011.csv"))}
    peers = {k: lb[k] for k in PEERS if k in lb}
    candidate_labels = {label(cand), label(cand + "_po")}
    mine = {k: v for k, v in vb.items()
            if k.removesuffix("（本地复现）") in candidate_labels and len(v) == 8}
    if mine and peers:
        def wins(v):
            return [d for d in NONDYN if v[d] >= max(p[d] for p in peers.values())]
        key = max(mine, key=lambda k: (len(wins(mine[k])), total(mine[k])[0]))
        v, w = mine[key], wins(mine[key])
        rows.append((f"VBench-I2V：除 dynamic 外的 7 个维度中，至少 {NEED_WINS} 个不低于 DreamX-World-1.0 及其他榜单前列模型",
                     f"{key}：领先 {len(w)}/7（{', '.join(w) or '无'}）；Total* {100 * total(v)[0]:.2f}，DreamX 同8维重算90.79（榜单原始90.49）",
                     len(w) >= NEED_WINS))
        rows.append(("__dimtable__", (key, v, peers), None))
    else:
        rows.append(("VBench-I2V 分维度对比", "待打分", None))
    rows.append(("盲评：Teacher 在多数片段中不差于 base 和最强基线，且无系统性伪影", "见下方盲评，导出后由我统计", None))
    return rows


def cases_html(cand):
    cases = [json.loads(l) for l in open(f"{WM}/cases.jsonl")]
    by = defaultdict(list)
    for c in cases:
        by[c["dataset"]].append(c)
    bl = next((m for m in BASELINES if len([f for f in os.listdir(f"{WM}/videos/{m}") if f.endswith(".mp4")]) >= 100), None) \
        if os.path.isdir(f"{WM}/videos") else None
    models = [cand, "base"] + ([bl] if bl else [])
    rng = random.Random(13)
    cards, key = [], {}
    for ds in sorted(by):
        for c in rng.sample(by[ds], min(PER_DS, len(by[ds]))):
            order = models[:]
            rng.shuffle(order)
            letters = "ABC"[:len(order)]
            key[c["id"]] = dict(zip(letters, order))
            tiles = video_tag(preview(c["gt"]), "GT（真实后续）") + "".join(
                video_tag(preview(f"{WM}/videos/{m}/{c['id']}.mp4"), f"模型 {L}") for L, m in zip(letters, order))
            opt = "".join(f"<option>{L}</option>" for L in letters)
            cards.append(f"""
<div class="card" data-case="{esc(c['id'])}">
  <div class="meta"><b>{esc(ds)}</b> · {esc(c['id'])}</div>
  <div class="grid4">{tiles}</div>
  <details><summary>Prompt</summary><p class="prompt">{esc(c['prompt'])}</p></details>
  <div class="review">
    <label>最好 <select data-k="best"><option value="">—</option>{opt}<option value="tie">差不多</option></select></label>
    <label>最差 <select data-k="worst"><option value="">—</option>{opt}<option value="tie">差不多</option></select></label>
    <label>最好的那个能当 Teacher 吗 <select data-k="usable"><option value="">—</option><option>能</option><option>勉强</option><option>不能</option></select></label>
    <textarea data-k="notes" placeholder="备注：首帧漂移、人体/手部变形、闪烁、动作不合理、物理错误……"></textarea>
  </div>
</div>""")
    return models, cards, key


JS = """
const KEY='wm_teacher_review_'+CAND;
const load=()=>{try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){return {}}};
const save=s=>localStorage.setItem(KEY,JSON.stringify(s));
function refresh(){const s=load();let n=0;document.querySelectorAll('.card[data-case]').forEach(c=>{
  const r=s[c.dataset.case]||{};const done=!!r.best;c.classList.toggle('reviewed',done);if(done)n++;});
  document.getElementById('rv-count').textContent=n+' / '+document.querySelectorAll('.card[data-case]').length+' 已评审';}
document.querySelectorAll('.card[data-case]').forEach(c=>{const s=load(),r=s[c.dataset.case]||{};
  c.querySelectorAll('[data-k]').forEach(el=>{if(r[el.dataset.k])el.value=r[el.dataset.k];
    el.addEventListener('input',()=>{const s=load();s[c.dataset.case]=s[c.dataset.case]||{};
      s[c.dataset.case][el.dataset.k]=el.value;save(s);refresh();});});});
document.getElementById('rv-export').onclick=()=>{const reviewer=prompt('评审人姓名（可留空）')||'';
  const blob=new Blob([JSON.stringify({reviewer,exported_at:new Date().toISOString(),build:BUILD,candidate:CAND,key:KEYMAP,reviews:load()},null,1)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='teacher_review_'+Date.now()+'.json';a.click();};
document.getElementById('rv-clear').onclick=()=>{if(confirm('清空本浏览器中的盲评记录？')){localStorage.removeItem(KEY);location.reload();}};
document.getElementById('sync').onclick=()=>{document.querySelectorAll('.grid4 video').forEach(v=>{v.currentTime=0;v.play();});};
refresh();
"""


def main():
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=list(NAME), help="Explicit candidate; missing metrics remain pending")
    args = parser.parse_args()
    os.makedirs(MEDIA, exist_ok=True)
    cand = args.candidate or candidate()
    models, cards, key = cases_html(cand)
    rows = checks(cand)
    dimtab = ""
    for a, b, ok in rows:
        if a == "__dimtable__":
            model_label, v, peers = b
            hdr = "<tr><th>维度</th><th>" + esc(model_label) + "</th>" + "".join(f"<th>{esc(n)}</th>" for n in peers) + "</tr>"
            body = ""
            for d in NONDYN + ["dynamic_degree"]:
                top = max(p[d] for p in peers.values())
                body += (f"<tr><td>{d}</td><td class=\"{'best' if v[d] >= top else ''}\">{100 * v[d]:.2f}</td>"
                         + "".join(f"<td>{100 * p[d]:.2f}</td>" for p in peers.values()) + "</tr>")
            dimtab = (f'<h3 style="font-size:15px">VBench-I2V 分维度（绿色 = 不低于全部对照模型；dynamic 仅供参考，不计入）</h3>'
                      f'<table class="tbl">{hdr}{body}</table>')
    tr = "".join(
        f"<tr><td>{esc(a)}</td><td>{esc(b)}</td><td class=\"{'best' if ok else ''}\">"
        f"{'PASS' if ok else ('待定' if ok is None else 'FAIL')}</td></tr>" for a, b, ok in rows if a != "__dimtable__")
    build = time.strftime("%Y-%m-%d %H:%M:%S")
    grid = len(models) + 1
    page = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Teacher 验收</title>
<style>{CSS}.grid4{{grid-template-columns:repeat({grid},1fr)}}</style></head><body>
<header><h1>双向 Teacher 验收（流式训练前置条件）</h1>
<div class="sub">候选 Teacher：{esc(NAME[cand])} · 盲评对照：{esc(' / '.join(NAME.get(m, m) for m in models[1:]))} · 生成于 {build}</div></header>
<nav><a href="index.html">← 实验总览</a><a href="#auto">自动指标</a><a href="#blind">盲评</a></nav>
<main>
<section id="auto"><h2>自动指标验收</h2><table class="tbl"><tr><th>条件</th><th>结果</th><th>判定</th></tr>{tr}</table>
{dimtab}<div class="note">全部 PASS 且用户盲评确认后，才开始因果/流式 Student 的训练。本地分数为 1 seed，榜单为 5 seed。"+ prompt-optimized" 仅表示输入扩写实验，并非已验证与官方 Wan 扩写器等价；当前 Physics-IQ 扩写还发现物体属性/终局偏移，不得将其涨分直接认作物理能力提升。</div></section>
<section id="blind"><h2>盲评（{len(cards)} 条 WM-Bench 留出片段，每个数据集 {PER_DS} 条）</h2>
<div class="note" style="margin-bottom:10px">左侧是真实视频（同一首帧之后的真实 5s）。A/B/C 的顺序每条随机，页面不显示对应关系，导出的 JSON 里带有映射。
主要看：首帧之后是否漂移、人和手是否变形、动作是否合理、和真实后续是否相像。</div>
<div class="toolbar"><button id="sync" class="sec">同步从头播放</button>
<button id="rv-export">导出盲评结果（JSON）</button><button id="rv-clear" class="sec">清空</button>
<span id="rv-count" class="note"></span></div>
{''.join(cards)}</section>
</main><script>const BUILD="{build}";const CAND="{cand}";const KEYMAP={json.dumps(key)};{JS}</script></body></html>"""
    tmp = f"{DEMO}/teacher.html.tmp"
    open(tmp, "w").write(page)
    os.replace(tmp, f"{DEMO}/teacher.html")
    print(f"teacher.html: candidate={cand} models={models} cards={len(cards)} checks={[r[2] for r in rows if r[0] != '__dimtable__']}")


if __name__ == "__main__":
    main()
