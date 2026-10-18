#!/usr/bin/env python3
"""
Build the review demo site for the Wan2.2-TI2V world-model experiments -> demo/index.html

Sections: experiment status, round-2 training loss, held-out val loss (base / round1 / round2),
TI2V generation comparison (GT | base | round1 | round2, with a per-sample review form whose
results are kept in the browser and exported as JSON), and round-2 training-data samples.

Videos are transcoded once into small previews under demo/media/ (cached by source path+mtime),
so the page is light and never reads the shared fuse mount at view time.
Before the round-2 eval exists, the comparison falls back to the round-1 121-frame samples.

Serve:  python3 -m http.server 8765 --directory demo   (see scripts/serve_demo.sh)
"""
import csv
import glob
import hashlib
import html
import json
import os
import imageio_ffmpeg
import random
import subprocess
import time
from collections import Counter, defaultdict

ROOT = "/srv/wm_teacher"
DEMO = f"{ROOT}/demo"
MEDIA = f"{DEMO}/media"
R1 = f"{ROOT}/models/train/round1_lora_r32"
R2 = f"{ROOT}/models/train/round2_5s_lora_r32"
EVAL = f"{R2}/eval"
STATUS = f"{ROOT}/dataset/download_logs/status_round2.log"
FFMPEG = os.environ.get("FFMPEG") or imageio_ffmpeg.get_ffmpeg_exe()
TOTAL_STEPS = 6437
SAMPLES_PER_DATASET = 2
esc = html.escape


def preview(src):
    """Small H.264 preview in demo/media (relative URL), cached; None if the source is missing."""
    if not src or not os.path.exists(src):
        return None
    key = hashlib.md5(f"{src}:{os.path.getmtime(src)}".encode()).hexdigest()[:16]
    dst = f"{MEDIA}/{key}.mp4"
    if not os.path.exists(dst):
        tmp = dst + ".tmp.mp4"
        cmd = [FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", src, "-vf", "scale=480:-2",
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", tmp]
        r = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=300)
        if r.returncode != 0 or not os.path.exists(tmp):
            return None
        os.replace(tmp, dst)
    return f"media/{os.path.basename(dst)}"


def video_tag(url, label):
    if not url:
        return f'<div class="v"><div class="lbl">{esc(label)}</div><div class="missing">暂无</div></div>'
    return (f'<div class="v"><div class="lbl">{esc(label)}</div>'
            f'<video src="{url}" controls loop muted playsinline preload="metadata"></video></div>')


# ---------------------------------------------------------------- status
def status():
    try:
        last = open(STATUS).read().strip().splitlines()[-1]
    except Exception:
        return {"line": "无状态记录"}
    out = {"line": last, "time": last.split(" | ")[0]}
    import re
    m = re.search(r"(\d+)/(\d+) \[([^<\]]*)<([^,\]]*)", last)
    if m:
        out.update(step=int(m.group(1)), total=int(m.group(2)), elapsed=m.group(3), eta=m.group(4))
    m = re.search(r"new_shards=(\d+)", last)
    if m:
        out["new_shards"] = int(m.group(1))
    out["train_done"] = os.path.exists("/dockerdata/wm_logs/ROUND2_TRAIN_DONE")
    out["eval_done"] = "EVAL DONE" in (open(f"{ROOT}/dataset/download_logs/eval_round2.log").read()
                                       if os.path.exists(f"{ROOT}/dataset/download_logs/eval_round2.log") else "")
    return out


# ---------------------------------------------------------------- loss curve
def loss_series():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return []
    pts, offset = [], 0
    for run in sorted(glob.glob(f"{R2}/run*"), key=lambda p: int(p.rsplit("run", 1)[1])):
        last = 0
        for ev in sorted(glob.glob(f"{run}/tensorboard_log/*")):
            ea = EventAccumulator(ev, size_guidance={"scalars": 0})
            ea.Reload()
            if "loss" not in ea.Tags().get("scalars", []):
                continue
            for s in ea.Scalars("loss"):
                pts.append((offset + s.step, s.value))
                last = max(last, s.step)
        offset += last
    return pts


def loss_svg(pts, w=900, h=260, win=50):
    if len(pts) < 2:
        return '<div class="missing">暂无 loss 数据</div>'
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    sm = []
    acc = 0.0
    for i, y in enumerate(ys):  # trailing moving average
        acc += y
        if i >= win:
            acc -= ys[i - win]
        sm.append(acc / min(i + 1, win))
    step = max(1, len(pts) // 400)
    lo, hi = min(sm), max(sm)
    pad = (hi - lo) * 0.15 or 0.01
    lo, hi = lo - pad, hi + pad
    x_max = max(TOTAL_STEPS, xs[-1])

    def X(x):
        return 50 + (w - 70) * x / x_max

    def Y(y):
        return 15 + (h - 45) * (1 - (y - lo) / (hi - lo))
    raw = " ".join(f"{X(xs[i]):.1f},{Y(min(max(ys[i], lo), hi)):.1f}" for i in range(0, len(xs), step))
    line = " ".join(f"{X(xs[i]):.1f},{Y(sm[i]):.1f}" for i in range(0, len(xs), step))
    grid = ""
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        grid += (f'<line x1="50" x2="{w - 20}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="g"/>'
                 f'<text x="44" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.3f}</text>')
    for k in range(0, x_max + 1, 1000):
        grid += f'<text x="{X(k):.1f}" y="{h - 12}" text-anchor="middle">{k}</text>'
    return (f'<svg viewBox="0 0 {w} {h}" class="chart">{grid}'
            f'<polyline points="{raw}" class="raw"/><polyline points="{line}" class="sm"/>'
            f'<line x1="{X(xs[-1]):.1f}" x2="{X(xs[-1]):.1f}" y1="15" y2="{h - 30}" class="now"/></svg>'
            f'<div class="note">浅色：单步 loss（裁剪到坐标范围）；深色：{win} 步滑动平均；竖线：当前步。'
            f'最近 {win} 步均值 {sm[-1]:.4f}，共 {len(pts)} 个点。</div>')


# ---------------------------------------------------------------- val loss
def val_loss_table():
    names = [("base", "Wan2.2-TI2V-5B（无 LoRA）"), ("round1", "Round1（49 帧，step-14921）"),
             ("round2_mid", "Round2 中间（step-3000）"), ("round2", "Round2 最终")]
    res = {}
    for key, _ in names:
        p = f"{EVAL}/loss_{key}.json"
        if os.path.exists(p):
            res[key] = json.load(open(p))
    if not res:
        return ('<div class="missing">评测尚未运行：训练结束后由 watchdog 自动执行 '
                '<code>scripts/eval_round2.sh</code>（120 条留出 5s 片段，固定时间步与噪声）。</div>')
    any_r = next(iter(res.values()))
    t_fracs = any_r["t_fracs"]
    dsets = sorted({d for r in res.values() for d in r["per_dataset"]})
    best = min(r["mean_loss"] for r in res.values())
    head = ("<tr><th>模型</th><th>平均 val loss</th>" + "".join(f"<th>t={t}</th>" for t in t_fracs)
            + "".join(f"<th>{esc(d)}</th>" for d in dsets) + "</tr>")
    rows = ""
    for key, label in names:
        if key not in res:
            continue
        r = res[key]
        cls = ' class="best"' if r["mean_loss"] == best else ""
        rows += (f"<tr><td>{esc(label)}</td><td{cls}>{r['mean_loss']:.4f}</td>"
                 + "".join(f"<td>{v:.4f}</td>" for v in r["mean_loss_per_t"])
                 + "".join(f"<td>{r['per_dataset'].get(d, float('nan')):.4f}</td>" for d in dsets) + "</tr>")
    return (f'<table class="tbl">{head}{rows}</table><div class="note">n={any_r["n"]} 条留出片段，'
            f'{any_r["num_frames"]} 帧；越低越好，绿色为最优。</div>')


# ---------------------------------------------------------------- generation comparison
def comparison_cases():
    """[(case_id, dataset, prompt, {label: src_path})], source description"""
    cases = []
    if os.path.exists(f"{EVAL}/gen_base/cases.csv"):
        for row in csv.DictReader(open(f"{EVAL}/gen_base/cases.csv")):
            ds = row["dataset"]
            cases.append((f"r2:{ds}", ds, row["prompt"], {
                "GT（真实视频）": f"{EVAL}/gen_base/{ds}_gt.mp4",
                "Base": f"{EVAL}/gen_base/{ds}_gen.mp4",
                "Round1": f"{EVAL}/gen_round1/{ds}_gen.mp4",
                "Round2": f"{EVAL}/gen_round2/{ds}_gen.mp4",
            }))
        return cases, "Round2 评测：round2_val 留出片段，GT 首帧 + caption → 121 帧，seed 0"
    want = ["hoigen1m", "noxi", "matrix"]
    seen = {}
    for row in csv.DictReader(open(f"{ROOT}/training_metadata/round1_val.csv")):
        ds = row["video"].split("/")[2].split("_")[0]
        if ds in want and ds not in seen:
            seen[ds] = row
    for ds in want:
        if ds not in seen:
            continue
        row = seen[ds]
        cases.append((f"r1:{ds}", ds, row["prompt"], {
            "GT（真实视频，4s）": f"{ROOT}/{row['video']}",
            "Base": f"{R1}/inference_baseline_121f/{ds}_lora.mp4",
            "Round1": f"{R1}/inference_final_121f/{ds}_lora.mp4",
            "Round2": None,
        }))
    return cases, "Round2 评测尚未完成，暂用 Round1 的 121 帧推理结果（round1_val，seed 0）；Round2 列待评测后自动填充"


def comparison_html():
    cases, src_desc = comparison_cases()
    cards = []
    for cid, ds, prompt, vids in cases:
        tiles = "".join(video_tag(preview(p) if p else None, label) for label, p in vids.items())
        models = [k for k in vids if not k.startswith("GT")]
        opts = "".join(f'<option value="{esc(m)}">{esc(m)}</option>' for m in models)
        cards.append(f"""
<div class="card" data-case="{esc(cid)}">
  <div class="meta"><b>{esc(ds)}</b> · {esc(cid)}</div>
  <div class="grid4">{tiles}</div>
  <details><summary>Prompt</summary><p class="prompt">{esc(prompt)}</p></details>
  <div class="review">
    <label>最佳模型 <select data-k="best"><option value="">—</option>{opts}<option value="tie">差不多</option></select></label>
    <label>Round2 运动质量 <select data-k="motion"><option value="">—</option><option>好</option><option>一般</option><option>差</option></select></label>
    <label>与 Prompt 一致 <select data-k="prompt_follow"><option value="">—</option><option>好</option><option>一般</option><option>差</option></select></label>
    <label>首帧一致 <select data-k="first_frame"><option value="">—</option><option>好</option><option>一般</option><option>差</option></select></label>
    <textarea data-k="notes" placeholder="备注：伪影、闪烁、物体变形、动作不合理……"></textarea>
  </div>
</div>""")
    return src_desc, "".join(cards)


# ---------------------------------------------------------------- data samples
def data_html():
    tr = f"{ROOT}/training_metadata/round2/round2_train.csv"
    va = f"{ROOT}/training_metadata/round2/round2_val.csv"
    cnt_tr, cnt_va = Counter(), Counter()
    by_ds = defaultdict(list)
    for path, cnt in ((tr, cnt_tr), (va, cnt_va)):
        if not os.path.exists(path):
            continue
        for row in csv.DictReader(open(path)):
            ds = row["video"].split("/")[2].rsplit("_5s", 1)[0]
            cnt[ds] += 1
            if path == va:
                by_ds[ds].append(row)
    total = sum(cnt_tr.values()) or 1
    rows = "".join(f"<tr><td>{esc(d)}</td><td>{cnt_tr[d]}</td><td>{100 * cnt_tr[d] / total:.1f}%</td>"
                   f"<td>{cnt_va[d]}</td></tr>" for d in sorted(cnt_tr, key=lambda d: -cnt_tr[d]))
    table = (f'<table class="tbl"><tr><th>数据集</th><th>train</th><th>占比</th><th>val</th></tr>{rows}'
             f'<tr><td><b>合计</b></td><td><b>{sum(cnt_tr.values())}</b></td><td></td>'
             f'<td><b>{sum(cnt_va.values())}</b></td></tr></table>')
    rng = random.Random(0)
    cards = []
    for ds in sorted(by_ds):
        for row in rng.sample(by_ds[ds], min(SAMPLES_PER_DATASET, len(by_ds[ds]))):
            url = preview(f"{ROOT}/{row['video']}")
            cards.append(f'<div class="dcard">{video_tag(url, ds)}<p class="prompt small">{esc(row["prompt"])}</p></div>')
    return table, "".join(cards)


# ---------------------------------------------------------------- page
CSS = """
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;margin:0;background:#f4f5f7;color:#222}
header{background:#1f2937;color:#fff;padding:18px 28px}
header h1{margin:0;font-size:20px} header .sub{color:#cbd5e1;font-size:13px;margin-top:4px}
nav{position:sticky;top:0;background:#fff;border-bottom:1px solid #e5e7eb;padding:8px 28px;z-index:5}
nav a{margin-right:18px;color:#2563eb;text-decoration:none;font-size:14px}
main{padding:20px 28px;max-width:1500px}
section{background:#fff;border-radius:10px;padding:18px 22px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
h2{font-size:17px;margin:0 0 12px}
.kpis{display:flex;gap:14px;flex-wrap:wrap}
.kpi{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:10px 14px;min-width:150px}
.kpi .k{font-size:12px;color:#64748b} .kpi .v{font-size:20px;font-weight:600;margin-top:2px}
.bar{height:8px;background:#e5e7eb;border-radius:4px;overflow:hidden;margin-top:6px}
.bar>div{height:100%;background:#22c55e}
.chart{width:100%;max-width:900px;height:auto;font-size:11px;fill:#64748b}
.chart .g{stroke:#eef0f3} .chart .raw{fill:none;stroke:#93c5fd;stroke-width:.6;opacity:.6}
.chart .sm{fill:none;stroke:#1d4ed8;stroke-width:2} .chart .now{stroke:#ef4444;stroke-dasharray:4 3}
.note{font-size:12px;color:#64748b;margin-top:6px}
.missing{color:#94a3b8;font-style:italic;padding:14px;background:#f8fafc;border-radius:6px}
.tbl{border-collapse:collapse;font-size:13px} .tbl td,.tbl th{border:1px solid #e5e7eb;padding:5px 9px;text-align:right}
.tbl th{background:#f8fafc} .tbl td:first-child,.tbl th:first-child{text-align:left} .best{background:#dcfce7;font-weight:600}
.card{border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin-bottom:16px}
.card.reviewed{border-color:#22c55e}
.meta{font-size:13px;color:#475569;margin-bottom:8px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.v .lbl{font-size:12px;color:#334155;margin-bottom:3px;font-weight:600}
.v video{width:100%;background:#000;border-radius:4px}
.prompt{font-size:13px;line-height:1.55;color:#334155;margin:6px 0} .small{font-size:12px}
.review{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:8px;font-size:13px}
.review textarea{flex:1 1 100%;min-height:40px;font-size:13px;padding:6px}
.dgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;margin-top:14px}
.dcard{border:1px solid #e5e7eb;border-radius:8px;padding:8px}
.toolbar{display:flex;gap:10px;align-items:center;margin-bottom:12px}
button{background:#2563eb;color:#fff;border:0;border-radius:6px;padding:7px 14px;cursor:pointer;font-size:13px}
button.sec{background:#e5e7eb;color:#111}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px}
"""

JS = """
const KEY='wm_round2_review_v1';
const load=()=>{try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){return {}}};
const save=s=>localStorage.setItem(KEY,JSON.stringify(s));
function refresh(){const s=load();let n=0;document.querySelectorAll('.card[data-case]').forEach(c=>{
  const r=s[c.dataset.case]||{};const done=Object.values(r).some(v=>v);c.classList.toggle('reviewed',done);if(done)n++;});
  document.getElementById('rv-count').textContent=n+' / '+document.querySelectorAll('.card[data-case]').length+' 已评审';}
document.querySelectorAll('.card[data-case]').forEach(c=>{const s=load(),r=s[c.dataset.case]||{};
  c.querySelectorAll('[data-k]').forEach(el=>{if(r[el.dataset.k])el.value=r[el.dataset.k];
    el.addEventListener('input',()=>{const s=load();s[c.dataset.case]=s[c.dataset.case]||{};
      s[c.dataset.case][el.dataset.k]=el.value;save(s);refresh();});});});
document.getElementById('rv-export').onclick=()=>{const reviewer=prompt('评审人姓名（可留空）')||'';
  const blob=new Blob([JSON.stringify({reviewer,exported_at:new Date().toISOString(),build:BUILD,reviews:load()},null,1)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='round2_review_'+Date.now()+'.json';a.click();};
document.getElementById('rv-clear').onclick=()=>{if(confirm('清空本浏览器中的所有评审记录？')){localStorage.removeItem(KEY);location.reload();}};
document.getElementById('sync').onclick=()=>{const vs=[...document.querySelectorAll('.grid4 video')];
  vs.forEach(v=>{v.currentTime=0;v.play();});};
refresh();
"""


def main():
    os.makedirs(MEDIA, exist_ok=True)
    t0 = time.time()
    st = status()
    step, total = st.get("step", 0), st.get("total", TOTAL_STEPS)
    pct = 100 * step / total if total else 0
    shards = 88 + st.get("new_shards", 0)
    train_state = "已完成" if st["train_done"] else f"{step}/{total}"
    kpis = f"""
<div class="kpis">
  <div class="kpi"><div class="k">Round2 训练步数</div><div class="v">{train_state}</div>
    <div class="bar"><div style="width:{100 if st['train_done'] else pct:.1f}%"></div></div></div>
  <div class="kpi"><div class="k">已用时 / 预计剩余</div><div class="v">{esc(st.get('elapsed', '-'))} / {esc(st.get('eta', '-'))}</div></div>
  <div class="kpi"><div class="k">评测</div><div class="v">{'已完成' if st['eval_done'] else '待训练结束'}</div></div>
  <div class="kpi"><div class="k">HF 5s 分片（nzc-415/wan22-5s）</div><div class="v">{shards} / ~1060</div>
    <div class="bar"><div style="width:{100 * shards / 1060:.1f}%"></div></div></div>
</div>
<div class="note">状态快照：{esc(st.get('time', '-'))}（来自 dataset/download_logs/status_round2.log）</div>"""
    ckpts = sorted(glob.glob(f"{R2}/run*/step-*.safetensors"), key=os.path.getmtime)
    ck = ", ".join(esc(os.path.relpath(c, R2)) for c in ckpts) or "暂无"
    src_desc, cmp_cards = comparison_html()
    data_table, data_cards = data_html()
    build = time.strftime("%Y-%m-%d %H:%M:%S")
    page = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>World Model · Round2 Review</title>
<style>{CSS}</style></head><body>
<header><h1>全双工交互世界模型 · Wan2.2-TI2V-5B LoRA Round2 评审</h1>
<div class="sub">121 帧（5s）· 1280×704 · LoRA r32 · 从 Round1 初始化 · 15×A100-40GB · 页面生成于 {build}</div></header>
<nav><a href="#status">实验状态</a><a href="#loss">训练曲线</a><a href="#val">验证集 Loss</a>
<a href="#gen">生成对比评审</a><a href="#data">训练数据</a><a href="teacher.html"><b>Teacher 验收 →</b></a></nav>
<main>
<section id="status"><h2>实验状态</h2>{kpis}
<div class="note">Checkpoints：{ck}</div></section>
<section id="loss"><h2>Round2 训练 Loss</h2>{loss_svg(loss_series())}</section>
<section id="val"><h2>留出验证集 Loss（base vs Round1 vs Round2）</h2>{val_loss_table()}</section>
<section id="gen"><h2>TI2V 生成对比与评审</h2>
<div class="note" style="margin-bottom:10px">{esc(src_desc)}</div>
<div class="toolbar"><button id="sync" class="sec">同步从头播放</button>
<button id="rv-export">导出评审结果（JSON）</button><button id="rv-clear" class="sec">清空</button>
<span id="rv-count" class="note"></span></div>
<div class="note" style="margin-bottom:10px">评审记录自动保存在本浏览器（localStorage），完成后点“导出评审结果”发给实验负责人。</div>
{cmp_cards}</section>
<section id="data"><h2>Round2 训练数据（5s 窗口 + Qwen3.5-35B-A3B 稠密 caption）</h2>
{data_table}<div class="note">下方为 val 集随机样例（每数据集 {SAMPLES_PER_DATASET} 条）。</div>
<div class="dgrid">{data_cards}</div></section>
</main><script>const BUILD="{build}";{JS}</script></body></html>"""
    tmp = f"{DEMO}/index.html.tmp"
    with open(tmp, "w") as f:
        f.write(page)
    os.replace(tmp, f"{DEMO}/index.html")
    n_media = len(glob.glob(f"{MEDIA}/*.mp4"))
    print(f"demo/index.html built in {time.time() - t0:.0f}s, {n_media} preview videos", flush=True)


if __name__ == "__main__":
    main()
