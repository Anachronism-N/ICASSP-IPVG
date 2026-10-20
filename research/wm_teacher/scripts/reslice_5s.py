#!/usr/bin/env python3
"""
Re-slice processed clips into exact 5s windows (121 frames @24fps) for Wan2.2-TI2V.

Source runs come from training_metadata/inventory_5s.jsonl (built by inventory_5s.py):
each run = contiguous 4s segments of one source video. MIRA runs are rebuilt from
manifest_mira_full.jsonl (chunks of one match+player, contiguity verified via physics.jsonl).

For each selected run: ffmpeg concat demuxer -> libx264 re-encode -> segment muxer with
exact frame boundaries (-segment_frames 121,242,...) -> keep only full 121-frame windows.

Selection controls diversity: --max-per-source caps windows per source video, windows are
spread evenly along the source (not only from its start).

Usage:
  python3 scripts/reslice_5s.py --datasets charades,noxi --target 1000 --out-root dataset/wan22_5s
  python3 scripts/reslice_5s.py --all --target-json '{"hoigen1m":20000,...}'
"""
import argparse
import json
import os
import random
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = "/srv/wm_teacher"
FFMPEG = ("/srv/PRIVATE_STORAGE"
          "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")
W, SEG, FPS = 121, 96, 24
MIRA_RE = re.compile(r"(.+)_c(\d{5})\.p(\d+)_720p\.mp4$")


def load_runs(datasets, orient):
    runs = defaultdict(list)
    with open(f"{ROOT}/training_metadata/inventory_5s.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["dataset"] in datasets and r["windows"] > 0 and orient in ("all", r["orient"]):
                runs[r["dataset"]].append(r)
    if "mira" in datasets and orient in ("all", "landscape"):
        streams = defaultdict(dict)
        with open(f"{ROOT}/training_metadata/manifest_mira_full.jsonl") as f:
            for line in f:
                vp = json.loads(line).get("video_path", "")
                m = MIRA_RE.search(vp)
                if m:
                    streams[(m.group(1), m.group(3))][int(m.group(2))] = vp
        for (match, p), chunks in streams.items():
            idxs = sorted(chunks)
            cur = [idxs[0]]
            for a, b in zip(idxs, idxs[1:]):
                if b == a + 1:
                    cur.append(b); continue
                if len(cur) * SEG >= W:
                    runs["mira"].append(_mira_run(match, p, cur, chunks))
                cur = [b]
            if len(cur) * SEG >= W:
                runs["mira"].append(_mira_run(match, p, cur, chunks))
    return runs


def _mira_run(match, p, cur, chunks):
    return {"dataset": "mira", "source": f"{match}_p{p}_c{cur[0]:05d}", "split": True,
            "segs": [chunks[c] for c in cur], "total_frames": SEG * len(cur),
            "windows": SEG * len(cur) // W, "width": 1280, "height": 720, "orient": "landscape"}


def plan(runs_by_ds, targets, max_per_source, seed):
    """Pick (run, window_indices) so that each dataset reaches its target with max source diversity."""
    rng = random.Random(seed)
    jobs = []
    for ds, runs in runs_by_ds.items():
        want = targets.get(ds, 0)
        if want <= 0:
            continue
        rng.shuffle(runs)
        # round-robin: pass k gives each run its k-th window (evenly spaced), up to max_per_source
        picks = defaultdict(list)
        got, k = 0, 0
        while got < want and k < max_per_source:
            added = False
            for i, r in enumerate(runs):
                n = r["windows"]
                cap = min(n, max_per_source)
                if k < cap:
                    # spread the chosen windows evenly along the source instead of taking only its start
                    order = [round((m + 0.5) * n / cap - 0.5) for m in range(cap)]
                    picks[i].append(order[k])
                    got += 1; added = True
                    if got >= want:
                        break
            if not added:
                break
            k += 1
        planned = 0
        for i, wins in picks.items():
            jobs.append((runs[i], sorted(set(wins))))
            planned += len(set(wins))
        print(f"[plan] {ds}: target={want} planned={planned} "
              f"from {len(picks)} sources (available runs={len(runs)})", flush=True)
    return jobs


TMP_DIR = "/dockerdata/wm_tmp"  # local disk: keep temp encode traffic off ceph


def shard_of(safe):
    """256 sub-dirs per dataset: one flat dir with ~1M files serializes every create on the ceph dir lock."""
    import hashlib
    return hashlib.md5(safe.encode()).hexdigest()[:2]


def window_key(ds, basename):
    """Path-independent id of a window: '<dataset>:<safe_source>_wNNN' (works for flat and sharded paths)."""
    return f"{ds}:{basename[:-4] if basename.endswith('.mp4') else basename}"


def _blocks(wins):
    """Split sorted window indices into contiguous blocks."""
    out, cur = [], [wins[0]]
    for a, b in zip(wins, wins[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            out.append(cur); cur = [b]
    out.append(cur)
    return out


def validate_run_names(jobs):
    identities = {}
    for run, _ in jobs:
        safe = re.sub(r"[^A-Za-z0-9_.+-]", "_", run["source"])[:150]
        key = (run["dataset"], safe)
        identity = (run["source"], tuple(run["segs"]))
        if key in identities and identities[key] != identity:
            raise ValueError(
                "Ambiguous window namespace " + repr(key) + ": distinct runs share an output name; "
                "preserve existing data and prepare a run-qualified manifest before slicing")
        identities[key] = identity


def process(job, out_root):
    run, wins = job
    rows, fails = [], 0
    for blk in _blocks(sorted(wins)):
        r, f = _process_block(run, blk, out_root)
        rows += r; fails += f
    return run["dataset"], rows, fails


def _process_block(run, wins, out_root):
    """Encode a contiguous block of windows, reading only the segments that cover it."""
    ds = run["dataset"]
    safe = re.sub(r"[^A-Za-z0-9_.+-]", "_", run["source"])[:150]
    out_dir = os.path.join(ROOT, out_root, f"{ds}_5s", shard_of(safe))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    w0, w1 = wins[0], wins[-1]
    segs = run["segs"]
    s0 = min(w0 * W // SEG, len(segs) - 1)
    s1 = min(((w1 + 1) * W - 1) // SEG, len(segs) - 1)
    off = w0 * W - s0 * SEG          # frames to skip inside the first segment
    nfr = (w1 - w0 + 1) * W
    with tempfile.TemporaryDirectory(dir=TMP_DIR) as td:
        lst = os.path.join(td, "list.txt")
        with open(lst, "w") as f:
            for s in segs[s0:s1 + 1]:
                f.write(f"file '{os.path.join(ROOT, s)}'\n")
        # -nostdin + stdin=DEVNULL: keep ffmpeg from ever touching the terminal.
        base = [FFMPEG, "-nostdin", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                "-vf", f"trim=start_frame={off}:end_frame={off + nfr},setpts=PTS-STARTPTS",
                "-frames:v", str(nfr), "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-threads", "4", "-r", str(FPS)]
        if len(wins) == 1:
            cmd = base + [os.path.join(td, "w000.mp4")]
        else:
            cuts = ",".join(str(W * (i + 1)) for i in range(len(wins) - 1))
            cmd = base + ["-force_key_frames", f"expr:eq(mod(n,{W}),0)",
                          "-f", "segment", "-segment_frames", cuts, "-reset_timestamps", "1",
                          os.path.join(td, "w%03d.mp4")]
        r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                           timeout=300 + nfr // 10)
        rows, fails = [], 0
        for i, w in enumerate(wins):
            src = os.path.join(td, f"w{i:03d}.mp4")
            dst = os.path.join(out_dir, f"{safe}_w{w:03d}.mp4")
            n, ow, oh, med, mx = _analyze(src) if os.path.exists(src) else (0, 0, 0, 0, 0)
            if n == W:
                try:
                    os.replace(src, dst)  # same local filesystem -> atomic rename
                except OSError:
                    _copy(src, dst)
                row = _row(run, w, out_root, safe)
                # real output geometry (ffmpeg applies rotation side-data, e.g. Aria clips)
                row.update(width=ow, height=oh, orient="portrait" if oh > ow else "landscape",
                           diff_median=round(med, 2), diff_max=round(mx, 2),
                           scene_cut=bool(mx > max(5 * med, 20.0)))
                rows.append(row)
            else:
                fails += 1
        if fails and r.returncode != 0:
            print(f"[ffmpeg-err] {run['source']} w{w0}-{w1}: {r.stderr[-200:]}", flush=True)
    return rows, fails


def _copy(src, dst):
    import shutil
    shutil.copyfile(src, dst)


def _analyze(path):
    """Decode once at low res: (frame_count, width, height, median |dF|, max |dF|)."""
    import av
    import numpy as np
    try:
        with av.open(path) as c:
            s = c.streams.video[0]
            w, h = s.width, s.height
            prev, diffs, n = None, [], 0
            for f in c.decode(video=0):
                g = f.reformat(width=64, height=36, format="gray").to_ndarray().astype(np.float32)
                if prev is not None:
                    diffs.append(float(np.abs(g - prev).mean()))
                prev = g
                n += 1
        if not diffs:
            return n, w, h, 0.0, 0.0
        return n, w, h, float(np.median(diffs)), float(max(diffs))
    except Exception:
        return 0, 0, 0, 0.0, 0.0


def _row(run, w, out_root, safe):
    first_seg = (w * W) // SEG
    last_seg = (w * W + W - 1) // SEG
    return {"video_path": f"{out_root}/{run['dataset']}_5s/{shard_of(safe)}/{safe}_w{w:03d}.mp4",
            "source_dataset": run["dataset"],
            "source_sequence_id": f"{run['dataset']}:{run['source']}",
            "window_idx": w, "start_frame": w * W, "num_frames": W, "fps": FPS,
            "width": run["width"], "height": run["height"], "orient": run["orient"],
            "src_segments": run["segs"][first_seg:last_seg + 1], "prompt": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="")
    ap.add_argument("--target", type=int, default=0, help="same target for every dataset")
    ap.add_argument("--target-json", default="", help='per-dataset targets, e.g. {"hoigen1m":20000}')
    ap.add_argument("--orient", default="all", choices=["all", "landscape", "portrait"])
    ap.add_argument("--max-per-source", type=int, default=8)
    ap.add_argument("--out-root", default="dataset/wan22_5s")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=12, help="max windows per ffmpeg job (--all-windows)")
    ap.add_argument("--all-windows", action="store_true",
                    help="slice EVERY 5s window of every source (datasets given by --datasets, in order)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.all_windows:
        order = [d for d in a.datasets.split(",") if d]
        runs = load_runs(set(order), a.orient)
        jobs = []
        for ds in order:  # dataset order = processing priority
            rs = runs.get(ds, [])
            for r in rs:  # chunk long sources so every job reads only ~15 segments
                n = r["windows"]
                jobs += [(r, list(range(k, min(k + a.chunk, n)))) for k in range(0, n, a.chunk)]
            print(f"[plan] {ds}: runs={len(rs)} windows={sum(r['windows'] for r in rs)}", flush=True)
    else:
        targets = json.loads(a.target_json) if a.target_json else {d: a.target for d in a.datasets.split(",") if d}
        runs = load_runs(set(targets), a.orient)
        jobs = plan(runs, targets, a.max_per_source, a.seed)
    print(f"[plan] total windows={sum(len(w) for _, w in jobs)} jobs={len(jobs)}", flush=True)
    validate_run_names(jobs)
    if a.dry_run:
        return

    os.makedirs(os.path.join(ROOT, a.out_root), exist_ok=True)
    mans = {o: os.path.join(ROOT, a.out_root, f"manifest_5s_{o}.jsonl") for o in ("landscape", "portrait")}
    seen = set()  # window keys (path-independent, so flat rows from earlier runs still count)
    for man in mans.values():  # resume: never write the same window twice
        if os.path.exists(man):
            with open(man) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        seen.add(window_key(r["source_dataset"], os.path.basename(r["video_path"])))
                    except Exception:
                        pass
    print(f"[resume] {len(seen)} windows already in manifests", flush=True)

    def left(j):
        safe = re.sub(r"[^A-Za-z0-9_.+-]", "_", j[0]["source"])[:150]
        return [w for w in j[1] if window_key(j[0]["dataset"], f"{safe}_w{w:03d}.mp4") not in seen]
    jobs = [(j[0], l) for j in jobs for l in [left(j)] if l]
    print(f"[resume] {len(jobs)} jobs left, {sum(len(w) for _, w in jobs)} windows", flush=True)
    ok = fail = cuts = 0
    t0 = time.time()
    fos = {o: open(m, "a") for o, m in mans.items()}
    try:
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(process, j, a.out_root) for j in jobs]
            for i, fu in enumerate(as_completed(futs), 1):
                try:
                    ds, rows, f = fu.result()
                except Exception as e:
                    print(f"[job-err] {type(e).__name__}: {str(e)[:200]}", flush=True); fail += 1; continue
                for r in rows:
                    k = window_key(r["source_dataset"], os.path.basename(r["video_path"]))
                    if k not in seen:
                        seen.add(k)
                        fos[r["orient"]].write(json.dumps(r, ensure_ascii=False) + "\n")
                        cuts += r.get("scene_cut", False)
                ok += len(rows); fail += f
                if i % 200 == 0 or i == len(jobs):
                    for fo in fos.values():
                        fo.flush()
                    rate = ok / max(time.time() - t0, 1)
                    print(f"[progress] {time.strftime('%H:%M:%S')} jobs {i}/{len(jobs)} windows ok={ok} "
                          f"fail={fail} scene_cut={cuts} {rate:.1f} win/s", flush=True)
    finally:
        for fo in fos.values():
            fo.close()
    print(f"[done] windows ok={ok} fail={fail} scene_cut={cuts} -> {list(mans.values())}", flush=True)


if __name__ == "__main__":
    main()
