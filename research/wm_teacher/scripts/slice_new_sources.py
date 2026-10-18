"""Cut the supplementary sources into Wan2.2 5s windows (1280x704 cover-crop, 24 fps, 121 frames, H.264)
and write a manifest in the reslice_5s.py format (consumed by caption_5s_extra.py for dense Qwen3.5 captions).

  uk_dashcam : every ~3 min 1080p30 file, windows every --dash-stride s           -> ukdash_5s
  robomind   : lerobot v3 episodes (static front/top cams), <=2 windows / episode   -> robomind_5s
  egodex     : lerobot v3 episodes (egocentric hands), <=2 windows / episode        -> egodex_5s
  openvidgen : already-sliced OpenVidHD general clips (only added to the manifest, for recaptioning)
Caps per source via --cap-<name>. Episodes shorter than 5.2 s are skipped (no window crosses an episode
boundary). Resume-safe (existing outputs are kept). Output manifest: dataset/wan22_5s_extra/manifest_5s_landscape.jsonl
Usage: python slice_new_sources.py [--procs 48]"""
import argparse, glob, json, os, random, re, subprocess
from multiprocessing import Pool
import av
import pandas as pd

ROOT = "/srv/wm_teacher"
SRC = f"{ROOT}/dataset/new_sources"
MAN = f"{ROOT}/dataset/wan22_5s_extra/manifest_5s_landscape.jsonl"
FF = "/dockerdata/wm_tmp/ffmpeg"
VF = "scale=1280:704:force_original_aspect_ratio=increase,crop=1280:704,fps=24"


def lerobot_windows(name, root, keys, cap, rng):
    out = []
    for info in glob.glob(f"{root}/**/meta/info.json", recursive=True):
        base = os.path.dirname(os.path.dirname(info))
        sub = os.path.relpath(base, root).replace("/", "_")
        eps = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{base}/meta/episodes/*/*.parquet"))])
        for key in keys:
            col = f"videos/{key}"
            if f"{col}/from_timestamp" not in eps.columns:
                continue
            for e in eps.itertuples(index=False, name=None):
                row = dict(zip(eps.columns, e))
                t0, t1 = row[f"{col}/from_timestamp"], row[f"{col}/to_timestamp"]
                src = f"{base}/videos/{key}/chunk-{row[f'{col}/chunk_index']:03d}/file-{row[f'{col}/file_index']:03d}.mp4"
                if t1 - t0 < 5.2 or not os.path.exists(src):
                    continue
                starts = [t0 + 0.05] + ([t0 + (t1 - t0 - 5.1) / 2] if t1 - t0 >= 10.4 else [])
                cam = key.split(".")[-1]
                for i, s in enumerate(starts):
                    out.append(dict(name=name, src=src, start=round(s, 3),
                                    out=f"dataset/wan22_5s/{name}_5s/{sub}/{cam}_ep{row['episode_index']:06d}_w{i}.mp4",
                                    seq=f"{name}:{sub}:{cam}:{row['episode_index']}"))
    rng.shuffle(out)
    return out[:cap]


def dash_windows(cap, stride, rng):
    out = []
    for f in sorted(glob.glob(f"{SRC}/uk_dashcam/**/*.MP4", recursive=True) + glob.glob(f"{SRC}/uk_dashcam/**/*.mp4", recursive=True)):
        try:
            c = av.open(f); dur = float(c.duration / 1e6); c.close()
        except Exception:
            continue
        stem = os.path.splitext(os.path.basename(f))[0]
        s, i = 1.0, 0
        while s + 5.2 <= dur:
            out.append(dict(name="ukdash", src=f, start=s, out=f"dataset/wan22_5s/ukdash_5s/{stem}_w{i:03d}.mp4", seq=f"ukdash:{stem}"))
            s += stride; i += 1
    rng.shuffle(out)
    return out[:cap]


def cut(w):
    dst = f"{ROOT}/{w['out']}"
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        r = subprocess.run([FF, "-nostdin", "-loglevel", "error", "-y", "-ss", str(w["start"]), "-i", w["src"], "-t", "5.2",
                            "-an", "-vf", VF, "-frames:v", "121", "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                            "-pix_fmt", "yuv420p", "-threads", "2", dst + ".tmp.mp4"], timeout=900)
        if r.returncode != 0:
            return None
        os.replace(dst + ".tmp.mp4", dst)
    try:
        c = av.open(dst); n = sum(1 for _ in c.decode(video=0)); c.close()
    except Exception:
        n = 0
    if n < 121:
        os.remove(dst)
        return None
    return {"video_path": w["out"], "source_dataset": w["name"], "source_sequence_id": w["seq"], "window_idx": 0,
            "num_frames": 121, "fps": 24, "width": 1280, "height": 704, "orient": "landscape", "scene_cut": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--cap-ukdash", type=int, default=5000)
    ap.add_argument("--cap-robomind", type=int, default=6000)
    ap.add_argument("--cap-egodex", type=int, default=6000)
    ap.add_argument("--dash-stride", type=float, default=10.0)
    a = ap.parse_args()
    rng = random.Random(0)
    W = dash_windows(a.cap_ukdash, a.dash_stride, rng)
    W += lerobot_windows("robomind", f"{SRC}/robomind", ["observation.images.camera_front", "observation.images.camera_top"], a.cap_robomind, rng)
    W += lerobot_windows("egodex", f"{SRC}/egodex", ["observation.images.camera"], a.cap_egodex, rng)
    print(f"{len(W)} windows to cut: " + ", ".join(f"{n}={sum(w['name'] == n for w in W)}" for n in ("ukdash", "robomind", "egodex")), flush=True)
    os.makedirs(os.path.dirname(MAN), exist_ok=True)
    have = set()
    if os.path.exists(MAN):
        have = {json.loads(l)["video_path"] for l in open(MAN)}
    todo = [w for w in W if w["out"] not in have]
    with open(MAN, "a") as fm, Pool(a.procs) as pool:
        ok = 0
        for i, r in enumerate(pool.imap_unordered(cut, todo, chunksize=4), 1):
            if r:
                fm.write(json.dumps(r) + "\n"); ok += 1
            if i % 1000 == 0:
                fm.flush(); print(f"{i}/{len(todo)} ok={ok}", flush=True)
        # OpenVidHD general clips: already in 5s format, just register them for dense recaptioning
        import csv
        n = 0
        for row in csv.DictReader(open(f"{ROOT}/training_metadata/ft2/openvid_general.csv")):
            if row["video"] in have:
                continue
            fm.write(json.dumps({"video_path": row["video"], "source_dataset": "openvidgen",
                                 "source_sequence_id": "openvidgen:" + os.path.basename(row["video"]).rsplit("_", 2)[0],
                                 "window_idx": 0, "num_frames": 121, "fps": 24, "width": 1280, "height": 704,
                                 "orient": "landscape", "scene_cut": False}) + "\n"); n += 1
    print(f"done: cut ok={ok}, openvid registered={n} -> {MAN}", flush=True)
    open(f"{ROOT}/dataset/wan22_5s_extra/SLICE_DONE", "w").write("done\n")


if __name__ == "__main__":
    main()
