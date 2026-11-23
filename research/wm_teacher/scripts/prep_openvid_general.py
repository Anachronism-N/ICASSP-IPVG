"""General-domain anti-forgetting data for FT2: OpenVidHD clips from the locally complete zip parts.
Select clips >= 5.1 s, aesthetic >= 6.0, known camera label, <= 2 clips per source video; keep all
camera-moving clips (they carry a text label) and fill with static ones up to --n. Extract from the zip,
transcode to the Wan2.2 5s format (1280x704 cover-crop, 24 fps, 121 frames) and write a manifest whose
prompt = OpenVid caption + an explicit camera sentence.
Output: dataset/wan22_5s/openvidgen_5s/*.mp4, training_metadata/ft2/openvid_general.csv (video,prompt)
Usage: python prep_openvid_general.py [--n 12000] [--procs 16]"""
import argparse, csv, json, os, re, random, subprocess, zipfile
from collections import defaultdict
from multiprocessing import Pool
import pandas as pd

ROOT = "/srv/wm_teacher"
SRC = f"{ROOT}/dataset/general_video/openvidhd/OpenVidHD"
OUTD = "dataset/wan22_5s/openvidgen_5s"
TMP = "/dockerdata/wm_tmp/openvid_extract"
FF = "/dockerdata/wm_tmp/ffmpeg"
CAM = {"static": "The camera is static.", "pan_left": "The camera pans left.", "pan_right": "The camera pans right.",
       "tilt_up": "The camera tilts up.", "tilt_down": "The camera tilts down.", "zoom_in": "The camera zooms in.",
       "zoom_out": "The camera zooms out."}


def cam_sentence(label):
    parts = [CAM.get(p) for p in label.split("+")]
    if any(p is None for p in parts):
        return None
    if len(parts) == 1:
        return parts[0]
    return "The camera " + " and ".join(p[len("The camera "):-1] for p in parts) + "."


def job(a):
    part, name, out = a
    dst = f"{ROOT}/{out}"
    if os.path.exists(dst):
        return out
    os.makedirs(TMP, exist_ok=True)
    tmp = f"{TMP}/{os.getpid()}_{name}"
    try:
        with zipfile.ZipFile(f"{SRC}/OpenVidHD_part_{part}.zip") as z, open(tmp, "wb") as f:
            f.write(z.read(name))
        r = subprocess.run([FF, "-nostdin", "-loglevel", "error", "-y", "-i", tmp, "-an", "-vf",
                            "scale=1280:704:force_original_aspect_ratio=increase,crop=1280:704,fps=24",
                            "-frames:v", "121", "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                            "-pix_fmt", "yuv420p", "-threads", "2", dst + ".tmp.mp4"], timeout=600)
        if r.returncode == 0:
            os.replace(dst + ".tmp.mp4", dst)
            return out
    except Exception:
        pass
    finally:
        for p in (tmp, dst + ".tmp.mp4"):
            if os.path.exists(p):
                os.remove(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    d = pd.read_csv("/dockerdata/wm_eval/openvid/data/train/OpenVidHD.csv")
    v2p = {v: int(k.replace("part", "")) for e in json.load(open(f"{SRC}/OpenVidHD.json")) for k, vs in e.items() for v in vs}
    zips = {int(m.group(1)) for f in os.listdir(SRC) if (m := re.match(r"OpenVidHD_part_(\d+)\.zip$", f))}
    d["part"] = d.video.map(v2p)
    d = d[d.part.isin(zips) & (d.seconds >= 5.1) & (d["aesthetic score"] >= 6.0)]
    d["cam"] = d["camera motion"].map(cam_sentence)
    d = d[d.cam.notna()]
    rng = random.Random(0)
    rows = d.sample(frac=1, random_state=0).to_dict("records")
    per_src, moving, static = defaultdict(int), [], []
    for r in rows:
        src = r["video"].rsplit("_", 2)[0]  # <youtube id>_<scene>_<a>to<b>.mp4 -> youtube id
        if per_src[src] >= 2:
            continue
        per_src[src] += 1
        (static if r["camera motion"] == "static" else moving).append(r)
    sel = moving[: a.n] + static[: max(0, a.n - len(moving))]
    rng.shuffle(sel)
    print(f"candidates {len(d)}, selected {len(sel)} (moving {min(len(moving), a.n)}, static {len(sel) - min(len(moving), a.n)})", flush=True)

    # zip member names: the file name inside each part archive
    names = {}
    for p in sorted({r["part"] for r in sel}):
        with zipfile.ZipFile(f"{SRC}/OpenVidHD_part_{p}.zip") as z:
            for n in z.namelist():
                names[os.path.basename(n)] = n
    os.makedirs(f"{ROOT}/{OUTD}", exist_ok=True)
    jobs, meta = [], {}
    for r in sel:
        if r["video"] not in names:
            continue
        out = f"{OUTD}/{r['video']}"
        jobs.append((r["part"], names[r["video"]], out))
        meta[out] = f"{r['caption'].strip()} {r['cam']}"
    done = []
    with Pool(a.procs) as pool:
        for i, out in enumerate(pool.imap_unordered(job, jobs, chunksize=4), 1):
            if out:
                done.append(out)
            if i % 500 == 0:
                print(f"{i}/{len(jobs)} ok={len(done)}", flush=True)
    os.makedirs(f"{ROOT}/training_metadata/ft2", exist_ok=True)
    with open(f"{ROOT}/training_metadata/ft2/openvid_general.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "prompt"])
        for out in sorted(done):
            w.writerow([out, meta[out]])
    print(f"done: {len(done)} clips -> training_metadata/ft2/openvid_general.csv", flush=True)


if __name__ == "__main__":
    main()
