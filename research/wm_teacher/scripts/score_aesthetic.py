"""LAION aesthetic score (same CLIP ViT-L/14 + linear head as VBench aesthetic_quality) for 5s windows.

  build : python score_aesthetic.py build [--per-ds 25000]   -> training_metadata/round4/aes_pool.txt
  score : python score_aesthetic.py score SHARD NSHARD       -> training_metadata/round4/aes_scores_SHARD.jsonl
Score = mean over frames {0, 60, 120} of linear(normalize(clip_img_emb)) (raw 0-10 scale).
Resumable: already-scored windows in the shard output are skipped.
"""
import csv
import json
import os
import random
import sys
from collections import defaultdict

ROOT = "/srv/wm_teacher"
OUT = f"{ROOT}/training_metadata/round4"
POOL = f"{OUT}/aes_pool.txt"
CACHE = "/dockerdata/vbench_cache"
FRAMES = (0, 60, 120)


def build(per_ds):
    os.makedirs(OUT, exist_ok=True)
    val_videos = {r["video"] for r in csv.DictReader(open(f"{ROOT}/training_metadata/round2/round2_val.csv"))}
    rows = [json.loads(l) for l in open(f"{ROOT}/dataset/wan22_5s/final_5s_landscape.jsonl")]
    val_seqs = {m["source_sequence_id"] for m in rows if m["video_path"] in val_videos}
    print("val videos", len(val_videos), "val seqs", len(val_seqs))
    by = defaultdict(list)
    for m in rows:
        if m.get("scene_cut") or m["source_sequence_id"] in val_seqs:
            continue
        by[m["source_dataset"]].append(m["video_path"])
    rng = random.Random(4)
    pool = []
    for ds, v in sorted(by.items()):
        rng.shuffle(v)
        pool += v[:per_ds]
        print(f"{ds:10s} avail {len(v):8d} pooled {min(len(v), per_ds):6d}")
    rng.shuffle(pool)
    open(POOL, "w").write("\n".join(pool) + "\n")
    print("pool", len(pool))


def score(shard, nshard):
    import av
    import clip
    import torch
    from PIL import Image

    out = f"{OUT}/aes_scores_{shard}.jsonl"
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["video"] for l in open(out)}
    todo = [v for i, v in enumerate(open(POOL).read().split()) if i % nshard == shard and v not in done]
    print(f"shard {shard}/{nshard}: {len(todo)} todo, {len(done)} done", flush=True)

    model, preprocess = clip.load(f"{CACHE}/clip_model/ViT-L-14.pt", device="cuda")
    head = torch.nn.Linear(768, 1)
    head.load_state_dict(torch.load(f"{CACHE}/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth", map_location="cpu"))
    head = head.cuda().eval()

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, i):
            v = todo[i]
            try:
                with av.open(f"{ROOT}/{v}") as c:
                    imgs = [f.to_image() for k, f in enumerate(c.decode(video=0)) if k in FRAMES]
                    # decode() stops naturally at the end; FRAMES max is 120 of 121 frames
                if len(imgs) != len(FRAMES):
                    raise ValueError(f"got {len(imgs)} frames")
                return v, torch.stack([preprocess(im) for im in imgs])
            except Exception as e:
                print(f"skip {v}: {str(e)[:80]}", flush=True)
                return v, None

    dl = torch.utils.data.DataLoader(DS(), batch_size=16, num_workers=12, collate_fn=lambda b: b, prefetch_factor=4)
    with open(out, "a") as f, torch.no_grad():
        for k, batch in enumerate(dl):
            ok = [(v, x) for v, x in batch if x is not None]
            if not ok:
                continue
            x = torch.cat([x for _, x in ok]).cuda()
            e = model.encode_image(x).float()
            e = e / e.norm(dim=-1, keepdim=True)
            s = head(e).view(len(ok), len(FRAMES)).mean(1).tolist()
            for (v, _), sc in zip(ok, s):
                f.write(json.dumps({"video": v, "aes": round(sc, 4)}) + "\n")
            f.flush()
            if k % 100 == 0:
                print(f"shard {shard}: batch {k}/{len(dl)}", flush=True)
    print(f"shard {shard} done", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(int(sys.argv[3]) if len(sys.argv) > 3 else 25000)
    else:
        score(int(sys.argv[2]), int(sys.argv[3]))
