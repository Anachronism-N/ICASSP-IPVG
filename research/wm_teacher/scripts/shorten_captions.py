"""Make a short (VBench-style, 8-20 words) caption for every Round5 window from its long caption (text-only LLM).

In:  training_metadata/round5/round5_pool.csv
Out: training_metadata/round5/round5_pool_short.csv (+ column 'short'); resumable via a jsonl cache.
Usage: python shorten_captions.py --ports 8001
"""
import argparse
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests

OUT = "/srv/wm_teacher/training_metadata/round5"
CACHE = f"{OUT}/short_cache.jsonl"
INSTR = ("Summarize this video description as ONE short prompt of 8 to 20 words, like a user would type into a "
         "video generator: main subject + main action (+ setting if important). No camera details unless the camera "
         "motion is the main event. Output only the prompt.\n\nDescription: {c}")


def ask(port, row):
    body = {"model": "qwen35-vl", "max_tokens": 60, "temperature": 0.3,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": INSTR.format(c=row["prompt"])}]}
    for _ in range(3):
        try:
            r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=120,
                              proxies={"http": None, "https": None})
            t = re.sub(r"\s+", " ", r.json()["choices"][0]["message"]["content"]).strip().strip('"').rstrip(".")
            if 5 <= len(t.split()) <= 30:
                return row["video"], t
        except Exception:
            pass
    return row["video"], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", default="8001")
    ap.add_argument("--conc", type=int, default=128)
    a = ap.parse_args()
    ports = [int(p) for p in a.ports.split(",")]
    rows = list(csv.DictReader(open(f"{OUT}/round5_pool.csv")))
    done = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            d = json.loads(l)
            done[d["video"]] = d["short"]
    todo = [r for r in rows if r["video"] not in done]
    print(f"{len(rows)} rows, {len(todo)} to shorten", flush=True)
    with open(CACHE, "a") as f, ThreadPoolExecutor(a.conc) as ex:
        for i, (v, t) in enumerate(ex.map(lambda x: ask(ports[hash(x[1]["video"]) % len(ports)], x[1]),
                                          enumerate(todo)), 1):
            if t:
                done[v] = t
                f.write(json.dumps({"video": v, "short": t}) + "\n")
            if i % 2000 == 0:
                f.flush()
                print(f"{i}/{len(todo)}", flush=True)
    with open(f"{OUT}/round5_pool_short.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["short"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, short=done.get(r["video"], "")))
    print(f"done: {sum(1 for r in rows if done.get(r['video']))}/{len(rows)} shortened", flush=True)


if __name__ == "__main__":
    main()
