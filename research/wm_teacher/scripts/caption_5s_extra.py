#!/usr/bin/env python3
"""
Detailed VLM captioning for the 5s (121-frame) windows produced by reslice_5s.py.

- Follows the growing manifests (dataset/wan22_5s/manifest_5s_{landscape,portrait}.jsonl)
  while slicing is still running; exits when slicing is done and every window is captioned.
- Sends the whole clip as video (file:// URL, vLLM samples 16 frames) so the model sees motion.
- Resume-safe: output is append-only JSONL, keyed by video_path.
- Quality gate: rejects refusals / too-short answers and retries (up to 3 attempts).

Usage:
  python3 scripts/caption_5s.py --ports 8001-8008 --concurrency 384
"""
import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import requests

ROOT = "/srv/wm_teacher"
MAN_DIR = f"{ROOT}/dataset/wan22_5s_extra"
OUT = f"{ROOT}/dataset/wan22_5s_extra/captions_5s.jsonl"
FAIL = f"{ROOT}/dataset/wan22_5s_extra/captions_5s_failed.jsonl"
RESLICE_LOG = "/dockerdata/wm_logs/reslice_5s.log"
MODEL = "qwen35-vl"
NOPROXY = {"http": None, "https": None}

PROMPT = """You are annotating a 5-second video clip for training a text-and-image-to-video world model. The first frame will be given to the generator as an image, so your description must make clear what happens OVER TIME, not just what the first frame looks like.

Write ONE detailed English paragraph of about 120-200 words that covers, in this order:
1. Scene and setting: location, environment, key background objects, lighting, time of day, overall visual style (real footage, video game, egocentric headset, studio, etc.).
2. Main subjects: every important person, hand, animal, vehicle, object or game element, with concrete appearance details (clothing, color, size, material, position in frame, relative positions).
3. Actions and dynamics in temporal order (beginning, middle, end): what each subject does, how objects move or change state, contacts and manipulations (grasping, pushing, pouring, opening, driving, jumping, talking, gesturing), speed and direction of motion, and the result by the end of the clip.
4. Interactions: between people, between people and objects, between game entities.
5. Camera: viewpoint (first-person, third-person, fixed, handheld), camera motion (static, pan, tilt, zoom, tracking, shake) and framing changes.

Rules:
- Describe only what is actually visible; be specific (colors, counts, left/right, near/far).
- Describe only what IS present. Never list things that are absent (no "there are no other people", "no text appears", "no camera movement occurs"); if the camera is static just say it is static.
- If on-screen text, HUD, score or subtitles are visible, mention them briefly.
- If there is a scene cut or abrupt transition, say so and describe both parts.
- Spend most of the words on motion and change over time; for near-static clips describe the subtle movements (head turns, blinking, hand gestures, lip movement, breathing, small object motion).
- Do not mention the clip length, "5 seconds", frames, or that this is a video/clip; do not start with "The video shows", "This video", "In this clip"; start directly with the scene.
- No speculation about identities, names, emotions that are not visible, or audio.
- Output only the paragraph, no lists, no headings."""

BAD = re.compile(r"(i'?m sorry|i cannot|i can'?t|unable to (view|see|process)|as an ai)", re.I)
# Sentences that only state absences ("No other people appear", "There are no scene cuts") carry no
# signal for the generator and bias every caption toward the same boilerplate -> drop them.
ABSENCE = re.compile(
    r"^(no |none |nothing |neither |there (is|are|were|was) no |at no point|without any )"
    r"|\b(no|without) (other |additional |visible |discernible |significant |notable )*"
    r"(people|persons?|objects?|text|on-?screen text|hud|subtitles?|captions?|scene cuts?|cuts|"
    r"transitions?|camera (movement|motion)|audio|sound|interactions?|animals?)\b"
    r"[^.]*\b(appear|appears|visible|present|occur|occurs|observed|seen|shown)\b", re.I)
META = re.compile(r"\b(\d+|five)[- ]second(s)?\b|\b(the|this) (video|clip|footage|sequence)\b", re.I)
# trailing absence clause inside an informative sentence: ", and there are no significant changes ..."
ABSENCE_CLAUSE = re.compile(
    r"\s*[,;]\s*(and\s+|with\s+|while\s+|but\s+)?(there (is|are|were|was) )?"
    r"(no|without any|nothing) (other |additional |visible |significant |notable |discernible |major |further )*"
    r"[^,.;]*(?=[.;]|$)", re.I)


def sanitize(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = []
    for s in sents:
        s = s.strip()
        if ABSENCE.search(s):
            continue
        s2 = ABSENCE_CLAUSE.sub("", s).strip()
        if s2 and s2[-1] not in ".!?":
            s2 += "."
        if len(s2.split()) >= 3:
            kept.append(s2)
    out = " ".join(kept)
    for _ in range(2):  # 2 passes: "the 5-second clip" -> "the clip" -> "the scene"
        out = META.sub(lambda m: "the scene" if m.group(3) else "", out)
        out = re.sub(r"\s{2,}", " ", out)
    return re.sub(r"\s+([,.;])", r"\1", out).strip()


class Ports:
    def __init__(self, ports):
        self.ports = ports
        self.i = 0
        self.lock = threading.Lock()

    def next(self):
        with self.lock:
            self.i += 1
            return self.ports[self.i % len(self.ports)]


def caption(row, ports, max_tokens):
    path = os.path.join(ROOT, row["video_path"])
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": f"file://{path}"}},
            {"type": "text", "text": PROMPT}]}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_err = ""
    for attempt in range(3):
        port = ports.next()
        try:
            r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload,
                              timeout=300, proxies=NOPROXY)
            if r.status_code != 200:
                last_err = f"http {r.status_code}: {r.text[:200]}"
                time.sleep(2 + attempt * 3)
                continue
            ch = r.json()["choices"][0]
            text = (ch["message"].get("content") or "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            text = sanitize(text)
            n_words = len(text.split())
            if BAD.search(text) or n_words < 60:
                last_err = f"rejected ({n_words} words): {text[:120]}"
                payload["temperature"] = 0.5
                continue
            return text, ch.get("finish_reason"), n_words, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(2 + attempt * 3)
    return None, None, 0, last_err


def load_done():
    done = set()
    for p in (OUT,):
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["video_path"])
                    except Exception:
                        pass
    return done


def reslice_finished():
    return os.path.exists(f"{MAN_DIR}/SLICE_DONE")  # written by slice_new_sources.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", default="8001-8008")
    ap.add_argument("--concurrency", type=int, default=384)
    ap.add_argument("--max-tokens", type=int, default=450)
    ap.add_argument("--limit", type=int, default=0, help="stop after N captions (benchmark)")
    a = ap.parse_args()
    lo, _, hi = a.ports.partition("-")
    ports = Ports(list(range(int(lo), int(hi or lo) + 1)))

    done = load_done()
    print(f"[start] {len(done)} already captioned, ports={ports.ports}", flush=True)
    offsets = {}
    fo = open(OUT, "a", buffering=1)  # line-buffered: every caption hits disk immediately
    ff = open(FAIL, "a", buffering=1)
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "words": 0, "trunc": 0, "queued": 0}
    t0 = time.time()
    inflight = set()
    state = {"slicing_done": False, "stop": False}

    def reporter():  # independent of the (possibly blocked) submit loop
        while not state["stop"]:
            time.sleep(60)
            with lock:
                el = time.time() - t0
                print(f"[progress] {time.strftime('%H:%M:%S')} ok={stats['ok']} fail={stats['fail']} "
                      f"queued={stats['queued']} {stats['ok']/max(el,1):.2f} cap/s "
                      f"avg_words={stats['words']/max(stats['ok'],1):.0f} truncated={stats['trunc']} "
                      f"slicing_done={state['slicing_done']}", flush=True)
    threading.Thread(target=reporter, daemon=True).start()

    def job(row):
        text, finish, nw, err = caption(row, ports, a.max_tokens)
        with lock:
            if text:
                out = {"video_path": row["video_path"], "prompt": text,
                       "source_dataset": row["source_dataset"],
                       "source_sequence_id": row["source_sequence_id"],
                       "orient": row["orient"], "width": row["width"], "height": row["height"],
                       "num_frames": row["num_frames"], "fps": row["fps"],
                       "scene_cut": row.get("scene_cut", False),
                       "caption_model": "Qwen3.5-35B-A3B", "caption_words": nw,
                       "finish_reason": finish}
                fo.write(json.dumps(out, ensure_ascii=False) + "\n")
                stats["ok"] += 1
                stats["words"] += nw
                stats["trunc"] += finish == "length"
            else:
                ff.write(json.dumps({"video_path": row["video_path"], "error": err}) + "\n")
                stats["fail"] += 1

    with ThreadPoolExecutor(a.concurrency) as ex:
        while True:
            finished_slicing = reslice_finished()  # read BEFORE scanning so we never miss tail rows
            state["slicing_done"] = finished_slicing
            new_rows = []
            for o in ("landscape", "portrait"):
                p = f"{MAN_DIR}/manifest_5s_{o}.jsonl"
                if not os.path.exists(p):
                    continue
                with open(p) as f:
                    f.seek(offsets.get(p, 0))
                    while True:
                        pos = f.tell()
                        line = f.readline()
                        if not line:
                            break
                        if not line.endswith("\n"):  # partial line being written
                            f.seek(pos)
                            break
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        if r["video_path"] not in done:
                            done.add(r["video_path"])
                            new_rows.append(r)
                    offsets[p] = f.tell()
            random.shuffle(new_rows)  # mix datasets so throughput stats are representative
            for r in new_rows:
                while len(inflight) >= a.concurrency * 2:
                    d, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    inflight.difference_update(d)
                inflight.add(ex.submit(job, r))
                stats["queued"] += 1
                if a.limit and stats["ok"] + stats["fail"] + len(inflight) >= a.limit:
                    break
            if a.limit and stats["ok"] + stats["fail"] >= a.limit:
                break
            if finished_slicing and not new_rows:
                wait(inflight)
                inflight.clear()
                # one more scan in case the final rows landed after our read
                if not any(os.path.getsize(p) > offsets.get(p, 0)
                           for p in [f"{MAN_DIR}/manifest_5s_{o}.jsonl" for o in ("landscape", "portrait")]
                           if os.path.exists(p)):
                    break
            if not new_rows:
                time.sleep(20)
            inflight = {x for x in inflight if not x.done()}
        wait(inflight)
    state["stop"] = True
    fo.close(); ff.close()
    el = time.time() - t0
    print(f"[done] ok={stats['ok']} fail={stats['fail']} in {el/3600:.2f}h "
          f"({stats['ok']/max(el,1):.2f} cap/s) avg_words={stats['words']/max(stats['ok'],1):.0f}", flush=True)


if __name__ == "__main__":
    main()
