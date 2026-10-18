"""Prompt optimization for VBench-I2V (the "prompt-optimized" protocol used by the leaderboard's Wan entries).

VBench prompts are ~10 words, our Teacher was trained on ~135-word temporal captions (caption_5s.py).
Qwen3.5-35B-A3B looks at the input image + the short prompt and writes a caption in the training
style that keeps the short prompt's subject/action and describes plausible motion over time.

Needs a vLLM server (qwen35-vl) that accepts one image per request.
Usage: python rewrite_prompts_vbench.py --port 8001 [--suite vbench|physiq|wmb] -> eval/<suite dir>/prompts_po.json
Resumable (existing entries are kept).
"""
import argparse
import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = "/srv/wm_teacher"
EVROOT = f"{ROOT}/eval"
BENCH = "/dockerdata/wm_eval/bench"


def load_cases(suite):
    """[{id, image, prompt}] for vbench (cases.jsonl), physiq (198 take-1 switch frames + original
    descriptions) and wmb (350 first frames + text_instruction; the image-side text_first_frame is
    given to the VLM as extra context, as WorldModelBench provides it)."""
    if suite == "vbench":
        return [json.loads(l) for l in open(f"{EVROOT}/vbench_i2v/cases.jsonl")]
    if suite == "physiq":
        import csv
        rows = [r for r in csv.DictReader(open(f"{BENCH}/physics-IQ-benchmark/descriptions/descriptions_original.csv"))
                if "_take-1_" in r["scenario"]]
        fr = {f[:4]: f for f in os.listdir(f"{EVROOT}/physiq/inputs")}
        return [{"id": r["scenario"][:4], "image": f"{EVROOT}/physiq/inputs/{fr[r['scenario'][:4]]}", "prompt": r["description"]}
                for r in rows]
    if suite == "wmb":
        d = json.load(open(f"{BENCH}/WorldModelBench/worldmodelbench.json"))
        out = []
        for x in d:
            n = os.path.splitext(os.path.basename(x["first_frame"]))[0]
            out.append({"id": n, "image": f"{EVROOT}/wmb/inputs/{os.path.basename(x['first_frame'])}",
                        "prompt": x["text_instruction"]})
        return out
    raise ValueError(suite)

INSTR = """The image is the FIRST FRAME of a 5-second video that a text-and-image-to-video model will generate. The user's short description of the video is:

"{p}"

Rewrite it as ONE detailed English paragraph of about 120-180 words in this style:
1. Scene and setting as seen in the image: location, background, lighting, visual style.
2. Main subjects with concrete appearance details (colors, materials, positions in frame).
3. Actions and dynamics in temporal order (beginning, middle, end): describe clear, natural, physically plausible motion that fulfils the short description - how each subject moves, speed and direction, how things change by the end.
4. Camera: viewpoint and camera motion (if the short description names a camera move, use it; otherwise a steady camera with at most slight natural movement).

Rules: preserve every explicitly stated action, object, material, color, direction, and camera constraint from the short description. If the image appears ambiguous, retain the description instead of changing the subject. Do not invent a collision outcome, deformation, floating or sinking, a final resting state, or any other physical consequence that the description does not specify. Describe an unspecified interaction only as an approach or onset, leaving its outcome open. Do not add subjects or change what happens. Describe only visible scene details; do not mention seconds, frames, or that this is a video; do not start with "The video"; output only the paragraph."""


def ask(port, case):
    img = base64.b64encode(open(case["image"], "rb").read()).decode()
    ext = os.path.splitext(case["image"])[1].lstrip(".").replace("jpg", "jpeg")
    body = {"model": "qwen35-vl", "max_tokens": 400, "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{img}"}},
                {"type": "text", "text": INSTR.format(p=case["prompt"])}]}]}
    for attempt in range(4):
        try:
            r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=300,
                              proxies={"http": None, "https": None})
            t = r.json()["choices"][0]["message"]["content"].strip()
            t = re.sub(r"\s+", " ", t)
            if 60 <= len(t.split()) <= 260:
                return case["id"], t
        except Exception as e:
            print(f"retry {case['id'][:40]}: {str(e)[:100]}", flush=True)
    return case["id"], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--conc", type=int, default=48)
    ap.add_argument("--suite", default="vbench", choices=["vbench", "physiq", "wmb"])
    ap.add_argument("--out", help="Separate output path for a new prompt protocol; existing entries remain unchanged")
    a = ap.parse_args()
    OUT = a.out or f"{EVROOT}/{ {'vbench': 'vbench_i2v'}.get(a.suite, a.suite) }/prompts_po.json"
    cases = load_cases(a.suite)
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(cases)} cases, {len(todo)} to rewrite", flush=True)
    with ThreadPoolExecutor(a.conc) as ex:
        for i, (cid, t) in enumerate(ex.map(lambda c: ask(a.port, c), todo), 1):
            if t:
                done[cid] = t
            if i % 50 == 0 or i == len(todo):
                json.dump(done, open(OUT + ".tmp", "w"), ensure_ascii=False, indent=0)
                os.replace(OUT + ".tmp", OUT)
                print(f"{i}/{len(todo)} rewritten, total {len(done)}", flush=True)
    json.dump(done, open(OUT + ".tmp", "w"), ensure_ascii=False, indent=0)
    os.replace(OUT + ".tmp", OUT)
    w = sorted(len(v.split()) for v in done.values())
    print(f"done {len(done)}/{len(cases)}, words median {w[len(w) // 2] if w else 0}", flush=True)


if __name__ == "__main__":
    main()
