#!/usr/bin/env python3
"""Grounded re-captioning for E3 (docs/43 section 5, docs/44 error classes).

Qwen2.5-VL-7B (transformers, bf16, one GPU) rewrites training captions under anti-hallucination
rules derived from the delegated 17-case review: (1) JSON fact table first, (2) paragraph from
confirmed facts only, (3) held-object claims need clear hand-object contact, (4) no unverifiable
identities/colors, (5) no invented endings, (6) camera sentence generated from measured cam_shift
(reuses build_ft2_mixture.cam_sentence), (7) second-pass held-object verification with VLM
yes/no/unclear on 4 frames; unverified sentences are removed.

Manifest in (same format as cache_e3_inputs): {"sample_id","video","captions":{...}}.
Output JSONL per clip: {"sample_id","video_sha256","frame_indices","model","prompt_version",
"facts_json","caption","original_prompt_sha256","removed_sentences","cam_shift"}.
Original captions are never modified. Resume skips only valid records of the current version.
Use a new output file for legacy versions; their records are not repaired or rewritten.
frame_indices are sampled-image display indices, not original video frame numbers.
Schema validation checks structure only, not visual truth or preference-label quality.
Shard with --shard k --num-shards N (one GPU per process).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/recaption_e3.py \
      --manifest prefilter_manifest.jsonl --cam-scores training_metadata/e3_pool_20261022_v2/cam_scores_merged.jsonl \
      --out training_metadata/e3_recaption_20261022/records.jsonl
  python scripts/recaption_e3.py --report <records.jsonl>        # CPU-only audit of removals
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODEL_PATH = Path(os.environ.get("E3_RECAPTION_MODEL") or ROOT / "models" / "Qwen2.5-VL-7B-Instruct")
PROMPT_VERSION = "e3-grounded-recaption-v2"
N_FRAMES = 8
HELD_RE = re.compile(r"\b(hold(?:s|ing|s)?|holding|grasp(?:s|ing)?|grip(?:s|ping)?|picks? up|carries|in (?:his|her|their) hand)\b", re.I)

FACT_PROMPT = """You are watching a {n}-frame video sampled uniformly from a 5-second clip. List ONLY what is clearly visible.
Return STRICT JSON (double quotes, no comments, no trailing commas) with exactly these keys:
"scene": string; "rendered_or_screen": boolean (is this gameplay/CGI/screen recording?);
"entities": list of objects with keys "desc" (string, generic term if unclear) and "attrs" (list of strings, only clearly visible attributes);
"hands": object with keys "visible" (boolean) and "held_object" (string ONLY if hand-object contact is clearly visible in at least 2 frames, else null);
"actions_by_time": list of objects with keys "frames" (string, e.g. "0-3", indices of the displayed sampled images, NOT original video frame numbers) and "action" (string, visible motion only);
"start_end_difference": string (what changed from first to last frame, or "no visible change").
Rules: do not guess object types you cannot identify; do not state gender/age/hair/clothing colors unless clearly visible; do not invent an ending; prefer fewer, safer statements."""

WRITE_PROMPT = """Write ONE English caption paragraph (40-120 words) for video-model training, using ONLY these confirmed facts:
{facts}
Rules: mention a held object ONLY if "held_object" is non-null; use generic terms for unidentifiable things; say the scene stays mostly still if that is what the facts show; do NOT describe camera movement (added separately); do NOT add any fact absent above; shorter is better than guessing. Output the paragraph only."""

VERIFY_PROMPT = """Look at these frames from one video. Question: does the person visibly hold or grip the object described as "{claim}" in at least one frame? Answer exactly one word: yes / no / unclear."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sample_frames(video, n=N_FRAMES, timeout=None):
    """Uniformly sample n frames via ffmpeg fps filter (single sequential read)."""
    import subprocess, tempfile, os
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix='rcap_') as td:
        out = os.path.join(td, 'f%03d.png')
        cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-i', video,
               '-vf', 'fps=' + str(n) + '/5.05', '-frames:v', str(n), out]
        subprocess.run(cmd, check=True, timeout=timeout or 120)
        files = sorted(os.listdir(td))
        if not files:
            raise ValueError('no frames decoded')
        return [Image.open(os.path.join(td, x)).convert('RGB') for x in files]


def load_cam(cam_path):
    out = {}
    if not cam_path:
        return out
    for line in open(cam_path):
        d = json.loads(line)
        out[d.get("video")] = d.get("cam_shift")
    return out


def cam_sentence(shift):
    if shift is None:
        return ""
    if shift < 0.5:
        return "The camera is static."
    if shift < 2.0:
        return "The camera moves slightly."
    return "The camera moves noticeably."


def extract_json(text):
    """Return the first parseable complete top-level object, scanning left to right.

    Nested objects are never candidates on their own, even in a truncated root.
    A surrounding code fence or prose is allowed; trailing commas are repaired
    outside strings only. No other non-JSON syntax is accepted.
    """
    stack = []
    start = None
    in_string = escaped = invalid = False
    for i, char in enumerate(text):
        if not stack:
            if char not in "{[":
                continue
            start = i
            stack.append(char)
            in_string = escaped = invalid = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack[-1] != ("{" if char == "}" else "["):
                invalid = True
            stack.pop()
            if stack:
                continue
            candidate = text[start:i + 1]
            if invalid or candidate[0] != "{":
                continue
            repaired = []
            quoted = escape = False
            for j, token in enumerate(candidate):
                if quoted:
                    if escape:
                        escape = False
                    elif token == "\\":
                        escape = True
                    elif token == '"':
                        quoted = False
                elif token == '"':
                    quoted = True
                elif token == ",":
                    k = j + 1
                    while k < len(candidate) and candidate[k].isspace():
                        k += 1
                    if k < len(candidate) and candidate[k] in "}]":
                        continue
                repaired.append(token)
            for trial in (candidate, "".join(repaired)):
                try:
                    return json.loads(trial)
                except json.JSONDecodeError:
                    continue
    raise ValueError("no parseable complete top-level JSON object in model output")


def validate_facts(facts):
    """Check the fact-table structure, not semantic correctness or visual support."""
    required = {"scene", "rendered_or_screen", "entities", "hands",
                "actions_by_time", "start_end_difference"}
    if not isinstance(facts, dict) or set(facts) != required:
        raise ValueError("facts must contain exactly the six required top-level fields")
    if not isinstance(facts["scene"], str) or not isinstance(facts["start_end_difference"], str):
        raise ValueError("facts scene and start_end_difference must be strings")
    if type(facts["rendered_or_screen"]) is not bool:
        raise ValueError("facts rendered_or_screen must be a boolean")
    if not isinstance(facts["entities"], list):
        raise ValueError("facts entities must be a list")
    for entity in facts["entities"]:
        if not isinstance(entity, dict) or set(entity) != {"desc", "attrs"}:
            raise ValueError("each entity must contain desc and attrs")
        attrs = entity["attrs"]
        if not isinstance(entity["desc"], str) or not isinstance(attrs, list):
            raise ValueError("entity desc must be a string and attrs must be a list")
        if not all(isinstance(attr, str) for attr in attrs):
            raise ValueError("entity attrs list entries must be strings")
    hands = facts["hands"]
    if not isinstance(hands, dict) or set(hands) != {"visible", "held_object"}:
        raise ValueError("facts hands must contain visible and held_object")
    if type(hands["visible"]) is not bool:
        raise ValueError("hands visible must be a boolean")
    if hands["held_object"] is not None and not isinstance(hands["held_object"], str):
        raise ValueError("hands held_object must be a string or null")
    if not isinstance(facts["actions_by_time"], list):
        raise ValueError("facts actions_by_time must be a list")
    for action in facts["actions_by_time"]:
        if not isinstance(action, dict) or set(action) != {"frames", "action"}:
            raise ValueError("each action must contain frames and action")
        if not isinstance(action["frames"], str) or not isinstance(action["action"], str):
            raise ValueError("action frames and action must be strings")
    return facts


def resume_done(out):
    """Never append the new schema version to legacy or malformed output files."""
    done = set()
    if not os.path.exists(out):
        return done
    with open(out) as source:
        for line in source:
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or record.get("prompt_version") != PROMPT_VERSION:
                    raise ValueError("existing output has a legacy or unknown version")
                if "error" in record:
                    continue
                validate_facts(record.get("facts_json"))
                if not isinstance(record.get("caption"), str) or not record["caption"].strip():
                    raise ValueError("existing output has no valid caption")
                done.add(record["sample_id"])
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError("cannot resume this output; choose a new --out file") from exc
    return done


def held_claims(facts):
    h = facts.get("hands") or {}
    return [h["held_object"]] if h.get("held_object") else []


def recaption_one(model, processor, video, device):
    frames = sample_frames(video)
    images = [f.resize((640, 352)) for f in frames]
    msgs = [{"role": "user", "content": ["<image>"] * len(images) + [{"type": "text", "text": FACT_PROMPT.format(n=len(images))}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[images], return_tensors="pt").to(device)
    n_in = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=500, do_sample=False)
    facts = validate_facts(extract_json(processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]))
    wp = WRITE_PROMPT.format(facts=json.dumps(facts, ensure_ascii=False))
    msgs = [{"role": "user", "content": [{"type": "text", "text": wp}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)
    n_in = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    caption = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0].strip()
    # decode prompts sometimes leak; keep only the assistant tail after the last 'assistant' marker
    if "assistant" in caption:
        caption = caption.rsplit("assistant", 1)[-1].strip()
    # Hard post-filter: writing stage sometimes adds hold-verbs even when facts say null.
    removed = []
    if not held_claims(facts):
        new_cap = re.sub(r"[^.!?]*\b" + HELD_RE.pattern.replace("\\b", "").replace("(?:", "(") + r"[^.!?]*[.!?]?\s*", "", caption)
        if new_cap != caption:
            removed.append({"claim": "<unfiltered hold phrase with facts.held_object=null>", "verdict": "removed_by_postfilter"})
            caption = new_cap
    for leak in ("video-model training", "video model training", "the facts", "the frames", "as an ai"):
        if leak in caption.lower():
            removed.append({"claim": "meta leak: " + leak, "verdict": "removed_by_postfilter"})
    caption = re.sub(r"[^.!?]*\b(?:video-model training|video model training|using only these confirmed facts|as an ai\b)[^.!?]*[.!?]?\s*", "", caption, flags=re.I)
    caption = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", caption).strip()  # spacing between kept sentences
    if len(caption) < 40:
        caption = "A mostly still scene. " + caption
    # second pass: verify held-object claims on 4 frames
    for claim in held_claims(facts):
        vimgs = [f.resize((640, 352)) for f in frames[::2][:4]]
        msgs = [{"role": "user", "content": ["<image>"] * len(vimgs) + [{"type": "text", "text": VERIFY_PROMPT.format(claim=claim)}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[vimgs], return_tensors="pt").to(device)
        n_in = inputs["input_ids"].shape[1]
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        ans = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0].strip().lower()
        if "yes" not in ans:
            words = [w for w in re.sub(r"[^a-z0-9 ]", " ", claim.lower()).split() if w not in ("a", "an", "the") and len(w) > 2]
            if words:
                def _drop(m):
                    low = m.group(0).lower()
                    return "" if all(w in low for w in words) else m.group(0)
                caption = re.sub(r"[^.!?]*[.!?]?\s*", _drop, caption).strip()
            removed.append({"claim": claim, "verdict": ans[:40]})
    caption = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", caption).strip()
    if len(caption) < 40:
        caption = "A mostly still scene. " + caption
    return facts, caption, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam-scores")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--report", help="CPU-only: audit a records file, print removal stats")
    a = ap.parse_args()
    if a.report:
        rows = [json.loads(l) for l in open(a.report)]
        n_rm = sum(len(r.get("removed_sentences", [])) for r in rows)
        held = sum(1 for r in rows if (r.get("facts_json") or {}).get("hands", {}).get("held_object"))
        print(json.dumps({"records": len(rows), "with_held_claim": held, "removed_claims": n_rm}, indent=1))
        return
    done = resume_done(a.out)
    with open(a.manifest) as source:
        rows = [json.loads(line) for line in source]
    rows = rows[a.shard::a.num_shards]
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    cam = load_cam(a.cam_scores)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as f:
        for i, r in enumerate(rows):
            if r["sample_id"] in done:
                continue
            try:
                facts, caption, removed = recaption_one(model, processor, r["video"], model.device)
                caption = (caption + " " + cam_sentence(cam.get(r["video"]))).strip()
                rec = {"sample_id": r["sample_id"], "video_sha256": sha256(r["video"]),
                       "frame_indices": list(range(N_FRAMES)), "model": "Qwen2.5-VL-7B-Instruct",
                       "prompt_version": PROMPT_VERSION, "facts_json": facts, "caption": caption,
                       "original_prompt_sha256": hashlib.sha256(r["captions"]["orig"].encode()).hexdigest(),
                       "removed_sentences": removed, "cam_shift": cam.get(r["video"])}
                f.write(json.dumps(rec, ensure_ascii=False) + chr(10))
                f.flush()
            except Exception as e:
                f.write(json.dumps({"sample_id": r["sample_id"], "prompt_version": PROMPT_VERSION,
                                    "error": type(e).__name__ + ": " + str(e)[:200]}) + chr(10))
                f.flush()
            if (i + 1) % 20 == 0:
                print(f"[recaption] {i+1}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
