"""Local Track 2 VACE temporal-guidance pilot; resource metrics are not quality scores."""
import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path


MODES = ("flatten", "window", "swapped")
DEFAULTS = {"width": 320, "height": 192, "num_frames": 33, "fps": 8,
            "num_inference_steps": 20, "cfg_scale": 5.0}


def build_event_masks(frames, fps, timeline, reference_prefix=1):
    """Half-open event occupancy in Wan groups [0], [1..4], [5..8], ... ."""
    import torch
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1 or frames % 4 != 1:
        raise ValueError("frames must be a positive 4k+1 integer")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if isinstance(reference_prefix, bool) or not isinstance(reference_prefix, int) or reference_prefix < 0:
        raise ValueError("reference_prefix must be a nonnegative integer")
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("timeline must contain events")
    duration = frames / fps
    ranges = []
    for event in timeline:
        if not isinstance(event, dict):
            raise ValueError("timeline event must be an object")
        start, end = event.get("start_s"), event.get("end_s")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (start, end)):
            raise ValueError("timeline endpoints must be finite numeric")
        if not 0 <= start < end <= duration:
            raise ValueError("timeline range must lie within the video duration")
        if not isinstance(event.get("action"), str) or not event["action"].strip():
            raise ValueError("timeline action must be nonempty text")
        ranges.append((start, end))
    ordered = sorted(ranges)
    if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("timeline windows must not overlap")
    groups = [[0]] + [list(range(start, start + 4)) for start in range(1, frames, 4)]
    masks = []
    for start, end in ranges:
        weights = [0.0] * reference_prefix
        weights += [sum(start <= frame / fps < end for frame in group) / len(group) for group in groups]
        masks.append(torch.tensor(weights, dtype=torch.float32).reshape(1, 1, -1, 1, 1))
    if bool((torch.stack(masks).sum(dim=0) > 1).any()):
        raise ValueError("event mask occupancy must sum to at most one")
    return masks


def flatten_prompt(prompt, timeline):
    return prompt + "\n" + "; ".join("%s-%ss: %s" % (event["start_s"], event["end_s"], event["action"])
                                        for event in timeline)


def sample_seed(row, parameters):
    seed = row.get("seed")
    if seed is None:
        base = parameters.get("seed", 42)
        if isinstance(base, bool) or not isinstance(base, int) or base < 0:
            raise ValueError("seed must be a nonnegative integer")
        seed = (base + int(hashlib.sha256(row["sample_id"].encode()).hexdigest()[:8], 16)) % 2**31
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("sample seed must be an integer in [0, 2**31)")
    return seed


class WindowGuidance:
    """Wrap the pipeline's original model_fn without cropping its reference prefix."""
    def __init__(self, original, mode, masks, event_contexts=()):
        import torch
        if mode not in MODES:
            raise ValueError("timeline_guidance_mode must be flatten, window or swapped")
        if len(masks) != 2 or any(not isinstance(mask, torch.Tensor) or mask.ndim != 5
                                  or tuple(mask.shape[:2]) != (1, 1) or tuple(mask.shape[-2:]) != (1, 1)
                                  for mask in masks):
            raise ValueError("exactly two broadcastable torch event masks are required")
        self.original, self.mode = original, mode
        self.mask_weights = [mask.detach().cpu().flatten().tolist() for mask in masks]
        if any(len(weights) != len(self.mask_weights[0]) or any(not math.isfinite(w) or w < 0 or w > 1 for w in weights)
               for weights in self.mask_weights):
            raise ValueError("event mask weights must be finite and in [0, 1]")
        if any(sum(weights) > 1 for weights in zip(*self.mask_weights)):
            raise ValueError("event mask occupancy must sum to at most one")
        self.masks = list(reversed(masks)) if mode == "swapped" else list(masks)
        self.event_contexts = list(event_contexts)
        self.positive_calls = self.negative_calls = self.positive_steps = 0
        self.same_latents = self.same_timestep = self.same_other_kwargs = True
        self.negative_same_step_inputs = True
        self.latent_shape = None
        self.context_shapes = {}
        self.last_positive_inputs = None

    def _positive_prediction(self, kwargs, context):
        call = dict(kwargs, context=context)
        self.same_latents &= call["latents"] is kwargs["latents"]
        self.same_timestep &= call["timestep"] is kwargs["timestep"]
        self.same_other_kwargs &= all(call[key] is value for key, value in kwargs.items() if key != "context")
        self.positive_calls += 1
        prediction = self.original(**call)
        if tuple(prediction.shape) != tuple(kwargs["latents"].shape):
            raise RuntimeError("model_fn must return the complete latent sequence including the reference prefix")
        return prediction

    def __call__(self, **kwargs):
        if kwargs.get("cfg_merge", False) or kwargs.get("tea_cache") is not None:
            raise ValueError("window guidance requires cfg_merge=False and no TeaCache")
        positive = kwargs.get("positive")
        if positive is False:
            self.negative_calls += 1
            if self.last_positive_inputs is not None:
                self.negative_same_step_inputs &= all(kwargs.get(key) is value
                                                     for key, value in self.last_positive_inputs.items())
            return self.original(**kwargs)
        if positive is not True:
            raise ValueError("model_fn calls must retain the upstream positive boolean")
        latents = kwargs["latents"]
        if latents.ndim != 5 or any(mask.shape[2] != latents.shape[2] for mask in self.masks):
            raise ValueError("event mask length must match the full reference-prefixed latent sequence")
        self.latent_shape = list(latents.shape)
        self.positive_steps += 1
        self.last_positive_inputs = {key: kwargs.get(key) for key in ("latents", "timestep", "vace_context")}
        self.context_shapes["global" if self.mode != "flatten" else "flatten"] = list(kwargs["context"].shape)
        global_prediction = self._positive_prediction(kwargs, kwargs["context"])
        if self.mode == "flatten":
            self._positive_prediction(kwargs, kwargs["context"])
            self._positive_prediction(kwargs, kwargs["context"])
            return global_prediction
        if len(self.event_contexts) != len(self.masks):
            raise RuntimeError("event prompts must be encoded once before denoising")
        prediction = global_prediction
        for index, (mask, context) in enumerate(zip(self.masks, self.event_contexts)):
            if mask.device != latents.device or mask.dtype != latents.dtype or context.device != latents.device:
                raise ValueError("masks and contexts must already be on the latent device and dtype")
            self.context_shapes["event_%d" % index] = list(context.shape)
            event_prediction = self._positive_prediction(kwargs, context)
            prediction = prediction + mask * (event_prediction - global_prediction)
        return prediction

    def metrics(self):
        return {"positive_denoiser_calls": self.positive_calls,
                "negative_denoiser_calls": self.negative_calls,
                "positive_steps": self.positive_steps,
                "context_shapes": self.context_shapes,
                "latent_shape": self.latent_shape,
                "event_mask_weights": self.mask_weights,
                "mask_event_indices": [1, 0] if self.mode == "swapped" else [0, 1],
                "same_latents_identity": self.same_latents,
                "same_timestep_identity": self.same_timestep,
                "same_non_context_kwargs_identity": self.same_other_kwargs,
                "negative_same_step_inputs_identity": self.negative_same_step_inputs}


def generate_sample(pipe, embedder, row, parameters, reference, masks, seed):
    """Install only per-sample hooks and restore both original bound attributes."""
    import torch
    mode = parameters["timeline_guidance_mode"]
    original_model_fn, original_process = pipe.model_fn, embedder.process
    guidance = WindowGuidance(original_model_fn, mode, masks)
    prompt = flatten_prompt(row["prompt"], row["timeline"]) if mode == "flatten" else row["prompt"]
    encoding_counts = {"positive": 0, "negative": 0, "event": 0}

    def encode_once(pipe, prompt, positive):
        with torch.no_grad():
            result = original_process(pipe, prompt, positive)
            encoding_counts["positive" if positive else "negative"] += 1
            if positive:
                guidance.masks = [mask.to(device=pipe.device, dtype=pipe.torch_dtype) for mask in guidance.masks]
                if mode != "flatten":
                    guidance.event_contexts = [embedder.encode_prompt(pipe, row["prompt"] + "\n" + event["action"])
                                               for event in row["timeline"]]
                    encoding_counts["event"] += len(guidance.event_contexts)
            return result

    try:
        pipe.model_fn, embedder.process = guidance, encode_once
        with torch.no_grad():
            video = pipe(prompt=prompt, negative_prompt=parameters.get("negative_prompt", ""),
                         vace_reference_image=reference, seed=seed, tiled=True,
                         height=parameters["height"], width=parameters["width"],
                         num_frames=parameters["num_frames"],
                         num_inference_steps=parameters["num_inference_steps"],
                         cfg_scale=parameters["cfg_scale"], cfg_merge=False,
                         tea_cache_l1_thresh=None)
        steps = parameters["num_inference_steps"]
        if guidance.positive_calls != steps * 3 or guidance.negative_calls != steps:
            raise RuntimeError("actual denoiser calls differ from the equal-budget 3 positive + 1 negative per step")
        if encoding_counts != {"positive": 1, "negative": 1, "event": 0 if mode == "flatten" else 2}:
            raise RuntimeError("prompt contexts were not encoded exactly once")
        if len(video) != parameters["num_frames"]:
            raise RuntimeError("generated frame count differs from the requested count")
        return video, dict(guidance.metrics(), prompt_encoding_calls=encoding_counts)
    finally:
        pipe.model_fn, embedder.process = original_model_fn, original_process


def main():
    from ipvg.common import load_json, positive, resolve_path, safe_name, write_json
    from ipvg.upstream import check_upstream, local_model_paths, video_parameters
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = load_json(args.request)
    parameters = dict(DEFAULTS, **request["parameters"])
    root = request["repo_root"]
    mode = parameters.get("timeline_guidance_mode")
    if mode not in MODES:
        raise ValueError("explicit timeline_guidance_mode=flatten/window/swapped is required")
    if request.get("task") != "inference" or request.get("resources", {}).get("gpus", 1) != 1:
        raise ValueError("this adapter only supports single-GPU inference")
    if parameters.get("cfg_merge", False) or parameters.get("tea_cache_l1_thresh") is not None:
        raise ValueError("cfg_merge and TeaCache are not supported")
    if parameters.get("lora_path") or parameters.get("adapter_checkpoint"):
        raise ValueError("this pilot uses the unchanged pretrained VACE model")
    video_parameters(parameters)
    positive(parameters["num_inference_steps"], "num_inference_steps", integer=True)
    positive(parameters["cfg_scale"], "cfg_scale")
    if parameters["cfg_scale"] == 1:
        raise ValueError("cfg_scale=1 disables the negative call and violates the equal-budget pilot")
    prepared = []
    seen = set()
    for row in request["samples"]:
        sample_id = safe_name(row["sample_id"], "sample_id")
        if sample_id in seen:
            raise ValueError("duplicate sample_id")
        seen.add(sample_id)
        if row.get("track") != 2 or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("each sample must have Track 2 and a global prompt")
        if row.get("control_video_path") or row.get("input_image") or row.get("input_video"):
            raise ValueError("only VACE reference conditioning is supported; no I2V or input video")
        if not isinstance(row.get("timeline"), list) or len(row["timeline"]) != 2:
            raise ValueError("the equal-budget pilot requires exactly two timeline events")
        masks = build_event_masks(parameters["num_frames"], parameters["fps"], row["timeline"])
        index = parameters.get("reference_index", 0)
        references = row.get("reference_images", [])
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(references):
            raise ValueError("reference_index out of range")
        path = resolve_path(references[index], root)
        if not path.is_file():
            raise ValueError("reference image is not a local file")
        prepared.append((row, masks, path, sample_seed(row, parameters)))
    if not prepared:
        raise ValueError("no selected samples")
    model_paths, tokenizer = local_model_paths(parameters, root)
    upstream = check_upstream(parameters, root)
    sys.path.insert(0, str(upstream))
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from PIL import Image
    from diffsynth.pipelines.wan_video import WanVideoPipeline, WanVideoUnit_PromptEmbedder, ModelConfig
    from diffsynth.utils.data import save_video
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda",
            model_configs=[ModelConfig(path=path) for path in model_paths],
            tokenizer_config=ModelConfig(path=tokenizer), redirect_common_files=False)
    torch.cuda.synchronize()
    profile = {"kind": "resource_profile_not_quality_score", "timeline_guidance_mode": mode,
               "model_load_seconds": time.monotonic() - started,
               "model_load_peak_allocated_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
               "model_load_peak_reserved_vram_gb": torch.cuda.max_memory_reserved() / 1e9,
               "gpu_name": torch.cuda.get_device_name(), "samples": []}
    metrics_path = Path(request["run_dir"]) / "backend_metrics.json"
    write_json(metrics_path, profile)
    if pipe.vace is None:
        raise RuntimeError("loaded pipeline has no VACE reference module")
    embedders = [unit for unit in pipe.units if isinstance(unit, WanVideoUnit_PromptEmbedder)]
    if len(embedders) != 1:
        raise RuntimeError("expected exactly one actual WanVideoUnit_PromptEmbedder")
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for row, masks, reference_path, seed in prepared:
        with Image.open(reference_path) as image:
            reference = image.convert("RGB").resize((parameters["width"], parameters["height"]))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        video, observations = generate_sample(pipe, embedders[0], row, parameters, reference, masks, seed)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        save_video(video, str(output / (row["sample_id"] + ".mp4")), fps=parameters["fps"], quality=5)
        profile["samples"].append(dict(observations, sample_id=row["sample_id"], seed=seed,
                    generation_seconds=seconds, generated_frames=len(video),
                    reference_index=parameters.get("reference_index", 0),
                    reference_pixels_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                    peak_allocated_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
                    peak_reserved_vram_gb=torch.cuda.max_memory_reserved() / 1e9))
        write_json(metrics_path, profile)


if __name__ == "__main__":
    main()
