"""Single-GPU local VACE reference inference. GPU behavior requires a server pilot."""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from ipvg.common import load_json, resolve_path, write_json, sha256_file, positive
from ipvg.upstream import check_upstream, local_model_paths, video_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = load_json(args.request)
    parameters, root = request["parameters"], request["repo_root"]
    alpha = positive(parameters.get("lora_alpha", 1), "lora_alpha")
    if alpha > 1 or ("lora_alpha" in parameters and not parameters.get("lora_path")):
        raise ValueError("lora_alpha requires lora_path and 0 < lora_alpha <= 1")
    if parameters.get("lora_path") and parameters.get("adapter_checkpoint"):
        raise ValueError("Choose one trained LoRA or adapter checkpoint")
    if request.get("task") != "inference":
        raise ValueError("VACE reference adapter only supports inference")
    if request.get("resources", {}).get("gpus",1) != 1:
        raise ValueError("This adapter is single-GPU; use a separately verified distributed backend for 8-GPU inference")
    if any(row.get("control_video_path") for row in request["samples"]):
        raise ValueError("This reference-only adapter does not consume control_video_path; use a separately validated control-video backend")
    if any(row["track"] == 2 for row in request["samples"]) and parameters.get("timeline_mode") != "flatten":
        raise ValueError("This baseline requires explicit timeline_mode=flatten for Track 2; structured timing is not implemented")
    model_paths, tokenizer = local_model_paths(parameters, root)
    upstream = check_upstream(parameters, root)
    video_parameters(parameters)
    sys.path.insert(0, str(upstream))
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from PIL import Image
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.utils.data import save_video
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    load_started = time.monotonic()
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[ModelConfig(path=value) for value in model_paths],
        tokenizer_config=ModelConfig(path=tokenizer), redirect_common_files=False)
    torch.cuda.synchronize()
    profile = {"kind": "resource_profile_not_quality_score",
               "model_load_seconds": time.monotonic() - load_started,
               "model_load_peak_allocated_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
               "model_load_peak_reserved_vram_gb": torch.cuda.max_memory_reserved() / 1e9,
               "gpu_name": torch.cuda.get_device_name(),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    metrics_path = Path(request["run_dir"]) / "backend_metrics.json"
    write_json(metrics_path, profile)
    if pipe.vace is None:
        raise RuntimeError("The loaded pipeline has no VACE reference module")
    condition_calls = []
    def observe_condition(module, inputs):
        condition_calls.append({"context_shape": list(inputs[1].shape),
                                "context_nonzero": bool(torch.count_nonzero(inputs[1]).item())})
    condition_hook = pipe.vace.register_forward_pre_hook(observe_condition)
    if parameters.get("lora_path"):
        from diffsynth.core import load_state_dict
        lora_path = resolve_path(parameters["lora_path"],root)
        state = load_state_dict(str(lora_path), device="cpu")
        loader = pipe.lora_loader(torch_dtype=pipe.torch_dtype, device=pipe.device)
        pairs = loader.get_name_dict(state)
        if not pairs or set(state) != {key for pair in pairs.values() for key in pair if key is not None}:
            raise ValueError("LoRA checkpoint contains incomplete or unconsumed parameter pairs")
        converted = loader.convert_state_dict(state)
        snapshots = {}
        for name in pairs:
            module = pipe.vace.get_submodule(name)
            a, b = converted[name + ".lora_A.weight"], converted[name + ".lora_B.weight"]
            if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0] or tuple(module.weight.shape) != (b.shape[0],a.shape[1]):
                raise ValueError("LoRA shape mismatch for " + name)
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("LoRA checkpoint contains nonfinite parameters")
            snapshots[name] = module.weight.detach().cpu().clone()
        pipe.load_lora(pipe.vace, str(lora_path), alpha=alpha)
        changed = sum(not torch.equal(before, pipe.vace.get_submodule(name).weight.detach().cpu())
                      for name,before in snapshots.items())
        profile["lora_reload"] = {"checkpoint_sha256":sha256_file(lora_path), "alpha":alpha,
                                  "matched_pairs":len(pairs),"changed_base_tensors":changed}
        if not changed:
            raise RuntimeError("LoRA reload produced no measurable base-weight change")
    if parameters.get("adapter_checkpoint"):
        from diffsynth.core import load_state_dict
        pipe.vace.load_state_dict(load_state_dict(str(resolve_path(parameters["adapter_checkpoint"],root))), strict=True)
    metrics = []
    profile["samples"] = metrics
    write_json(metrics_path, profile)
    output = Path(request["output_dir"])
    output.mkdir(parents=True,exist_ok=True)
    for row in request["samples"]:
        prompt = row["prompt"]
        if row["track"] == 2:
            prompt += "\n" + "; ".join("%s-%ss: %s" % (e["start_s"],e["end_s"],e["action"]) for e in row["timeline"])
        index = parameters.get("reference_index",0)
        if not isinstance(index,int) or index < 0 or index >= len(row["reference_images"]):
            raise ValueError("reference_index out of range")
        with Image.open(row["reference_images"][index]) as image:
            reference = image.convert("RGB").resize((parameters.get("width",832),parameters.get("height",480)))
        seed = (int(parameters.get("seed",42)) + int(hashlib.sha256(row["sample_id"].encode()).hexdigest()[:8],16)) % 2**31
        condition_calls.clear()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        video = pipe(prompt=prompt, negative_prompt=parameters.get("negative_prompt",""),
                     vace_reference_image=reference, seed=seed, tiled=True,
                     height=parameters.get("height",480), width=parameters.get("width",832),
                     num_frames=parameters.get("num_frames",49),
                     num_inference_steps=parameters.get("num_inference_steps",30),
                     cfg_scale=parameters.get("cfg_scale",5.0))
        torch.cuda.synchronize()
        seconds = time.monotonic()-started
        if not condition_calls or not any(call["context_nonzero"] for call in condition_calls):
            raise RuntimeError("The reference image produced no active VACE context")
        if len(video) != parameters.get("num_frames",49):
            raise RuntimeError("The generated frame count differs from the requested count")
        save_video(video,str(output/(row["sample_id"]+".mp4")),fps=parameters.get("fps",16),quality=5)
        metrics.append({"sample_id":row["sample_id"],"seed":seed,"generation_seconds":seconds,
                        "reference_index":index,"reference_pixels_sha256":hashlib.sha256(reference.tobytes()).hexdigest(),
                        "vace_forward_calls":len(condition_calls),"vace_context_shape":condition_calls[0]["context_shape"],
                        "generated_frames":len(video),
                        "peak_allocated_vram_gb":torch.cuda.max_memory_allocated()/1e9,
                        "peak_reserved_vram_gb":torch.cuda.max_memory_reserved()/1e9})
        write_json(metrics_path, profile)
    condition_hook.remove()


if __name__ == "__main__":
    main()
