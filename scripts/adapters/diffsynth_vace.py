"""Single-GPU local VACE reference inference. GPU behavior requires a server pilot."""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from ipvg.common import load_json, resolve_path, write_json
from ipvg.upstream import check_upstream, local_model_paths, video_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = load_json(args.request)
    parameters, root = request["parameters"], request["repo_root"]
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
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[ModelConfig(path=value) for value in model_paths],
        tokenizer_config=ModelConfig(path=tokenizer), redirect_common_files=False)
    if parameters.get("lora_path") and parameters.get("adapter_checkpoint"):
        raise ValueError("Choose one trained LoRA or adapter checkpoint")
    if parameters.get("lora_path"):
        pipe.load_lora(pipe.vace, str(resolve_path(parameters["lora_path"],root)), alpha=1)
    if parameters.get("adapter_checkpoint"):
        from diffsynth.core import load_state_dict
        pipe.vace.load_state_dict(load_state_dict(str(resolve_path(parameters["adapter_checkpoint"],root))), strict=True)
    metrics = []
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
        save_video(video,str(output/(row["sample_id"]+".mp4")),fps=parameters.get("fps",16),quality=5)
        metrics.append({"sample_id":row["sample_id"],"seed":seed,"generation_seconds":seconds,
                        "peak_allocated_vram_gb":torch.cuda.max_memory_allocated()/1e9})
    write_json(Path(request["run_dir"])/"backend_metrics.json", {"kind":"resource_profile_not_quality_score","samples":metrics})


if __name__ == "__main__":
    main()
