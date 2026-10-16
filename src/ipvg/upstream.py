"""Local-only adapters for the pinned DiffSynth API; no torch dependency here."""
import csv
import json
import subprocess
import sys
from pathlib import Path
from .common import load_json, positive, resolve_path
from .data import read_manifest, validate_manifest


def export_vace_rows(samples, output, *, root=".", reference_index=0):
    rows = list(samples)
    validate_manifest(rows, root=root, check_paths=True)
    positive(reference_index, "reference_index", allow_zero=True, integer=True)
    if not rows:
        raise ValueError("No selected training rows")
    target = Path(output)
    if target.exists():
        raise FileExistsError("Refusing to replace an existing metadata CSV")
    controls = [bool(row.get("control_video_path")) for row in rows]
    if any(controls) and not all(controls):
        raise ValueError("Control video conditioning must be consistently present or absent")
    fields = ["video", "vace_reference_image", "prompt"] + (["vace_video"] if all(controls) else [])
    converted = []
    for row in rows:
        if not row.get("video_path"):
            raise ValueError("training requires video_path: %s" % row["sample_id"])
        if reference_index >= len(row["reference_images"]):
            raise ValueError("reference_index out of range: %s" % row["sample_id"])
        item = {"video": str(resolve_path(row["video_path"], root)),
                "vace_reference_image": str(resolve_path(row["reference_images"][reference_index], root)),
                "prompt": row["prompt"]}
        if all(controls):
            control = resolve_path(row["control_video_path"], root)
            if not control.is_file():
                raise ValueError("control_video_path not found: %s" % control)
            item["vace_video"] = str(control)
        converted.append(item)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(converted)
    return {"samples": len(converted), "csv": str(target.resolve()), "fields": fields,
            "reference_index": reference_index, "limitation": "One reference per row; no timestamped-action conditioning is added"}


def export_vace_csv(manifest, output, *, root=".", split="train", reference_index=0):
    rows = read_manifest(manifest)
    report = validate_manifest(rows)
    if not report["leakage"]["safe"]:
        raise ValueError("Manifest contains split leakage")
    return export_vace_rows([r for r in rows if r.get("split") == split], output, root=root, reference_index=reference_index)


def local_model_paths(parameters, root):
    models = parameters.get("model_files")
    if not isinstance(models, list) or len(models) != 3:
        raise ValueError("model_files must contain [transformer file(s), text encoder file, VAE file]")
    def convert(value):
        if isinstance(value, list):
            if not value:
                raise ValueError("empty model shard list")
            return [convert(v) for v in value]
        if not isinstance(value, str):
            raise ValueError("model paths must be strings")
        path = resolve_path(value, root)
        if not path.is_file():
            raise ValueError("local model file not found: %s" % path)
        return str(path)
    resolved = [convert(value) for value in models]
    tokenizer = resolve_path(parameters["tokenizer_dir"], root)
    if not tokenizer.is_dir():
        raise ValueError("local tokenizer_dir not found")
    return resolved, str(tokenizer)


def check_upstream(parameters, root):
    upstream = resolve_path(parameters["upstream_dir"], root)
    lock = load_json(Path(root) / "resources/upstream.lock.json")["repositories"]["diffsynth"]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True, text=True, timeout=10)
    if result.returncode or result.stdout.strip() != lock["commit"]:
        raise ValueError("DiffSynth checkout does not match resources/upstream.lock.json")
    return upstream


def video_parameters(parameters):
    values = {"width": parameters.get("width", 832), "height": parameters.get("height", 480),
              "num_frames": parameters.get("num_frames", 49), "fps": parameters.get("fps", 16)}
    for key, value in values.items():
        positive(value, key, integer=key != "fps")
    if values["width"] % 16 or values["height"] % 16 or values["num_frames"] % 4 != 1:
        raise ValueError("VACE dimensions must be divisible by 16; num_frames must be 4k+1")
    return values


def build_training_command(request, metadata_report, model_paths, tokenizer, upstream):
    """Construct argv against checked parsers.py, never a shell string."""
    parameters = request["parameters"]
    video = video_parameters(parameters)
    processes = parameters.get("num_processes", 1)
    positive(processes, "num_processes", integer=True)
    declared = request.get("resources", {}).get("gpus", processes)
    if processes > declared:
        raise ValueError("num_processes exceeds declared experiment GPUs")
    if parameters.get("mode", "lora") not in ("adapter", "lora"):
        raise ValueError("training mode must be adapter or lora")
    for key, default in [("num_epochs",1),("dataset_repeat",1),("save_steps",50),
                         ("gradient_accumulation_steps",1),("lora_rank",16)]:
        positive(parameters.get(key, default), key, integer=True)
    positive(parameters.get("learning_rate",5e-5), "learning_rate")
    command = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", str(processes),
               "--mixed_precision", "bf16"]
    if processes > 1:
        command.append("--multi_gpu")
    if parameters.get("accelerate_config"):
        raise ValueError("Custom FSDP/DeepSpeed launchers need a separate validated backend config; this adapter is DDP only")
    command.extend([str(Path(request["repo_root"]) / "scripts/adapters/training_entry.py"),
                    "--upstream-train", str(upstream / "examples/wanvideo/model_training/train.py"),
                    "--seed", str(parameters.get("seed",42)), "--",
                    "--dataset_base_path", request["repo_root"],
                    "--dataset_metadata_path", metadata_report["csv"],
                    "--data_file_keys", ",".join(x for x in metadata_report["fields"] if x != "prompt"),
                    "--extra_inputs", ",".join(x for x in metadata_report["fields"] if x not in ("prompt", "video")),
                    "--model_paths", json.dumps(model_paths), "--tokenizer_path", tokenizer,
                    "--output_path", request["output_dir"], "--remove_prefix_in_ckpt", "pipe.vace.",
                    "--height", str(video["height"]), "--width", str(video["width"]),
                    "--num_frames", str(video["num_frames"]),
                    "--num_epochs", str(parameters.get("num_epochs",1)),
                    "--dataset_repeat", str(parameters.get("dataset_repeat",1)),
                    "--learning_rate", str(parameters.get("learning_rate",5e-5)),
                    "--save_steps", str(parameters.get("save_steps",50)),
                    "--gradient_accumulation_steps", str(parameters.get("gradient_accumulation_steps",1)),
                    "--use_gradient_checkpointing_offload"])
    if parameters.get("mode", "lora") == "adapter":
        command.extend(["--trainable_models", "vace"])
    else:
        command.extend(["--lora_base_model", "vace", "--lora_rank", str(parameters.get("lora_rank",16)),
                        "--lora_target_modules", "q,k,v,o,ffn.0,ffn.2"])
    return command
