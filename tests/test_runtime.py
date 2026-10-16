"""Budget, real-media CPU orchestration and upstream adapter contract checks."""
import csv
import json
import shutil
import importlib.util
import sys
import types
from pathlib import Path
import pytest
from ipvg.budget import inference_budget
from ipvg.cli import main
from ipvg.common import write_json
from ipvg.experiments import run_experiment
from ipvg.upstream import export_vace_csv, local_model_paths, video_parameters, build_training_command


def row(split="train"):
    return {"sample_id":"one","identity_id":"subject","source_video_id":"source","reference_images":["face.png"],
            "video_path":"source.mp4","prompt":"A subject moves, then stops.","track":1,"split":split,"timeline":[]}


def test_official_reference_budget_and_partial_waves():
    result = inference_budget(pairs=1000,runs=3,seconds_per_video=121.6,nodes=2)
    assert result["theoretical_gpu_hours"] == pytest.approx(810.6666667)
    assert result["planned_wall_hours"] == pytest.approx(72.38095238)
    assert inference_budget(pairs=1,seconds_per_video=3600,nodes=4,efficiency=1)["planned_wall_hours"] == 1


@pytest.mark.parametrize("change", [{"runs":4},{"pairs":0},{"pairs":1.2},{"efficiency":1.1},
                                     {"seconds_per_video":float("nan")},{"gpus_per_job":16}])
def test_budget_rejects_invalid_inputs(change):
    args = {"pairs":1000,"seconds_per_video":121.6}
    args.update(change)
    with pytest.raises(ValueError):
        inference_budget(**args)


def test_export_vace_preserves_prompt_and_local_fields(tmp_path):
    (tmp_path/"face.png").write_bytes(b"fixture")
    (tmp_path/"source.mp4").write_bytes(b"fixture")
    manifest=tmp_path/"manifest.jsonl"
    manifest.write_text(json.dumps(row())+"\n",encoding="utf-8")
    output=tmp_path/"metadata.csv"
    result=export_vace_csv(manifest,output,root=tmp_path)
    with output.open(encoding="utf-8",newline="") as handle:
        exported=list(csv.DictReader(handle))
    assert result["samples"] == 1
    assert exported[0]["prompt"] == row()["prompt"]
    assert Path(exported[0]["video"]) == tmp_path/"source.mp4"
    assert exported[0]["vace_reference_image"] == str(tmp_path/"face.png")
    with pytest.raises(FileExistsError):
        export_vace_csv(manifest,output,root=tmp_path)


def test_export_refuses_leaking_splits(tmp_path):
    train,dev=row(),row("dev")
    dev["sample_id"]="two"
    manifest=tmp_path/"manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(x) for x in [train,dev]),encoding="utf-8")
    with pytest.raises(ValueError,match="leakage"):
        export_vace_csv(manifest,tmp_path/"out.csv",root=tmp_path)


def test_local_model_paths_and_frame_constraints(tmp_path):
    for name in ["dit","t5","vae"]:
        (tmp_path/name).write_bytes(b"fixture")
    (tmp_path/"tokenizer").mkdir()
    paths,tokenizer=local_model_paths({"model_files":[["dit"],"t5","vae"],"tokenizer_dir":"tokenizer"},tmp_path)
    assert paths[0]==[str(tmp_path/"dit")]
    assert tokenizer==str(tmp_path/"tokenizer")
    with pytest.raises(ValueError):
        video_parameters({"width":831})
    with pytest.raises(ValueError):
        video_parameters({"num_frames":48})


def test_training_argv_is_local_bounded_and_uses_known_flags(tmp_path):
    request={"parameters":{"num_processes":8},"resources":{"gpus":8},"repo_root":str(tmp_path),"output_dir":str(tmp_path/"output")}
    metadata={"csv":str(tmp_path/"metadata.csv"),"fields":["video","vace_reference_image","prompt"]}
    command=build_training_command(request,metadata,["dit","t5","vae"],"tokenizer",tmp_path/"upstream")
    assert "--model_paths" in command
    assert "--model_id_with_origin_paths" not in command
    assert "--multi_gpu" in command
    assert command[command.index("--extra_inputs")+1] == "vace_reference_image"
    request["resources"]["gpus"]=1
    with pytest.raises(ValueError,match="declared"):
        build_training_command(request,metadata,["dit","t5","vae"],"tokenizer",tmp_path)


def test_training_entry_seeds_before_accelerator_without_device_state(monkeypatch,tmp_path):
    entry=Path(__file__).resolve().parents[1]/"scripts/adapters/training_entry.py"
    spec=importlib.util.spec_from_file_location("ipvg_training_entry_test",entry)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls=[]
    fake=types.ModuleType("accelerate.utils")
    fake.set_seed=lambda seed,device_specific: calls.append((seed,device_specific))
    monkeypatch.setitem(sys.modules,"accelerate.utils",fake)
    monkeypatch.setenv("RANK","3")
    monkeypatch.setattr(sys,"argv",[str(entry),"--upstream-train",str(tmp_path/"train.py"),"--seed","42","--","--height","480"])
    executed=[]
    monkeypatch.setattr(module.runpy,"run_path",lambda path,run_name:executed.append((path,run_name,list(sys.argv))))
    module.main()
    assert calls==[(45,False)]
    assert executed[0][2]==[str(tmp_path/"train.py"),"--height","480"]


def test_reference_adapter_refuses_ignored_control_before_gpu_import(monkeypatch,tmp_path):
    entry=Path(__file__).resolve().parents[1]/"scripts/adapters/diffsynth_vace.py"
    spec=importlib.util.spec_from_file_location("ipvg_vace_adapter_test",entry)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request={"task":"inference","parameters":{},"repo_root":str(tmp_path),"resources":{"gpus":1},
             "samples":[{"control_video_path":"pose.mp4"}]}
    request_path=tmp_path/"request.json"
    write_json(request_path,request)
    monkeypatch.setattr(sys,"argv",[str(entry),"--request",str(request_path)])
    with pytest.raises(ValueError,match="reference-only"):
        module.main()


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),reason="FFmpeg tools not installed")
def test_real_synthetic_video_cli_evaluation_and_draft(tmp_path,capsys):
    manifest=tmp_path/"manifest.jsonl"
    manifest.write_text(json.dumps(row("dev"))+"\n",encoding="utf-8")
    config={"schema_version":1,"run_id":"cpu_test","hypothesis":"Verify engineering only", "manifest":str(manifest),
            "backend":{"kind":"synthetic"},"resources":{"gpus":0,"timeout_s":30,"max_gpu_hours":0},
            "parameters":{"width":64,"height":64,"fps":8,"duration_s":0.5}}
    config_path=tmp_path/"experiment.json"
    write_json(config_path,config)
    result=run_experiment(config_path,root=tmp_path)
    assert result["status"] == "completed",result
    assert result["synthetic"] is True
    assert result["estimated_backend_gpu_hours"] == 0
    run_dir=tmp_path/"runs/cpu_test"
    assert main(["evaluate",str(manifest),str(run_dir/"outputs"),"--output",str(tmp_path/"evaluation.json")]) == 0
    assert main(["submission",str(manifest),str(run_dir/"outputs"),"--archive",str(tmp_path/"draft.zip")]) == 0
    assert (tmp_path/"draft.zip").is_file()
    capsys.readouterr()
