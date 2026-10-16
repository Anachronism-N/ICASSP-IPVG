"""Launch pinned upstream VACE DDP training on selected local train records."""
import argparse
import os
import subprocess
from pathlib import Path
from ipvg.common import load_json, write_json
from ipvg.upstream import check_upstream, local_model_paths, export_vace_rows, build_training_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",required=True)
    args = parser.parse_args()
    request = load_json(args.request)
    if request.get("task") != "training" or any(r.get("split") != "train" for r in request["samples"]):
        raise ValueError("Training backend requires explicitly selected train records")
    parameters,root = request["parameters"],request["repo_root"]
    models,tokenizer = local_model_paths(parameters,root)
    upstream = check_upstream(parameters,root)
    csv = export_vace_rows(request["samples"],Path(request["run_dir"])/"training_metadata.csv",root=root,
                           reference_index=parameters.get("reference_index",0))
    command = build_training_command(request,csv,models,tokenizer,upstream)
    write_json(Path(request["run_dir"])/"upstream_command.json",{"argv":command,"gpu_verified":False})
    env = os.environ.copy()
    env["PYTHONPATH"] = str(upstream)+os.pathsep+env.get("PYTHONPATH","")
    env.update({"DIFFSYNTH_SKIP_DOWNLOAD":"True","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
    # The experiment controller owns the process group and kills descendants at timeout.
    subprocess.run(command,cwd=root,env=env,check=True,stdin=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
