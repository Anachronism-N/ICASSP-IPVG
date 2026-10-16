"""Run the CPU engineering workflow, including real MP4 decoding and draft ZIP."""
import argparse
import json
from pathlib import Path
from uuid import uuid4
from ipvg.common import load_json, write_json
from ipvg.doctor import inventory
from ipvg.experiments import run_experiment, build_report
from ipvg.submission import create_draft_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_json(root/"configs/experiments/synthetic_smoke.json")
    config["run_id"] = args.run_id or "smoke_"+uuid4().hex[:12]
    config_path = root/"runs/_configs"/(config["run_id"]+".json")
    write_json(config_path,config)
    run_dir = root/"runs"/config["run_id"]
    result = run_experiment(config_path,root=root)
    if result["status"] != "completed":
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 1
    write_json(run_dir/"doctor.json",inventory(root))
    draft = create_draft_submission(run_dir/"manifest.jsonl",run_dir/"outputs",run_dir/"draft.zip",
        expected_width=64,expected_height=64,expected_fps=8,expected_duration=1)
    build_report(root/"runs",root/"runs/REPORT.md")
    print(json.dumps({"status":"passed","synthetic_only":True,"run_id":config["run_id"],
                      "run_dir":str(run_dir),"draft_status":draft["status"]},ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
