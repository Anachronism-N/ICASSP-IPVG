"""Command line interface. Planning is read-only; execution is an explicit command."""
import argparse
import json
import sys
from pathlib import Path
from .common import load_json, write_json


def parser():
    top = argparse.ArgumentParser(prog="ipvg", description="IPVG data, experiment and local evaluation tools")
    sub = top.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Read-only CPU/GPU/tool inventory")
    doctor.add_argument("--root", default=".")
    doctor.add_argument("--output")
    data = sub.add_parser("data").add_subparsers(dest="data_command", required=True)
    validate = data.add_parser("validate")
    validate.add_argument("manifest")
    validate.add_argument("--root", default=".")
    validate.add_argument("--check-paths", action="store_true")
    split = data.add_parser("split")
    split.add_argument("manifest")
    split.add_argument("--output", required=True)
    split.add_argument("--report")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    split.add_argument("--respect-existing", action="store_true")
    export = data.add_parser("export-vace")
    export.add_argument("manifest")
    export.add_argument("--output", required=True)
    export.add_argument("--root", default=".")
    export.add_argument("--split", choices=["train", "dev"], default="train")
    export.add_argument("--reference-index", type=int, default=0)
    resource = sub.add_parser("resources").add_subparsers(dest="resource_command", required=True)
    plan = resource.add_parser("plan")
    plan.add_argument("--repo-id", required=True)
    plan.add_argument("--repo-type", choices=["model", "dataset"], default="dataset")
    plan.add_argument("--revision", required=True, help="Full immutable 40-character commit SHA")
    plan.add_argument("--allow-pattern", action="append", required=True, dest="allow_patterns")
    plan.add_argument("--max-bytes", type=int, required=True)
    plan.add_argument("--metadata", help="Use a saved HF metadata JSON without a network request")
    plan.add_argument("--output", required=True)
    download = resource.add_parser("download")
    download.add_argument("plan")
    download.add_argument("--target", required=True)
    download.add_argument("--execute", action="store_true")
    download.add_argument("--access-approved", action="store_true")
    budget = sub.add_parser("budget")
    for name, default, kind in [("pairs",1000,int),("runs",1,int),("candidates",1,int),("seconds-per-video",None,float),
                                ("gpus-per-job",8,int),("nodes",2,int),("gpus-per-node",8,int),("efficiency",0.7,float)]:
        budget.add_argument("--"+name, type=kind, default=default, required=default is None)
    exp = sub.add_parser("experiment").add_subparsers(dest="experiment_command", required=True)
    for name in ("plan", "run"):
        child = exp.add_parser(name)
        child.add_argument("config")
        child.add_argument("--root", default=".")
        if name == "run":
            child.add_argument("--resume", action="store_true")
    camp = sub.add_parser("campaign").add_subparsers(dest="campaign_command", required=True).add_parser("run")
    camp.add_argument("config")
    camp.add_argument("--root", default=".")
    camp.add_argument("--resume", action="store_true")
    for name in ("evaluate", "submission"):
        child = sub.add_parser(name)
        child.add_argument("manifest")
        child.add_argument("outputs_dir")
        child.add_argument("--ffprobe", default="ffprobe")
        for field, kind in [("width",int),("height",int),("fps",float),("duration",float)]:
            child.add_argument("--expected-"+field, type=kind)
        child.add_argument("--duration-tolerance", type=float, default=0.25)
        child.add_argument("--fps-tolerance", type=float, default=0.01)
        if name == "evaluate":
            child.add_argument("--proxy-scores", dest="proxy_scores_path")
            child.add_argument("--output", required=True)
            child.add_argument("--fail-on-extra", action="store_true")
        else:
            child.add_argument("--archive", required=True)
    report = sub.add_parser("report")
    report.add_argument("runs_dir")
    report.add_argument("--output", required=True)
    return top


def dispatch(args):
    if args.command == "doctor":
        from .doctor import inventory
        result = inventory(args.root)
        if args.output:
            write_json(args.output, result)
        return result
    if args.command == "data":
        from .data import read_manifest, validate_manifest, split_manifest, write_manifest
        if args.data_command == "validate":
            return validate_manifest(read_manifest(args.manifest), root=args.root, check_paths=args.check_paths)
        if args.data_command == "split":
            rows, report = split_manifest(read_manifest(args.manifest), ratios=args.ratios, seed=args.seed,
                                          respect_existing=args.respect_existing)
            output = Path(args.output).resolve()
            source = Path(args.manifest).resolve()
            report_path = Path(args.report or str(args.output)+".report.json").resolve()
            if output == source:
                raise ValueError("split output must not overwrite the source manifest")
            if report_path in (source, output):
                raise ValueError("split report must not overwrite the source or output manifest")
            write_manifest(args.output, rows)
            write_json(report_path, report)
            return report
        from .upstream import export_vace_csv
        return export_vace_csv(args.manifest, args.output, root=args.root, split=args.split, reference_index=args.reference_index)
    if args.command == "resources":
        from .resources import fetch_hf_metadata, build_download_plan, download_plan
        if args.resource_command == "plan":
            metadata = load_json(args.metadata) if args.metadata else fetch_hf_metadata(args.repo_id, revision=args.revision, repo_type=args.repo_type)
            plan = build_download_plan(metadata, repo_id=args.repo_id, revision=args.revision, repo_type=args.repo_type,
                                       allow_patterns=args.allow_patterns, max_bytes=args.max_bytes)
            write_json(args.output, plan)
            return plan
        return download_plan(load_json(args.plan), args.target, execute=args.execute, access_approved=args.access_approved)
    if args.command == "budget":
        from .budget import inference_budget
        keys = ("pairs","runs","candidates","seconds_per_video","gpus_per_job","nodes","gpus_per_node","efficiency")
        return inference_budget(**{key:getattr(args,key) for key in keys})
    if args.command == "experiment":
        from .experiments import plan_experiment, run_experiment
        if args.experiment_command == "plan":
            return plan_experiment(args.config, root=args.root)
        return run_experiment(args.config, root=args.root, resume=args.resume)
    if args.command == "campaign":
        from .experiments import run_campaign
        return run_campaign(args.config, root=args.root, resume=args.resume)
    if args.command in ("evaluate", "submission"):
        keys = ("ffprobe","expected_width","expected_height","expected_fps","expected_duration","duration_tolerance","fps_tolerance")
        options = {key:getattr(args,key) for key in keys}
        if args.command == "evaluate":
            from .evaluate import evaluate_outputs
            return evaluate_outputs(args.manifest,args.outputs_dir,args.output,
                                    proxy_scores_path=args.proxy_scores_path,
                                    fail_on_extra=args.fail_on_extra,**options)
        from .submission import create_draft_submission
        return create_draft_submission(args.manifest,args.outputs_dir,args.archive,**options)
    if args.command == "report":
        from .experiments import build_report
        return build_report(args.runs_dir,args.output)
    raise ValueError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if result.get("status") == "failed" or result.get("ok") is False else 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"error":str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2
