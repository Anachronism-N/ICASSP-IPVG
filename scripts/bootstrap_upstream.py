"""Fetch one pinned code repository. Does not install packages or fetch models."""
import argparse
import subprocess
from pathlib import Path
from ipvg.common import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=["diffsynth"], required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    record = load_json(root/"resources/upstream.lock.json")["repositories"][args.name]
    target = Path(args.target).resolve()
    if target.exists():
        raise FileExistsError("Choose a new target; existing directories are not replaced")
    target.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(["git","clone","--filter=blob:none","--no-checkout",record["url"],str(target)],check=True,timeout=180)
    subprocess.run(["git","-C",str(target),"fetch","--depth","1","origin",record["commit"]],check=True,timeout=180)
    subprocess.run(["git","-C",str(target),"checkout","--detach",record["commit"]],check=True,timeout=120)
    print("Pinned code is ready. Install GPU dependencies in a separate environment after checking the server CUDA version.")


if __name__ == "__main__":
    main()
