"""Set a per-rank training seed before executing the pinned upstream entrypoint."""
import argparse
import os
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-train",required=True)
    parser.add_argument("--seed",type=int,required=True)
    args,remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    from accelerate.utils import set_seed
    # accelerate launch sets RANK but has not created AcceleratorState yet.
    # device_specific=True here would require an initialized Accelerator.
    set_seed(args.seed + int(os.environ.get("RANK", "0")),device_specific=False)
    sys.argv = [args.upstream_train]+remaining
    runpy.run_path(args.upstream_train,run_name="__main__")


if __name__ == "__main__":
    main()
