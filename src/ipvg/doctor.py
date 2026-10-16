"""Read-only server inventory without importing large GPU frameworks."""
import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from .common import utc_now


def inventory(root="."):
    root = Path(root).resolve()
    disk = shutil.disk_usage(root)
    result = {"timestamp_utc": utc_now(), "python": sys.version.split()[0], "platform": platform.platform(),
              "cpu_count": __import__("os").cpu_count(), "disk_free_gb": round(disk.free / 1e9, 2),
              "tools": {name: shutil.which(name) for name in ("git", "ffmpeg", "ffprobe", "nvidia-smi", "srun", "sbatch")},
              "packages": {name: importlib.util.find_spec(name) is not None for name in ("torch", "diffsynth", "huggingface_hub")},
              "gpu_query": None, "interconnect": "Run nvidia-smi topo -m and a site-approved NCCL test on the actual allocation"}
    if result["tools"]["nvidia-smi"]:
        command = [result["tools"]["nvidia-smi"], "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]
        try:
            probe = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result["gpu_query"] = probe.stdout.strip() if probe.returncode == 0 else "query_failed"
        except (OSError, subprocess.TimeoutExpired):
            result["gpu_query"] = "query_failed"
    return result
