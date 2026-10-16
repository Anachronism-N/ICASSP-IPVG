"""Budget arithmetic. Input throughput must come from a relevant pilot."""
import math
from .common import positive


def inference_budget(*, pairs, runs=1, candidates=1, seconds_per_video, gpus_per_job=8,
                     nodes=2, gpus_per_node=8, efficiency=0.7):
    for label, value in [("pairs", pairs), ("runs", runs), ("candidates", candidates),
                         ("gpus_per_job", gpus_per_job), ("nodes", nodes), ("gpus_per_node", gpus_per_node)]:
        positive(value, label, integer=True)
    positive(seconds_per_video, "seconds_per_video")
    positive(efficiency, "efficiency")
    if runs > 3:
        raise ValueError("The current public rules permit at most three complete submission runs per track")
    if efficiency > 1:
        raise ValueError("efficiency must be in (0, 1]")
    concurrent = int(nodes) * (int(gpus_per_node) // int(gpus_per_job))
    if concurrent < 1:
        raise ValueError("This estimator supports independent jobs fitting within one node")
    videos = int(pairs * runs * candidates)
    ideal_gpu_hours = videos * seconds_per_video * gpus_per_job / 3600
    # ceil includes the final partial wave, important for small pilots.
    planned_hours = math.ceil(videos / concurrent) * seconds_per_video / 3600 / efficiency
    return {"videos": videos, "concurrent_jobs": concurrent, "theoretical_gpu_hours": ideal_gpu_hours,
            "planned_wall_hours": planned_hours, "planned_days": planned_hours / 24,
            "efficiency_assumption": efficiency, "exceeds_seven_days": planned_hours >= 168,
            "scope": "Generation only; no identity preprocessing, scoring or upload. No cross-node speedup assumed."}
