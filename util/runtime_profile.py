import csv
import os
from typing import List, Tuple

import numpy as np

from util.online_curve import results_dir


def init_runtime_log(args, method_name: str) -> Tuple[object, csv.writer, str]:
    out_dir = results_dir(args, method_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "runtime_profile.csv")
    handle = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    writer.writerow(["step", "ms_per_step", "samples_per_sec"])
    return handle, writer, out_dir


def log_runtime_step(writer: csv.writer, step: int, ms_per_step: float, samples_per_sec: float) -> None:
    writer.writerow([step, ms_per_step, samples_per_sec])


def finalize_runtime_log(handle, out_dir: str, ms_values: List[float], sps_values: List[float]) -> None:
    handle.close()
    if not ms_values:
        return
    ms = np.array(ms_values, dtype=float)
    sps = np.array(sps_values, dtype=float)
    summary_path = os.path.join(out_dir, "runtime_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["metric", "mean", "std", "median", "p90", "count"])
        writer.writerow([
            "ms_per_step",
            float(ms.mean()),
            float(ms.std()),
            float(np.median(ms)),
            float(np.percentile(ms, 90)),
            int(ms.size),
        ])
        writer.writerow([
            "samples_per_sec",
            float(sps.mean()),
            float(sps.std()),
            float(np.median(sps)),
            float(np.percentile(sps, 90)),
            int(sps.size),
        ])
