import csv
import os
from typing import Dict, Optional

from util.online_curve import results_dir, resolve_method_name


def save_test_summary(
    args,
    method_name: str,
    metrics: Dict[str, float],
    params_total: Optional[int] = None,
    params_trainable: Optional[int] = None,
    memory_mb: Optional[float] = None,
) -> None:
    method = resolve_method_name(args, override=method_name)
    out_dir = results_dir(args, method)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "test_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
        if params_total is not None:
            writer.writerow(["params_total", int(params_total)])
        if params_trainable is not None:
            writer.writerow(["params_trainable", int(params_trainable)])
        if memory_mb is not None:
            writer.writerow(["memory_mb", float(memory_mb)])
