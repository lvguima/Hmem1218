import csv
import os

import numpy as np


def resolve_method_name(args, override=None, default="frozen"):
    name = override if override else getattr(args, "online_method", None) or default
    return str(name).lower()


def results_dir(args, method_name):
    dataset = str(getattr(args, "dataset", "data")).lower()
    folder = f"{dataset}_{method_name}_{args.seq_len}_{args.pred_len}"
    return os.path.join("results", folder)


def compute_step_mse(pred, true, target_variate=None):
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    if target_variate is not None:
        pred = pred[:, :, target_variate]
        if true.dim() == 3:
            true = true[:, :, target_variate]
    diff = pred - true
    if diff.dim() == 2:
        mse = (diff ** 2).mean(dim=1)
    else:
        mse = (diff ** 2).mean(dim=(1, 2))
    return mse.detach().cpu().numpy().tolist()


def compute_rolling(values, window):
    if not values:
        return []
    window = max(1, int(window))
    rolling = []
    running_sum = 0.0
    window_vals = []
    for value in values:
        value = float(value)
        window_vals.append(value)
        running_sum += value
        if len(window_vals) > window:
            running_sum -= window_vals.pop(0)
        rolling.append(running_sum / len(window_vals))
    return rolling


def compute_segments(values):
    count = len(values)
    if count == 0:
        return [("Early", float("nan")), ("Middle", float("nan")), ("Late", float("nan"))]
    first = int(count * 0.33)
    second = int(count * 0.66)
    segments = [
        ("Early", values[:first]),
        ("Middle", values[first:second]),
        ("Late", values[second:]),
    ]
    results = []
    for name, segment in segments:
        if not segment:
            results.append((name, float("nan")))
        else:
            results.append((name, float(np.mean(segment))))
    return results


def save_online_curve_csv(args, raw_mse, method_name=None, window=500):
    method = resolve_method_name(args, override=method_name)
    out_dir = results_dir(args, method)
    os.makedirs(out_dir, exist_ok=True)

    rolling = compute_rolling(raw_mse, window)
    rolling_path = os.path.join(out_dir, "rolling_mse.csv")
    with open(rolling_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "method", "mse", "rolling_mse"])
        for step, (mse, roll) in enumerate(zip(raw_mse, rolling)):
            writer.writerow([step, method, float(mse), float(roll)])

    segment_path = os.path.join(out_dir, "segment_mse.csv")
    with open(segment_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["segment", "method", "mse_mean"])
        for segment, mean in compute_segments(raw_mse):
            writer.writerow([segment, method, mean])
