import csv
import math
import os

import torch

from util.online_curve import results_dir


def open_chrc_logs(args, method_name="hmem"):
    out_dir = results_dir(args, method_name)
    os.makedirs(out_dir, exist_ok=True)

    scatter_path = os.path.join(out_dir, "chrc_scatter.csv")
    correction_path = os.path.join(out_dir, "correction_norm.csv")
    similarity_path = os.path.join(out_dir, "retrieval_similarity.csv")

    scatter_handle = open(scatter_path, "w", newline="", encoding="utf-8")
    correction_handle = open(correction_path, "w", newline="", encoding="utf-8")
    similarity_handle = open(similarity_path, "w", newline="", encoding="utf-8")

    horizon_path = os.path.join(out_dir, "horizon_effect.csv")
    horizon_handle = open(horizon_path, "w", newline="", encoding="utf-8")

    time_bucket_path = os.path.join(out_dir, "time_buckets_mse.csv")
    time_bucket_handle = open(time_bucket_path, "w", newline="", encoding="utf-8")

    scatter_writer = csv.writer(scatter_handle)
    correction_writer = csv.writer(correction_handle)
    similarity_writer = csv.writer(similarity_handle)
    horizon_writer = csv.writer(horizon_handle)
    time_bucket_writer = csv.writer(time_bucket_handle)

    scatter_writer.writerow(["step", "horizon", "mse_base", "mse_corrected"])
    correction_writer.writerow(["step", "horizon", "correction_norm"])
    similarity_writer.writerow(["step", "mean_similarity"])
    horizon_writer.writerow(["horizon", "metric", "value"])
    time_bucket_writer.writerow(["bucket", "sample_id", "mse"])

    horizon_len = int(getattr(args, "pred_len", 0))
    horizon_acc = {
        "count": 0,
        "correction_sum": [0.0] * horizon_len,
        "mse_improve_sum": [0.0] * horizon_len,
    }

    return {
        "scatter": (scatter_handle, scatter_writer),
        "correction": (correction_handle, correction_writer),
        "similarity": (similarity_handle, similarity_writer),
        "horizon_effect": (horizon_handle, horizon_writer, horizon_acc),
        "time_buckets": (time_bucket_handle, time_bucket_writer),
    }


def close_chrc_logs(handles):
    horizon_effect = handles.get("horizon_effect")
    if horizon_effect is not None:
        handle, writer, acc = horizon_effect
        count = acc.get("count", 0)
        correction_sum = acc.get("correction_sum", [])
        mse_improve_sum = acc.get("mse_improve_sum", [])
        horizon_len = min(len(correction_sum), len(mse_improve_sum))
        for idx in range(horizon_len):
            horizon = idx + 1
            if count:
                correction_mean = correction_sum[idx] / count
                mse_improve_mean = mse_improve_sum[idx] / count
            else:
                correction_mean = float("nan")
                mse_improve_mean = float("nan")
            writer.writerow([horizon, "correction_norm", correction_mean])
            writer.writerow([horizon, "mse_improve", mse_improve_mean])
        handle.close()

    for key, value in handles.items():
        if key == "horizon_effect":
            continue
        handle = value[0]
        handle.close()


def _mean_horizon(values: torch.Tensor) -> torch.Tensor:
    if values.dim() == 2:
        return values.mean(dim=0)
    return values.mean(dim=2).mean(dim=0)


def log_chrc_step(handles, step, base_pred, pred, true, similarities=None, valid_mask=None, bucket_id=None):
    if base_pred is None or pred is None or true is None:
        return

    base_mse = _mean_horizon((base_pred - true) ** 2)
    corrected_mse = _mean_horizon((pred - true) ** 2)
    correction_norm = _mean_horizon(torch.norm(pred - base_pred, dim=2))

    base_mse = base_mse.detach().cpu().tolist()
    corrected_mse = corrected_mse.detach().cpu().tolist()
    correction_norm = correction_norm.detach().cpu().tolist()

    scatter_writer = handles["scatter"][1]
    correction_writer = handles["correction"][1]
    for idx, (base_val, corr_val, norm_val) in enumerate(zip(base_mse, corrected_mse, correction_norm), 1):
        scatter_writer.writerow([step, idx, float(base_val), float(corr_val)])
        correction_writer.writerow([step, idx, float(norm_val)])

    horizon_effect = handles.get("horizon_effect")
    if horizon_effect is not None:
        acc = horizon_effect[2]
        horizon_len = len(correction_norm)
        if len(acc.get("correction_sum", [])) != horizon_len:
            acc["correction_sum"] = [0.0] * horizon_len
            acc["mse_improve_sum"] = [0.0] * horizon_len
            acc["count"] = 0
        for idx in range(horizon_len):
            acc["correction_sum"][idx] += float(correction_norm[idx])
            acc["mse_improve_sum"][idx] += float(base_mse[idx] - corrected_mse[idx])
        acc["count"] += 1

    time_buckets = handles.get("time_buckets")
    if time_buckets is not None and bucket_id is not None:
        mse_per_sample = ((pred - true) ** 2).mean(dim=(1, 2)).detach().cpu().tolist()
        if isinstance(bucket_id, (list, tuple)):
            bucket_ids = list(bucket_id)
        else:
            bucket_ids = [bucket_id] * len(mse_per_sample)
        writer = time_buckets[1]
        for idx, (bucket, mse_val) in enumerate(zip(bucket_ids, mse_per_sample)):
            sample_id = step * len(mse_per_sample) + idx
            writer.writerow([int(bucket), sample_id, float(mse_val)])

    mean_similarity = math.nan
    if similarities is not None and valid_mask is not None:
        sims = similarities.detach()
        mask = valid_mask.detach()
        if mask.any():
            mean_similarity = sims[mask].mean().item()
    handles["similarity"][1].writerow([step, mean_similarity])
