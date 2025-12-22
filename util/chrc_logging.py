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

    scatter_writer = csv.writer(scatter_handle)
    correction_writer = csv.writer(correction_handle)
    similarity_writer = csv.writer(similarity_handle)

    scatter_writer.writerow(["step", "horizon", "mse_base", "mse_corrected"])
    correction_writer.writerow(["step", "horizon", "correction_norm"])
    similarity_writer.writerow(["step", "mean_similarity"])

    return {
        "scatter": (scatter_handle, scatter_writer),
        "correction": (correction_handle, correction_writer),
        "similarity": (similarity_handle, similarity_writer),
    }


def close_chrc_logs(handles):
    for handle, _writer in handles.values():
        handle.close()


def _mean_horizon(values: torch.Tensor) -> torch.Tensor:
    if values.dim() == 2:
        return values.mean(dim=0)
    return values.mean(dim=2).mean(dim=0)


def log_chrc_step(handles, step, base_pred, pred, true, similarities=None, valid_mask=None):
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

    mean_similarity = math.nan
    if similarities is not None and valid_mask is not None:
        sims = similarities.detach()
        mask = valid_mask.detach()
        if mask.any():
            mean_similarity = sims[mask].mean().item()
    handles["similarity"][1].writerow([step, mean_similarity])
