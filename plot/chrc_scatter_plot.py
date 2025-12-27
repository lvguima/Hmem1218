import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
#python plot\chrc_scatter_plot.py --dataset flotation --seq-len 64 --pred-len 24 --plot-style paired --base-size 7 --corrected-size 7 --line-width 1
def parse_figsize(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("figsize must be in 'width,height' format")
    return float(parts[0]), float(parts[1])


def read_pairs(path, stride=1, horizon_filter=None, step_stride=1):
    steps = []
    bases = []
    corrected = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            step_val = None
            raw_step = row.get("step", None)
            if raw_step is not None:
                try:
                    step_val = int(raw_step)
                except (TypeError, ValueError):
                    step_val = None
            if step_stride > 1:
                if step_val is None:
                    continue
                if step_val % step_stride != 0:
                    continue
            if horizon_filter is not None:
                try:
                    horizon = int(row.get("horizon", 0))
                except (TypeError, ValueError):
                    continue
                if horizon != horizon_filter:
                    continue
            if stride > 1 and idx % stride != 0:
                continue
            try:
                base_val = float(row.get("mse_base", "nan"))
                corrected_val = float(row.get("mse_corrected", "nan"))
            except (TypeError, ValueError):
                continue
            if step_val is None:
                step_val = idx
            steps.append(step_val)
            bases.append(base_val)
            corrected.append(corrected_val)
    return steps, bases, corrected


def infer_name_parts(args, csv_path):
    dataset = args.dataset
    seq_len = args.seq_len if args.seq_len else None
    pred_len = args.pred_len if args.pred_len else None
    if (not dataset or seq_len is None or pred_len is None) and csv_path is not None:
        parts = csv_path.parent.name.split("_")
        if len(parts) >= 3:
            inferred_pred = parts[-1]
            inferred_seq = parts[-2]
            inferred_dataset = "_".join(parts[:-3])
            if not dataset:
                dataset = inferred_dataset
            if seq_len is None:
                seq_len = inferred_seq
            if pred_len is None:
                pred_len = inferred_pred
    return dataset, seq_len, pred_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="", help="Path to chrc_scatter.csv (optional).")
    parser.add_argument("--results-dir", default="results", help="Root results directory.")
    parser.add_argument("--dataset", default="", help="Dataset name, e.g. ETTm1.")
    parser.add_argument("--seq-len", type=int, default=0, help="Sequence length.")
    parser.add_argument("--pred-len", type=int, default=0, help="Prediction length.")
    parser.add_argument("--figsize", default="5,5", help="Figure size, width,height.")
    parser.add_argument("--title", default="", help="Plot title.")
    parser.add_argument("--font-size", type=float, default=15, help="Base font size.")
    parser.add_argument("--plot-style", default="paired", choices=["paired", "scatter"], help="Plot style.")
    parser.add_argument("--dot-size", type=float, default=8, help="Scatter dot size.")
    parser.add_argument("--base-size", type=float, default=0.0, help="Base point size (0 = use dot-size).")
    parser.add_argument("--corrected-size", type=float, default=0.0, help="Corrected point size (0 = use dot-size).")
    parser.add_argument("--alpha", type=float, default=0.4, help="Scatter alpha.")
    parser.add_argument("--stride", type=int, default=1, help="Row sampling stride.")
    parser.add_argument("--step-stride", type=int, default=1, help="Only keep steps divisible by N.")
    parser.add_argument("--horizon", type=int, default=0, help="Filter by horizon (0 = all).")
    parser.add_argument("--line-color", default="#444444", help="Diagonal reference line color.")
    parser.add_argument("--line-width", type=float, default=1.0, help="Pair line width.")
    parser.add_argument("--pair-line-color", default="#444444", help="Pair line color.")
    parser.add_argument("--dot-color", default="#1f77b4", help="Scatter dot color.")
    parser.add_argument("--base-color", default="#1f77b4", help="Base point color.")
    parser.add_argument("--corrected-color", default="#d62728", help="Corrected point color.")
    parser.add_argument("--out", default="", help="Output image path (optional).")
    parser.add_argument("--dpi", type=int, default=550, help="Output DPI.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if args.input:
        csv_path = Path(args.input)
    else:
        if not args.dataset or not args.seq_len or not args.pred_len:
            raise ValueError("When --input is not provided, --dataset/--seq-len/--pred-len are required.")
        results_dir = Path(args.results_dir)
        if not results_dir.exists() and args.results_dir == "results":
            fallback = project_root / "results"
            if fallback.exists():
                results_dir = fallback
        folder = results_dir / f"{args.dataset}_hmem_{args.seq_len}_{args.pred_len}"
        csv_path = folder / "chrc_scatter.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing csv: {csv_path}")

    horizon_filter = args.horizon if args.horizon > 0 else None
    steps, bases, corrected = read_pairs(
        csv_path,
        stride=args.stride,
        horizon_filter=horizon_filter,
        step_stride=args.step_stride,
    )
    if not bases:
        raise RuntimeError("No data loaded. Check filters or file content.")

    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )
    figsize = parse_figsize(args.figsize)
    fig, ax = plt.subplots(figsize=figsize)

    if args.plot_style == "scatter":
        ax.scatter(
            bases,
            corrected,
            s=args.dot_size,
            alpha=args.alpha,
            color=args.dot_color,
            marker="o",
            edgecolors="none",
            linewidths=0,
        )
        min_val = min(min(bases), min(corrected))
        max_val = max(max(bases), max(corrected))
        ax.plot([min_val, max_val], [min_val, max_val], color=args.line_color, linewidth=1.0)
        ax.set_xlabel("Base MSE")
        ax.set_ylabel("Corrected MSE")
    else:
        base_size = args.base_size if args.base_size > 0 else args.dot_size
        corrected_size = args.corrected_size if args.corrected_size > 0 else args.dot_size
        for step_val, base_val, corrected_val in zip(steps, bases, corrected):
            ax.plot(
                [step_val, step_val],
                [base_val, corrected_val],
                color=args.pair_line_color,
                linewidth=args.line_width,
                alpha=args.alpha,
            )
        ax.scatter(
            steps,
            bases,
            s=base_size,
            alpha=args.alpha,
            color=args.base_color,
            label="Base MSE",
            marker="o",
            edgecolors="none",
            linewidths=0,
            zorder=3,
        )
        ax.scatter(
            steps,
            corrected,
            s=corrected_size,
            alpha=args.alpha,
            color=args.corrected_color,
            label="Corrected MSE",
            marker="o",
            edgecolors="none",
            linewidths=0,
            zorder=3,
        )
        ax.set_xlabel("Sample")
        ax.set_ylabel("MSE")
        ax.legend(loc="upper right", frameon=False, markerscale=3)
    if args.title:
        ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    y_formatter = ScalarFormatter(useOffset=False)
    y_formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(y_formatter)

    if args.out:
        out_path = Path(args.out)
    else:
        dataset, seq_len, pred_len = infer_name_parts(args, csv_path)
        if not dataset or seq_len is None or pred_len is None:
            raise ValueError("Provide --dataset/--seq-len/--pred-len or --out to name the output.")
        name = f"{dataset}_{seq_len}_{pred_len}_chrc_scatter.jpg"
        out_path = script_dir / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, format="jpg", bbox_inches="tight")


if __name__ == "__main__":
    main()
