import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
#python plot\correction_norm_plot.py --dataset ETTm1 --seq-len 512 --pred-len 24 --kde

def parse_figsize(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("figsize must be in 'width,height' format")
    return float(parts[0]), float(parts[1])


def read_correction_norm(path, horizon_filter=None, step_stride=1):
    values = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if step_stride > 1:
                try:
                    step = int(row.get("step", 0))
                except (TypeError, ValueError):
                    continue
                if step % step_stride != 0:
                    continue
            if horizon_filter is not None:
                try:
                    horizon = int(row.get("horizon", 0))
                except (TypeError, ValueError):
                    continue
                if horizon != horizon_filter:
                    continue
            try:
                values.append(float(row.get("correction_norm", "nan")))
            except (TypeError, ValueError):
                continue
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="", help="Path to correction_norm.csv (optional).")
    parser.add_argument("--results-dir", default="results", help="Root results directory.")
    parser.add_argument("--dataset", default="", help="Dataset name, e.g. ETTm1.")
    parser.add_argument("--seq-len", type=int, default=0, help="Sequence length.")
    parser.add_argument("--pred-len", type=int, default=0, help="Prediction length.")
    parser.add_argument("--figsize", default="6,4", help="Figure size, width,height.")
    parser.add_argument("--title", default="", help="Plot title.")
    parser.add_argument("--font-size", type=float, default=12, help="Base font size.")
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Histogram alpha.")
    parser.add_argument("--color", default="#1f77b4", help="Histogram color.")
    parser.add_argument("--kde", action="store_true", help="Overlay KDE curve.")
    parser.add_argument("--kde-color", default="#d62728", help="KDE line color.")
    parser.add_argument("--kde-width", type=float, default=1.5, help="KDE line width.")
    parser.add_argument("--step-stride", type=int, default=1, help="Only keep steps divisible by N.")
    parser.add_argument("--horizon", type=int, default=0, help="Filter by horizon (0 = all).")
    parser.add_argument("--log-y", action="store_true", help="Log scale for y-axis.")
    parser.add_argument("--no-grid", action="store_true", help="Disable grid.")
    parser.add_argument("--out", default="", help="Output image path (optional).")
    parser.add_argument("--dpi", type=int, default=450, help="Output DPI.")
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
        csv_path = folder / "correction_norm.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing csv: {csv_path}")

    horizon_filter = args.horizon if args.horizon > 0 else None
    values = read_correction_norm(csv_path, horizon_filter=horizon_filter, step_stride=args.step_stride)
    if not values:
        raise RuntimeError("No data loaded. Check filters or file content.")

    plt.rcParams.update({"font.size": args.font_size})
    figsize = parse_figsize(args.figsize)
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(values, bins=args.bins, color=args.color, alpha=args.alpha)
    if args.kde:
        data = np.array(values, dtype=float)
        std = float(np.std(data))
        n = data.size
        if n > 1 and std > 0:
            bw = 1.06 * std * (n ** (-1.0 / 5.0))
            if bw > 0:
                grid = np.linspace(data.min(), data.max(), 200)
                diff = (grid[:, None] - data[None, :]) / bw
                kde = np.exp(-0.5 * diff * diff).sum(axis=1)
                kde /= (n * bw * np.sqrt(2 * np.pi))
                bin_width = (data.max() - data.min()) / max(1, args.bins)
                ax.plot(grid, kde * n * bin_width, color=args.kde_color, linewidth=args.kde_width)
    ax.set_xlabel("Correction Norm")
    ax.set_ylabel("Count")
    if args.title:
        ax.set_title(args.title)
    if args.log_y:
        ax.set_yscale("log")
    if not args.no_grid:
        ax.grid(True, axis="y", alpha=0.3)

    if args.out:
        out_path = Path(args.out)
    else:
        name = f"{args.dataset}_{args.seq_len}_{args.pred_len}_correction_norm.jpg"
        out_path = script_dir / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, format="jpg")


if __name__ == "__main__":
    main()
