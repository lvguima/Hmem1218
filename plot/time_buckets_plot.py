import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

#python plot\time_buckets_plot.py --dataset ETTm1 --seq-len 512 --pred-len 24

def parse_figsize(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("figsize must be in 'width,height' format")
    return float(parts[0]), float(parts[1])


def read_time_buckets(path):
    data = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                bucket = int(row.get("bucket", 0))
                mse = float(row.get("mse", "nan"))
            except (TypeError, ValueError):
                continue
            data.setdefault(bucket, []).append(mse)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="", help="Path to time_buckets_mse.csv (optional).")
    parser.add_argument("--results-dir", default="results", help="Root results directory.")
    parser.add_argument("--dataset", default="", help="Dataset name, e.g. ETTm1.")
    parser.add_argument("--seq-len", type=int, default=0, help="Sequence length.")
    parser.add_argument("--pred-len", type=int, default=0, help="Prediction length.")
    parser.add_argument("--bucket-count", type=int, default=4, help="Number of buckets to show.")
    parser.add_argument("--figsize", default="6,4", help="Figure size, width,height.")
    parser.add_argument("--title", default="", help="Plot title.")
    parser.add_argument("--font-size", type=float, default=12, help="Base font size.")
    parser.add_argument("--show-means", action="store_true", help="Show mean markers.")
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
        csv_path = folder / "time_buckets_mse.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing csv: {csv_path}")

    data = read_time_buckets(csv_path)
    if not data:
        raise RuntimeError("No time bucket data loaded.")

    bucket_count = max(1, args.bucket_count)
    buckets = list(range(bucket_count))
    values = [data.get(b, []) for b in buckets]

    plt.rcParams.update({"font.size": args.font_size})
    figsize = parse_figsize(args.figsize)
    fig, ax = plt.subplots(figsize=figsize)

    ax.boxplot(
        values,
        labels=[str(b) for b in buckets],
        showmeans=args.show_means,
        meanline=False,
        patch_artist=True,
        boxprops={"facecolor": "#1f77b4", "alpha": 0.6},
        medianprops={"color": "#d62728", "linewidth": 1.5},
        whiskerprops={"color": "#444444"},
        capprops={"color": "#444444"},
    )

    ax.set_xlabel("Bucket")
    ax.set_ylabel("MSE")
    if args.title:
        ax.set_title(args.title)
    if not args.no_grid:
        ax.grid(True, axis="y", alpha=0.3)

    if args.out:
        out_path = Path(args.out)
    else:
        name = f"{args.dataset}_{args.seq_len}_{args.pred_len}_time_buckets.jpg"
        out_path = script_dir / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, format="jpg")


if __name__ == "__main__":
    main()
