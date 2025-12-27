import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
#python plot\horizon_effect_plot.py --dataset ETTm1 --seq-len 512 --pred-len 24

def parse_figsize(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("figsize must be in 'width,height' format")
    return float(parts[0]), float(parts[1])


def parse_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def read_horizon_effect(path):
    data = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                horizon = int(row.get("horizon", 0))
                metric = row.get("metric", "")
                value = float(row.get("value", "nan"))
            except (TypeError, ValueError):
                continue
            if not metric:
                continue
            data.setdefault(metric, {})[horizon] = value
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="", help="Path to horizon_effect.csv (optional).")
    parser.add_argument("--results-dir", default="results", help="Root results directory.")
    parser.add_argument("--dataset", default="", help="Dataset name, e.g. ETTm1.")
    parser.add_argument("--seq-len", type=int, default=0, help="Sequence length.")
    parser.add_argument("--pred-len", type=int, default=0, help="Prediction length.")
    parser.add_argument("--metrics", default="correction_norm,mse_improve", help="Comma-separated metric names.")
    parser.add_argument("--figsize", default="6,4", help="Figure size, width,height.")
    parser.add_argument("--title", default="", help="Plot title.")
    parser.add_argument("--font-size", type=float, default=12, help="Base font size.")
    parser.add_argument(
        "--palette",
        default="#1f77b4,#d62728,#2ca02c,#ff7f0e",
        help="Comma-separated color list.",
    )
    parser.add_argument("--line-width", type=float, default=1.6, help="Line width.")
    parser.add_argument("--legend-loc", default="best", help="Legend location.")
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
        csv_path = folder / "horizon_effect.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing csv: {csv_path}")

    data = read_horizon_effect(csv_path)
    if not data:
        raise RuntimeError("No horizon effect data loaded.")

    metrics = parse_list(args.metrics)
    palette = parse_list(args.palette)
    plt.rcParams.update({"font.size": args.font_size})
    figsize = parse_figsize(args.figsize)
    fig, ax = plt.subplots(figsize=figsize)

    plotted = 0
    for idx, metric in enumerate(metrics):
        series = data.get(metric, {})
        if not series:
            continue
        horizons = sorted(series.keys())
        values = [series[h] for h in horizons]
        color = palette[idx % len(palette)] if palette else None
        ax.plot(horizons, values, label=metric, linewidth=args.line_width, color=color)
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No matching metrics to plot.")

    ax.set_xlabel("Horizon")
    ax.set_ylabel("Value")
    if args.title:
        ax.set_title(args.title)
    if not args.no_grid:
        ax.grid(True, alpha=0.3)
    ax.legend(loc=args.legend_loc)

    if args.out:
        out_path = Path(args.out)
    else:
        name = f"{args.dataset}_{args.seq_len}_{args.pred_len}_horizon_effect.jpg"
        out_path = script_dir / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, format="jpg")


if __name__ == "__main__":
    main()
