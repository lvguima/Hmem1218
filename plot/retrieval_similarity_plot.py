import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#python plot\retrieval_similarity_plot.py --dataset Grinding --seq-len 300 --pred-len 15 --kde

DEFAULT_MULTI_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def parse_figsize(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("figsize must be in 'width,height' format")
    return float(parts[0]), float(parts[1])


def parse_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_config_item(value):
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Config '{value}' must be in 'dataset:seq:pred' format.")
    dataset = parts[0].strip()
    if not dataset:
        raise ValueError(f"Config '{value}' must include a dataset name.")
    try:
        seq_len = int(parts[1])
        pred_len = int(parts[2])
    except ValueError as exc:
        raise ValueError(f"Config '{value}' must use integer seq/pred lengths.") from exc
    if seq_len <= 0 or pred_len <= 0:
        raise ValueError(f"Config '{value}' must use positive seq/pred lengths.")
    return dataset, seq_len, pred_len


def read_similarity(path, step_stride=1):
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
            try:
                values.append(float(row.get("mean_similarity", "nan")))
            except (TypeError, ValueError):
                continue
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="", help="Path to retrieval_similarity.csv (optional).")
    parser.add_argument("--results-dir", default="results", help="Root results directory.")
    parser.add_argument("--dataset", default="", help="Dataset name, e.g. ETTm1.")
    parser.add_argument("--seq-len", type=int, default=0, help="Sequence length.")
    parser.add_argument("--pred-len", type=int, default=0, help="Prediction length.")
    parser.add_argument("--configs", default="", help="Comma-separated dataset:seq:pred configs.")
    parser.add_argument("--labels", default="", help="Comma-separated labels for configs.")
    parser.add_argument("--colors", default="", help="Comma-separated colors for configs.")
    parser.add_argument("--edge-colors", default="", help="Comma-separated edge colors for configs.")
    parser.add_argument("--kde-colors", default="", help="Comma-separated KDE colors for configs.")
    parser.add_argument("--figsize", default="6,4", help="Figure size, width,height.")
    parser.add_argument("--title", default="", help="Plot title.")
    parser.add_argument("--font-size", type=float, default=15, help="Base font size.")
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Histogram alpha.")
    parser.add_argument(
        "--histtype",
        default="auto",
        choices=("auto", "bar", "step", "stepfilled"),
        help="Histogram type (auto uses step for multi-series).",
    )
    parser.add_argument("--color", default="#5198e9", help="Histogram color.")
    parser.add_argument("--edge-color", default="#9ad9fa", help="Histogram edge color.")
    parser.add_argument("--edge-width", type=float, default=0.6, help="Histogram edge line width.")
    parser.add_argument("--kde", action="store_true", help="Overlay KDE curve (default on).")
    parser.add_argument("--no-kde", action="store_true", help="Disable KDE curve.")
    parser.add_argument("--kde-color", default="#2d7998", help="KDE line color.")
    parser.add_argument("--kde-width", type=float, default=1.5, help="KDE line width.")
    parser.add_argument("--step-stride", type=int, default=1, help="Only keep steps divisible by N.")
    parser.add_argument("--log-y", action="store_true", help="Log scale for y-axis.")
    parser.add_argument("--no-grid", action="store_true", help="Disable grid.")
    parser.add_argument("--out", default="", help="Output image path (optional).")
    parser.add_argument("--dpi", type=int, default=450, help="Output DPI.")
    args = parser.parse_args()
    if args.no_kde:
        args.kde = False
    elif not args.kde:
        args.kde = True

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if args.input and args.configs:
        raise ValueError("Use either --input for a single file or --configs for multiple datasets, not both.")

    if args.input:
        csv_path = Path(args.input)
        csv_paths = [(csv_path, "")]
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.exists() and args.results_dir == "results":
            fallback = project_root / "results"
            if fallback.exists():
                results_dir = fallback
        if args.configs:
            configs = [parse_config_item(item) for item in parse_list(args.configs)]
        else:
            if not args.dataset or not args.seq_len or not args.pred_len:
                raise ValueError("When --input is not provided, --dataset/--seq-len/--pred-len are required.")
            configs = [(args.dataset, args.seq_len, args.pred_len)]
        csv_paths = []
        for dataset, seq_len, pred_len in configs:
            folder = results_dir / f"{dataset}_hmem_{seq_len}_{pred_len}"
            csv_paths.append((folder / "retrieval_similarity.csv", f"{dataset}"))

    multi = len(csv_paths) > 1
    if args.labels:
        label_list = parse_list(args.labels)
        if len(label_list) != len(csv_paths):
            raise ValueError("Number of --labels entries must match --configs.")
        csv_paths = [(path, label) for (path, _), label in zip(csv_paths, label_list)]

    values_list = []
    for csv_path, label in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing csv: {csv_path}")
        values = read_similarity(csv_path, step_stride=args.step_stride)
        if not values:
            raise RuntimeError(f"No data loaded from {csv_path}. Check filters or file content.")
        values_list.append((values, label))

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

    color_list = parse_list(args.colors)
    edge_color_list = parse_list(args.edge_colors)
    kde_color_list = parse_list(args.kde_colors)
    if not color_list and multi:
        color_list = DEFAULT_MULTI_COLORS
    if not edge_color_list and multi:
        edge_color_list = color_list
    if not kde_color_list and multi:
        kde_color_list = color_list

    histtype = args.histtype
    if histtype == "auto":
        histtype = "step" if multi else "bar"
    line_width = args.edge_width
    if multi and histtype in ("step", "stepfilled") and line_width < 1.2:
        line_width = 1.2

    for idx, (values, label) in enumerate(values_list):
        if multi:
            color = color_list[idx % len(color_list)] if color_list else args.color
            edge_color = edge_color_list[idx % len(edge_color_list)] if edge_color_list else args.edge_color
            kde_color = kde_color_list[idx % len(kde_color_list)] if kde_color_list else args.kde_color
            legend_label = label or f"Series {idx + 1}"
        else:
            color = args.color
            edge_color = args.edge_color
            kde_color = args.kde_color
            legend_label = ""
        ax.hist(
            values,
            bins=args.bins,
            color=color,
            alpha=args.alpha,
            edgecolor=edge_color,
            linewidth=line_width,
            histtype=histtype,
            label=legend_label if multi else None,
        )
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
                    ax.plot(
                        grid,
                        kde * n * bin_width,
                        color=kde_color,
                        linewidth=args.kde_width,
                        zorder=3,
                    )
            elif n > 0:
                mean_val = float(np.mean(data))
                ax.axvline(mean_val, color=kde_color, linewidth=args.kde_width, zorder=3)

    ax.set_xlabel("Mean Similarity")
    ax.set_ylabel("Count")
    if args.title:
        ax.set_title(args.title)
    if args.log_y:
        ax.set_yscale("log")
    if not args.no_grid:
        ax.grid(True, axis="y", alpha=0.3)
    if multi:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
        )

    if args.out:
        out_path = Path(args.out)
    else:
        if multi:
            name = "multi_retrieval_similarity.jpg"
        else:
            name = f"{args.dataset}_{args.seq_len}_{args.pred_len}_retrieval_similarity.jpg"
        out_path = script_dir / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, format="jpg", bbox_inches="tight")


if __name__ == "__main__":
    main()
