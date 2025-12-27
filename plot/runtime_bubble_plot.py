import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
#python plot\runtime_bubble_plot.py --dataset Flotation --seq-len 64 --pred-len 2 --x-metric ms_per_step --y-metric mse --size-metric memory_mb --label

LEGEND_ORDER = ["frozen", "online", "er", "derpp", "acl", "clser", "mir", "solid", "hmem", "rmem"]
LEGEND_LABELS = {
    "frozen": "Frozen",
    "online": "Online",
    "er": "Er",
    "derpp": "Derpp",
    "acl": "Acl",
    "clser": "Clser",
    "mir": "Mir",
    "solid": "Solid",
    "hmem": "Rmem",
    "rmem": "Rmem",
}

def parse_figsize(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("figsize must be in 'width,height' format")
    return float(parts[0]), float(parts[1])


def parse_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def find_result_dirs(results_dir, dataset, seq_len, pred_len, methods):
    matches = []
    for item in results_dir.iterdir():
        if not item.is_dir():
            continue
        parts = item.name.split("_")
        if len(parts) < 4:
            continue
        pred = parts[-1]
        seq = parts[-2]
        method = parts[-3]
        ds = "_".join(parts[:-3])
        if ds.lower() != dataset.lower():
            continue
        if seq != str(seq_len) or pred != str(pred_len):
            continue
        if methods and method.lower() not in methods:
            continue
        matches.append((method, item))
    if methods:
        order = {name: idx for idx, name in enumerate(methods)}
        matches.sort(key=lambda x: order.get(x[0].lower(), 999))
    else:
        matches.sort(key=lambda x: x[0].lower())
    return matches


def read_runtime_summary(path):
    data = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric = row.get("metric", "")
            if not metric:
                continue
            try:
                data[metric] = float(row.get("mean", "nan"))
            except (TypeError, ValueError):
                continue
    return data


def read_test_summary(path):
    data = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric = row.get("metric", "")
            if not metric:
                continue
            try:
                data[metric] = float(row.get("value", "nan"))
            except (TypeError, ValueError):
                continue
    return data


def scale_sizes(values, min_size, max_size):
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [0.5 * (min_size + max_size)] * len(values)
    sizes = []
    for v in values:
        frac = (v - vmin) / (vmax - vmin)
        sizes.append(min_size + frac * (max_size - min_size))
    return sizes


def sort_method_key(method):
    method_lower = method.lower()
    if method_lower in LEGEND_ORDER:
        return (0, LEGEND_ORDER.index(method_lower))
    return (1, method_lower)


def format_method_label(method):
    return LEGEND_LABELS.get(method.lower(), method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", help="Root results directory.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. ETTm1.")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length.")
    parser.add_argument("--pred-len", type=int, required=True, help="Prediction length.")
    parser.add_argument("--methods", default="", help="Comma-separated methods to include.")
    parser.add_argument("--x-metric", default="ms_per_step", help="X-axis metric: ms_per_step or samples_per_sec.")
    parser.add_argument("--y-metric", default="mse", help="Y-axis metric: mse or mae.")
    parser.add_argument("--size-metric", default="params_total", help="Bubble size metric: params_total, params_trainable, memory_mb.")
    parser.add_argument("--min-size", type=float, default=80, help="Min bubble size.")
    parser.add_argument("--max-size", type=float, default=500, help="Max bubble size.")
    parser.add_argument("--figsize", default="6,4", help="Figure size, width,height.")
    parser.add_argument("--title", default="", help="Plot title.")
    parser.add_argument("--font-size", type=float, default=15, help="Base font size.")
    parser.add_argument(
        "--palette",
        default="#1f77b4,#ff7f0e,#2ca02c,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf",
        help="Comma-separated color list.",
    )
    parser.add_argument("--label", action="store_true", help="Show method legend on the right.")
    parser.add_argument("--no-grid", action="store_true", help="Disable grid.")
    parser.add_argument("--out", default="", help="Output image path (optional).")
    parser.add_argument("--dpi", type=int, default=450, help="Output DPI.")
    args = parser.parse_args()
    args.x_metric = args.x_metric.lower()
    args.y_metric = args.y_metric.lower()
    args.size_metric = args.size_metric.lower()
    if args.x_metric not in ("ms_per_step", "samples_per_sec"):
        raise ValueError("x-metric must be 'ms_per_step' or 'samples_per_sec'")
    if args.y_metric not in ("mse", "mae"):
        raise ValueError("y-metric must be 'mse' or 'mae'")
    if args.size_metric not in ("params_total", "params_trainable", "memory_mb"):
        raise ValueError("size-metric must be 'params_total', 'params_trainable', or 'memory_mb'")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    results_dir = Path(args.results_dir)
    if not results_dir.exists() and args.results_dir == "results":
        fallback = project_root / "results"
        if fallback.exists():
            results_dir = fallback
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    methods = [name.lower() for name in parse_list(args.methods)]
    matches = find_result_dirs(results_dir, args.dataset, args.seq_len, args.pred_len, methods)
    if not matches:
        raise FileNotFoundError("No matching result folders found.")
    matches.sort(key=lambda item: sort_method_key(item[0]))

    palette = parse_list(args.palette)
    points = []
    size_values = []
    for method, folder in matches:
        summary_path = folder / "runtime_summary.csv"
        test_summary_path = folder / "test_summary.csv"
        if not summary_path.exists() or not test_summary_path.exists():
            continue
        summary = read_runtime_summary(summary_path)
        test_summary = read_test_summary(test_summary_path)
        x_value = summary.get(args.x_metric)
        y_value = test_summary.get(args.y_metric)
        size_value = test_summary.get(args.size_metric)
        if x_value is None or y_value is None or size_value is None:
            continue
        points.append((method, x_value, y_value))
        size_values.append(size_value)

    if not points:
        raise RuntimeError("No runtime summary data found.")

    sizes = scale_sizes(size_values, args.min_size, args.max_size)

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

    handles = []
    labels = []
    for idx, ((method, x_value, y_value), size) in enumerate(zip(points, sizes)):
        color = palette[idx % len(palette)] if palette else None
        ax.scatter(x_value, y_value, s=size, color=color, alpha=0.8)
        if args.label:
            handles.append(Patch(facecolor=color or "black", edgecolor="black"))
            labels.append(format_method_label(method))
    if args.label and handles:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            ncol=1,
            frameon=False,
        )

    x_label = "ms/step" if args.x_metric == "ms_per_step" else "samples/sec"
    y_label = args.y_metric.upper()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if args.title:
        ax.set_title(args.title)
    if not args.no_grid:
        ax.grid(True, alpha=0.3)

    if args.out:
        out_path = Path(args.out)
    else:
        name = f"{args.dataset}_{args.seq_len}_{args.pred_len}_runtime_bubble.jpg"
        out_path = script_dir / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, format="jpg", bbox_inches="tight")


if __name__ == "__main__":
    main()
