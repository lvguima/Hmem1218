import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
#python plot\rolling_mse_plot.py --dataset Grinding --seq-len 300 --pred-len 15 --methods Frozen,Online,Er,Derpp,Acl,Clser,Mir,Solid,Rmem --step-start 2000 --step-end 16000

METHOD_ALIASES = {
    # keep compatibility with existing result folder names
    "rmem": "hmem",
}

LABEL_MAP = {
    "hmem": "Rmem",
    "rmem": "Rmem",
}


def normalize_method_name(name: str) -> str:
    normalized = name.strip().lower()
    return METHOD_ALIASES.get(normalized, normalized)

def parse_csv(path, value_col):
    steps = []
    values = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                steps.append(int(row.get("step", 0)))
                values.append(float(row.get(value_col, "nan")))
            except (TypeError, ValueError):
                continue
    return steps, values


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
        if methods and normalize_method_name(method) not in methods:
            continue
        matches.append((method, item))
    if methods:
        order = {name: idx for idx, name in enumerate(methods)}
        matches.sort(key=lambda x: order.get(normalize_method_name(x[0]), 999))
    else:
        matches.sort(key=lambda x: x[0].lower())
    return matches


def tail_slice(steps, values, fraction):
    if not values:
        return steps, values
    fraction = min(max(float(fraction), 0.0), 1.0)
    if fraction <= 0:
        return steps, values
    tail_len = max(1, int(len(values) * fraction))
    start = max(0, len(values) - tail_len)
    return steps[start:], values[start:]


def filter_step_range(steps, values, step_start, step_end):
    if step_start is None and step_end is None:
        return steps, values
    filtered_steps = []
    filtered_values = []
    for step, value in zip(steps, values):
        if step_start is not None and step < step_start:
            continue
        if step_end is not None and step > step_end:
            continue
        filtered_steps.append(step)
        filtered_values.append(value)
    return filtered_steps, filtered_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", help="Root results directory.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. ETTm1.")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length.")
    parser.add_argument("--pred-len", type=int, required=True, help="Prediction length.")
    parser.add_argument("--methods", default="", help="Comma-separated methods to include.")
    parser.add_argument("--value-col", default="rolling_mse", help="CSV column to plot.")
    parser.add_argument("--stride", type=int, default=1, help="Plot every N points.")
    parser.add_argument("--tail-fraction", type=float, default=0.1, help="Plot the last fraction of points.")
    parser.add_argument(
        "--step-start",
        type=int,
        default=None,
        help="Only plot points with step >= this value (disables tail-fraction).",
    )
    parser.add_argument(
        "--step-end",
        type=int,
        default=None,
        help="Only plot points with step <= this value (disables tail-fraction).",
    )
    parser.add_argument("--figsize", default="10,4", help="Figure size, width,height.")
    parser.add_argument("--title", default="", help="Plot title.")
    parser.add_argument("--font-size", type=float, default=12, help="Base font size.")
    parser.add_argument("--font-family", default="Times New Roman", help="Font family.")
    parser.add_argument("--font-weight", default="bold", help="Font weight, e.g. normal/bold.")
    parser.add_argument("--line-width", type=float, default=1.6, help="Line width.")
    parser.add_argument(
        "--palette",
        default="#1f77b4,#ff7f0e,#2ca02c,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf",
        help="Comma-separated color list.",
    )
    parser.add_argument("--legend-loc", default="upper center", help="Legend location.")
    parser.add_argument("--legend-ncol", type=int, default=0, help="Legend columns (0 = auto, single row).")
    parser.add_argument("--legend-anchor-y", type=float, default=1.2, help="Legend anchor Y position.")
    parser.add_argument("--no-grid", action="store_true", help="Disable grid.")
    parser.add_argument("--tight-layout", action="store_true", help="Apply tight layout.")
    parser.add_argument("--out", default="", help="Output image path (optional).")
    parser.add_argument("--dpi", type=int, default=450, help="Output DPI.")
    parser.add_argument("--debug", action="store_true", help="Print debug information.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    results_dir = Path(args.results_dir)
    if not results_dir.exists() and args.results_dir == "results":
        fallback = project_root / "results"
        if fallback.exists():
            results_dir = fallback
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    methods = [normalize_method_name(name) for name in parse_list(args.methods)]
    matches = find_result_dirs(results_dir, args.dataset, args.seq_len, args.pred_len, methods)
    if not matches:
        raise FileNotFoundError("No matching result folders found.")

    palette = parse_list(args.palette)
    figsize = parse_figsize(args.figsize)

    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "font.family": args.font_family,
            "font.weight": args.font_weight,
            "axes.labelweight": args.font_weight,
            "axes.titleweight": args.font_weight,
        }
    )
    fig, ax = plt.subplots(figsize=figsize)

    if args.debug:
        print(f"[debug] cwd: {Path.cwd()}")
        print(f"[debug] results_dir: {results_dir}")
        print(f"[debug] matches: {[m[0] for m in matches]}")
        print(f"[debug] value_col: {args.value_col} stride: {args.stride} tail: {args.tail_fraction}")
        print(f"[debug] step range: {args.step_start} - {args.step_end}")

    plotted = 0
    for idx, (method, folder) in enumerate(matches):
        csv_path = folder / "rolling_mse.csv"
        if not csv_path.exists():
            if args.debug:
                print(f"[debug] missing csv: {csv_path}")
            continue
        steps, values = parse_csv(csv_path, args.value_col)
        tail_fraction = args.tail_fraction
        if args.step_start is not None or args.step_end is not None:
            tail_fraction = 0.0
        steps, values = tail_slice(steps, values, tail_fraction)
        steps, values = filter_step_range(steps, values, args.step_start, args.step_end)
        if args.debug:
            print(f"[debug] {method}: points={len(values)}")
        if args.stride > 1:
            steps = steps[:: args.stride]
            values = values[:: args.stride]
        color = palette[idx % len(palette)] if palette else None
        method_lower = method.lower() if method else ""
        method_label = LABEL_MAP.get(method_lower) or (f"{method[:1].upper()}{method[1:]}" if method else method)
        ax.plot(steps, values, label=method_label, linewidth=args.line_width, color=color)
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No data series plotted. Check CSV files and value column.")

    ax.set_xlabel("Sample")
    ax.tick_params(axis="x", which="both", labelbottom=False)
    ax.set_ylabel(args.value_col.replace("_", " ").title())
    if args.title:
        ax.set_title(args.title)
    if not args.no_grid:
        ax.grid(True, alpha=0.3)
    # Default: place legend above in a single horizontal row.
    legend_ncol = args.legend_ncol if args.legend_ncol > 0 else max(1, plotted)
    ax.legend(
        loc=args.legend_loc,
        bbox_to_anchor=(0.5, args.legend_anchor_y),
        ncol=legend_ncol,
        frameon=False,
    )

    if args.out:
        out_path = Path(args.out)
    else:
        filename = f"{args.dataset}_{args.seq_len}_{args.pred_len}_rolling_mse.jpg"
        out_path = script_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.debug:
        print(f"[debug] out_path: {out_path}")
    if args.tight_layout:
        fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, format="jpg", bbox_inches="tight")


if __name__ == "__main__":
    main()
