import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
#python plot\segment_mse_plot.py --dataset ETTm1 --seq-len 512 --pred-len 24 --methods frozen,online,er,derpp,acl,clser,mir,solid,rmem --legend-anchor-y 1.15

LABEL_MAP = {
    "hmem": "Rmem",
    "rmem": "Rmem",
}

METHOD_ALIASES = {
    "rmem": "hmem",
}


def normalize_method_name(name: str) -> str:
    normalized = name.strip().lower()
    return METHOD_ALIASES.get(normalized, normalized)
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


def read_segment_mse(path, value_col):
    data = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            segment = row.get("segment", "")
            if not segment:
                continue
            try:
                value = float(row.get(value_col, "nan"))
            except (TypeError, ValueError):
                continue
            data[segment] = value
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", help="Root results directory.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. ETTm1.")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length.")
    parser.add_argument("--pred-len", type=int, required=True, help="Prediction length.")
    parser.add_argument("--methods", default="", help="Comma-separated methods to include.")
    parser.add_argument("--segments", default="Early,Middle,Late", help="Comma-separated segment order.")
    parser.add_argument("--value-col", default="mse_mean", help="CSV column to plot.")
    parser.add_argument("--figsize", default="8,4", help="Figure size, width,height.")
    parser.add_argument("--title", default="", help="Plot title.")
    parser.add_argument("--font-size", type=float, default=13, help="Base font size.")
    parser.add_argument("--font-family", default="Times New Roman", help="Font family.")
    parser.add_argument("--font-weight", default="bold", help="Font weight, e.g. normal/bold.")
    parser.add_argument("--bar-width", type=float, default=0.0, help="Bar width (0 = auto).")
    parser.add_argument("--alpha", type=float, default=0.9, help="Bar transparency.")
    parser.add_argument(
        "--palette",
        default="#1f77b4,#ff7f0e,#2ca02c,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf",
        help="Comma-separated color list.",
    )
    parser.add_argument("--legend-loc", default="above", help="Legend location.")
    parser.add_argument("--legend-ncol", type=int, default=0, help="Legend columns (0 = auto, single row).")
    parser.add_argument("--legend-anchor-x", type=float, default=0.5, help="Legend anchor X position.")
    parser.add_argument("--legend-anchor-y", type=float, default=1.15, help="Legend anchor Y position.")
    parser.add_argument("--no-grid", action="store_true", help="Disable grid.")
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

    segments = parse_list(args.segments)
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

    if args.debug:
        print(f"[debug] results_dir: {results_dir}")
        print(f"[debug] matches: {[m[0] for m in matches]}")
        print(f"[debug] segments: {segments}")

    method_series = []
    for method, folder in matches:
        csv_path = folder / "segment_mse.csv"
        if not csv_path.exists():
            if args.debug:
                print(f"[debug] missing csv: {csv_path}")
            continue
        data = read_segment_mse(csv_path, args.value_col)
        values = [data.get(seg, float("nan")) for seg in segments]
        method_series.append((method, values))

    if not method_series:
        raise RuntimeError("No segment MSE data found.")

    method_count = len(method_series)
    x = np.arange(len(segments))
    bar_width = args.bar_width if args.bar_width > 0 else (0.8 / max(1, method_count))
    offsets = (np.arange(method_count) - (method_count - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=figsize)
    for idx, (method, values) in enumerate(method_series):
        color = palette[idx % len(palette)] if palette else None
        method_str = str(method)
        method_lower = method_str.lower()
        method_label = LABEL_MAP.get(method_lower) or (f"{method_str[:1].upper()}{method_str[1:]}" if method_str else method_str)
        ax.bar(x + offsets[idx], values, width=bar_width, label=method_label, color=color, alpha=args.alpha)

    # ax.set_xlabel("Segment")
    ax.set_ylabel(args.value_col.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(segments)
    if args.title:
        ax.set_title(args.title)
    if not args.no_grid:
        ax.grid(True, axis="y", alpha=0.3)
    if args.legend_loc.lower() == "above":
        # Default: place legend above in a single horizontal row.
        legend_cols = args.legend_ncol if args.legend_ncol > 0 else max(1, method_count)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(args.legend_anchor_x, args.legend_anchor_y),
            ncol=legend_cols,
            frameon=False,
        )
    else:
        ax.legend(loc=args.legend_loc)

    if args.out:
        out_path = Path(args.out)
    else:
        filename = f"{args.dataset}_{args.seq_len}_{args.pred_len}_segment_mse.jpg"
        out_path = script_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.debug:
        print(f"[debug] out_path: {out_path}")
    fig.savefig(out_path, dpi=args.dpi, format="jpg", bbox_inches="tight")


if __name__ == "__main__":
    main()
