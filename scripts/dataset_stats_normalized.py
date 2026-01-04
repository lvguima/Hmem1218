import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


EXCLUDE_NAME_HINTS = {"date", "time", "timestamp"}
LAST_COLUMN_ONLY_DATASETS = {"flotation", "grinding"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute normalized non-stationarity statistics for datasets. "
            "Metrics are computed per variable on z-scored series (per variable), "
            "then averaged within each dataset. For flotation/grinding, only the OT column is used."
        )
    )
    parser.add_argument("--data-dir", default="dataset", help="Directory with CSV datasets.")
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset names (without .csv). Default: all CSVs in data-dir.",
    )
    parser.add_argument("--segments", type=int, default=10, help="Number of segments for drift/slope.")
    parser.add_argument(
        "--output",
        default="dataset_stats_normalized.csv",
        help="Output dataset-level CSV path.",
    )
    parser.add_argument(
        "--per-variable-out",
        default="",
        help="Optional per-variable metrics CSV path.",
    )
    return parser.parse_args()


def list_datasets(data_dir, names):
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not names:
        return csv_files
    wanted = {name.strip().lower() for name in names.split(",") if name.strip()}
    return [p for p in csv_files if p.stem.lower() in wanted]


def select_numeric_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c.lower() not in EXCLUDE_NAME_HINTS]


def zscore(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 0 or not math.isfinite(std):
        return None
    return (values - mean) / std


def dominant_period_fft(values):
    n = len(values)
    if n < 4:
        return math.nan
    x = values - np.mean(values)
    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    power[0] = 0.0
    idx = int(np.argmax(power))
    if freqs[idx] <= 0:
        return math.nan
    return float(1.0 / freqs[idx])


def linear_trend(values):
    n = len(values)
    if n < 2:
        return math.nan, math.nan, np.full(n, np.nan)
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, values, 1)
    trend = slope * t + intercept
    return float(slope), float(intercept), trend


def seasonal_strength(values, period, trend):
    n = len(values)
    if not math.isfinite(period) or n < 4:
        return math.nan
    period_int = int(round(period))
    if period_int < 2 or period_int > n // 2:
        return math.nan
    detrended = values - trend
    phases = np.arange(n) % period_int
    seasonal_means = np.zeros(period_int, dtype=float)
    for phase in range(period_int):
        phase_vals = detrended[phases == phase]
        if phase_vals.size:
            seasonal_means[phase] = np.mean(phase_vals)
    seasonal = seasonal_means[phases]
    denom = np.var(detrended)
    if denom <= 0:
        return math.nan
    return float(np.var(seasonal) / denom)


def segment_bounds(n, segments):
    if n <= 0:
        return []
    bounds = np.linspace(0, n, segments + 1, dtype=int)
    return list(zip(bounds[:-1], bounds[1:]))


def mean_abs_adjacent_diff(values):
    clean = [v for v in values if math.isfinite(v)]
    if len(clean) < 2:
        return math.nan
    diffs = np.abs(np.diff(clean))
    if diffs.size == 0:
        return math.nan
    return float(np.mean(diffs))


def piecewise_slopes(values, segments):
    n = len(values)
    if n < 2:
        return [math.nan]
    slopes = []
    for start, end in segment_bounds(n, segments):
        if end - start < 2:
            slopes.append(0.0)
            continue
        t = np.arange(end - start, dtype=float)
        seg = values[start:end]
        slope, _ = np.polyfit(t, seg, 1)
        slopes.append(float(slope))
    return slopes


def compute_variable_metrics(raw_series, segments):
    values = zscore(raw_series)
    if values is None or values.size < 4:
        return None

    # Segment drift: mean abs adjacent difference between segment statistics.
    means = []
    variances = []
    for start, end in segment_bounds(len(values), segments):
        seg = values[start:end]
        if seg.size == 0:
            means.append(math.nan)
            variances.append(math.nan)
            continue
        means.append(float(np.mean(seg)))
        variances.append(float(np.var(seg)))

    delta_mu = mean_abs_adjacent_diff(means)
    delta_sigma2 = mean_abs_adjacent_diff(variances)

    slopes = piecewise_slopes(values, segments)
    grad_abs = float(np.mean(np.abs(slopes))) if slopes else math.nan

    _, _, trend = linear_trend(values)
    s = seasonal_strength(values, dominant_period_fft(values), trend)

    return {
        "delta_mu": delta_mu,
        "delta_sigma2": delta_sigma2,
        "grad_abs": grad_abs,
        "seasonal_strength": s,
    }


def aggregate(values):
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return math.nan
    return float(np.mean(clean))


def main():
    args = parse_args()
    datasets = list_datasets(args.data_dir, args.datasets)
    if not datasets:
        raise SystemExit("No datasets found.")

    dataset_rows = []
    per_var_rows = []

    for csv_path in datasets:
        df = pd.read_csv(csv_path)
        numeric_cols = select_numeric_columns(df)
        if not numeric_cols:
            continue

        dataset_name = csv_path.stem.lower()
        used_cols = numeric_cols
        if dataset_name in LAST_COLUMN_ONLY_DATASETS:
            used_cols = ["OT"] if "OT" in df.columns else [numeric_cols[-1]]

        var_metrics = []
        for col in used_cols:
            metrics = compute_variable_metrics(df[col].values, args.segments)
            if metrics is None:
                continue
            var_metrics.append((col, metrics))
            per_var_rows.append({"dataset": csv_path.stem, "variable": col, **metrics})

        if not var_metrics:
            continue

        dataset_rows.append(
            {
                "dataset": csv_path.stem,
                "length": len(df),
                "dims_total": len(numeric_cols),
                "dims_used": len(used_cols),
                "delta_mu": aggregate([m["delta_mu"] for _, m in var_metrics]),
                "delta_sigma2": aggregate([m["delta_sigma2"] for _, m in var_metrics]),
                "grad_abs": aggregate([m["grad_abs"] for _, m in var_metrics]),
                "seasonal_strength": aggregate([m["seasonal_strength"] for _, m in var_metrics]),
            }
        )

    if not dataset_rows:
        raise SystemExit("No metrics computed.")

    out_df = pd.DataFrame(dataset_rows).sort_values("dataset")
    out_df.to_csv(args.output, index=False)
    print(f"Wrote dataset summary: {args.output}")

    if args.per_variable_out:
        per_df = pd.DataFrame(per_var_rows).sort_values(["dataset", "variable"])
        per_df.to_csv(args.per_variable_out, index=False)
        print(f"Wrote per-variable: {args.per_variable_out}")


if __name__ == "__main__":
    main()

