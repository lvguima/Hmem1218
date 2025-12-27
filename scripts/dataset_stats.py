import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


EXCLUDE_NAME_HINTS = {"date", "time", "timestamp"}
LAST_COLUMN_ONLY_DATASETS = {"flotation", "grinding"}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute dataset property metrics.")
    parser.add_argument("--data-dir", default="dataset", help="Directory with CSV datasets.")
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset names (without .csv). Default: all CSVs in data-dir.",
    )
    parser.add_argument("--output", default="dataset_stats.csv", help="Output summary CSV path.")
    parser.add_argument(
        "--per-variable-out",
        default="",
        help="Optional per-variable metrics CSV path.",
    )
    parser.add_argument("--segments", type=int, default=10, help="Number of segments.")
    parser.add_argument("--max-lag", type=int, default=2000, help="Max lag for ACF.")
    parser.add_argument("--quantiles", default="0.1,0.5,0.9", help="Quantiles for drift.")
    return parser.parse_args()


def list_datasets(data_dir, names):
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not names:
        return csv_files
    wanted = {name.strip().lower() for name in names.split(",") if name.strip()}
    results = []
    for csv_path in csv_files:
        stem = csv_path.stem.lower()
        if stem in wanted:
            results.append(csv_path)
    return results


def select_numeric_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    filtered = []
    for col in numeric_cols:
        if col.lower() in EXCLUDE_NAME_HINTS:
            continue
        filtered.append(col)
    return filtered


def dominant_period_acf(values, max_lag):
    n = len(values)
    if n < 4:
        return math.nan
    max_lag = min(max_lag, n - 1)
    x = values - np.mean(values)
    nfft = 1 << (2 * n - 1).bit_length()
    fft = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(fft * np.conj(fft), n=nfft)[: max_lag + 1]
    if acf[0] == 0:
        return math.nan
    acf = acf / acf[0]
    peak_lag = int(np.argmax(acf[1:])) + 1
    return float(peak_lag)


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
    period = 1.0 / freqs[idx]
    return float(period)


def linear_trend(values):
    n = len(values)
    if n < 2:
        return math.nan, math.nan, np.full(n, np.nan)
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, values, 1)
    trend = slope * t + intercept
    return float(slope), float(intercept), trend


def piecewise_trend(values, segments):
    n = len(values)
    if n < 2:
        return [math.nan], np.full(n, np.nan)
    bounds = np.linspace(0, n, segments + 1, dtype=int)
    slopes = []
    trend = np.zeros(n, dtype=float)
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end - start < 2:
            seg_vals = values[start:end]
            if seg_vals.size:
                trend[start:end] = np.mean(seg_vals)
            slopes.append(0.0)
            continue
        t = np.arange(end - start, dtype=float)
        seg = values[start:end]
        slope, intercept = np.polyfit(t, seg, 1)
        trend[start:end] = slope * t + intercept
        slopes.append(float(slope))
    return slopes, trend


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


def segment_stats(values, segments, quantiles):
    n = len(values)
    bounds = np.linspace(0, n, segments + 1, dtype=int)
    means = []
    variances = []
    quantile_map = {q: [] for q in quantiles}
    for start, end in zip(bounds[:-1], bounds[1:]):
        seg = values[start:end]
        if seg.size == 0:
            means.append(math.nan)
            variances.append(math.nan)
            for q in quantiles:
                quantile_map[q].append(math.nan)
            continue
        means.append(float(np.mean(seg)))
        variances.append(float(np.var(seg)))
        for q in quantiles:
            quantile_map[q].append(float(np.quantile(seg, q)))
    return means, variances, quantile_map


def drift_range(values):
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return math.nan
    return float(max(clean) - min(clean))


def compute_metrics(series, segments, max_lag, quantiles):
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return None

    acf_period = dominant_period_acf(values, max_lag)
    fft_period = dominant_period_fft(values)

    slope, intercept, trend = linear_trend(values)
    trend_var_ratio = math.nan
    denom = np.var(values)
    if denom > 0 and np.all(np.isfinite(trend)):
        trend_var_ratio = float(np.var(trend) / denom)

    piece_slopes, piece_trend = piecewise_trend(values, segments)
    piece_trend_var_ratio = math.nan
    if denom > 0 and np.all(np.isfinite(piece_trend)):
        piece_trend_var_ratio = float(np.var(piece_trend) / denom)

    season_strength = seasonal_strength(values, fft_period, trend)

    means, variances, quantile_map = segment_stats(values, segments, quantiles)
    drift_mean = drift_range(means)
    drift_var = drift_range(variances)
    drift_quantiles = {q: drift_range(vals) for q, vals in quantile_map.items()}

    return {
        "acf_period": acf_period,
        "fft_period": fft_period,
        "seasonal_strength": season_strength,
        "segment_mean_drift": drift_mean,
        "segment_var_drift": drift_var,
        "segment_quantile_drift": drift_quantiles,
        "linear_slope": slope,
        "linear_trend_var_ratio": trend_var_ratio,
        "piecewise_slopes": piece_slopes,
        "piecewise_trend_var_ratio": piece_trend_var_ratio,
    }


def aggregate(values):
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return math.nan, math.nan
    return float(np.mean(clean)), float(np.median(clean))


def main():
    args = parse_args()
    quantiles = [float(q.strip()) for q in args.quantiles.split(",") if q.strip()]
    datasets = list_datasets(args.data_dir, args.datasets)
    if not datasets:
        raise SystemExit("No datasets found.")

    summary_rows = []
    per_var_rows = []

    for csv_path in datasets:
        df = pd.read_csv(csv_path)
        numeric_cols = select_numeric_columns(df)
        if not numeric_cols:
            continue
        length = len(df)
        dataset_name = csv_path.stem.lower()
        metric_cols = numeric_cols
        if dataset_name in LAST_COLUMN_ONLY_DATASETS:
            metric_cols = [numeric_cols[-1]]
        per_var = []
        for col in metric_cols:
            metrics = compute_metrics(df[col].values, args.segments, args.max_lag, quantiles)
            if metrics is None:
                continue
            per_var.append((col, metrics))
            per_var_rows.append(
                {
                    "dataset": csv_path.stem,
                    "variable": col,
                    "acf_period": metrics["acf_period"],
                    "fft_period": metrics["fft_period"],
                    "seasonal_strength": metrics["seasonal_strength"],
                    "segment_mean_drift": metrics["segment_mean_drift"],
                    "segment_var_drift": metrics["segment_var_drift"],
                    **{
                        f"segment_q{int(q*100)}_drift": metrics["segment_quantile_drift"][q]
                        for q in quantiles
                    },
                    "linear_slope": metrics["linear_slope"],
                    "linear_trend_var_ratio": metrics["linear_trend_var_ratio"],
                    "piecewise_slope_abs_mean": float(np.mean(np.abs(metrics["piecewise_slopes"]))),
                    "piecewise_trend_var_ratio": metrics["piecewise_trend_var_ratio"],
                }
            )

        if not per_var:
            continue

        metrics_by_key = {}
        for _, metrics in per_var:
            for key in [
                "acf_period",
                "fft_period",
                "seasonal_strength",
                "segment_mean_drift",
                "segment_var_drift",
                "linear_slope",
                "linear_trend_var_ratio",
                "piecewise_trend_var_ratio",
            ]:
                metrics_by_key.setdefault(key, []).append(metrics[key])
            for q in quantiles:
                metrics_by_key.setdefault(f"segment_q{int(q*100)}_drift", []).append(
                    metrics["segment_quantile_drift"][q]
                )
            metrics_by_key.setdefault("piecewise_slope_abs_mean", []).append(
                float(np.mean(np.abs(metrics["piecewise_slopes"])))
            )

        row = {
            "dataset": csv_path.stem,
            "length": length,
            "variables": len(numeric_cols),
        }
        for key, values in metrics_by_key.items():
            mean_val, median_val = aggregate(values)
            row[f"{key}_mean"] = mean_val
            row[f"{key}_median"] = median_val

        summary_rows.append(row)

    if not summary_rows:
        raise SystemExit("No metrics computed.")

    summary_df = pd.DataFrame(summary_rows).sort_values("dataset")
    summary_df.to_csv(args.output, index=False)

    if args.per_variable_out:
        per_var_df = pd.DataFrame(per_var_rows).sort_values(["dataset", "variable"])
        per_var_df.to_csv(args.per_variable_out, index=False)

    print(f"Wrote summary: {args.output}")
    if args.per_variable_out:
        print(f"Wrote per-variable: {args.per_variable_out}")


if __name__ == "__main__":
    main()
