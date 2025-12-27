import sys
sys.path.insert(0, 'd:/pyproject/OnlineTSF-main')

# Run the plot script
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Set parameters
results_dir = Path("d:/pyproject/OnlineTSF-main/results")
dataset = "ETTm1"
seq_len = 512
pred_len = 96
methods = [name.lower() for name in parse_list("frozen,online,er,derpp,acl,clser,mir,solid,hmem")]
value_col = "rolling_mse"
stride = 1
figsize = (10, 4)
font_size = 12
line_width = 1.6
palette = parse_list("#1f77b4,#ff7f0e,#2ca02c,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf")
legend_loc = "best"
dpi = 450

print(f"Results directory: {results_dir}")
print(f"Dataset: {dataset}, seq_len: {seq_len}, pred_len: {pred_len}")
print(f"Methods: {methods}")

matches = find_result_dirs(results_dir, dataset, seq_len, pred_len, methods)
print(f"\nFound {len(matches)} matching folders:")
for method, folder in matches:
    print(f"  - {method}: {folder}")

if not matches:
    print("ERROR: No matching result folders found.")
    sys.exit(1)

plt.rcParams.update({"font.size": font_size})
fig, ax = plt.subplots(figsize=figsize)

for idx, (method, folder) in enumerate(matches):
    csv_path = folder / "rolling_mse.csv"
    if not csv_path.exists():
        print(f"WARNING: {csv_path} does not exist, skipping...")
        continue
    steps, values = parse_csv(csv_path, value_col)
    print(f"Loaded {len(steps)} data points for {method}")
    if stride > 1:
        steps = steps[:: stride]
        values = values[:: stride]
    color = palette[idx % len(palette)] if palette else None
    ax.plot(steps, values, label=method, linewidth=line_width, color=color)

ax.set_xlabel("Step")
ax.set_ylabel(value_col.replace("_", " ").title())
ax.grid(True, alpha=0.3)
ax.legend(loc=legend_loc)

out_path = Path("d:/pyproject/OnlineTSF-main/plot") / f"{dataset}_{seq_len}_{pred_len}_rolling_mse.jpg"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
print(f"\nSaving plot to: {out_path}")
fig.savefig(out_path, dpi=dpi, format="jpg")
print("Plot saved successfully!")
