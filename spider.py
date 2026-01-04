import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'

INDICATOR_KEYS = ['MSE', 'MAE', 'RMSE', 'MAPE', 'R2', 'RSE']
INDICATOR_LABELS = ['MSE', 'MAE', 'RMSE', 'MAPE', r'$R^2$', 'RSE']
METHODS = ['R-ARC', 'w/o Buckets', 'w/o Mask', 'w/o Refinement', 'w/o Gate']
LOWER_IS_BETTER = {'MSE', 'MAE', 'RMSE', 'MAPE', 'RSE'}


def normalize_with_reference(raw: np.ndarray, reference_ranges: dict) -> np.ndarray:
    """Normalize each metric to [0, 1] using dataset-specific reference ranges."""
    scores = np.zeros_like(raw, dtype=float)
    for j, metric in enumerate(INDICATOR_KEYS):
        col = raw[:, j].astype(float)
        ref_min, ref_max = reference_ranges[metric]
        col_clipped = np.clip(col, ref_min, ref_max)
        if metric in LOWER_IS_BETTER:
            scores[:, j] = 1.0 - (col_clipped - ref_min) / (ref_max - ref_min)
        else:
            scores[:, j] = (col_clipped - ref_min) / (ref_max - ref_min)
    return scores


def plot_radar(dataset_name: str, raw_metrics: dict, reference_ranges: dict, output_prefix: str):
    raw = np.array([[raw_metrics[m][k] for k in INDICATOR_KEYS] for m in METHODS], dtype=float)
    data = normalize_with_reference(raw, reference_ranges)

    # Keep a small radius floor so worst methods do not collapse to a line.
    radius_floor = 0.05
    data = radius_floor + (1.0 - radius_floor) * data

    angles = np.linspace(0, 2 * np.pi, len(INDICATOR_KEYS), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    colors = ['#2E86AB', '#F18F01', '#C73E1D', '#6A4C93', '#2A9D8F']

    ax.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Plot ablations first, then R-ARC on top.
    plot_order = list(range(1, len(METHODS))) + [0]
    for idx in plot_order:
        values = np.concatenate((data[idx], [data[idx][0]]))
        if idx == 0:
            ax.plot(angles, values, 'o-', linewidth=3, label=METHODS[idx], color=colors[idx], zorder=5)
            ax.fill(angles, values, alpha=0.15, color=colors[idx], zorder=4)
        else:
            ax.plot(angles, values, 'o-', linewidth=2, label=METHODS[idx], color=colors[idx], zorder=2)
            ax.fill(angles, values, alpha=0.18, color=colors[idx], zorder=1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(INDICATOR_LABELS, fontsize=24, fontweight='bold')
    ax.set_ylim(0.0, 1.0)

    # Remove radial tick labels for cleaner visualization
    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=700, bbox_inches='tight')

    legend_fig, legend_ax = plt.subplots(figsize=(12, 2.2))
    legend_handles = [
        plt.Line2D([0], [0], color=colors[i], lw=2, marker='o', markersize=4, label=METHODS[i])
        for i in range(len(METHODS))
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc='center',
        ncol=5,
        fontsize=28,
        prop={'family': 'Times New Roman', 'weight': 'bold', 'style': 'normal'}
    )
    legend_ax.axis('off')
    legend_fig.savefig(f'{output_prefix}_legend.png', dpi=700, bbox_inches='tight')


if __name__ == "__main__":
    # ETTm1 (L=512, H=96)
    ett_metrics = {
        'R-ARC':          {'MSE': 0.708000, 'MAE': 0.536000, 'RMSE': 0.842000, 'MAPE': 2.585000, 'R2': 0.726000, 'RSE': 0.523000},
        'w/o Buckets':    {'MSE': 0.726312, 'MAE': 0.544627, 'RMSE': 0.849239, 'MAPE': 2.636662, 'R2': 0.719563, 'RSE': 0.529563},
        'w/o Mask':       {'MSE': 0.724049, 'MAE': 0.537663, 'RMSE': 0.851911, 'MAPE': 2.605471, 'R2': 0.720437, 'RSE': 0.538737},
        'w/o Refinement': {'MSE': 0.786091, 'MAE': 0.569200, 'RMSE': 0.886618, 'MAPE': 2.726443, 'R2': 0.696482, 'RSE': 0.550925},
        'w/o Gate':       {'MSE': 0.775419, 'MAE': 0.550380, 'RMSE': 0.876239, 'MAPE': 2.726698, 'R2': 0.696741, 'RSE': 0.540689},
    }
    ett_ranges = {
        'MSE': (0.70, 0.80),
        'MAE': (0.53, 0.58),
        'RMSE': (0.83, 0.90),
        'MAPE': (2.55, 2.75),
        'RSE': (0.52, 0.56),
        'R2': (0.69, 0.74),
    }

    # Weather (L=512, H=96)
    weather_metrics = {
        'R-ARC':          {'MSE': 1.514000, 'MAE': 0.682000, 'RMSE': 1.230000, 'MAPE': 2.431000, 'R2': 0.751000, 'RSE': 0.499000},
        'w/o Buckets':    {'MSE': 1.540398, 'MAE': 0.705848, 'RMSE': 1.241128, 'MAPE': 2.755080, 'R2': 0.746338, 'RSE': 0.503648},
        'w/o Mask':       {'MSE': 1.610728, 'MAE': 0.713414, 'RMSE': 1.269145, 'MAPE': 3.197021, 'R2': 0.734757, 'RSE': 0.515018},
        'w/o Refinement': {'MSE': 1.546056, 'MAE': 0.692491, 'RMSE': 1.243405, 'MAPE': 2.366037, 'R2': 0.745407, 'RSE': 0.504573},
        'w/o Gate':       {'MSE': 1.580708, 'MAE': 0.709320, 'RMSE': 1.257262, 'MAPE': 3.207815, 'R2': 0.739700, 'RSE': 0.510196},
    }
    weather_ranges = {
        'MSE': (1.50, 1.65),
        'MAE': (0.67, 0.72),
        'RMSE': (1.22, 1.28),
        'MAPE': (2.30, 3.30),
        'RSE': (0.49, 0.52),
        'R2': (0.73, 0.76),
    }

    # Flotation (L=64, H=24)
    flotation_metrics = {
        'R-ARC':          {'MSE': 1.118000, 'MAE': 0.680000, 'RMSE': 1.057000, 'MAPE': 3.892000, 'R2': 0.658000, 'RSE': 0.585000},
        'w/o Buckets':    {'MSE': 1.154000, 'MAE': 0.687000, 'RMSE': 1.062000, 'MAPE': 3.921000, 'R2': 0.650000, 'RSE': 0.588000},
        'w/o Mask':       {'MSE': 1.120000, 'MAE': 0.689583, 'RMSE': 1.056596, 'MAPE': 3.907880, 'R2': 0.641480, 'RSE': 0.589168},
        'w/o Refinement': {'MSE': 1.124060, 'MAE': 0.690601, 'RMSE': 1.059436, 'MAPE': 3.896043, 'R2': 0.652110, 'RSE': 0.589238},
        'w/o Gate':       {'MSE': 1.121000, 'MAE': 0.706304, 'RMSE': 1.086712, 'MAPE': 4.046857, 'R2': 0.639017, 'RSE': 0.600818},
    }
    flotation_ranges = {
        'MSE': (1.10, 1.20),
        'MAE': (0.67, 0.72),
        'RMSE': (1.05, 1.10),
        'MAPE': (3.85, 4.10),
        'RSE': (0.58, 0.61),
        'R2': (0.63, 0.67),
    }

    plot_radar('ETTm1', ett_metrics, ett_ranges, 'radar_ett_ablation')
    plot_radar('Weather', weather_metrics, weather_ranges, 'radar_weather_ablation')
    plot_radar('Flotation', flotation_metrics, flotation_ranges, 'radar_flotation_ablation')

    # Grinding (L=300, H=15)
    grinding_metrics = {
        'R-ARC':          {'MSE': 0.633000, 'MAE': 0.131000, 'RMSE': 0.796000, 'MAPE': 0.589000, 'R2': 0.981000, 'RSE': 0.170000},
        'w/o Buckets':    {'MSE': 0.652610, 'MAE': 0.137573, 'RMSE': 0.817843, 'MAPE': 0.616637, 'R2': 0.974333, 'RSE': 0.174242},
        'w/o Mask':       {'MSE': 0.648024, 'MAE': 0.134765, 'RMSE': 0.798764, 'MAPE': 0.597700, 'R2': 0.967960, 'RSE': 0.170307},
        'w/o Refinement': {'MSE': 0.719727, 'MAE': 0.141353, 'RMSE': 0.848368, 'MAPE': 0.638057, 'R2': 0.967281, 'RSE': 0.184883},
        'w/o Gate':       {'MSE': 1.201089, 'MAE': 0.180954, 'RMSE': 1.095942, 'MAPE': 0.794810, 'R2': 0.945399, 'RSE': 0.233669},
    }
    grinding_ranges = {
        'MSE': (0.60, 1.25),
        'MAE': (0.13, 0.19),
        'RMSE': (0.78, 1.12),
        'MAPE': (0.58, 0.82),
        'RSE': (0.16, 0.25),
        'R2': (0.94, 0.98),
    }

    plot_radar('Grinding', grinding_metrics, grinding_ranges, 'radar_grinding_ablation')
