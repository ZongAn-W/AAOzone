import csv
import glob
import os
import re
import sys

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from phase_warp_frontend import PhaseWarpFrontEnd

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def set_sci_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 300,
        }
    )


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
train_log_dir = os.path.join(base_dir, "models", "训练过程")
result_dir = os.path.join(base_dir, "models", "训练结果")
os.makedirs(train_log_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)


class Logger:
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


EXPERIMENT_TAG = "PredRNNv2_PhaseWarp_SyncVerification"
LOG_PATH = os.path.join(train_log_dir, f"{EXPERIMENT_TAG}.txt")
sys.stdout = Logger(LOG_PATH)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training Device: {device}")

openmars_dir = os.path.join(base_dir, "Dataset", "OpenMars")
mcd_dir = os.path.join(base_dir, "Dataset", "MCDALL")

window = 3
horizon = 3
batch_size = 16
epochs = 15
learning_rate = 1e-4
train_split_ratio = 0.8

SYNC_EXPERIMENTS = (
    {
        "enabled": True,
        "feature_name": "V_sync",
        "feature_key": "sync",
        "alpha_mode": "match_reference_std",
        "manual_alpha": 1.0,
        "reference_feature_key": "temp",
        "noise_ratio": 0.05,
        "noise_seed": 20260420,
    },
    {
        "enabled": True,
        "feature_name": "V_sync_2",
        "feature_key": "sync2",
        "alpha_mode": "match_reference_std",
        "manual_alpha": 1.0,
        "reference_feature_key": "temp",
        "noise_ratio": 0.08,
        "noise_seed": 20260421,
    },
    {
        "enabled": True,
        "feature_name": "V_sync_3",
        "feature_key": "sync3",
        "alpha_mode": "match_reference_std",
        "manual_alpha": 1.0,
        "reference_feature_key": "temp",
        "noise_ratio": 0.12,
        "noise_seed": 20260422,
    },
)

ACTIVE_SYNC_EXPERIMENTS = tuple(
    config for config in SYNC_EXPERIMENTS if config.get("enabled", False)
)


def natural_sort_key(value):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", value)]


def sanitize_filename(name):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def validate_sync_experiments(sync_experiments):
    feature_names = [config["feature_name"] for config in sync_experiments]
    feature_keys = [config["feature_key"] for config in sync_experiments]
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("SYNC_EXPERIMENTS contains duplicate feature_name values.")
    if len(feature_keys) != len(set(feature_keys)):
        raise ValueError("SYNC_EXPERIMENTS contains duplicate feature_key values.")


def merge_sol_hour(x):
    sols, hours, lat_dim, lon_dim = x.shape
    return x.reshape(sols * hours, lat_dim, lon_dim)


def clean_invalid(x, name):
    x = np.asarray(x, dtype=np.float32)
    bad = ~np.isfinite(x) | (np.abs(x) > 1e10)
    if np.any(bad):
        print(f"{name}: cleaned {bad.sum()} invalid values")
        x[bad] = np.nan
    return np.nan_to_num(x, nan=0.0)


def unwrap_ls(ls_array):
    ls_unwrapped = np.array(ls_array, dtype=np.float32, copy=True)
    year_offset = 0.0
    for idx in range(1, len(ls_unwrapped)):
        if ls_array[idx] < ls_array[idx - 1] - 180.0:
            year_offset += 360.0
        ls_unwrapped[idx] += year_offset
    return ls_unwrapped


def build_sync_variable(o3_raw, reference_raw, train_time_idx, config):
    o3_train = o3_raw[:train_time_idx]
    o3_std = float(np.std(o3_train) + 1e-6)

    if config["alpha_mode"] == "manual":
        alpha = float(config["manual_alpha"])
    elif config["alpha_mode"] == "match_reference_std":
        ref_train = reference_raw[:train_time_idx]
        ref_std = float(np.std(ref_train))
        alpha = ref_std / o3_std
    else:
        raise ValueError(f"Unsupported alpha_mode: {config['alpha_mode']}")

    base_signal = alpha * o3_raw
    base_std = float(np.std(base_signal[:train_time_idx]))
    noise_std = config["noise_ratio"] * (base_std + 1e-6)
    rng = np.random.default_rng(config["noise_seed"])
    noise = rng.normal(loc=0.0, scale=noise_std, size=o3_raw.shape).astype(np.float32)
    sync_raw = (base_signal + noise).astype(np.float32)

    stats = {
        "alpha": alpha,
        "noise_std": noise_std,
        "signal_std": base_std,
        "train_corr_with_o3": float(
            np.corrcoef(o3_train.reshape(-1), sync_raw[:train_time_idx].reshape(-1))[0, 1]
        ),
    }
    return sync_raw, stats


def tensor_to_effective_triplet(w_tensor, b_tensor, k_tensor):
    w_np = w_tensor.detach().cpu().numpy().squeeze()
    b_np = b_tensor.detach().cpu().numpy().squeeze()
    k_np = torch.tanh(k_tensor).detach().cpu().numpy().squeeze()
    return w_np, b_np, k_np


def summarize_abs_distribution(abs_values, suffix, include_threshold_fractions=False):
    abs_values = np.asarray(abs_values, dtype=np.float32).ravel()
    metrics = {
        f"mean_abs_{suffix}": float(np.mean(abs_values)),
        f"median_abs_{suffix}": float(np.median(abs_values)),
        f"std_abs_{suffix}": float(np.std(abs_values)),
        f"rms_abs_{suffix}": float(np.sqrt(np.mean(np.square(abs_values.astype(np.float64))))),
        f"max_abs_{suffix}": float(np.max(abs_values)),
        f"p95_abs_{suffix}": float(np.percentile(abs_values, 95)),
        f"p99_abs_{suffix}": float(np.percentile(abs_values, 99)),
    }
    if include_threshold_fractions:
        metrics.update(
            {
                f"frac_abs_{suffix}_lt_0.05": float(np.mean(abs_values < 0.05)),
                f"frac_abs_{suffix}_gt_0.10": float(np.mean(abs_values > 0.10)),
                f"frac_abs_{suffix}_gt_0.20": float(np.mean(abs_values > 0.20)),
            }
        )
    return metrics


def summarize_triplet(w_np, b_np, k_np):
    return {
        **summarize_abs_distribution(np.abs(w_np), "W"),
        **summarize_abs_distribution(np.abs(b_np), "B", include_threshold_fractions=True),
        **summarize_abs_distribution(np.abs(k_np), "K", include_threshold_fractions=True),
    }


def summarize_absolute_values(abs_w, abs_b, abs_k):
    return {
        **summarize_abs_distribution(abs_w, "W"),
        **summarize_abs_distribution(abs_b, "B", include_threshold_fractions=True),
        **summarize_abs_distribution(abs_k, "K", include_threshold_fractions=True),
    }


def collect_auxiliary_param_stats(model):
    triplets = model.phase_warp.get_auxiliary_param_triplets()
    branch_rows = []
    channel_rows = []

    for feature_name, component_triplets in triplets.items():
        abs_w_parts = []
        abs_b_parts = []
        abs_k_parts = []

        for component_name, tensors in component_triplets.items():
            w_np, b_np, k_np = tensor_to_effective_triplet(*tensors)
            branch_stats = summarize_triplet(w_np, b_np, k_np)
            branch_rows.append(
                {
                    "feature_name": feature_name,
                    "component": component_name.upper(),
                    **branch_stats,
                }
            )
            abs_w_parts.append(np.abs(w_np).ravel())
            abs_b_parts.append(np.abs(b_np).ravel())
            abs_k_parts.append(np.abs(k_np).ravel())

        channel_stats = summarize_absolute_values(
            np.concatenate(abs_w_parts),
            np.concatenate(abs_b_parts),
            np.concatenate(abs_k_parts),
        )
        channel_rows.append(
            {
                "feature_name": feature_name,
                "component": "ALL",
                **channel_stats,
            }
        )

    return branch_rows, channel_rows


def assign_descending_ranks(rows, metric_key, rank_key):
    for rank_idx, row in enumerate(
        sorted(rows, key=lambda current_row: current_row[metric_key], reverse=True),
        start=1,
    ):
        row[rank_key] = rank_idx


def write_stats_csv(path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_sync_evidence_lines(channel_rows, sync_feature_name, metric_key):
    sorted_rows = sorted(channel_rows, key=lambda row: row[metric_key], reverse=True)
    sync_row = next((row for row in channel_rows if row["feature_name"] == sync_feature_name), None)
    if sync_row is None:
        return []
    rank_map = {row["feature_name"]: idx for idx, row in enumerate(sorted_rows, start=1)}

    lines = [
        f"{metric_key} ranking (descending, sin/cos combined):",
    ]
    for rank_idx, row in enumerate(sorted_rows, start=1):
        marker = "  <== target" if row["feature_name"] == sync_feature_name else ""
        lines.append(
            f"  {rank_idx}. {row['feature_name']:<12} "
            f"{row[metric_key]:.6f}{marker}"
        )

    lines.append(
        f"  => {sync_feature_name} rank = "
        f"{rank_map[sync_feature_name]}/{len(sorted_rows)}"
    )
    for row in sorted_rows:
        if row["feature_name"] == sync_feature_name:
            continue
        ratio = sync_row[metric_key] / (row[metric_key] + 1e-12)
        lines.append(
            f"  => {sync_feature_name} / {row['feature_name']} = {ratio:.3f}x"
        )

    return lines


def build_channel_table_lines(channel_rows):
    lines = [
        "Feature-level comparison of |W|, |B| and |K| (sin/cos combined):",
        "  "
        f"{'Feature':<14}{'mean|W|':>12}{'rank':>8}"
        f"{'mean|B|':>12}{'rank':>8}{'mean|K|':>12}{'rank':>8}",
    ]
    for row in channel_rows:
        lines.append(
            "  "
            f"{row['feature_name']:<14}"
            f"{row['mean_abs_W']:>12.6f}{row['rank_mean_abs_W']:>8d}"
            f"{row['mean_abs_B']:>12.6f}{row['rank_mean_abs_B']:>8d}"
            f"{row['mean_abs_K']:>12.6f}{row['rank_mean_abs_K']:>8d}"
        )
    return lines


def build_additional_channel_table_lines(channel_rows):
    lines = [
        "Additional channel metrics for |B| and |K|:",
        "  "
        f"{'Feature':<14}{'median|B|':>12}{'median|K|':>12}"
        f"{'std|B|':>12}{'std|K|':>12}{'rms|B|':>12}{'rms|K|':>12}",
    ]
    for row in channel_rows:
        lines.append(
            "  "
            f"{row['feature_name']:<14}"
            f"{row['median_abs_B']:>12.6f}{row['median_abs_K']:>12.6f}"
            f"{row['std_abs_B']:>12.6f}{row['std_abs_K']:>12.6f}"
            f"{row['rms_abs_B']:>12.6f}{row['rms_abs_K']:>12.6f}"
        )

    lines.append("")
    lines.append(
        "  "
        f"{'Feature':<14}{'p99|B|':>12}{'p99|K|':>12}"
        f"{'frac|B|>0.10':>16}{'frac|K|>0.10':>16}"
        f"{'frac|B|>0.20':>16}{'frac|K|>0.20':>16}"
    )
    for row in channel_rows:
        lines.append(
            "  "
            f"{row['feature_name']:<14}"
            f"{row['p99_abs_B']:>12.6f}{row['p99_abs_K']:>12.6f}"
            f"{row['frac_abs_B_gt_0.10']:>16.6f}{row['frac_abs_K_gt_0.10']:>16.6f}"
            f"{row['frac_abs_B_gt_0.20']:>16.6f}{row['frac_abs_K_gt_0.20']:>16.6f}"
        )
    return lines


def build_branch_table_lines(branch_rows):
    lines = [
        "Branch-level comparison of |W|, |B| and |K|:",
        "  "
        f"{'Feature':<14}{'Comp':<8}{'mean|W|':>12}{'mean|B|':>12}{'mean|K|':>12}",
    ]
    for row in sorted(branch_rows, key=lambda current_row: (current_row["feature_name"], current_row["component"])):
        lines.append(
            "  "
            f"{row['feature_name']:<14}{row['component']:<8}"
            f"{row['mean_abs_W']:>12.6f}"
            f"{row['mean_abs_B']:>12.6f}{row['mean_abs_K']:>12.6f}"
        )
    return lines


def save_sync_summary(model, sync_feature_name, sync_generation_stats):
    triplets = model.phase_warp.get_auxiliary_param_triplets()
    if sync_feature_name not in triplets:
        return

    feature_slug = sanitize_filename(sync_feature_name)
    summary_path = os.path.join(result_dir, f"{EXPERIMENT_TAG}_{feature_slug}_SyncSummary.txt")
    channel_csv_path = os.path.join(
        result_dir, f"{EXPERIMENT_TAG}_{feature_slug}_SyncChannelComparison.csv"
    )
    branch_csv_path = os.path.join(
        result_dir, f"{EXPERIMENT_TAG}_{feature_slug}_SyncBranchComparison.csv"
    )
    branch_triplets = triplets[sync_feature_name]
    branch_rows, channel_rows = collect_auxiliary_param_stats(model)
    assign_descending_ranks(channel_rows, "mean_abs_W", "rank_mean_abs_W")
    assign_descending_ranks(channel_rows, "mean_abs_B", "rank_mean_abs_B")
    assign_descending_ranks(channel_rows, "mean_abs_K", "rank_mean_abs_K")

    lines = [
        f"Experiment: {EXPERIMENT_TAG}",
        f"Feature: {sync_feature_name}",
        "",
        "Synthetic variable generation:",
    ]
    if sync_generation_stats is not None:
        lines.extend(
            [
                f"  alpha = {sync_generation_stats['alpha']:.6f}",
                f"  signal_std = {sync_generation_stats['signal_std']:.6f}",
                f"  noise_std = {sync_generation_stats['noise_std']:.6f}",
                f"  train_corr_with_o3 = {sync_generation_stats['train_corr_with_o3']:.6f}",
                "",
            ]
        )
    else:
        lines.extend(["  unavailable", ""])

    for component_name, tensors in branch_triplets.items():
        w_np, b_np, k_np = tensor_to_effective_triplet(*tensors)
        stats = summarize_triplet(w_np, b_np, k_np)
        lines.append(f"{component_name.upper()} component:")
        for key, value in stats.items():
            lines.append(f"  {key} = {value:.6f}")
        lines.append("")

    lines.extend(build_channel_table_lines(channel_rows))
    lines.append("")
    lines.extend(build_additional_channel_table_lines(channel_rows))
    lines.append("")
    lines.extend(build_sync_evidence_lines(channel_rows, sync_feature_name, "mean_abs_W"))
    lines.append("")
    lines.extend(build_sync_evidence_lines(channel_rows, sync_feature_name, "mean_abs_B"))
    lines.append("")
    lines.extend(build_sync_evidence_lines(channel_rows, sync_feature_name, "mean_abs_K"))
    lines.append("")
    lines.extend(build_sync_evidence_lines(channel_rows, sync_feature_name, "median_abs_B"))
    lines.append("")
    lines.extend(build_sync_evidence_lines(channel_rows, sync_feature_name, "median_abs_K"))
    lines.append("")
    lines.extend(build_sync_evidence_lines(channel_rows, sync_feature_name, "p99_abs_B"))
    lines.append("")
    lines.extend(build_sync_evidence_lines(channel_rows, sync_feature_name, "p99_abs_K"))
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  Ratio < 1.0x means V_sync is smaller than that channel.")
    lines.append("  Ratio > 1.0x means V_sync is larger than that channel.")
    lines.append("")
    lines.extend(build_branch_table_lines(branch_rows))

    with open(summary_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines).strip() + "\n")

    write_stats_csv(channel_csv_path, channel_rows)
    write_stats_csv(branch_csv_path, branch_rows)

    print(f"Saved sync-variable summary to: {summary_path}")
    print(f"Saved sync-channel comparison CSV to: {channel_csv_path}")
    print(f"Saved sync-branch comparison CSV to: {branch_csv_path}")


def plot_spatial_modulation(w_np, b_np, k_np, var_name):
    set_sci_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Spatial Modulation Parameters: {var_name}", fontsize=18, fontweight="bold", y=1.02)

    configs = [
        {"data": w_np, "title": "(a) Amplitude W", "cmap": "RdBu_r", "label": "Amplitude Weights"},
        {"data": b_np, "title": "(b) Phase Delay B", "cmap": "twilight", "label": "Phase Shift (Radians)"},
        {"data": k_np, "title": "(c) Warping Coeff. K", "cmap": "PuOr", "label": "Warping Coefficient"},
    ]

    for idx, cfg in enumerate(configs):
        ax = axes[idx]
        image = ax.imshow(
            cfg["data"],
            extent=[-180, 180, -90, 90],
            cmap=cfg["cmap"],
            origin="upper",
            aspect="auto",
        )
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.set_xticklabels(["180W", "120W", "60W", "0", "60E", "120E", "180E"])
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
        ax.set_yticklabels(["90S", "60S", "30S", "0", "30N", "60N", "90N"])
        ax.set_title(cfg["title"], pad=15, fontweight="bold")
        cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cfg["label"], labelpad=10)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(
        result_dir, f"{EXPERIMENT_TAG}_{sanitize_filename(var_name)}_SpatialModulation.png"
    )
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved spatial modulation map: {save_path}")
    plt.show()


def plot_sync_branch_summary(model, sync_feature_name):
    triplets = model.phase_warp.get_auxiliary_param_triplets()
    if sync_feature_name not in triplets:
        return

    set_sci_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{sync_feature_name} Branch Sanity Check",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )

    component_items = [("Sin", triplets[sync_feature_name]["sin"]), ("Cos", triplets[sync_feature_name]["cos"])]
    panel_titles = ["Amplitude W", "Phase Delay B", "Warping Coeff. K"]
    panel_cmaps = ["RdBu_r", "twilight", "PuOr"]

    for row_idx, (component_name, tensors) in enumerate(component_items):
        w_np, b_np, k_np = tensor_to_effective_triplet(*tensors)
        for col_idx, data in enumerate([w_np, b_np, k_np]):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                data,
                extent=[-180, 180, -90, 90],
                cmap=panel_cmaps[col_idx],
                origin="upper",
                aspect="auto",
            )
            ax.set_title(f"{component_name}: {panel_titles[col_idx]}", fontweight="bold")
            ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
            ax.set_xticklabels(["180W", "120W", "60W", "0", "60E", "120E", "180E"])
            ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
            ax.set_yticklabels(["90S", "60S", "30S", "0", "30N", "60N", "90N"])
            ax.grid(True, linestyle="--", alpha=0.3)
            plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(
        result_dir, f"{EXPERIMENT_TAG}_{sanitize_filename(sync_feature_name)}_BranchSummary.png"
    )
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved sync-branch summary figure: {save_path}")
    plt.show()


def plot_zonal_mean_error(true, pred, forecast_horizon):
    set_sci_style()
    true_first = true[:, 0, 0, :, :]
    pred_short = pred[:, 0, 0, :, :]
    pred_long = pred[:, -1, 0, :, :]

    rmse_short = np.sqrt(np.mean((true_first - pred_short) ** 2, axis=(0, 2)))
    rmse_long = np.sqrt(np.mean((true_first - pred_long) ** 2, axis=(0, 2)))
    latitudes = np.linspace(90, -90, len(rmse_short))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(latitudes, rmse_short, "b--", marker="o", markersize=4, label="Short-term (H=1)")
    ax.plot(latitudes, rmse_long, "r-", marker="s", markersize=4, label=f"Long-term (H={forecast_horizon})")
    ax.axvspan(-90, -60, color="gray", alpha=0.15, label="Polar Vortex Regions")
    ax.axvspan(60, 90, color="gray", alpha=0.15)
    ax.set_xlim(90, -90)
    ax.set_xticks([90, 60, 30, 0, -30, -60, -90])
    ax.set_xticklabels(["90N", "60N", "30N", "0", "30S", "60S", "90S"])
    ax.set_xlabel("Latitude", fontweight="bold")
    ax.set_ylabel("RMSE (Normalized)", fontweight="bold")
    ax.set_title("Zonal Mean Error Analysis", pad=15)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(result_dir, f"{EXPERIMENT_TAG}_ZonalMeanError.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved zonal-mean error figure: {save_path}")
    plt.show()


def evaluate_rmse(model, loader, current_device, target_std, target_mean):
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for xb, lsb, yb in loader:
            xb = xb.to(current_device)
            lsb = lsb.to(current_device)
            pred = model(xb, lsb).cpu().numpy()
            all_preds.append(pred)
            all_trues.append(yb.numpy())

    preds = np.concatenate(all_preds, axis=0) * target_std + target_mean
    trues = np.concatenate(all_trues, axis=0) * target_std + target_mean
    return np.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))


print("\n[Step 1] Loading OpenMars Data...")
o3_list = []
om_ls_list = []

file_list = sorted(glob.glob(os.path.join(openmars_dir, "*.nc")), key=natural_sort_key)
if not file_list:
    raise FileNotFoundError("No OpenMars NetCDF files found.")

with nc.Dataset(file_list[0]) as ref_ds:
    om_lats = ref_ds.variables["lat"][:] if "lat" in ref_ds.variables else ref_ds.variables["latitude"][:]
    om_lons = ref_ds.variables["lon"][:] if "lon" in ref_ds.variables else ref_ds.variables["longitude"][:]
print(f"OpenMars grid size: lat={len(om_lats)}, lon={len(om_lons)}")

for file_path in file_list:
    with nc.Dataset(file_path) as ds:
        o3_list.append(ds.variables["o3col"][:])
        if "Ls" in ds.variables:
            om_ls_list.append(ds.variables["Ls"][:])
        elif "ls" in ds.variables:
            om_ls_list.append(ds.variables["ls"][:])
        else:
            raise ValueError(f"Missing Ls/ls in OpenMars file: {file_path}")

o3col = np.concatenate(o3_list, axis=0)
om_ls_raw = np.concatenate(om_ls_list, axis=0)
print(f"OpenMars O3 shape: {o3col.shape}")

print("\n[Step 2] Loading MCD Data...")
mcd_data_list = {"u": [], "v": [], "temp": [], "fluxsurf_dn_sw": []}
mcd_ls_list = []
target_files = [
    os.path.join(mcd_dir, "MCD_MY27_Lat-90-90_real.nc"),
    os.path.join(mcd_dir, "MCD_MY28_Lat-90-90_real.nc"),
]

for file_path in target_files:
    if not os.path.exists(file_path):
        continue

    print(f"Reading: {os.path.basename(file_path)}")
    with nc.Dataset(file_path) as ds:
        mcd_data_list["u"].append(merge_sol_hour(ds.variables["U_Wind"][:]))
        mcd_data_list["v"].append(merge_sol_hour(ds.variables["V_Wind"][:]))
        mcd_data_list["temp"].append(merge_sol_hour(ds.variables["Temperature"][:]))
        mcd_data_list["fluxsurf_dn_sw"].append(merge_sol_hour(ds.variables["Solar_Flux_DN"][:]))

        ls_tmp = ds.variables["Ls"][:] if "Ls" in ds.variables else ds.variables["ls"][:]
        sols, hours = ds.variables["U_Wind"].shape[:2]
        if ls_tmp.ndim == 1 and len(ls_tmp) == sols:
            ls_expanded = np.zeros(sols * hours, dtype=np.float32)
            for idx in range(sols):
                ls_start = float(ls_tmp[idx])
                if idx < sols - 1:
                    ls_end = float(ls_tmp[idx + 1])
                    if ls_end < ls_start:
                        ls_end += 360.0
                else:
                    ls_end = ls_start + (float(ls_tmp[1] - ls_tmp[0]) if sols > 1 else 0.5)
                ls_expanded[idx * hours : (idx + 1) * hours] = np.linspace(
                    ls_start, ls_end, hours, endpoint=False
                )
            mcd_ls_list.append(ls_expanded % 360.0)
        else:
            mcd_ls_list.append(ls_tmp.flatten().astype(np.float32))

if not any(mcd_data_list.values()):
    raise FileNotFoundError("No MCD NetCDF files were loaded.")

vars_dict = {key: np.concatenate(value, axis=0) for key, value in mcd_data_list.items()}
mcd_ls_raw = np.concatenate(mcd_ls_list, axis=0)

y_raw = clean_invalid(o3col, "OpenMars O3")
for key in vars_dict:
    vars_dict[key] = clean_invalid(vars_dict[key], key)

if "fluxsurf_dn_sw" in vars_dict:
    vars_dict["fluxsurf_dn_sw"] /= np.max(vars_dict["fluxsurf_dn_sw"]) + 1e-6

print("\n[Step 3] Interpolating MCD variables to the OpenMars timeline...")
om_ls_continuous = unwrap_ls(om_ls_raw)
mcd_ls_continuous = unwrap_ls(mcd_ls_raw)

for key in vars_dict:
    interpolator = interp1d(
        mcd_ls_continuous,
        vars_dict[key],
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    vars_dict[key] = interpolator(om_ls_continuous).astype(np.float32)

train_time_idx = int(train_split_ratio * len(y_raw))
auxiliary_feature_arrays = {
    "u": vars_dict["u"],
    "v": vars_dict["v"],
    "t": vars_dict["temp"],
    "f": vars_dict["fluxsurf_dn_sw"],
}

auxiliary_variable_specs = [
    {"key": "u", "display_name": "U_Wind"},
    {"key": "v", "display_name": "V_Wind"},
    {"key": "t", "display_name": "Temperature"},
    {"key": "f", "display_name": "Solar_Flux"},
]

validate_sync_experiments(ACTIVE_SYNC_EXPERIMENTS)
sync_generation_stats_map = {}
for sync_config in ACTIVE_SYNC_EXPERIMENTS:
    reference_key = sync_config["reference_feature_key"]
    reference_raw = auxiliary_feature_arrays.get(reference_key, y_raw)
    sync_raw, sync_generation_stats = build_sync_variable(
        y_raw, reference_raw, train_time_idx, sync_config
    )
    auxiliary_feature_arrays[sync_config["feature_key"]] = sync_raw
    auxiliary_variable_specs.append(
        {
            "key": sync_config["feature_key"],
            "display_name": sync_config["feature_name"],
        }
    )
    sync_generation_stats_map[sync_config["feature_name"]] = sync_generation_stats
    print(
        f"Added synthetic feature {sync_config['feature_name']} "
        f"(alpha={sync_generation_stats['alpha']:.4f}, "
        f"noise_std={sync_generation_stats['noise_std']:.4f}, "
        f"train_corr={sync_generation_stats['train_corr_with_o3']:.4f})"
    )

print("\n[Step 4] Building scaled sequences...")
feature_arrays = [y_raw]
feature_names = ["Prev_O3"]
for spec in auxiliary_variable_specs:
    feature_arrays.append(auxiliary_feature_arrays[spec["key"]])
    feature_names.append(spec["display_name"])

X_raw = np.stack(feature_arrays, axis=-1).astype(np.float32)
T, H, W, C = X_raw.shape
print(f"Raw feature tensor shape: {X_raw.shape}")

X_train_raw = X_raw[:train_time_idx]
y_train_raw = y_raw[:train_time_idx]

X_scaled = np.zeros_like(X_raw, dtype=np.float32)
for channel_idx in range(C):
    scaler = StandardScaler()
    scaler.fit(X_train_raw[..., channel_idx].reshape(train_time_idx, -1))
    X_scaled[..., channel_idx] = scaler.transform(X_raw[..., channel_idx].reshape(T, -1)).reshape(T, H, W)

y_mean = float(y_train_raw.mean())
y_std = float(y_train_raw.std() + 1e-6)
y_scaled = ((y_raw - y_mean) / y_std).astype(np.float32)

X_seq = []
y_seq = []
ls_seq = []
for start_idx in range(T - window - horizon + 1):
    end_idx = start_idx + window + horizon
    X_seq.append(X_scaled[start_idx:end_idx])
    y_seq.append(y_scaled[start_idx + window : end_idx])
    ls_seq.append(om_ls_continuous[start_idx:end_idx])

X_torch = torch.tensor(np.asarray(X_seq, dtype=np.float32)).permute(0, 1, 4, 2, 3)
y_torch = torch.tensor(np.asarray(y_seq, dtype=np.float32)).unsqueeze(2)
ls_torch = torch.tensor(np.asarray(ls_seq, dtype=np.float32))

split_sample_idx = max(0, min(train_time_idx - window - horizon + 1, len(X_torch)))
train_dataset = TensorDataset(
    X_torch[:split_sample_idx],
    ls_torch[:split_sample_idx],
    y_torch[:split_sample_idx],
)
test_dataset = TensorDataset(
    X_torch[split_sample_idx:],
    ls_torch[split_sample_idx:],
    y_torch[split_sample_idx:],
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if len(train_loader) == 0 or len(test_loader) == 0:
    raise ValueError("Train/test loaders are empty. Check the sequence construction settings.")

print(f"Training samples: {len(train_dataset)} | Validation samples: {len(test_dataset)}")
print(f"Feature order: {feature_names}")


class SpatioTemporalLSTMCellv2(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size):
        super().__init__()
        padding = filter_size // 2
        self.conv_x = nn.Conv2d(in_channel, num_hidden * 7, filter_size, padding=padding)
        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, filter_size, padding=padding)
        self.conv_m = nn.Conv2d(num_hidden, num_hidden * 3, filter_size, padding=padding)
        self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, filter_size, padding=padding)
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, 1)
        self.num_hidden = num_hidden

    def forward(self, x, h, c, m):
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)
        i_x, f_x, g_x, i_xp, f_xp, g_xp, o_x = torch.split(x_concat, self.num_hidden, 1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, 1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, 1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + 1.0)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c + i_t * g_t

        i_tp = torch.sigmoid(i_xp + i_m)
        f_tp = torch.sigmoid(f_xp + f_m + 1.0)
        g_tp = torch.tanh(g_xp + g_m)
        m_new = f_tp * m + i_tp * g_tp

        mem = torch.cat([c_new, m_new], dim=1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new


class PredRNNv2(nn.Module):
    def __init__(
        self,
        auxiliary_variable_specs,
        hidden_dims=None,
        height=H,
        width=W,
        forecast_horizon=horizon,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims or [64, 64, 64]
        self.phase_warp = PhaseWarpFrontEnd(
            spatial_shape=(height, width),
            auxiliary_variable_specs=auxiliary_variable_specs,
        )
        self.input_dim = self.phase_warp.get_output_channels()
        self.horizon = forecast_horizon

        self.layers = nn.ModuleList()
        for layer_idx, hidden_dim in enumerate(self.hidden_dims):
            in_channels = self.input_dim if layer_idx == 0 else self.hidden_dims[layer_idx - 1]
            self.layers.append(SpatioTemporalLSTMCellv2(in_channels, hidden_dim, height, width, 3))

        self.conv_last = nn.Conv2d(self.hidden_dims[-1], 1, 1)

    def load_state_dict(self, state_dict, strict=True):
        phase_warp_keys = set(self.phase_warp.state_dict().keys())
        if any(key in phase_warp_keys for key in state_dict) and not any(
            key.startswith("phase_warp.") for key in state_dict
        ):
            state_dict = {
                (f"phase_warp.{key}" if key in phase_warp_keys else key): value
                for key, value in state_dict.items()
            }
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x, ls):
        batch_size_local, total_steps, _, height_dim, width_dim = x.shape
        encoder_steps = total_steps - self.horizon
        x_new = self.phase_warp(x, ls)

        h = [torch.zeros(batch_size_local, dim, height_dim, width_dim, device=x.device) for dim in self.hidden_dims]
        c = [torch.zeros_like(h_state) for h_state in h]
        m = torch.zeros_like(h[0])

        for step_idx in range(encoder_steps):
            inp = x_new[:, step_idx]
            for layer_idx, cell in enumerate(self.layers):
                h[layer_idx], c[layer_idx], m = cell(inp, h[layer_idx], c[layer_idx], m)
                inp = h[layer_idx]

        preds = []
        current_o3 = x[:, encoder_steps - 1 : encoder_steps, 0:1, :, :]
        for step_idx in range(self.horizon):
            full_step_idx = encoder_steps + step_idx
            real_aux_features = x_new[:, full_step_idx, 1:, :, :]
            current_o3_fused = self.phase_warp.fuse_o3(current_o3, ls[:, full_step_idx : full_step_idx + 1])
            inp = torch.cat([current_o3_fused.squeeze(1), real_aux_features], dim=1)

            for layer_idx, cell in enumerate(self.layers):
                h[layer_idx], c[layer_idx], m = cell(inp, h[layer_idx], c[layer_idx], m)
                inp = h[layer_idx]

            pred_o3 = self.conv_last(h[-1])
            preds.append(pred_o3)
            current_o3 = pred_o3.unsqueeze(1)

        return torch.stack(preds, dim=1)


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0, path="checkpoint.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


model = PredRNNv2(auxiliary_variable_specs=auxiliary_variable_specs, height=H, width=W).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.SmoothL1Loss()

checkpoint_path = os.path.join(result_dir, f"{EXPERIMENT_TAG}_best.pth")
final_model_path = os.path.join(result_dir, f"{EXPERIMENT_TAG}.pth")
early_stopping = EarlyStopping(patience=5, verbose=True, path=checkpoint_path)

print("\n[Step 5] Start training...")
for epoch_idx in range(epochs):
    model.train()
    train_loss_sum = 0.0
    for xb, lsb, yb in train_loader:
        xb = xb.to(device)
        lsb = lsb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred = model(xb, lsb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()

    avg_train_loss = train_loss_sum / len(train_loader)

    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for xb, lsb, yb in test_loader:
            xb = xb.to(device)
            lsb = lsb.to(device)
            yb = yb.to(device)
            pred = model(xb, lsb)
            val_loss_sum += criterion(pred, yb).item()

    avg_val_loss = val_loss_sum / len(test_loader)
    print(
        f"Epoch {epoch_idx + 1}/{epochs} | "
        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
    )

    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

print("Loading best model state from checkpoint...")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

print("\n[Step 6] Evaluation...")
model.eval()
preds = []
trues = []

with torch.no_grad():
    for xb, lsb, yb in test_loader:
        xb = xb.to(device)
        lsb = lsb.to(device)
        y_pred = model(xb, lsb).cpu().numpy()
        preds.append(y_pred)
        trues.append(yb.numpy())

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)

y_pred_phys = preds * y_std + y_mean
y_true_phys = trues * y_std + y_mean
pred_flat = y_pred_phys.reshape(-1)
true_flat = y_true_phys.reshape(-1)

mse = float(np.mean((pred_flat - true_flat) ** 2))
rmse = float(np.sqrt(mse))
mae = float(np.mean(np.abs(pred_flat - true_flat)))
ss_res = float(np.sum((true_flat - pred_flat) ** 2))
ss_tot = float(np.sum((true_flat - np.mean(true_flat)) ** 2) + 1e-6)
r2 = float(1.0 - ss_res / ss_tot)

print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R2  : {r2:.4f}")

print("\nAdvanced Metrics")
threshold = 0.1
mask = true_flat > threshold
if np.sum(mask) > 0:
    mape_filtered = float(np.mean(np.abs((true_flat[mask] - pred_flat[mask]) / true_flat[mask])))
    print(f"Filtered MAPE (>{threshold}): {mape_filtered:.2%}")

smape = float(np.mean(2.0 * np.abs(pred_flat - true_flat) / (np.abs(true_flat) + np.abs(pred_flat) + 1e-6)))
print(f"SMAPE: {smape:.2%}")

print("\n[Step 7] Visualization...")
try:
    for var_name, w_param, b_param, k_param in model.phase_warp.get_plot_configs():
        w_np, b_np, k_np = tensor_to_effective_triplet(w_param, b_param, k_param)
        plot_spatial_modulation(w_np, b_np, k_np, var_name)

    for sync_config in ACTIVE_SYNC_EXPERIMENTS:
        sync_feature_name = sync_config["feature_name"]
        plot_sync_branch_summary(model, sync_feature_name)
        save_sync_summary(
            model,
            sync_feature_name,
            sync_generation_stats_map.get(sync_feature_name),
        )

    plot_zonal_mean_error(trues, preds, horizon)

    plt.figure(figsize=(9, 8), facecolor="white")
    hb = plt.hexbin(true_flat, pred_flat, gridsize=100, cmap="viridis", mincnt=1, bins="log")
    max_val = max(np.max(true_flat), np.max(pred_flat))
    min_val = min(np.min(true_flat), np.min(pred_flat))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", lw=2, label="1:1 Line")
    plt.xlabel("True O3 Column", fontsize=12)
    plt.ylabel("Predicted O3 Column", fontsize=12)
    plt.title(f"True vs Predicted O3 (Horizon={horizon})", fontsize=14, fontweight="bold")
    plt.text(
        0.05,
        0.95,
        f"RMSE: {rmse:.4f}\nR2: {r2:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    plt.colorbar(hb).set_label("Sample Density (log10)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    scatter_path = os.path.join(result_dir, f"{EXPERIMENT_TAG}_O3Scatter.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    print(f"Saved scatter figure: {scatter_path}")
    plt.show()
except Exception as exc:
    print(f"Visualization failed: {exc}")

torch.save(model.state_dict(), final_model_path)
print(f"\nSaved model weights to: {final_model_path}")

print("\n[Step 8] Permutation Feature Importance...")
baseline_rmse = evaluate_rmse(model, test_loader, device, y_std, y_mean)
print(f"Baseline RMSE: {baseline_rmse:.4f}")

pfi_scores = []
for feature_idx, feature_name in enumerate(feature_names):
    X_test_tensor = test_dataset.tensors[0].clone()
    ls_test_tensor = test_dataset.tensors[1]
    y_test_tensor = test_dataset.tensors[2]

    perm_idx = torch.randperm(X_test_tensor.size(0))
    X_test_tensor[:, :, feature_idx, :, :] = X_test_tensor[perm_idx, :, feature_idx, :, :]

    permuted_dataset = TensorDataset(X_test_tensor, ls_test_tensor, y_test_tensor)
    permuted_loader = DataLoader(permuted_dataset, batch_size=batch_size, shuffle=False)

    permuted_rmse = evaluate_rmse(model, permuted_loader, device, y_std, y_mean)
    importance = float(permuted_rmse - baseline_rmse)
    pfi_scores.append(importance)
    print(f"Feature [{feature_name}] -> permuted RMSE: {permuted_rmse:.4f}, increase: {importance:.4f}")

try:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=pfi_scores, y=feature_names, palette="viridis")
    plt.title("Permutation Feature Importance (PFI)")
    plt.xlabel("Increase in RMSE")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    pfi_path = os.path.join(result_dir, f"{EXPERIMENT_TAG}_PFI.png")
    plt.savefig(pfi_path, bbox_inches="tight")
    print(f"Saved PFI figure: {pfi_path}")
    plt.show()
except Exception as exc:
    print(f"PFI plotting failed: {exc}")
