"""
PredRNNv2 ablation runner for O3 + configurable UVST subsets.

Each run keeps O3 history as a fixed input, selects one UVST subset,
and evaluates two matched branches:
1. Raw inputs
2. PhaseWarp front-end + the same PredRNNv2 backbone
"""

import glob
import os
import re
import sys
from itertools import combinations

import netCDF4 as nc
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from phase_warp_frontend import PhaseWarpFrontEnd, normalize_active_uvst


UVST_ORDER = ("U", "V", "S", "T")
RUN_ALL_UVST_ABLATIONS = True
ACTIVE_UVST_SUBSET = ("U", "V", "S", "T")
SUMMARY_COLUMNS = (
    "subset_tag",
    "active_uvst",
    "raw_rmse",
    "phase_rmse",
    "rmse_improve",
    "raw_mae",
    "phase_mae",
    "raw_r2",
    "phase_r2",
    "raw_smape",
    "phase_smape",
)


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


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
            print(f"Validation loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", s)]


def merge_sol_hour(x):
    sols, hours, lat_size, lon_size = x.shape
    return x.reshape(sols * hours, lat_size, lon_size)


def clean_invalid(x, name):
    x = np.array(x, dtype=np.float32)
    bad = ~np.isfinite(x) | (np.abs(x) > 1e10)
    if np.any(bad):
        print(f"{name}: cleaned {bad.sum()} invalid values")
        x[bad] = np.nan
    return np.nan_to_num(x, nan=0.0)


def unwrap_ls(ls_array):
    ls_unwrapped = np.copy(ls_array)
    year_offset = 0
    for idx in range(1, len(ls_unwrapped)):
        if ls_array[idx] < ls_array[idx - 1] - 180:
            year_offset += 360
        ls_unwrapped[idx] += year_offset
    return ls_unwrapped


def build_all_uvst_subsets():
    subsets = []
    for size in range(len(UVST_ORDER) + 1):
        subsets.extend(combinations(UVST_ORDER, size))
    return tuple(tuple(subset) for subset in subsets)


def resolve_target_subsets():
    if RUN_ALL_UVST_ABLATIONS:
        return build_all_uvst_subsets()
    return (normalize_active_uvst(ACTIVE_UVST_SUBSET),)


def build_subset_tag(active_uvst):
    active_uvst = normalize_active_uvst(active_uvst)
    if not active_uvst:
        return "o3only"
    return "o3_" + "".join(channel.lower() for channel in active_uvst)


def format_active_uvst(active_uvst):
    active_uvst = normalize_active_uvst(active_uvst)
    if not active_uvst:
        return "()"
    return ",".join(active_uvst)


def load_aligned_cube(base_dir):
    openmars_dir = os.path.join(base_dir, "Dataset", "OpenMars")
    mcd_dir = os.path.join(base_dir, "Dataset", "MCDALL")

    print("\n[Step 1] Loading OpenMars Data...")
    file_list = sorted(glob.glob(os.path.join(openmars_dir, "*.nc")), key=natural_sort_key)
    if not file_list:
        raise FileNotFoundError("OpenMars files were not found.")

    o3_list = []
    om_ls_list = []
    for file_path in file_list:
        ds = nc.Dataset(file_path)
        o3_list.append(ds.variables["o3col"][:])
        if "Ls" in ds.variables:
            om_ls_list.append(ds.variables["Ls"][:])
        elif "ls" in ds.variables:
            om_ls_list.append(ds.variables["ls"][:])
        else:
            raise ValueError(f"Missing Ls variable in {file_path}")
        ds.close()

    y_raw = clean_invalid(np.concatenate(o3_list, axis=0), "OpenMars O3")
    om_ls_raw = np.concatenate(om_ls_list, axis=0)
    print(f"OpenMars final shape: {y_raw.shape}")

    print("\n[Step 2] Loading MCD Data...")
    target_files = [
        os.path.join(mcd_dir, "MCD_MY27_Lat-90-90_real.nc"),
        os.path.join(mcd_dir, "MCD_MY28_Lat-90-90_real.nc"),
    ]

    mcd_data_list = {
        "u": [],
        "v": [],
        "temp": [],
        "fluxsurf_dn_sw": [],
    }
    mcd_ls_list = []

    for file_path in target_files:
        if not os.path.exists(file_path):
            continue

        print(f"Loading: {os.path.basename(file_path)}")
        ds = nc.Dataset(file_path)

        mcd_data_list["u"].append(merge_sol_hour(ds.variables["U_Wind"][:]))
        mcd_data_list["v"].append(merge_sol_hour(ds.variables["V_Wind"][:]))
        mcd_data_list["temp"].append(merge_sol_hour(ds.variables["Temperature"][:]))
        mcd_data_list["fluxsurf_dn_sw"].append(merge_sol_hour(ds.variables["Solar_Flux_DN"][:]))

        ls_tmp = ds.variables["Ls"][:] if "Ls" in ds.variables else ds.variables["ls"][:]
        sols, hours = ds.variables["U_Wind"].shape[:2]

        if ls_tmp.ndim == 1 and len(ls_tmp) == sols:
            ls_expanded = np.zeros(sols * hours, dtype=np.float32)
            for idx in range(sols):
                ls_start = ls_tmp[idx]
                if idx < sols - 1:
                    ls_end = ls_tmp[idx + 1]
                    if ls_end < ls_start:
                        ls_end += 360.0
                else:
                    ls_end = ls_start + (ls_tmp[1] - ls_tmp[0] if sols > 1 else 0.5)
                ls_expanded[idx * hours:(idx + 1) * hours] = np.linspace(
                    ls_start,
                    ls_end,
                    hours,
                    endpoint=False,
                )
            mcd_ls_list.append(ls_expanded % 360.0)
        else:
            mcd_ls_list.append(ls_tmp.flatten())

        ds.close()

    if not mcd_ls_list:
        raise FileNotFoundError("MCD files were not found or did not contain usable variables.")

    vars_dict = {}
    for key, values in mcd_data_list.items():
        if not values:
            raise ValueError(f"MCD variable {key} was not loaded from the available files.")
        vars_dict[key] = clean_invalid(np.concatenate(values, axis=0), key)

    vars_dict["fluxsurf_dn_sw"] /= (np.max(vars_dict["fluxsurf_dn_sw"]) + 1e-6)

    print("\n[Step 3] Aligning MCD to OpenMars time axis...")
    om_ls_continuous = unwrap_ls(om_ls_raw)
    mcd_ls_continuous = unwrap_ls(np.concatenate(mcd_ls_list, axis=0))

    for key in vars_dict:
        interpolator = interp1d(
            mcd_ls_continuous,
            vars_dict[key],
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        vars_dict[key] = interpolator(om_ls_continuous)

    feature_cubes = {
        "O3": y_raw,
        "U": vars_dict["u"],
        "V": vars_dict["v"],
        "S": vars_dict["fluxsurf_dn_sw"],
        "T": vars_dict["temp"],
    }
    return feature_cubes, y_raw, om_ls_continuous


def build_grid_dataloaders(feature_cubes, y_raw, ls_raw, active_uvst, window, horizon, batch_size):
    active_uvst = normalize_active_uvst(active_uvst)
    input_keys = ("O3",) + active_uvst
    x_raw = np.stack([feature_cubes[key] for key in input_keys], axis=-1).astype(np.float32)

    time_steps, lat_size, lon_size, channels = x_raw.shape
    split_time_idx = int(0.8 * time_steps)

    x_train_raw = x_raw[:split_time_idx]
    y_train_raw = y_raw[:split_time_idx]

    x_scaled = np.zeros_like(x_raw, dtype=np.float32)
    for channel_idx in range(channels):
        scaler = StandardScaler()
        scaler.fit(x_train_raw[..., channel_idx].reshape(split_time_idx, -1))
        x_scaled[..., channel_idx] = scaler.transform(x_raw[..., channel_idx].reshape(time_steps, -1)).reshape(
            time_steps,
            lat_size,
            lon_size,
        )

    y_mean = float(y_train_raw.mean())
    y_std = float(y_train_raw.std())
    y_scaled = (y_raw - y_mean) / (y_std + 1e-6)

    x_seq = []
    y_seq = []
    ls_seq = []
    for idx in range(time_steps - window - horizon + 1):
        x_seq.append(x_scaled[idx: idx + window])
        y_seq.append(y_scaled[idx + window: idx + window + horizon])
        ls_seq.append(ls_raw[idx: idx + window])

    x_torch = torch.tensor(np.asarray(x_seq, dtype=np.float32)).permute(0, 1, 4, 2, 3)
    y_torch = torch.tensor(np.asarray(y_seq, dtype=np.float32))
    ls_torch = torch.tensor(np.asarray(ls_seq, dtype=np.float32))

    split_sample_idx = max(0, min(split_time_idx - window - horizon + 1, len(x_torch)))
    train_dataset = TensorDataset(x_torch[:split_sample_idx], ls_torch[:split_sample_idx], y_torch[:split_sample_idx])
    test_dataset = TensorDataset(x_torch[split_sample_idx:], ls_torch[split_sample_idx:], y_torch[split_sample_idx:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, y_mean, y_std, input_keys


class SpatioTemporalLSTMCellV2(nn.Module):
    def __init__(self, input_dim, hidden_dim, filter_size=3):
        super().__init__()
        padding = filter_size // 2
        self.hidden_dim = hidden_dim

        self.conv_x = nn.Conv2d(input_dim, hidden_dim * 7, filter_size, padding=padding)
        self.conv_h = nn.Conv2d(hidden_dim, hidden_dim * 4, filter_size, padding=padding)
        self.conv_m = nn.Conv2d(hidden_dim, hidden_dim * 3, filter_size, padding=padding)
        self.conv_o = nn.Conv2d(hidden_dim * 2, hidden_dim, filter_size, padding=padding)
        self.conv_last = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)

    def forward(self, x, h, c, m):
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)

        i_x, f_x, g_x, i_xp, f_xp, g_xp, o_x = torch.split(x_concat, self.hidden_dim, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.hidden_dim, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.hidden_dim, dim=1)

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


class PredRNNv2Forecaster(nn.Module):
    def __init__(self, pred_len, lat_size, lon_size, active_uvst, use_phase_warp, hidden_dims, filter_size=3):
        super().__init__()
        self.pred_len = pred_len
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.active_uvst = normalize_active_uvst(active_uvst)
        self.use_phase_warp = use_phase_warp
        self.hidden_dims = list(hidden_dims)
        self.raw_input_dim = 1 + len(self.active_uvst)

        if use_phase_warp:
            self.phase_warp = PhaseWarpFrontEnd(
                spatial_shape=(lat_size, lon_size),
                active_uvst=self.active_uvst,
            )
            input_dim = 1 + 2 * len(self.active_uvst)
        else:
            self.phase_warp = None
            input_dim = self.raw_input_dim

        self.model_input_dim = input_dim
        self.cells = nn.ModuleList()
        for layer_idx, hidden_dim in enumerate(self.hidden_dims):
            cur_input_dim = input_dim if layer_idx == 0 else self.hidden_dims[layer_idx - 1]
            self.cells.append(SpatioTemporalLSTMCellV2(cur_input_dim, hidden_dim, filter_size=filter_size))

        self.forecast_head = nn.Conv2d(self.hidden_dims[-1], pred_len, kernel_size=1)

    def _init_states(self, batch_size, device):
        h_states = []
        c_states = []
        for hidden_dim in self.hidden_dims:
            h_state = torch.zeros(batch_size, hidden_dim, self.lat_size, self.lon_size, device=device)
            c_state = torch.zeros_like(h_state)
            h_states.append(h_state)
            c_states.append(c_state)

        memory = torch.zeros(batch_size, self.hidden_dims[0], self.lat_size, self.lon_size, device=device)
        return h_states, c_states, memory

    def forward(self, x, ls):
        if x.size(2) != self.raw_input_dim:
            raise ValueError(f"Expected {self.raw_input_dim} raw channels, got {x.size(2)}")

        features = self.phase_warp(x, ls) if self.phase_warp is not None else x
        if features.size(2) != self.model_input_dim:
            raise ValueError(f"Expected {self.model_input_dim} model channels, got {features.size(2)}")

        batch_size, seq_len, _, _, _ = features.shape
        h_states, c_states, memory = self._init_states(batch_size, features.device)

        for time_idx in range(seq_len):
            current = features[:, time_idx]
            for layer_idx, cell in enumerate(self.cells):
                h_next, c_next, memory = cell(current, h_states[layer_idx], c_states[layer_idx], memory)
                h_states[layer_idx] = h_next
                c_states[layer_idx] = c_next
                current = h_next

        last_hidden = h_states[-1]
        return self.forecast_head(last_hidden)


def evaluate_metrics(model, loader, device, y_std, y_mean):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, lsb, yb in loader:
            xb = xb.to(device)
            lsb = lsb.to(device)
            pred = model(xb, lsb).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    y_pred_phys = preds * (y_std + 1e-6) + y_mean
    y_true_phys = trues * (y_std + 1e-6) + y_mean

    pred_flat = y_pred_phys.reshape(-1)
    true_flat = y_true_phys.reshape(-1)

    mse = np.mean((pred_flat - true_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_flat - true_flat))
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    smape = np.mean(2.0 * np.abs(pred_flat - true_flat) / (np.abs(true_flat) + np.abs(pred_flat) + 1e-6))

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "smape": float(smape),
    }


def train_and_evaluate(
    branch_name,
    subset_tag,
    active_uvst,
    use_phase_warp,
    train_loader,
    test_loader,
    device,
    y_std,
    y_mean,
    lat_size,
    lon_size,
    horizon,
    hidden_dims,
    filter_size,
    epochs,
    learning_rate,
    early_stopping_patience,
    results_dir,
):
    display_name = f"PredRNNv2_{branch_name}_{subset_tag}"
    print(f"\n[Experiment] {display_name}")

    model = PredRNNv2Forecaster(
        pred_len=horizon,
        lat_size=lat_size,
        lon_size=lon_size,
        active_uvst=active_uvst,
        use_phase_warp=use_phase_warp,
        hidden_dims=hidden_dims,
        filter_size=filter_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()
    file_stem = display_name.lower()
    checkpoint_path = os.path.join(results_dir, f"{file_stem}_checkpoint.pth")
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=checkpoint_path)

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
                loss = criterion(pred, yb)
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / len(test_loader)
        print(
            f"{display_name} | Epoch {epoch_idx + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"{display_name} triggered early stopping.")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    metrics = evaluate_metrics(model, test_loader, device, y_std, y_mean)

    save_path = os.path.join(results_dir, f"{file_stem}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"{display_name} weights saved to: {save_path}")
    print(
        f"{display_name} Metrics | RMSE: {metrics['rmse']:.4f} | "
        f"MAE: {metrics['mae']:.4f} | R^2: {metrics['r2']:.4f} | SMAPE: {metrics['smape']:.2%}"
    )
    return metrics


def build_summary_record(subset_tag, active_uvst, raw_metrics, phase_metrics):
    return {
        "subset_tag": subset_tag,
        "active_uvst": format_active_uvst(active_uvst),
        "raw_rmse": f"{raw_metrics['rmse']:.6f}",
        "phase_rmse": f"{phase_metrics['rmse']:.6f}",
        "rmse_improve": f"{raw_metrics['rmse'] - phase_metrics['rmse']:.6f}",
        "raw_mae": f"{raw_metrics['mae']:.6f}",
        "phase_mae": f"{phase_metrics['mae']:.6f}",
        "raw_r2": f"{raw_metrics['r2']:.6f}",
        "phase_r2": f"{phase_metrics['r2']:.6f}",
        "raw_smape": f"{raw_metrics['smape']:.6f}",
        "phase_smape": f"{phase_metrics['smape']:.6f}",
    }


def summary_sort_key(record):
    active_text = record["active_uvst"]
    active_uvst = tuple(active_text.split(",")) if active_text and active_text != "()" else ()
    membership = tuple(1 if key in active_uvst else 0 for key in UVST_ORDER)
    return (len(active_uvst), membership, record["subset_tag"])


def update_summary_file(summary_path, record):
    existing_records = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                if line.split("\t")[0] == SUMMARY_COLUMNS[0]:
                    continue
                parts = line.split("\t")
                if len(parts) != len(SUMMARY_COLUMNS):
                    continue
                parsed_record = dict(zip(SUMMARY_COLUMNS, parts))
                existing_records[parsed_record["subset_tag"]] = parsed_record

    existing_records[record["subset_tag"]] = record
    ordered_records = sorted(existing_records.values(), key=summary_sort_key)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\t".join(SUMMARY_COLUMNS) + "\n")
        for row in ordered_records:
            f.write("\t".join(row[column] for column in SUMMARY_COLUMNS) + "\n")


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_subset(
    active_uvst,
    subset_index,
    total_subsets,
    feature_cubes,
    y_raw,
    ls_raw,
    device,
    logs_dir,
    results_dir,
    summary_path,
    window,
    horizon,
    batch_size,
    hidden_dims,
    filter_size,
    epochs,
    learning_rate,
    early_stopping_patience,
    seed,
):
    active_uvst = normalize_active_uvst(active_uvst)
    subset_tag = build_subset_tag(active_uvst)
    log_path = os.path.join(logs_dir, f"PredRNNv2_Ablation_{subset_tag}.txt")

    original_stdout = sys.stdout
    logger = Logger(log_path)
    sys.stdout = logger

    try:
        set_random_seed(seed)
        lat_size, lon_size = y_raw.shape[1], y_raw.shape[2]
        print(f"Training Device: {device}")
        print(f"Running subset {subset_index}/{total_subsets}")
        print(f"Configured ACTIVE_UVST_SUBSET: {active_uvst}")
        print(f"Subset tag: {subset_tag}")
        train_loader, test_loader, y_mean, y_std, input_keys = build_grid_dataloaders(
            feature_cubes=feature_cubes,
            y_raw=y_raw,
            ls_raw=ls_raw,
            active_uvst=active_uvst,
            window=window,
            horizon=horizon,
            batch_size=batch_size,
        )
        print(f"Input channels for this run: {input_keys}")

        raw_metrics = train_and_evaluate(
            branch_name="Raw",
            subset_tag=subset_tag,
            active_uvst=active_uvst,
            use_phase_warp=False,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            y_std=y_std,
            y_mean=y_mean,
            lat_size=lat_size,
            lon_size=lon_size,
            horizon=horizon,
            hidden_dims=hidden_dims,
            filter_size=filter_size,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            results_dir=results_dir,
        )

        phase_metrics = train_and_evaluate(
            branch_name="PhaseWarp",
            subset_tag=subset_tag,
            active_uvst=active_uvst,
            use_phase_warp=True,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            y_std=y_std,
            y_mean=y_mean,
            lat_size=lat_size,
            lon_size=lon_size,
            horizon=horizon,
            hidden_dims=hidden_dims,
            filter_size=filter_size,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            results_dir=results_dir,
        )

        print("\n[Comparison Summary]")
        print(f"RMSE improvement: {raw_metrics['rmse'] - phase_metrics['rmse']:.4f}")
        print(f"MAE improvement : {raw_metrics['mae'] - phase_metrics['mae']:.4f}")
        print(f"R^2 gain        : {phase_metrics['r2'] - raw_metrics['r2']:.4f}")
        print(f"SMAPE gain      : {raw_metrics['smape'] - phase_metrics['smape']:.2%}")

        summary_record = build_summary_record(
            subset_tag=subset_tag,
            active_uvst=active_uvst,
            raw_metrics=raw_metrics,
            phase_metrics=phase_metrics,
        )
        update_summary_file(summary_path, summary_record)
        print(f"Summary updated: {summary_path}")
    finally:
        sys.stdout = original_stdout
        logger.close()


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, "models", "训练过程")
    results_dir = os.path.join(base_dir, "models", "训练结果")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "predrnnv2_uvst_ablation_summary.txt")
    target_subsets = resolve_target_subsets()

    seed = 42
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training Device: {device}")
    if RUN_ALL_UVST_ABLATIONS:
        print(f"Batch ablation mode enabled: {len(target_subsets)} UVST subsets will run sequentially.")
    else:
        print(f"Single-subset mode enabled: {target_subsets[0]}")

    window = 3
    horizon = 3
    batch_size = 4

    hidden_dim = 32
    num_layers = 2
    hidden_dims = [hidden_dim] * num_layers
    filter_size = 3

    epochs = 15
    learning_rate = 1e-3
    early_stopping_patience = 5

    feature_cubes, y_raw, ls_raw = load_aligned_cube(base_dir)

    for subset_index, active_uvst in enumerate(target_subsets, start=1):
        print(f"\n[Batch] Starting subset {subset_index}/{len(target_subsets)}: {active_uvst}")
        run_single_subset(
            active_uvst=active_uvst,
            subset_index=subset_index,
            total_subsets=len(target_subsets),
            feature_cubes=feature_cubes,
            y_raw=y_raw,
            ls_raw=ls_raw,
            device=device,
            logs_dir=logs_dir,
            results_dir=results_dir,
            summary_path=summary_path,
            window=window,
            horizon=horizon,
            batch_size=batch_size,
            hidden_dims=hidden_dims,
            filter_size=filter_size,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            seed=seed,
        )

    print("\nAll requested UVST ablation runs completed.")


if __name__ == "__main__":
    main()
