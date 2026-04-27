"""
本脚本比较在火星臭氧任务上两种逐点 DLinear 实验：

1. DLinear_Raw：
   将原始历史变量直接输入 DLinear 风格的骨干网络。
2. DLinear_PhaseWarp：
   在相同的 DLinear 骨干前加入 PhaseWarpFrontEnd。

设计思路：
- 原始 PredRNN 流程在完整的时空网格上工作 [B, T, C, H, W]。
- 通用时间序列骨干如 DLinear 更适配于 [B, T, C]。
- 为了衔接两者，脚本把每个经纬度网格点视为一个多变量序列，
  并在所有网格点共享同一套模型权重。

目标不是打造最强的 DLinear 变体，而是检验在更换预测骨干后，
相位扭曲思路是否仍能提升精度。
"""

import glob
import os
import re
import sys

import netCDF4 as nc
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Reuse the same phase-warp module that has been extracted from the PredRNN script.
# This keeps the "innovation module" consistent across different forecasting backbones.
from phase_warp_frontend import PhaseWarpFrontEnd


class Logger(object):
    """将 stdout 镜像到日志文件，以保证长时间实验可复现。"""

    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class EarlyStopping:
    """
    当验证损失不再下降时提前停止训练。

    这里实现保持极简，因为脚本的核心是模块对比，
    而非激进的训练技巧。
    """

    def __init__(self, patience=20, verbose=False, delta=0.0, path="checkpoint.pth"):
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
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def natural_sort_key(s):
    """按自然顺序对文件名排序，例如 2 在 10 前。"""

    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", s)]


def merge_sol_hour(x):
    """
    将 MCD 数据从 [sol, hour, lat, lon] 展平为 [time, lat, lon]。

    OpenMars 已经沿单一时间轴组织，因此在时间对齐前需将 MCD 合并为同样的形态。
    """

    sols, hours, lat_size, lon_size = x.shape
    return x.reshape(sols * hours, lat_size, lon_size)


def clean_invalid(x, name):
    """将 NaN / inf / 极端填充值替换为 0，以保证训练稳定。"""

    x = np.array(x, dtype=np.float32)
    bad = ~np.isfinite(x) | (np.abs(x) > 1e10)
    if np.any(bad):
        print(f"{name}: cleaned {bad.sum()} invalid values")
        x[bad] = np.nan
    return np.nan_to_num(x, nan=0.0)


def unwrap_ls(ls_array):
    """
    将循环的 Ls 值转换为连续轴。

    示例：
    ... 350, 355, 2, 7 ...
    变为
    ... 350, 355, 362, 367 ...

    这样可避免在 360 -> 0 处的插值伪影。
    """

    ls_unwrapped = np.copy(ls_array)
    year_offset = 0
    for i in range(1, len(ls_unwrapped)):
        if ls_array[i] < ls_array[i - 1] - 180:
            year_offset += 360
        ls_unwrapped[i] += year_offset
    return ls_unwrapped


def load_aligned_cube(base_dir):
    """
    加载臭氧及气象字段，并对齐至 OpenMars 时间轴。

    返回：
    - x_raw: [T, H, W, 5]
      通道顺序: [O3, U, V, Temperature, Solar_Flux]
    - y_raw: [T, H, W]
      目标臭氧场
    - om_ls_continuous: [T]
      与 x_raw / y_raw 对齐的连续 Ls 值
    """

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
    short_names = ["u", "v", "temp", "fluxsurf_dn_sw"]
    mcd_data_list = {key: [] for key in short_names}
    mcd_ls_list = []

    for file_path in target_files:
        if not os.path.exists(file_path):
            continue

        print(f"Loading: {os.path.basename(file_path)}")
        ds = nc.Dataset(file_path)

        # Keep the meteorological variables separate first, then stack them only after
        # they have all been cleaned and interpolated to the same time base.
        mcd_data_list["u"].append(merge_sol_hour(ds.variables["U_Wind"][:]))
        mcd_data_list["v"].append(merge_sol_hour(ds.variables["V_Wind"][:]))
        mcd_data_list["temp"].append(merge_sol_hour(ds.variables["Temperature"][:]))
        mcd_data_list["fluxsurf_dn_sw"].append(merge_sol_hour(ds.variables["Solar_Flux_DN"][:]))

        ls_tmp = ds.variables["Ls"][:] if "Ls" in ds.variables else ds.variables["ls"][:]
        sols, hours = ds.variables["U_Wind"].shape[:2]
        if ls_tmp.ndim == 1 and len(ls_tmp) == sols:
            # When MCD stores one Ls per sol, expand it to hourly resolution so it can
            # be interpolated against the OpenMars time axis.
            ls_expanded = np.zeros(sols * hours)
            for idx in range(sols):
                ls_start = ls_tmp[idx]
                if idx < sols - 1:
                    ls_end = ls_tmp[idx + 1]
                    if ls_end < ls_start:
                        ls_end += 360.0
                else:
                    ls_end = ls_start + (ls_tmp[1] - ls_tmp[0] if sols > 1 else 0.5)
                ls_expanded[idx * hours:(idx + 1) * hours] = np.linspace(ls_start, ls_end, hours, endpoint=False)
            mcd_ls_list.append(ls_expanded % 360.0)
        else:
            mcd_ls_list.append(ls_tmp.flatten())
        ds.close()

    vars_dict = {key: clean_invalid(np.concatenate(value, axis=0), key) for key, value in mcd_data_list.items()}
    if "fluxsurf_dn_sw" in vars_dict:
        # Flux is normalized to stabilize optimization and keep it comparable with
        # the other standardized inputs later.
        vars_dict["fluxsurf_dn_sw"] /= (np.max(vars_dict["fluxsurf_dn_sw"]) + 1e-6)

    print("\n[Step 3] Aligning MCD to OpenMars time axis...")
    om_ls_continuous = unwrap_ls(om_ls_raw)
    mcd_ls_continuous = unwrap_ls(np.concatenate(mcd_ls_list, axis=0))
    for key in vars_dict:
        # Interpolate each meteorological field to the exact OpenMars timestamps so the
        # model sees synchronized ozone / meteorology pairs.
        interpolator = interp1d(
            mcd_ls_continuous,
            vars_dict[key],
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        vars_dict[key] = interpolator(om_ls_continuous)

    x_raw = np.stack(
        [y_raw, vars_dict["u"], vars_dict["v"], vars_dict["temp"], vars_dict["fluxsurf_dn_sw"]],
        axis=-1,
    )
    return x_raw, y_raw, om_ls_continuous


def build_grid_dataloaders(x_raw, y_raw, ls_raw, window, horizon, batch_size):
    """
    为严格的历史预测设置构建序列样本。

    输入序列：
    - x[t : t + window]
    - ls[t : t + window]

    目标序列：
    - y[t + window : t + window + horizon]

    重要说明：
    本脚本不向模型输入未来的臭氧或气象数据，
    以保持对通用时间序列骨干的对比更为纯粹。
    """

    time_steps, lat_size, lon_size, channels = x_raw.shape
    split_time_idx = int(0.8 * time_steps)

    x_train_raw = x_raw[:split_time_idx]
    y_train_raw = y_raw[:split_time_idx]

    x_scaled = np.zeros_like(x_raw)
    for channel_idx in range(channels):
        scaler = StandardScaler()
        # Fit only on the training period to avoid temporal leakage.
        scaler.fit(x_train_raw[..., channel_idx].reshape(split_time_idx, -1))
        x_scaled[..., channel_idx] = scaler.transform(x_raw[..., channel_idx].reshape(time_steps, -1)).reshape(
            time_steps, lat_size, lon_size
        )

    y_mean = y_train_raw.mean()
    y_std = y_train_raw.std()
    y_scaled = (y_raw - y_mean) / y_std

    x_seq = []
    y_seq = []
    ls_seq = []
    for idx in range(time_steps - window - horizon + 1):
        # Historical input only: this is the safe formulation for cross-backbone
        # ablation when we want the phase-warp module, not future covariates,
        # to be the main changing factor.
        x_seq.append(x_scaled[idx: idx + window])
        y_seq.append(y_scaled[idx + window: idx + window + horizon])
        ls_seq.append(ls_raw[idx: idx + window])

    # x_torch: [N, window, 5, H, W]
    # y_torch: [N, horizon, H, W]
    # ls_torch: [N, window]
    x_torch = torch.tensor(np.array(x_seq)).permute(0, 1, 4, 2, 3).float()
    y_torch = torch.tensor(np.array(y_seq)).float()
    ls_torch = torch.tensor(np.array(ls_seq)).float()

    split_sample_idx = max(0, min(split_time_idx - window - horizon + 1, len(x_torch)))
    train_dataset = TensorDataset(x_torch[:split_sample_idx], ls_torch[:split_sample_idx], y_torch[:split_sample_idx])
    test_dataset = TensorDataset(x_torch[split_sample_idx:], ls_torch[split_sample_idx:], y_torch[split_sample_idx:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, y_mean, y_std


class MovingAverage(nn.Module):
    """DLinear风格分解所使用的移动平均块。"""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        if self.kernel_size <= 1:
            return x

        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    """将序列拆分为趋势和季节性残差部分。"""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend

class LinearLayer(nn.Module):
    """可调隐藏层数的全连接块。"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_hidden_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

class DLinearBackbone(nn.Module):
    """
    极简 DLinear 风格骨干网络。

    输入 / 输出形状:
    - input : [N_like, seq_len, feature_dim]
    - output: [N_like, pred_len, feature_dim]

    在这里，时间映射 seq_len -> pred_len 是跨特征通道共享的，
    这保持了骨干网络的简单性，并使消融实验的重点集中在相位扭曲
    前端上，而不是骨干网络的复杂性上。
    """

    def __init__(self, seq_len, pred_len, feature_dim, linear_hidden_layers=2):
        super().__init__()
        kernel_size = seq_len if seq_len % 2 == 1 else max(1, seq_len - 1)
        self.decomposition = SeriesDecomposition(max(1, kernel_size))
        # 使用可调隐藏层数的 LinearLayer
        hidden_dim = feature_dim * 2
        self.linear_seasonal = LinearLayer(seq_len, hidden_dim, pred_len, num_hidden_layers=linear_hidden_layers)
        self.linear_trend = LinearLayer(seq_len, hidden_dim, pred_len, num_hidden_layers=linear_hidden_layers)
        self.feature_dim = feature_dim

    def forward(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.transpose(1, 2)
        trend_init = trend_init.transpose(1, 2)
        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)
        return (seasonal_output + trend_output).transpose(1, 2)


class GridPointDLinearO3(nn.Module):
    """
    将共享的DLinear模型独立应用于每个网格点。

    为什么存在这个包装器：
    - 原始火星臭氧数据是网格化的：[B, T, C, H, W]
    - DLinear期望普通的多元序列：[B_like, T, C]

    该包装器将全局场重塑为许多逐点序列，在它们上面运行
    相同的骨干网络，然后将预测结果重塑回 [B, horizon, H, W]。
    """

    def __init__(self, seq_len, pred_len, lat_size, lon_size, use_phase_warp, linear_hidden_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.use_phase_warp = use_phase_warp
        self.linear_hidden_layers = linear_hidden_layers

        if use_phase_warp:
            # 这里没有传递空间形状，因此这变成了一个跨所有网格点共享参数的
            # 相位扭曲模块。这对于测试该想法是否在空间密集型 PredRNN 设置
            # 之外有效很有用。
            self.phase_warp = PhaseWarpFrontEnd()
            feature_dim = 9
        else:
            self.phase_warp = None
            feature_dim = 5

        self.backbone = DLinearBackbone(seq_len=seq_len, pred_len=pred_len, feature_dim=feature_dim, linear_hidden_layers=linear_hidden_layers)
        self.target_head = nn.Linear(feature_dim, 1)

    def forward(self, x, ls):
        batch_size, seq_len, channels, lat_size, lon_size = x.shape

        # 将全局网格转换为逐点序列：
        # [B, T, C, H, W] -> [B * H * W, T, C]
        x_point = x.permute(0, 3, 4, 1, 2).reshape(batch_size * lat_size * lon_size, seq_len, channels)

        # 把每个样本的历史Ls序列广播到每个网格点，以便
        # 相位扭曲模块可以一致地调制逐点序列。
        ls_point = ls[:, None, None, :].expand(batch_size, lat_size, lon_size, seq_len).reshape(
            batch_size * lat_size * lon_size, seq_len
        )

        if self.phase_warp is not None:
            # 原始的5通道输入在相位扭曲后变成9个通道：
            # [O3_fused, U_sin, U_cos, V_sin, V_cos, T_sin, T_cos, F_sin, F_cos]
            features = self.phase_warp(x_point, ls_point)
        else:
            features = x_point

        pred_features = self.backbone(features)
        pred_o3 = self.target_head(pred_features).squeeze(-1)
        # 恢复原始网格布局，以便指标与目标网格场保持直接可比。
        return pred_o3.view(batch_size, lat_size, lon_size, self.pred_len).permute(0, 3, 1, 2)


def evaluate_metrics(model, loader, device, y_std, y_mean):
    """在反转目标标准化之后，以物理臭氧单位进行评估。"""

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
    y_pred_phys = preds * y_std + y_mean
    y_true_phys = trues * y_std + y_mean

    pred_flat = y_pred_phys.reshape(-1)
    true_flat = y_true_phys.reshape(-1)

    mse = np.mean((pred_flat - true_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_flat - true_flat))
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
    r2 = 1 - ss_res / ss_tot
    smape = np.mean(2.0 * np.abs(pred_flat - true_flat) / (np.abs(true_flat) + np.abs(pred_flat) + 1e-6))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "smape": smape,
    }


def train_and_evaluate(
    label,
    use_phase_warp,
    train_loader,
    test_loader,
    device,
    y_std,
    y_mean,
    lat_size,
    lon_size,
    window,
    horizon,
    epochs,
    learning_rate,
    base_dir,
):
    """
    训练一个实验配置并返回其评估指标。

    实验之间唯一预期的开关是 `use_phase_warp`，因此任何指标
    差距都更容易归因于前端模块，而不是骨干网络的更改。
    """

    print(f"\n[Experiment] {label}")
    model = GridPointDLinearO3(
        seq_len=window,
        pred_len=horizon,
        lat_size=lat_size,
        lon_size=lon_size,
        use_phase_warp=use_phase_warp,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()
    checkpoint_path = os.path.join(base_dir, "models", "训练结果", f"{label.lower()}_checkpoint.pth")
    early_stopping = EarlyStopping(patience=early_stop_patience, verbose=True, path=checkpoint_path)

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
        print(f"{label} | Epoch {epoch_idx + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"{label} triggered early stopping.")
            break

    # 始终评估最佳的检查点，而不是最后一个epoch。
    model.load_state_dict(torch.load(checkpoint_path))
    metrics = evaluate_metrics(model, test_loader, device, y_std, y_mean)
    save_path = os.path.join(base_dir, "models", "训练结果", f"{label.lower()}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"{label} weights saved to: {save_path}")
    print(
        f"{label} Metrics | RMSE: {metrics['rmse']:.4f} | "
        f"MAE: {metrics['mae']:.4f} | R^2: {metrics['r2']:.4f} | SMAPE: {metrics['smape']:.2%}"
    )
    return metrics


# ---------------------------------------------------------------------------
# 主实验入口
# ---------------------------------------------------------------------------
# 该脚本有意在相同的数据集划分和超参数下连续运行两个实验：
# - DLinear_Raw
# - DLinear_PhaseWarp
#
# 如果第二个持续优于第一个，这就能证明
# 相位扭曲模块不仅仅绑定于PredRNN。
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(os.path.join(base_dir, "models", "训练过程"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "models", "训练结果"), exist_ok=True)
sys.stdout = Logger(os.path.join(base_dir, "models", "训练过程", "DLinear_PhaseWarp_Compare.txt"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training Device: {device}")

window = 3
horizon = 3
batch_size = 8
epochs = 50
learning_rate = 1e-3
# 早停耐心度（可调超参数）
early_stop_patience = 20
# Linear 隐藏层数（可调超参数）
linear_hidden_layers = 6

x_raw, y_raw, ls_raw = load_aligned_cube(base_dir)
lat_size, lon_size = y_raw.shape[1], y_raw.shape[2]
train_loader, test_loader, y_mean, y_std = build_grid_dataloaders(
    x_raw=x_raw,
    y_raw=y_raw,
    ls_raw=ls_raw,
    window=window,
    horizon=horizon,
    batch_size=batch_size,
)

raw_metrics = train_and_evaluate(
    label="DLinear_Raw",
    use_phase_warp=False,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    y_std=y_std,
    y_mean=y_mean,
    lat_size=lat_size,
    lon_size=lon_size,
    window=window,
    horizon=horizon,
    epochs=epochs,
    learning_rate=learning_rate,
    base_dir=base_dir,
)

# 相同的骨干网，相同的数据，相同的优化器设置；只有前端改变了。
phase_metrics = train_and_evaluate(
    label="DLinear_PhaseWarp",
    use_phase_warp=True,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    y_std=y_std,
    y_mean=y_mean,
    lat_size=lat_size,
    lon_size=lon_size,
    window=window,
    horizon=horizon,
    epochs=epochs,
    learning_rate=learning_rate,
    base_dir=base_dir,
)

print("\n[Comparison Summary]")
# 以下的正值代表相位扭曲变体在误差指标上表现更好，
# 而正的R^2增益意味着相位扭曲变体解释了更多的方差。
print(f"RMSE improvement: {raw_metrics['rmse'] - phase_metrics['rmse']:.4f}")
print(f"MAE improvement : {raw_metrics['mae'] - phase_metrics['mae']:.4f}")
print(f"R^2 gain        : {phase_metrics['r2'] - raw_metrics['r2']:.4f}")
print(f"SMAPE gain      : {raw_metrics['smape'] - phase_metrics['smape']:.2%}")
