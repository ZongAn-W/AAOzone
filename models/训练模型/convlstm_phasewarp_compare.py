"""
这个脚本用于在火星臭氧预测任务上比较两组 ConvLSTM 实验：

1. ConvLSTM_Raw
   不使用相位扭曲模块，直接把历史原始变量送入 ConvLSTM 主干。
2. ConvLSTM_PhaseWarp
   先经过 PhaseWarpFrontEnd，再送入完全相同的 ConvLSTM 主干。

设计目标：
- 尽量保持与 dlinear_phasewarp_compare.py 相同的数据切分、标准化和评价口径；
- 只改变是否插入 Phase Warp 前端，使精度差异更容易归因到你的创新模块；
- 用空间模型 ConvLSTM 再验证一次：相位扭曲的收益是否不依赖于 PredRNN。
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

from phase_warp_frontend import PhaseWarpFrontEnd


class Logger(object):
    """把终端输出同步写入日志文件，方便复现实验过程。"""

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
    基于验证集损失的简单早停器。

    这里保持实现尽量朴素，因为本脚本的主要目的不是追求训练技巧，
    而是做“是否加入相位扭曲模块”的受控对比。
    """

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
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def natural_sort_key(s):
    """按人类习惯排序文件名，例如 2 会排在 10 前面。"""

    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", s)]


def merge_sol_hour(x):
    """
    把 MCD 数据从 [sol, hour, lat, lon] 展平成 [time, lat, lon]。

    OpenMars 已经是单时间轴表示，因此 MCD 需要先整理成相同的时间组织方式，
    后面才能和 OpenMars 做时间对齐。
    """

    sols, hours, lat_size, lon_size = x.shape
    return x.reshape(sols * hours, lat_size, lon_size)


def clean_invalid(x, name):
    """把 NaN / inf / 异常填充值替换成 0，避免训练阶段数值不稳定。"""

    x = np.array(x, dtype=np.float32)
    bad = ~np.isfinite(x) | (np.abs(x) > 1e10)
    if np.any(bad):
        print(f"{name}: cleaned {bad.sum()} invalid values")
        x[bad] = np.nan
    return np.nan_to_num(x, nan=0.0)


def unwrap_ls(ls_array):
    """
    将循环的 Ls 序列展开成连续轴，用于后续插值。

    例如：
    350, 355, 2, 7
    会被展开成：
    350, 355, 362, 367

    这样可以避免在 360 -> 0 的跳变处产生错误插值。
    """

    ls_unwrapped = np.copy(ls_array)
    year_offset = 0
    for idx in range(1, len(ls_unwrapped)):
        if ls_array[idx] < ls_array[idx - 1] - 180:
            year_offset += 360
        ls_unwrapped[idx] += year_offset
    return ls_unwrapped


def load_aligned_cube(base_dir):
    """
    读取臭氧和气象变量，并将它们统一对齐到 OpenMars 的时间轴上。

    返回：
    - x_raw: [T, H, W, 5]
      通道顺序为 [O3, U, V, Temperature, Solar_Flux]
    - y_raw: [T, H, W]
      目标臭氧场
    - om_ls_continuous: [T]
      与 x_raw / y_raw 一一对齐的连续 Ls 序列
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

        # 先分别收集各个气象变量，等到清洗并插值完成后再统一堆叠成输入张量。
        mcd_data_list["u"].append(merge_sol_hour(ds.variables["U_Wind"][:]))
        mcd_data_list["v"].append(merge_sol_hour(ds.variables["V_Wind"][:]))
        mcd_data_list["temp"].append(merge_sol_hour(ds.variables["Temperature"][:]))
        mcd_data_list["fluxsurf_dn_sw"].append(merge_sol_hour(ds.variables["Solar_Flux_DN"][:]))

        ls_tmp = ds.variables["Ls"][:] if "Ls" in ds.variables else ds.variables["ls"][:]
        sols, hours = ds.variables["U_Wind"].shape[:2]
        if ls_tmp.ndim == 1 and len(ls_tmp) == sols:
            # 某些 MCD 文件只给出逐 sol 的 Ls，这里把它扩展到逐 hour 分辨率，
            # 这样后面才能和 OpenMars 的时间轴精细对齐。
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
        # 对太阳通量做一个简单归一化，避免量纲过大影响优化稳定性。
        vars_dict["fluxsurf_dn_sw"] /= (np.max(vars_dict["fluxsurf_dn_sw"]) + 1e-6)

    print("\n[Step 3] Aligning MCD to OpenMars time axis...")
    om_ls_continuous = unwrap_ls(om_ls_raw)
    mcd_ls_continuous = unwrap_ls(np.concatenate(mcd_ls_list, axis=0))
    for key in vars_dict:
        # 把每个气象变量都插值到 OpenMars 的时间点上，保证臭氧和气象是一一对应的。
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
    在“严格历史输入”的设定下构造样本序列。

    输入：
    - x[t : t + window]
    - ls[t : t + window]

    预测目标：
    - y[t + window : t + window + horizon]

    这里刻意不喂未来臭氧，也不喂未来气象变量，
    这样更适合做“相位扭曲模块是否跨 backbone 有效”的公平对比。
    """

    time_steps, lat_size, lon_size, channels = x_raw.shape
    split_time_idx = int(0.8 * time_steps)

    x_train_raw = x_raw[:split_time_idx]
    y_train_raw = y_raw[:split_time_idx]

    x_scaled = np.zeros_like(x_raw)
    for channel_idx in range(channels):
        scaler = StandardScaler()
        # 只用训练时间段拟合标准化器，避免时间泄露。
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
        # 历史窗口输入，对应未来 horizon 步臭氧预测。
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


class ConvLSTMCell(nn.Module):
    """
    单个 ConvLSTM 单元，对一个时间步的空间网格进行更新。

    输入：
    - x: 当前时刻输入特征图 [B, C_in, H, W]
    - h_cur: 上一时刻隐藏状态 [B, C_h, H, W]
    - c_cur: 上一时刻记忆状态 [B, C_h, H, W]
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        i_gate, f_gate, o_gate, g_gate = torch.chunk(gates, 4, dim=1)

        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)

        c_next = f_gate * c_cur + i_gate * g_gate
        h_next = o_gate * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMForecaster(nn.Module):
    """
    一个紧凑版的 ConvLSTM 臭氧预测主干。

    模型流程：
    1. 对历史时序网格做编码；
    2. 通过多层 ConvLSTM 提取时空隐藏状态；
    3. 用最后一层最后时刻的隐藏状态，通过 1x1 卷积头直接映射出 pred_len 张未来臭氧图。

    这里没有像 PredRNN 一样做自回归解码，而是刻意保持结构紧凑，
    使“是否加入 Phase Warp”成为主要变量，而不是复杂解码策略本身。
    """

    def __init__(self, pred_len, lat_size, lon_size, use_phase_warp, hidden_dims, kernel_size=3):
        super().__init__()
        self.pred_len = pred_len
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.use_phase_warp = use_phase_warp
        self.hidden_dims = list(hidden_dims)

        if use_phase_warp:
            # ConvLSTM 仍然工作在网格空间上，因此这里传入 spatial_shape，
            # 让相位扭曲参数可以保留逐格点的空间自由度。
            self.phase_warp = PhaseWarpFrontEnd(spatial_shape=(lat_size, lon_size))
            input_dim = 9
        else:
            self.phase_warp = None
            input_dim = 5

        self.cells = nn.ModuleList()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            cur_input_dim = input_dim if idx == 0 else self.hidden_dims[idx - 1]
            self.cells.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size=kernel_size))

        self.forecast_head = nn.Conv2d(self.hidden_dims[-1], pred_len, kernel_size=1)

    def _init_states(self, batch_size, device):
        """为每一层 ConvLSTM 初始化隐藏状态和记忆状态。"""

        states = []
        for hidden_dim in self.hidden_dims:
            h_state = torch.zeros(batch_size, hidden_dim, self.lat_size, self.lon_size, device=device)
            c_state = torch.zeros_like(h_state)
            states.append([h_state, c_state])
        return states

    def forward(self, x, ls):
        # x: [B, T, 5, H, W]
        # ls: [B, T]
        if self.phase_warp is not None:
            # 经过相位扭曲后，5 通道输入会被扩展为 9 通道：
            # [O3_fused, U_sin, U_cos, V_sin, V_cos, T_sin, T_cos, F_sin, F_cos]
            features = self.phase_warp(x, ls)
        else:
            features = x

        # features: [B, T, C, H, W]
        batch_size, seq_len, _, _, _ = features.shape
        states = self._init_states(batch_size, features.device)

        for t in range(seq_len):
            # current: [B, C, H, W]
            current = features[:, t]
            for layer_idx, cell in enumerate(self.cells):
                h_cur, c_cur = states[layer_idx]
                h_next, c_next = cell(current, h_cur, c_cur)
                states[layer_idx] = [h_next, c_next]
                current = h_next

        # 取最后一层最后时刻的隐藏状态，直接映射成未来 horizon 张臭氧图。
        last_hidden = states[-1][0]
        return self.forecast_head(last_hidden)


def evaluate_metrics(model, loader, device, y_std, y_mean):
    """把标准化空间里的预测还原到物理臭氧单位后再计算指标。"""

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
    horizon,
    hidden_dims,
    epochs,
    learning_rate,
    early_stopping_patience,
    base_dir,
):
    """
    训练一组 ConvLSTM 实验，并返回评估指标。

    受控变量只有一个：是否启用 phase-warp 前端。
    因此两组实验之间的性能差异，更容易解释为模块收益而非 backbone 变化。
    """

    print(f"\n[Experiment] {label}")
    model = ConvLSTMForecaster(
        pred_len=horizon,
        lat_size=lat_size,
        lon_size=lon_size,
        use_phase_warp=use_phase_warp,
        hidden_dims=hidden_dims,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()
    checkpoint_path = os.path.join(base_dir, "models", "训练结果", f"{label.lower()}_checkpoint.pth")
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
        print(f"{label} | Epoch {epoch_idx + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"{label} triggered early stopping.")
            break

    # 评估时始终加载验证集表现最好的 checkpoint，而不是最后一个 epoch。
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


def main():
    # -----------------------------------------------------------------------
    # 主实验入口
    # -----------------------------------------------------------------------
    # 这里顺序运行两组实验：
    # 1. ConvLSTM_Raw
    # 2. ConvLSTM_PhaseWarp
    #
    # 两组实验共用：
    # - 同一份数据
    # - 同一套时间切分
    # - 同一套超参数
    # - 同一套评价指标
    #
    # 这样最终差值就可以更直接地解释为：
    # “在 ConvLSTM backbone 上，引入 Phase Warp 是否带来提升”
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(os.path.join(base_dir, "models", "训练过程"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "models", "训练结果"), exist_ok=True)
    sys.stdout = Logger(os.path.join(base_dir, "models", "训练过程", "ConvLSTM_PhaseWarp_Compare.txt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Device: {device}")

    window = 3
    horizon = 3
    batch_size = 4
    hidden_dim = 32
    num_layers = 2
    hidden_dims = [hidden_dim] * num_layers
    epochs = 15





    learning_rate = 1e-3
    early_stopping_patience = 5

    # 读取并对齐原始数据，得到统一时间轴下的臭氧 / 气象 / Ls。
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
        label="ConvLSTM_Raw",
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
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        base_dir=base_dir,
    )

    # 同一个 ConvLSTM 主干，只把前端替换为 PhaseWarpFrontEnd。
    phase_metrics = train_and_evaluate(
        label="ConvLSTM_PhaseWarp",
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
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        base_dir=base_dir,
    )

    print("\n[Comparison Summary]")
    # 下面这几行中：
    # - 误差类指标 improvement > 0 代表 PhaseWarp 更好
    # - R^2 gain > 0 代表 PhaseWarp 解释了更多方差
    print(f"RMSE improvement: {raw_metrics['rmse'] - phase_metrics['rmse']:.4f}")
    print(f"MAE improvement : {raw_metrics['mae'] - phase_metrics['mae']:.4f}")
    print(f"R^2 gain        : {phase_metrics['r2'] - raw_metrics['r2']:.4f}")
    print(f"SMAPE gain      : {raw_metrics['smape'] - phase_metrics['smape']:.2%}")


if __name__ == "__main__":
    main()
