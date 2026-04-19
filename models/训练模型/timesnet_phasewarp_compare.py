"""
这个脚本用于在火星臭氧预测任务上比较两组 TimesNet 风格实验：
1. TimesNet_Raw
   不使用相位扭曲模块，直接把历史原始变量送入 TimesNet 主干。
2. TimesNet_PhaseWarp
   先经过 PhaseWarpFrontEnd，再送入完全相同的 TimesNet 主干。

设计目标：
- 尽量保持与 dlinear_phasewarp_compare.py 相同的数据切分、标准化和评价口径；
- 采用“格点共享的点序列建模”方式，让 TimesNet 更符合它的一维时序建模习惯；
- 保留 TimesNet 的核心思想：频域选周期、按周期重排、再用二维卷积提取时序模式；
- 只改变是否插入 Phase Warp 前端，使性能差异更容易归因到你的创新模块。
"""

import glob
import os
import re
import sys

import netCDF4 as nc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    这里保持实现尽量朴素，目的是做“是否加入相位扭曲模块”的受控对比。
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
    """把 MCD 数据从 [sol, hour, lat, lon] 展平为 [time, lat, lon]。"""

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
    例如 359 -> 1 会被处理成 359 -> 361。
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
      通道顺序 [O3, U, V, Temperature, Solar_Flux]
    - y_raw: [T, H, W]
      目标臭氧场
    - om_ls_continuous: [T]
      与 x_raw / y_raw 对齐的连续 Ls
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
        mcd_data_list["u"].append(merge_sol_hour(ds.variables["U_Wind"][:]))
        mcd_data_list["v"].append(merge_sol_hour(ds.variables["V_Wind"][:]))
        mcd_data_list["temp"].append(merge_sol_hour(ds.variables["Temperature"][:]))
        mcd_data_list["fluxsurf_dn_sw"].append(merge_sol_hour(ds.variables["Solar_Flux_DN"][:]))

        ls_tmp = ds.variables["Ls"][:] if "Ls" in ds.variables else ds.variables["ls"][:]
        sols, hours = ds.variables["U_Wind"].shape[:2]
        if ls_tmp.ndim == 1 and len(ls_tmp) == sols:
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
    """

    time_steps, lat_size, lon_size, channels = x_raw.shape
    split_time_idx = int(0.8 * time_steps)

    x_train_raw = x_raw[:split_time_idx]
    y_train_raw = y_raw[:split_time_idx]

    x_scaled = np.zeros_like(x_raw)
    for channel_idx in range(channels):
        scaler = StandardScaler()
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


class PositionalEmbedding(nn.Module):
    """标准正弦位置编码，为时间步提供顺序信息。"""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class DataEmbedding(nn.Module):
    """
    TimesNet 的输入嵌入层。
    这里使用“变量值线性映射 + 位置编码”的轻量实现。
    """

    def __init__(self, c_in, d_model, dropout, max_len):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class InceptionBlockV1(nn.Module):
    """
    TimesNet 中常用的 Inception 风格卷积块。
    在不同卷积核尺度上提取模式，再做平均融合。
    """

    def __init__(self, in_channels, out_channels, num_kernels=4):
        super().__init__()
        kernel_size_list = [1, 3, 5, 7][:num_kernels]
        self.kernels = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
                for kernel_size in kernel_size_list
            ]
        )

    def forward(self, x):
        outputs = [kernel(x) for kernel in self.kernels]
        return torch.stack(outputs, dim=-1).mean(dim=-1)


def fft_for_period(x, top_k):
    """
    利用 FFT 找到当前 batch 中最显著的若干周期。

    输入：
    - x: [B_like, T, d_model]

    返回：
    - period_list: [k]，每个候选周期的长度
    - period_weight: [B_like, k]，每个样本对这些周期的相对偏好
    """

    # 这里把 FFT 周期搜索固定放到 CPU 上做。
    # 原因不是算法需要，而是部分 Windows + CUDA + PyTorch 组合下，
    # complex 张量在 GPU 上做 rfft / abs 会触发 nvrtc 的架构编译报错：
    # "invalid value for --gpu-architecture (-arch)"。
    #
    # 对这个项目来说，window 很短，top_k 也很小，
    # 所以把“挑周期”这一步搬到 CPU，代价非常低，但能明显提升兼容性。
    x_fft = x.detach().to(device="cpu", dtype=torch.float32)
    xf = torch.fft.rfft(x_fft, dim=1)
    frequency_list = xf.abs().mean(0).mean(-1)
    if frequency_list.numel() > 0:
        frequency_list[0] = 0

    max_candidates = max(1, frequency_list.shape[0] - 1)
    top_k = min(top_k, max_candidates)
    top_list = torch.topk(frequency_list, top_k).indices
    top_list = top_list[top_list > 0]

    if top_list.numel() == 0:
        top_list = torch.tensor([1], device=x.device)

    period_list = torch.clamp(torch.div(x.shape[1], top_list, rounding_mode="floor"), min=1)
    period_weight = xf.abs().mean(-1)[:, top_list]

    # 后续卷积和加权融合还在原来的设备上进行，所以把结果搬回去。
    period_list = period_list.to(x.device)
    period_weight = period_weight.to(device=x.device, dtype=x.dtype)
    return period_list, period_weight


class TimesBlock(nn.Module):
    """
    TimesNet 的核心块：
    1. 先在频域中挑选最显著的若干周期；
    2. 对每个周期把时间轴重排成二维结构；
    3. 用二维卷积提取局部模式；
    4. 按周期权重做自适应融合。
    """

    def __init__(self, seq_len, pred_len, d_model, d_ff, top_k, num_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        batch_size, total_len, d_model = x.shape
        period_list, period_weight = fft_for_period(x, self.top_k)

        results = []
        for period in period_list:
            period = max(1, int(period.item()))

            if total_len % period != 0:
                pad_len = period - (total_len % period)
                padded = torch.cat([x, x[:, -1:, :].repeat(1, pad_len, 1)], dim=1)
            else:
                padded = x

            padded_len = padded.shape[1]

            # [B, T, C] -> [B, C, block_num, period]
            out = padded.reshape(batch_size, padded_len // period, period, d_model).permute(0, 3, 1, 2).contiguous()

            out = self.conv(out)

            # 再还原回一维时间序列。
            out = out.permute(0, 2, 3, 1).reshape(batch_size, padded_len, d_model)
            results.append(out[:, :total_len, :])

        results = torch.stack(results, dim=-1)
        period_weight = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1)
        period_weight = period_weight.repeat(1, total_len, d_model, 1)
        result = torch.sum(results * period_weight, dim=-1)
        return result + x


class TimesNetBackbone(nn.Module):
    """
    一个简化但结构上贴近 TimesNet 的主干。

    流程：
    1. 对输入做逐样本归一化；
    2. 用线性层把历史长度 seq_len 扩展到 seq_len + pred_len；
    3. 做 embedding；
    4. 叠加多个 TimesBlock；
    5. 投影回变量维度，再取最后 pred_len 个时间步作为预测。
    """

    def __init__(self, seq_len, pred_len, c_in, d_model, e_layers, d_ff, top_k, num_kernels, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.predict_linear = nn.Linear(seq_len, seq_len + pred_len)
        self.embedding = DataEmbedding(
            c_in=c_in,
            d_model=d_model,
            dropout=dropout,
            max_len=seq_len + pred_len + 8,
        )
        self.model = nn.ModuleList(
            [
                TimesBlock(
                    seq_len=seq_len,
                    pred_len=pred_len,
                    d_model=d_model,
                    d_ff=d_ff,
                    top_k=top_k,
                    num_kernels=num_kernels,
                )
                for _ in range(e_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, c_in)

    def forward(self, x):
        # x: [B_like, seq_len, n_vars]
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # predict_linear 是沿时间维工作的，所以先转成 [B_like, n_vars, seq_len]。
        x = self.predict_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.embedding(x)

        for block in self.model:
            x = self.layer_norm(block(x))

        x = self.projection(x)

        total_len = self.seq_len + self.pred_len
        x = x * stdev[:, 0, :].unsqueeze(1).repeat(1, total_len, 1)
        x = x + means[:, 0, :].unsqueeze(1).repeat(1, total_len, 1)
        return x[:, -self.pred_len:, :]


class GridPointTimesNetO3(nn.Module):
    """
    把每个经纬度格点看成一条多变量时间序列，并共享同一个 TimesNet 模型。

    输入：
    - x: [B, T, C, H, W]
    - ls: [B, T]

    中间：
    - reshape 为 [B * H * W, T, C]
    - 可选经过 PhaseWarpFrontEnd
    - TimesNet 主干预测未来每个特征的轨迹
    - 用 target_head 把未来特征压成未来 O3

    输出：
    - [B, horizon, H, W]
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        lat_size,
        lon_size,
        use_phase_warp,
        d_model,
        e_layers,
        d_ff,
        top_k,
        num_kernels,
        dropout,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.use_phase_warp = use_phase_warp

        if use_phase_warp:
            # 点序列模型这里使用共享参数版相位扭曲模块。
            self.phase_warp = PhaseWarpFrontEnd()
            n_vars = 9
        else:
            self.phase_warp = None
            n_vars = 5

        self.backbone = TimesNetBackbone(
            seq_len=seq_len,
            pred_len=pred_len,
            c_in=n_vars,
            d_model=d_model,
            e_layers=e_layers,
            d_ff=d_ff,
            top_k=top_k,
            num_kernels=num_kernels,
            dropout=dropout,
        )
        self.target_head = nn.Linear(n_vars, 1)

    def forward(self, x, ls):
        batch_size, seq_len, channels, lat_size, lon_size = x.shape

        # [B, T, C, H, W] -> [B * H * W, T, C]
        x_point = x.permute(0, 3, 4, 1, 2).reshape(batch_size * lat_size * lon_size, seq_len, channels)
        ls_point = ls[:, None, None, :].expand(batch_size, lat_size, lon_size, seq_len).reshape(
            batch_size * lat_size * lon_size, seq_len
        )

        if self.phase_warp is not None:
            features = self.phase_warp(x_point, ls_point)
        else:
            features = x_point

        pred_features = self.backbone(features)
        pred_o3 = self.target_head(pred_features).squeeze(-1)
        return pred_o3.view(batch_size, lat_size, lon_size, self.pred_len).permute(0, 3, 1, 2)


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
    window,
    horizon,
    d_model,
    e_layers,
    d_ff,
    top_k,
    num_kernels,
    dropout,
    epochs,
    learning_rate,
    early_stopping_patience,
    base_dir,
):
    """
    训练一组 TimesNet 实验，并返回评估指标。
    受控变量只有一个：是否启用 phase-warp 前端。
    """

    print(f"\n[Experiment] {label}")
    model = GridPointTimesNetO3(
        seq_len=window,
        pred_len=horizon,
        lat_size=lat_size,
        lon_size=lon_size,
        use_phase_warp=use_phase_warp,
        d_model=d_model,
        e_layers=e_layers,
        d_ff=d_ff,
        top_k=top_k,
        num_kernels=num_kernels,
        dropout=dropout,
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

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
    # 1. TimesNet_Raw
    # 2. TimesNet_PhaseWarp
    #
    # 两组实验共用同一套数据、超参数和评价指标，
    # 只有是否启用 PhaseWarpFrontEnd 不同。
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(os.path.join(base_dir, "models", "训练过程"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "models", "训练结果"), exist_ok=True)
    sys.stdout = Logger(os.path.join(base_dir, "models", "训练过程", "TimesNet_PhaseWarp_Compare.txt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Device: {device}")

    # -----------------------------------------------------------------------
    # 你后续最常调的超参数都集中放在这里
    # -----------------------------------------------------------------------
    window = 3
    horizon = 3
    batch_size = 4
    d_model = 64
    e_layers = 2
    d_ff = 128
    top_k = 3
    num_kernels = 4
    dropout = 0.1
    epochs = 20
    learning_rate = 1e-3
    early_stopping_patience = 5

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
        label="TimesNet_Raw",
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
        d_model=d_model,
        e_layers=e_layers,
        d_ff=d_ff,
        top_k=top_k,
        num_kernels=num_kernels,
        dropout=dropout,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        base_dir=base_dir,
    )

    phase_metrics = train_and_evaluate(
        label="TimesNet_PhaseWarp",
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
        d_model=d_model,
        e_layers=e_layers,
        d_ff=d_ff,
        top_k=top_k,
        num_kernels=num_kernels,
        dropout=dropout,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        base_dir=base_dir,
    )

    print("\n[Comparison Summary]")
    print(f"RMSE improvement: {raw_metrics['rmse'] - phase_metrics['rmse']:.4f}")
    print(f"MAE improvement : {raw_metrics['mae'] - phase_metrics['mae']:.4f}")
    print(f"R^2 gain        : {phase_metrics['r2'] - raw_metrics['r2']:.4f}")
    print(f"SMAPE gain      : {raw_metrics['smape'] - phase_metrics['smape']:.2%}")


if __name__ == "__main__":
    main()
