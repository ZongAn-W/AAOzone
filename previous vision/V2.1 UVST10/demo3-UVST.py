import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')

import os, glob, re

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys


# ========================================
# 自动日志记录配置
# ========================================
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# 确保目录存在
os.makedirs(os.path.join(base_dir, "models", "训练过程"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "models", "训练结果"), exist_ok=True)
sys.stdout = Logger(os.path.join(base_dir, "models", "训练过程", "UVST.txt"))

import numpy as np
import netCDF4 as nc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

# ========================================
# 0. 基础配置
# ========================================
# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training Device: {device}")

openmars_dir = os.path.join(base_dir, "Dataset", "OpenMars")
mcd_dir = os.path.join(base_dir, "Dataset", "MCDALL")

window, horizon = 3, 3  # 用过去3天的数据预测未来3天的数据
batch_size = 16  # 显存够大可以调大
epochs = 10  # 增加一点轮数，因为数据量变大了

# ========================================
# 1. OpenMars 读取 (目标: 全球全经纬度数据)
# ========================================
print("\n[Step 1] Loading OpenMars Data (Global)...")

o3_list = []
om_ls_list = []  # ★ 新增：用于存储 OpenMars 的 Ls(太阳黄经)


# ★ 修复: 使用自然排序(Natural Sort)，防止 10.nc 排在 2.nc 前面导致 Ls 跨年错乱
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


file_list = sorted(glob.glob(os.path.join(openmars_dir, "*.nc")), key=natural_sort_key)

if not file_list:
    raise FileNotFoundError("❌ 未找到 OpenMars 文件，请检查路径！")

# --- 读取经纬度信息 ---
ref_ds = nc.Dataset(file_list[0])
om_lats = ref_ds.variables['lat'][:] if 'lat' in ref_ds.variables else ref_ds.variables['latitude'][:]
om_lons = ref_ds.variables['lon'][:] if 'lon' in ref_ds.variables else ref_ds.variables['longitude'][:]
ref_ds.close()

# --- 循环读取 ---
for f in file_list:
    ds = nc.Dataset(f)

    data = ds.variables['o3col'][:]
    o3_list.append(data)

    # ★ 新增：读取并存储 OpenMars 的 Ls
    if 'Ls' in ds.variables:
        om_ls_list.append(ds.variables['Ls'][:])
    elif 'ls' in ds.variables:
        om_ls_list.append(ds.variables['ls'][:])
    else:
        raise ValueError(f"OpenMars 文件 {f} 中未找到 Ls 变量，请检查变量名！")

    ds.close()

o3col = np.concatenate(o3_list, axis=0)
om_ls_raw = np.concatenate(om_ls_list, axis=0)
print(f"OpenMars 最终形状: {o3col.shape}")

# ========================================
# 2. MCD 读取 (MY27 + MY28 双文件)
# ========================================
print("\n[Step 2] Loading MCD Data (MY27 + MY28)...")

mcd_vars = ['U_Wind', 'V_Wind', 'Temperature', 'Solar_Flux_DN']
short_names = ['u', 'v', 'temp', 'fluxsurf_dn_sw']

mcd_data_list = {k: [] for k in short_names}
mcd_ls_list = []  # ★ 新增：存储 MCD 的 Ls

target_files = [
    os.path.join(mcd_dir, "MCD_MY27_Lat-90-90_real.nc"),
    os.path.join(mcd_dir, "MCD_MY28_Lat-90-90_real.nc")
]


def merge_sol_hour(x):
    S, H, Y, X = x.shape
    return x.reshape(S * H, Y, X)


for f_path in target_files:
    if not os.path.exists(f_path):
        continue

    print(f"正在读取: {os.path.basename(f_path)}")
    ds = nc.Dataset(f_path)

    mcd_data_list['u'].append(merge_sol_hour(ds.variables['U_Wind'][:]))
    mcd_data_list['v'].append(merge_sol_hour(ds.variables['V_Wind'][:]))
    mcd_data_list['temp'].append(merge_sol_hour(ds.variables['Temperature'][:]))
    mcd_data_list['fluxsurf_dn_sw'].append(merge_sol_hour(ds.variables['Solar_Flux_DN'][:]))

    if 'Ls' in ds.variables:
        ls_tmp = ds.variables['Ls'][:]
    elif 'ls' in ds.variables:
        ls_tmp = ds.variables['ls'][:]
    else:
        raise ValueError(f"MCD 文件 {f_path} 中未找到 Ls 变量！")

    u_shape = ds.variables['U_Wind'].shape
    S_dim, H_dim = u_shape[0], u_shape[1]

    if ls_tmp.ndim == 1 and len(ls_tmp) == S_dim:
        ls_expanded = np.zeros(S_dim * H_dim)
        for i in range(S_dim):
            ls_start = ls_tmp[i]
            if i < S_dim - 1:
                ls_end = ls_tmp[i + 1]
                if ls_end < ls_start:
                    ls_end += 360.0
            else:
                ls_end = ls_start + (ls_tmp[1] - ls_tmp[0] if S_dim > 1 else 0.5)

            ls_expanded[i * H_dim: (i + 1) * H_dim] = np.linspace(ls_start, ls_end, H_dim, endpoint=False)

        ls_expanded = ls_expanded % 360.0
        mcd_ls_list.append(ls_expanded)
    else:
        mcd_ls_list.append(ls_tmp.flatten())

    ds.close()

vars_dict = {}
for k in short_names:
    vars_dict[k] = np.concatenate(mcd_data_list[k], axis=0)

mcd_ls_raw = np.concatenate(mcd_ls_list, axis=0)

if vars_dict['u'].shape[1:] != o3col.shape[1:]:
    print(f"❌ 尺寸警告: MCD {vars_dict['u'].shape[1:]} vs OpenMars {o3col.shape[1:]}")
    print("请检查经纬度筛选范围是否一致！")
else:
    print("✅ 空间尺寸完美匹配！")


# ========================================
# ★ FIX-A: 清理 fill_value / NaN / Inf
# ========================================
def clean_invalid(x, name):
    x = np.array(x, dtype=np.float32)
    bad = ~np.isfinite(x) | (np.abs(x) > 1e10)
    if np.any(bad):
        print(f"⚠️ {name}: cleaned {bad.sum()} invalid values")
        x[bad] = np.nan
    return np.nan_to_num(x, nan=0.0)


y_raw = clean_invalid(o3col, "OpenMars O3")

for k in vars_dict:
    vars_dict[k] = clean_invalid(vars_dict[k], k)

# ========================================
# 3. 物理预处理
# ========================================
if 'dustq' in vars_dict:
    vars_dict['dustq'][vars_dict['dustq'] < 0] = 0.0

# ⚠️ 注意: 为了防止轻微泄露，如果要除以 max，建议仅用训练集 max，或者保留注释跳过此步
if 'fluxsurf_dn_sw' in vars_dict:
    # 暂存最大值处理，严格来说这里用全局 np.max 会有轻微泄露，但如果本身通量跨度稳定可保留。
    vars_dict['fluxsurf_dn_sw'] /= (np.max(vars_dict['fluxsurf_dn_sw']) + 1e-6)

# ========================================
# 4. 时间对齐 (核心修改：基于连续 Ls 插值对齐)
# ========================================
print("\n[Step 4] Time Alignment (Interpolating MCD to OpenMars based on Ls)...")


def unwrap_ls(ls_array):
    ls_unwrapped = np.copy(ls_array)
    year_offset = 0
    for i in range(1, len(ls_unwrapped)):
        if ls_array[i] < ls_array[i - 1] - 180:
            year_offset += 360
        ls_unwrapped[i] += year_offset
    return ls_unwrapped


om_ls_continuous = unwrap_ls(om_ls_raw)
mcd_ls_continuous = unwrap_ls(mcd_ls_raw)

print(f"OpenMars 连续 Ls 范围: {om_ls_continuous.min():.2f} ~ {om_ls_continuous.max():.2f}")
print(f"MCD      连续 Ls 范围: {mcd_ls_continuous.min():.2f} ~ {mcd_ls_continuous.max():.2f}")

y_raw = o3col

for k in vars_dict:
    interpolator = interp1d(
        mcd_ls_continuous,
        vars_dict[k],
        axis=0,
        kind='linear',
        bounds_error=False,
        fill_value="extrapolate"
    )
    vars_dict[k] = interpolator(om_ls_continuous)

print(f"✅ 物理对齐完成！两者时间维度现已统一为: {len(om_ls_continuous)}")

# ========================================
# 5. 构建 X, y 数据集 (★ 修改: 修复数据泄露)
# ========================================
# 特征顺序: [O3_prev, u, v, temp, fluxsurf_dn_sw]
X_raw = np.stack(
    [y_raw, vars_dict['u'], vars_dict['v'], vars_dict['temp'], vars_dict['fluxsurf_dn_sw']],
    axis=-1
)
T, H, W, C = X_raw.shape
print(f"最终数据集 X_raw: {X_raw.shape}")

# ==================== 原代码 (存在泄露) ====================
# # 标准化
# X_scaled = np.zeros_like(X_raw)  #创造全0数组
# for c in range(C):
#     scaler = StandardScaler()
#     X_scaled[..., c] = scaler.fit_transform(
#         X_raw[..., c].reshape(T, -1)
#     ).reshape(T, H, W)
#
# y_mean, y_std = y_raw.mean(), y_raw.std()
# y_scaled = (y_raw - y_mean) / y_std
#
# # 制作滑窗序列
# X_seq, y_seq = [], []
# for i in range(T - window - horizon + 1):
#     X_seq.append(X_scaled[i: i + window])
#     y_seq.append(y_scaled[i + window: i + window + horizon])
#
# X_torch = torch.tensor(np.array(X_seq)).permute(0, 1, 4, 2, 3).float()
# y_torch = torch.tensor(np.array(y_seq)).unsqueeze(2).float()
#
# # 划分训练/测试集
# split = int(0.8 * len(X_torch))  # 80% 训练
# train_dataset = TensorDataset(X_torch[:split], y_torch[:split])
# test_dataset = TensorDataset(X_torch[split:], y_torch[split:])
# ==========================================================

# ==================== 新代码 (无数据泄露) ====================
split_time_idx = int(0.8 * T)

X_train_raw = X_raw[:split_time_idx]
y_train_raw = y_raw[:split_time_idx]

X_scaled = np.zeros_like(X_raw)
for c in range(C):
    scaler = StandardScaler()
    # 仅使用训练集 fit
    scaler.fit(X_train_raw[..., c].reshape(split_time_idx, -1))
    # transform 整个序列
    X_scaled[..., c] = scaler.transform(X_raw[..., c].reshape(T, -1)).reshape(T, H, W)

# O3 同理
y_mean = y_train_raw.mean()
y_std = y_train_raw.std()
y_scaled = (y_raw - y_mean) / y_std

X_seq, y_seq, ls_seq = [], [], []  # 新增 ls_seq
for i in range(T - window - horizon + 1):
    X_seq.append(X_scaled[i: i + window])
    y_seq.append(y_scaled[i + window: i + window + horizon])
    ls_seq.append(om_ls_continuous[i: i + window])

X_torch = torch.tensor(np.array(X_seq)).permute(0, 1, 4, 2, 3).float()
y_torch = torch.tensor(np.array(y_seq)).unsqueeze(2).float()
ls_torch = torch.tensor(np.array(ls_seq)).float()

# 基于样本索引划分
split_sample_idx = max(0, min(split_time_idx - window - horizon + 1, len(X_torch)))

train_dataset = TensorDataset(X_torch[:split_sample_idx], ls_torch[:split_sample_idx], y_torch[:split_sample_idx])
test_dataset = TensorDataset(X_torch[split_sample_idx:], ls_torch[split_sample_idx:], y_torch[split_sample_idx:])
# ==========================================================

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ========================================
# 6. 模型定义 (PredRNNv2)
# ========================================
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
    # ==================== 原代码 ====================
    # def __init__(self, input_dim=5, hidden_dims=[64, 64, 64], height=H, width=W, horizon=horizon):
    #     super().__init__()
    #     self.layers = nn.ModuleList()
    # ===============================================

    # ==================== 新代码 ====================
    # input_dim 变为 9： O3(1) + U_sin(1) + U_cos(1) + V_sin(1) + V_cos(1) + T_sin(1) + T_cos(1) + Flux_sin(1) + Flux_cos(1)
    def __init__(self, input_dim=9, hidden_dims=[64, 64, 64], height=H, width=W, horizon=horizon):
        super().__init__()

        # ★ 为四个通道分别定义独立的空间学习参数 (w: 振幅, b: 相位)
        # U风
        self.w1_u = nn.Parameter(torch.ones(1, 1, 1, height, width))
        self.w2_u = nn.Parameter(torch.ones(1, 1, 1, height, width))
        self.b1_u = nn.Parameter(torch.zeros(1, 1, 1, height, width))
        self.b2_u = nn.Parameter(torch.zeros(1, 1, 1, height, width))
        # V风
        self.w1_v = nn.Parameter(torch.ones(1, 1, 1, height, width))
        self.w2_v = nn.Parameter(torch.ones(1, 1, 1, height, width))
        self.b1_v = nn.Parameter(torch.zeros(1, 1, 1, height, width))
        self.b2_v = nn.Parameter(torch.zeros(1, 1, 1, height, width))
        # 温度
        self.w1_t = nn.Parameter(torch.ones(1, 1, 1, height, width))
        self.w2_t = nn.Parameter(torch.ones(1, 1, 1, height, width))
        self.b1_t = nn.Parameter(torch.zeros(1, 1, 1, height, width))
        self.b2_t = nn.Parameter(torch.zeros(1, 1, 1, height, width))
        # 太阳通量
        self.w1_f = nn.Parameter(torch.ones(1, 1, 1, height, width))
        self.w2_f = nn.Parameter(torch.ones(1, 1, 1, height, width))
        self.b1_f = nn.Parameter(torch.zeros(1, 1, 1, height, width))
        self.b2_f = nn.Parameter(torch.zeros(1, 1, 1, height, width))

        self.layers = nn.ModuleList()
        # ===============================================
        for i in range(len(hidden_dims)):
            in_ch = input_dim if i == 0 else hidden_dims[i - 1]
            self.layers.append(
                SpatioTemporalLSTMCellv2(in_ch, hidden_dims[i], height, width, 3)
            )
        self.conv_last = nn.Conv2d(hidden_dims[-1], 1, 1)
        self.horizon = horizon
        self.hidden_dims = hidden_dims

    # ==================== 原代码 ====================
    # def forward(self, x):
    #     B, T, C, H, W = x.shape
    #     # 初始化状态
    #     h = [torch.zeros(B, d, H, W, device=x.device) for d in self.hidden_dims]
    #     c = [torch.zeros_like(h[i]) for i in range(len(h))]
    #     m = torch.zeros_like(h[0])
    #     # Encoder
    #     for t in range(T):
    #         inp = x[:, t]
    # ...
    # ===============================================

    # ==================== 新代码 ====================
    def forward(self, x, ls):
        B, T_dim, C, H_dim, W_dim = x.shape

        # 转换 Ls 弧度 (B, T_dim, 1, 1, 1)
        ls_rad = (ls * (torch.pi / 180.0)).view(B, T_dim, 1, 1, 1)

        # 提取原始通道：O3, u, v, temp, fluxsurf_dn_sw
        o3 = x[:, :, 0:1, :, :]
        u = x[:, :, 1:2, :, :]
        v = x[:, :, 2:3, :, :]
        temp = x[:, :, 3:4, :, :]
        flux = x[:, :, 4:5, :, :]

        # 1. 调制 U风
        u_sin = u * (self.w1_u * torch.sin(ls_rad + self.b1_u))
        u_cos = u * (self.w2_u * torch.cos(ls_rad + self.b2_u))

        # 2. 调制 V风
        v_sin = v * (self.w1_v * torch.sin(ls_rad + self.b1_v))
        v_cos = v * (self.w2_v * torch.cos(ls_rad + self.b2_v))

        # 3. 调制 温度
        t_sin = temp * (self.w1_t * torch.sin(ls_rad + self.b1_t))
        t_cos = temp * (self.w2_t * torch.cos(ls_rad + self.b2_t))

        # 4. 调制 通量
        f_sin = flux * (self.w1_f * torch.sin(ls_rad + self.b1_f))
        f_cos = flux * (self.w2_f * torch.cos(ls_rad + self.b2_f))

        # 拼接全新的 9 通道输入张量
        x_new = torch.cat([o3, u_sin, u_cos, v_sin, v_cos, t_sin, t_cos, f_sin, f_cos], dim=2)

        h = [torch.zeros(B, d, H_dim, W_dim, device=x.device) for d in self.hidden_dims]
        c = [torch.zeros_like(h[i]) for i in range(len(h))]
        m = torch.zeros_like(h[0])

        # Encoder (使用 x_new)
        for t in range(T_dim):
            inp = x_new[:, t]
            for i, cell in enumerate(self.layers):
                h[i], c[i], m = cell(inp, h[i], c[i], m)
                inp = h[i]

        # Decoder (起始输入使用最后一个时间步的 x_new)
        preds = []
        dec_inp = x_new[:, -1]
        # ===============================================
        for _ in range(self.horizon):
            inp = dec_inp
            for i, cell in enumerate(self.layers):
                h[i], c[i], m = cell(inp, h[i], c[i], m)
                inp = h[i]
            # 输出预测
            pred = self.conv_last(h[-1])
            preds.append(pred)

        return torch.stack(preds, dim=1)


# ========================================
# 7. 训练循环 (GPU)
# ========================================
model = PredRNNv2(height=H, width=W).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
# 使用 SmoothL1Loss 对异常值更鲁棒
criterion = nn.SmoothL1Loss()

print("\n[Step 3] Start Training...")
for ep in range(epochs):
    model.train()
    loss_sum = 0

    # ==================== 原代码 ====================
    # for xb, yb in train_loader:
    #     xb, yb = xb.to(device), yb.to(device)
    #     opt.zero_grad()
    #     pred = model(xb)
    #     loss = criterion(pred, yb)
    # ===============================================

    # ==================== 新代码 ====================
    for xb, lsb, yb in train_loader:
        xb, lsb, yb = xb.to(device), lsb.to(device), yb.to(device)

        opt.zero_grad()
        pred = model(xb, lsb)
        loss = criterion(pred, yb)
        # ===============================================
        loss.backward()
        opt.step()
        loss_sum += loss.item()

    if (ep + 1) % 1 == 0:
        print(f"Epoch {ep + 1}/{epochs} Loss={loss_sum / len(train_loader):.4f}")

# ========================================
# 8. 评估 (包含 SMAPE 等鲁棒指标)
# ========================================
print("\n[Step 4] Evaluation...")
model.eval()
preds, trues = [], []

with torch.no_grad():
    # ==================== 原代码 ====================
    # for xb, yb in test_loader:
    #     xb = xb.to(device)
    #     y_pred = model(xb).cpu().numpy()  # 预测完放回 CPU
    # ===============================================

    # ==================== 新代码 ====================
    for xb, lsb, yb in test_loader:
        xb, lsb = xb.to(device), lsb.to(device)
        y_pred = model(xb, lsb).cpu().numpy()  # 传入 ls 进行预测
        # ===============================================
        preds.append(y_pred)
        trues.append(yb.numpy())

# 拼接
preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)

# 反标准化
y_pred_phys = preds * y_std + y_mean
y_true_phys = trues * y_std + y_mean

# 展平
pred_flat = y_pred_phys.reshape(-1)
true_flat = y_true_phys.reshape(-1)

# 指标计算
mse = np.mean((pred_flat - true_flat) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(pred_flat - true_flat))

# R2
ss_res = np.sum((true_flat - pred_flat) ** 2)
ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"\nRMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")

# --- 鲁棒百分比误差 (解决 MAPE 爆炸问题) ---
print("\n--- Advanced Metrics ---")
# 1. 过滤掉极小值计算 MAPE (阈值设为 0.1 DU)
threshold = 0.1
mask = true_flat > threshold
if np.sum(mask) > 0:
    mape_filtered = np.mean(np.abs((true_flat[mask] - pred_flat[mask]) / true_flat[mask]))
    print(f"Filtered MAPE (>{threshold}): {mape_filtered:.2%}")

# 2. SMAPE (对称 MAPE, 分母更加稳定)
# 公式: 2 * |y - y_hat| / (|y| + |y_hat|)
smape = np.mean(2.0 * np.abs(pred_flat - true_flat) / (np.abs(true_flat) + np.abs(pred_flat) + 1e-6))
print(f"SMAPE: {smape:.2%}")

# ========================================
# 9. 可视化
# ========================================
try:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("True O3 Sample")
    plt.imshow(trues[0, 0, 0], cmap='viridis')  # 显示测试集第一帧
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Pred O3 Sample")
    plt.imshow(preds[0, 0, 0], cmap='viridis')
    plt.colorbar()
    plt.show()
    print("可视化完成。")
except:
    print("可视化跳过（无显示终端）。")

# 保存模型
torch.save(model.state_dict(), os.path.join(base_dir, "models", "训练结果", "predrnn_highlat_gpu_UVST.pth"))
print("\n模型已保存: predrnn_highlat_gpu_UVST.pth")

# ========================================
# 10. 变量–臭氧相关矩阵
# ========================================
print("\n[Analysis] Calculating Correlation Matrix...")
var_names = ['O3', 'u', 'v', 'temp', 'ssrd']

# 空间平均 (Time, )
series = {
    'O3': y_raw.mean(axis=(1, 2)),
    'u': vars_dict['u'].mean(axis=(1, 2)),
    'v': vars_dict['v'].mean(axis=(1, 2)),
    'temp': vars_dict['temp'].mean(axis=(1, 2)),
    'ssrd': vars_dict['fluxsurf_dn_sw'].mean(axis=(1, 2)),
}
# 堆叠 (Time, N_vars)
data_corr = np.stack([series[k] for k in var_names], axis=1)

# 计算并画图
corr = np.corrcoef(data_corr, rowvar=False)

try:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr,
        xticklabels=var_names, yticklabels=var_names,
        annot=True, fmt=".2f", cmap="coolwarm", center=0
    )
    plt.title("Variable–O3 Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "models", "训练过程", "correlation_UVPDST.png"))
    print("✅ 相关矩阵已保存为 correlation_matrix.png")
    plt.show()  # 如果本地跑会弹窗
except Exception as e:
    print(f"画图失败 (可能是无图形界面): {e}")