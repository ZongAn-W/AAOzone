import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')

import os, glob, re
import sys
import numpy as np
import netCDF4 as nc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
sys.stdout = Logger(os.path.join(base_dir, "models", "训练过程", "UVPDST.txt"))

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
om_ls_list = []


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


file_list = sorted(glob.glob(os.path.join(openmars_dir, "*.nc")), key=natural_sort_key)

if not file_list:
    raise FileNotFoundError("❌ 未找到 OpenMars 文件，请检查路径！")

ref_ds = nc.Dataset(file_list[0])
om_lats = ref_ds.variables['lat'][:] if 'lat' in ref_ds.variables else ref_ds.variables['latitude'][:]
om_lons = ref_ds.variables['lon'][:] if 'lon' in ref_ds.variables else ref_ds.variables['longitude'][:]
ref_ds.close()

for f in file_list:
    ds = nc.Dataset(f)
    data = ds.variables['o3col'][:]
    o3_list.append(data)

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

short_names = ['temp']
mcd_data_list = {k: [] for k in short_names}
mcd_ls_list = []

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
    mcd_data_list['temp'].append(merge_sol_hour(ds.variables['Temperature'][:]))

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

if vars_dict['temp'].shape[1:] != o3col.shape[1:]:
    print(f"❌ 尺寸警告: MCD {vars_dict['temp'].shape[1:]} vs OpenMars {o3col.shape[1:]}")
else:
    print("✅ 空间尺寸完美匹配！")


# ========================================
# 清理 fill_value / NaN / Inf
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
# 4. 时间对齐 (基于连续 Ls 插值对齐)
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

y_raw = o3col
for k in vars_dict:
    interpolator = interp1d(
        mcd_ls_continuous, vars_dict[k], axis=0, kind='linear',
        bounds_error=False, fill_value="extrapolate"
    )
    vars_dict[k] = interpolator(om_ls_continuous)

print(f"✅ 物理对齐完成！两者时间维度现已统一为: {len(om_ls_continuous)}")

# ========================================
# 5. 构建 X, y 数据集 (无数据泄露)
# ========================================
X_raw = np.stack([y_raw, vars_dict['temp']], axis=-1)
T, H, W, C = X_raw.shape

split_time_idx = int(0.8 * T)
X_train_raw = X_raw[:split_time_idx]
y_train_raw = y_raw[:split_time_idx]

X_scaled = np.zeros_like(X_raw)
for c in range(C):
    scaler = StandardScaler()
    scaler.fit(X_train_raw[..., c].reshape(split_time_idx, -1))
    X_scaled[..., c] = scaler.transform(X_raw[..., c].reshape(T, -1)).reshape(T, H, W)

y_mean = y_train_raw.mean()
y_std = y_train_raw.std()
y_scaled = (y_raw - y_mean) / y_std

X_seq, y_seq, ls_seq = [], [], []
for i in range(T - window - horizon + 1):
    X_seq.append(X_scaled[i: i + window])
    y_seq.append(y_scaled[i + window: i + window + horizon])
    ls_seq.append(om_ls_continuous[i: i + window])

X_torch = torch.tensor(np.array(X_seq)).permute(0, 1, 4, 2, 3).float()
y_torch = torch.tensor(np.array(y_seq)).unsqueeze(2).float()
ls_torch = torch.tensor(np.array(ls_seq)).float()

split_sample_idx = max(0, min(split_time_idx - window - horizon + 1, len(X_torch)))

train_dataset = TensorDataset(X_torch[:split_sample_idx], ls_torch[:split_sample_idx], y_torch[:split_sample_idx])
test_dataset = TensorDataset(X_torch[split_sample_idx:], ls_torch[split_sample_idx:], y_torch[split_sample_idx:])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")


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
    def __init__(self, input_dim=3, hidden_dims=[64, 64, 64], height=H, width=W, horizon=horizon):
        super().__init__()

        # ★ 核心改动：定义空间异质性的可学习张量 (1, 1, 1, H, W)
        self.w1 = nn.Parameter(torch.ones(1, 1, 1, height, width))
        self.w2 = nn.Parameter(torch.ones(1, 1, 1, height, width))
        self.b1 = nn.Parameter(torch.zeros(1, 1, 1, height, width))
        self.b2 = nn.Parameter(torch.zeros(1, 1, 1, height, width))

        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            in_ch = input_dim if i == 0 else hidden_dims[i - 1]
            self.layers.append(
                SpatioTemporalLSTMCellv2(in_ch, hidden_dims[i], height, width, 3)
            )
        self.conv_last = nn.Conv2d(hidden_dims[-1], 1, 1)
        self.horizon = horizon
        self.hidden_dims = hidden_dims

    def forward(self, x, ls):
        B, T_dim, C, H_dim, W_dim = x.shape

        # 将 Ls 角度转换为弧度
        ls_rad = (ls * (torch.pi / 180.0)).view(B, T_dim, 1, 1, 1)

        # ★ PyTorch广播机制发挥作用：(B, T_dim, 1, 1, 1) + (1, 1, 1, H_dim, W_dim)
        # 每个网格会根据自己的专属振幅w和偏置b去调制季节因子
        mod_sin = self.w1 * torch.sin(ls_rad + self.b1)
        mod_cos = self.w2 * torch.cos(ls_rad + self.b2)

        o3 = x[:, :, 0:1, :, :]
        temp = x[:, :, 1:2, :, :]

        # 生成调制后的温度通道
        temp_sin = temp * mod_sin
        temp_cos = temp * mod_cos

        x_new = torch.cat([o3, temp_sin, temp_cos], dim=2)

        # 初始化状态
        h = [torch.zeros(B, d, H_dim, W_dim, device=x.device) for d in self.hidden_dims]
        c = [torch.zeros_like(h[i]) for i in range(len(h))]
        m = torch.zeros_like(h[0])

        # Encoder
        for t in range(T_dim):
            inp = x_new[:, t]
            for i, cell in enumerate(self.layers):
                h[i], c[i], m = cell(inp, h[i], c[i], m)
                inp = h[i]

        # Decoder
        preds = []
        dec_inp = x_new[:, -1]
        for _ in range(self.horizon):
            inp = dec_inp
            for i, cell in enumerate(self.layers):
                h[i], c[i], m = cell(inp, h[i], c[i], m)
                inp = h[i]
            pred = self.conv_last(h[-1])
            preds.append(pred)

        return torch.stack(preds, dim=1)


# ========================================
# 7. 训练循环 (GPU)
# ========================================
model = PredRNNv2(height=H, width=W).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.SmoothL1Loss()

print("\n[Step 3] Start Training...")
for ep in range(epochs):
    model.train()
    loss_sum = 0
    for xb, lsb, yb in train_loader:
        xb, lsb, yb = xb.to(device), lsb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb, lsb)
        loss = criterion(pred, yb)
        loss.backward()
        opt.step()
        loss_sum += loss.item()

    if (ep + 1) % 1 == 0:
        print(f"Epoch {ep + 1}/{epochs} Loss={loss_sum / len(train_loader):.4f}")

# ========================================
# 8. 评估
# ========================================
print("\n[Step 4] Evaluation...")
model.eval()
preds, trues = [], []

with torch.no_grad():
    for xb, lsb, yb in test_loader:
        xb, lsb = xb.to(device), lsb.to(device)
        y_pred = model(xb, lsb).cpu().numpy()
        preds.append(y_pred)
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

print(f"\nRMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")

# ========================================
# 9. 结果与空间物理参数可视化 (★ 新增)
# ========================================
try:
    # 1. 臭氧预测可视化
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("True O3 Sample")
    plt.imshow(trues[0, 0, 0], cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Pred O3 Sample")
    plt.imshow(preds[0, 0, 0], cmap='viridis')
    plt.colorbar()
    plt.savefig(os.path.join(base_dir, "models", "训练过程", "prediction_sample.png"))
    plt.show()

    # 2. ★ 空间异质性参数可视化 (w1 和 b1)
    # 将学习到的空间张量提取到CPU并去掉无用的维度
    learned_w1 = model.w1.detach().cpu().numpy()[0, 0, 0]
    learned_b1 = model.b1.detach().cpu().numpy()[0, 0, 0]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Learned Spatial Amplitude (w1)\nHigh value = More sensitive to season")
    # 使用 bwr 或 coolwarm 可以清晰看出权重分布
    plt.imshow(learned_w1, cmap='coolwarm')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Learned Spatial Phase (b1)\nSeasonal lag distribution")
    plt.imshow(learned_b1, cmap='twilight')  # twilight 适合展示带有周期相位的分布
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "models", "训练过程", "spatial_heterogeneity_params.png"))
    plt.show()
    print("✅ 可视化完成，空间物理参数图已保存。")
except Exception as e:
    print(f"可视化跳过（可能是无图形界面）: {e}")

# 保存模型
torch.save(model.state_dict(), os.path.join(base_dir, "models", "训练结果", "predrnn_highlat_gpu_T.pth"))
print("\n模型已保存: predrnn_highlat_gpu_T.pth")