import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')

import os, glob, re, sys
import numpy as np
import netCDF4 as nc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ========================================
# 自动日志记录配置 ( Baseline 版本 )
# ========================================
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 确保日志实时刷新到文件

    def flush(self):
        pass


os.makedirs(os.path.join(base_dir, "models", "训练过程"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "models", "训练结果"), exist_ok=True)
# 修改了日志文件名以区分 baseline
sys.stdout = Logger(os.path.join(base_dir, "models", "训练过程", "Baseline_No_Ls.txt"))

# ========================================
# 0. 基础配置
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training Device: {device} [MODE: BASELINE WITHOUT Ls]")

openmars_dir = os.path.join(base_dir, "Dataset", "OpenMars")
mcd_dir = os.path.join(base_dir, "Dataset", "MCDALL")

window, horizon = 3, 3
batch_size = 16
epochs = 50

# ========================================
# 1. OpenMars 读取
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
    o3_list.append(ds.variables['o3col'][:])
    if 'Ls' in ds.variables:
        om_ls_list.append(ds.variables['Ls'][:])
    elif 'ls' in ds.variables:
        om_ls_list.append(ds.variables['ls'][:])
    ds.close()

o3col = np.concatenate(o3_list, axis=0)
om_ls_raw = np.concatenate(om_ls_list, axis=0)

# ========================================
# 2. MCD 读取 (MY27 + MY28 双文件)
# ========================================
print("\n[Step 2] Loading MCD Data (MY27 + MY28)...")
mcd_vars = ['U_Wind', 'V_Wind', 'Temperature', 'Solar_Flux_DN']
short_names = ['u', 'v', 'temp', 'fluxsurf_dn_sw']

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
    if not os.path.exists(f_path): continue
    ds = nc.Dataset(f_path)
    mcd_data_list['u'].append(merge_sol_hour(ds.variables['U_Wind'][:]))
    mcd_data_list['v'].append(merge_sol_hour(ds.variables['V_Wind'][:]))
    mcd_data_list['temp'].append(merge_sol_hour(ds.variables['Temperature'][:]))
    mcd_data_list['fluxsurf_dn_sw'].append(merge_sol_hour(ds.variables['Solar_Flux_DN'][:]))

    ls_tmp = ds.variables['Ls'][:] if 'Ls' in ds.variables else ds.variables['ls'][:]
    u_shape = ds.variables['U_Wind'].shape
    S_dim, H_dim = u_shape[0], u_shape[1]

    if ls_tmp.ndim == 1 and len(ls_tmp) == S_dim:
        ls_expanded = np.zeros(S_dim * H_dim)
        for i in range(S_dim):
            ls_start = ls_tmp[i]
            if i < S_dim - 1:
                ls_end = ls_tmp[i + 1]
                if ls_end < ls_start: ls_end += 360.0
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


def clean_invalid(x, name):
    x = np.array(x, dtype=np.float32)
    bad = ~np.isfinite(x) | (np.abs(x) > 1e10)
    if np.any(bad):
        x[bad] = np.nan
    return np.nan_to_num(x, nan=0.0)


y_raw = clean_invalid(o3col, "OpenMars O3")
for k in vars_dict:
    vars_dict[k] = clean_invalid(vars_dict[k], k)

if 'dustq' in vars_dict:
    vars_dict['dustq'][vars_dict['dustq'] < 0] = 0.0

if 'fluxsurf_dn_sw' in vars_dict:
    vars_dict['fluxsurf_dn_sw'] /= (np.max(vars_dict['fluxsurf_dn_sw']) + 1e-6)

# ========================================
# 4. 时间对齐 (即便不注入模型，对齐依然需要 Ls 作为基准)
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

for k in vars_dict:
    interpolator = interp1d(mcd_ls_continuous, vars_dict[k], axis=0, kind='linear', bounds_error=False,
                            fill_value="extrapolate")
    vars_dict[k] = interpolator(om_ls_continuous)

# ========================================
# 5. 构建数据集
# ========================================
X_raw = np.stack([y_raw, vars_dict['u'], vars_dict['v'], vars_dict['temp'], vars_dict['fluxsurf_dn_sw']], axis=-1)
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

X_seq, y_seq = [], []
for i in range(T - window - horizon + 1):
    X_seq.append(X_scaled[i: i + window])
    y_seq.append(y_scaled[i + window: i + window + horizon])

X_torch = torch.tensor(np.array(X_seq)).permute(0, 1, 4, 2, 3).float()
y_torch = torch.tensor(np.array(y_seq)).unsqueeze(2).float()

split_sample_idx = max(0, min(split_time_idx - window - horizon + 1, len(X_torch)))
# 注意：Baseline 的 dataset 不再打包 Ls
train_dataset = TensorDataset(X_torch[:split_sample_idx], y_torch[:split_sample_idx])
test_dataset = TensorDataset(X_torch[split_sample_idx:], y_torch[split_sample_idx:])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ========================================
# 6. 模型定义 ( Baseline 版本，无 Ls 注入 )
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


class PredRNNv2_Baseline(nn.Module):
    # Baseline 输入维度回归为 5：O3(1) + U(1) + V(1) + Temp(1) + Flux(1)
    def __init__(self, input_dim=5, hidden_dims=[64, 64, 64], height=H, width=W, horizon=horizon):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            in_ch = input_dim if i == 0 else hidden_dims[i - 1]
            self.layers.append(SpatioTemporalLSTMCellv2(in_ch, hidden_dims[i], height, width, 3))

        self.conv_last = nn.Conv2d(hidden_dims[-1], 1, 1)
        self.horizon = horizon
        self.hidden_dims = hidden_dims

    # 移除 ls 参数
    def forward(self, x):
        B, T_dim, C, H_dim, W_dim = x.shape

        # 直接使用原始特征，无物理周期调制
        x_new = x

        h = [torch.zeros(B, d, H_dim, W_dim, device=x.device) for d in self.hidden_dims]
        c = [torch.zeros_like(h[i]) for i in range(len(h))]
        m = torch.zeros_like(h[0])

        # Encoder
        for t in range(T_dim):
            inp = x_new[:, t]
            for i, cell in enumerate(self.layers):
                h[i], c[i], m = cell(inp, h[i], c[i], m)
                inp = h[i]

        # Decoder (自回归预测)
        preds = []
        last_meteo = x_new[:, -1, 1:, :, :]  # 提取历史最后时刻的气象环境 (4通道)
        current_o3 = x_new[:, -1, 0:1, :, :]  # 提取历史最后时刻的 O3 (1通道)

        for _ in range(self.horizon):
            inp = torch.cat([current_o3, last_meteo], dim=1)  # 重新拼接成 5 通道
            for i, cell in enumerate(self.layers):
                h[i], c[i], m = cell(inp, h[i], c[i], m)
                inp = h[i]

            pred_o3 = self.conv_last(h[-1])
            preds.append(pred_o3)
            current_o3 = pred_o3

        return torch.stack(preds, dim=1)


# ========================================
# 7. 训练循环 ( Baseline )
# ========================================
model = PredRNNv2_Baseline(height=H, width=W).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.SmoothL1Loss()

print("\n[Step 3] Start Training Baseline Model...")
best_val_loss = float('inf')
patience = 5  # 早停容忍度
early_stop_counter = 0
best_model_path = os.path.join(base_dir, "models", "训练结果", "best_predrnn_baseline.pth")

for ep in range(epochs):
    model.train()
    loss_sum = 0
    # 移除 lsb 的解包
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb)  # 移除 ls 的输入
        loss = criterion(pred, yb)
        loss.backward()
        opt.step()
        loss_sum += loss.item()

    # 打印训练损失
    avg_train_loss = loss_sum / len(train_loader)
    print(f"Epoch {ep + 1}/{epochs} Train Loss={avg_train_loss:.4f}")

    # 计算验证集损失
    model.eval()
    val_loss_sum = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            vloss = criterion(pred, yb)
            val_loss_sum += vloss.item()
    val_loss = val_loss_sum / len(test_loader)
    print(f"Validation Loss={val_loss:.4f}")

    # 早停逻辑与最佳模型保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ 保存最佳基准模型至: {os.path.basename(best_model_path)}")
    else:
        early_stop_counter += 1
    
    if early_stop_counter >= patience:
        print(f"🛑 早停触发: 在 epoch {ep + 1} 后验证损失未提升 {patience} 次")
        break

# ========================================
# 8. 评估
# 加载验证集上表现最好的模型
best_model_path = os.path.join(base_dir, "models", "训练结果", "best_predrnn_baseline.pth")
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"✅ 加载最佳模型进行评估: {os.path.basename(best_model_path)}")
else:
    print("⚠️ 未找到最佳模型，使用当前模型权重进行评估")
# ========================================
print("\n[Step 4] Evaluation...")
model.eval()
preds, trues = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        y_pred = model(xb).cpu().numpy()
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

print("\n--- Advanced Metrics ---")
threshold = 0.1
mask = true_flat > threshold
if np.sum(mask) > 0:
    mape_filtered = np.mean(np.abs((true_flat[mask] - pred_flat[mask]) / true_flat[mask]))
    print(f"Filtered MAPE (>{threshold}): {mape_filtered:.2%}")

smape = np.mean(2.0 * np.abs(pred_flat - true_flat) / (np.abs(true_flat) + np.abs(pred_flat) + 1e-6))
print(f"SMAPE: {smape:.2%}")

# ========================================
# 9. 保存模型
# ========================================
save_path = os.path.join(base_dir, "models", "训练结果", "predrnn_highlat_gpu_baseline_No_Ls.pth")
torch.save(model.state_dict(), save_path)
print(f"\n模型已保存: {os.path.basename(save_path)}")

# ========================================
# 10. PFI (Permutation Feature Importance) 分析
# ========================================
print("\n[Analysis] Starting PFI for Baseline...")


def evaluate_rmse(model, loader, device, y_std, y_mean):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            all_preds.append(pred)
            all_trues.append(yb.numpy())

    preds = np.concatenate(all_preds, axis=0) * y_std + y_mean
    trues = np.concatenate(all_trues, axis=0) * y_std + y_mean
    return np.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))


baseline_rmse = evaluate_rmse(model, test_loader, device, y_std, y_mean)
print(f"基准 RMSE: {baseline_rmse:.4f}")

feature_names = ['Prev_O3', 'U_Wind', 'V_Wind', 'Temperature', 'Solar_Flux']
pfi_scores = []

for i, col_name in enumerate(feature_names):
    X_test_tensor = test_dataset.tensors[0].clone()
    y_test_tensor = test_dataset.tensors[1]

    perm_idx = torch.randperm(X_test_tensor.size(0))
    X_test_tensor[:, :, i, :, :] = X_test_tensor[perm_idx, :, i, :, :]

    temp_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

    permuted_rmse = evaluate_rmse(model, temp_loader, device, y_std, y_mean)
    importance = permuted_rmse - baseline_rmse
    pfi_scores.append(importance)

    print(f"特征 [{col_name}] 置换后 RMSE: {permuted_rmse:.4f}, 增加量: {importance:.4f}")

try:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=pfi_scores, y=feature_names, palette="Reds_r")
    plt.title("Permutation Feature Importance (Baseline WITHOUT Ls)")
    plt.xlabel("Increase in RMSE (Lower is more important)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "models", "训练结果", "PFI_Analysis_Baseline.png"))
    print("✅ PFI 分析图表已保存为 PFI_Analysis_Baseline.png")
    plt.show()
except Exception as e:
    print(f"PFI 绘图失败: {e}")