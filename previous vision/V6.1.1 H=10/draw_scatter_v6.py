import os, glob, re, sys
import numpy as np
import netCDF4 as nc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

# 设置中文字体 (Windows)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========================================
# 0. 基础路径与配置
# ========================================
# 当前脚本所在目录: d:\AAOzone\previous vision\V6.1.1 H=10
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(current_dir))

openmars_dir = os.path.join(base_dir, "Dataset", "OpenMars")
mcd_dir = os.path.join(base_dir, "Dataset", "MCDALL")
model_path = os.path.join(current_dir, "predrnn_highlat_gpu_UVST.pth")

window, horizon = 3, 10
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Using Device: {device}")

# ========================================
# 1. 数据读取逻辑 (同步 demo3-UVST.py)
# ========================================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

print("\n[Step 1] Loading Data...")
file_list = sorted(glob.glob(os.path.join(openmars_dir, "*.nc")), key=natural_sort_key)
if not file_list:
    raise FileNotFoundError("❌ 未找到 OpenMars 文件！")

o3_list, om_ls_list = [], []
ref_ds = nc.Dataset(file_list[0])
om_lats = ref_ds.variables['lat'][:] if 'lat' in ref_ds.variables else ref_ds.variables['latitude'][:]
om_lons = ref_ds.variables['lon'][:] if 'lon' in ref_ds.variables else ref_ds.variables['longitude'][:]
ref_ds.close()

for f in file_list:
    ds = nc.Dataset(f)
    o3_list.append(ds.variables['o3col'][:])
    ls_var = 'Ls' if 'Ls' in ds.variables else 'ls'
    om_ls_list.append(ds.variables[ls_var][:])
    ds.close()

o3col = np.concatenate(o3_list, axis=0)
om_ls_raw = np.concatenate(om_ls_list, axis=0)

# MCD 读取
mcd_vars = ['u', 'v', 'temp', 'fluxsurf_dn_sw']
mcd_data_list = {k: [] for k in mcd_vars}
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
            ls_end = ls_tmp[i+1] if i < S_dim-1 else ls_start + 0.5
            if ls_end < ls_start: ls_end += 360.0
            ls_expanded[i*H_dim : (i+1)*H_dim] = np.linspace(ls_start, ls_end, H_dim, endpoint=False)
        mcd_ls_list.append(ls_expanded % 360.0)
    else:
        mcd_ls_list.append(ls_tmp.flatten())
    ds.close()

vars_dict = {k: np.concatenate(mcd_data_list[k], axis=0) for k in mcd_vars}
mcd_ls_raw = np.concatenate(mcd_ls_list, axis=0)

def clean_invalid(x):
    x = np.array(x, dtype=np.float32)
    x[~np.isfinite(x) | (np.abs(x) > 1e10)] = np.nan
    return np.nan_to_num(x, nan=0.0)

y_raw = clean_invalid(o3col)
for k in vars_dict: vars_dict[k] = clean_invalid(vars_dict[k])
vars_dict['fluxsurf_dn_sw'] /= (np.max(vars_dict['fluxsurf_dn_sw']) + 1e-6)

# 时间对齐
def unwrap_ls(ls_array):
    ls_unwrapped = np.copy(ls_array)
    year_offset = 0
    for i in range(1, len(ls_unwrapped)):
        if ls_array[i] < ls_array[i-1] - 180: year_offset += 360
        ls_unwrapped[i] += year_offset
    return ls_unwrapped

om_ls_continuous = unwrap_ls(om_ls_raw)
mcd_ls_continuous = unwrap_ls(mcd_ls_raw)
for k in vars_dict:
    interpolator = interp1d(mcd_ls_continuous, vars_dict[k], axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
    vars_dict[k] = interpolator(om_ls_continuous)

# ========================================
# 2. 构建测试集 (内存优化版)
# ========================================
X_raw = np.stack([y_raw, vars_dict['u'], vars_dict['v'], vars_dict['temp'], vars_dict['fluxsurf_dn_sw']], axis=-1)
T, H, W, C = X_raw.shape
split_time_idx = int(0.8 * T)

X_train_raw = X_raw[:split_time_idx]
y_train_raw = y_raw[:split_time_idx]

# 缩放
X_scaled = np.zeros_like(X_raw, dtype=np.float32)
for c in range(C):
    scaler = StandardScaler()
    scaler.fit(X_train_raw[..., c].reshape(split_time_idx, -1))
    X_scaled[..., c] = scaler.transform(X_raw[..., c].reshape(T, -1)).reshape(T, H, W)

y_mean, y_std = y_train_raw.mean(), y_train_raw.std()
y_scaled = (y_raw - y_mean) / y_std

# 使用自定义 Dataset 避免内存冗余
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X_scaled, y_scaled, ls_cont, start_idx, end_idx, window, horizon):
        self.X = X_scaled
        self.y = y_scaled
        self.ls = ls_cont
        self.window = window
        self.horizon = horizon
        self.start_idx = start_idx
        # 有效样本数量
        self.num_samples = (end_idx - start_idx) - (window + horizon) + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        real_idx = self.start_idx + idx
        x_seq = self.X[real_idx : real_idx + self.window + self.horizon]
        y_seq = self.y[real_idx + self.window : real_idx + self.window + self.horizon]
        ls_seq = self.ls[real_idx : real_idx + self.window + self.horizon]
        
        # 转为 Tensor 并调整维度 (T, C, H, W)
        x_torch = torch.from_numpy(x_seq).permute(0, 3, 1, 2).float()
        y_torch = torch.from_numpy(y_seq).unsqueeze(1).float() # (horizon, 1, H, W)
        ls_torch = torch.from_numpy(ls_seq).float()
        return x_torch, ls_torch, y_torch

# 划分测试集范围
test_start = split_time_idx
test_end = T
test_dataset = SeqDataset(X_scaled, y_scaled, om_ls_continuous, test_start, test_end, window, horizon)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ========================================
# 3. 模型定义 (PredRNNv2)
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
        x_concat = self.conv_x(x); h_concat = self.conv_h(h); m_concat = self.conv_m(m)
        i_x, f_x, g_x, i_xp, f_xp, g_xp, o_x = torch.split(x_concat, self.num_hidden, 1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, 1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, 1)
        i_t = torch.sigmoid(i_x + i_h); f_t = torch.sigmoid(f_x + f_h + 1.0); g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c + i_t * g_t
        i_tp = torch.sigmoid(i_xp + i_m); f_tp = torch.sigmoid(f_xp + f_m + 1.0); g_tp = torch.tanh(g_xp + g_m)
        m_new = f_tp * m + i_tp * g_tp
        mem = torch.cat([c_new, m_new], dim=1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new

class PredRNNv2(nn.Module):
    def __init__(self, input_dim=9, hidden_dims=[64, 64, 64], height=H, width=W, horizon=horizon):
        super().__init__()
        self.w1_o3 = nn.Parameter(torch.zeros(1, 1, 1, height, width))
        self.b1_o3 = nn.Parameter(torch.zeros(1, 1, 1, height, width))
        self.w1_u = nn.Parameter(torch.ones(1, 1, 1, height, width)); self.w2_u = nn.Parameter(torch.ones(1, 1, 1, height, width))
        self.b1_u = nn.Parameter(torch.zeros(1, 1, 1, height, width)); self.b2_u = nn.Parameter(torch.zeros(1, 1, 1, height, width))
        self.w1_v = nn.Parameter(torch.ones(1, 1, 1, height, width)); self.w2_v = nn.Parameter(torch.ones(1, 1, 1, height, width))
        self.b1_v = nn.Parameter(torch.zeros(1, 1, 1, height, width)); self.b2_v = nn.Parameter(torch.zeros(1, 1, 1, height, width))
        self.w1_t = nn.Parameter(torch.ones(1, 1, 1, height, width)); self.w2_t = nn.Parameter(torch.ones(1, 1, 1, height, width))
        self.b1_t = nn.Parameter(torch.zeros(1, 1, 1, height, width)); self.b2_t = nn.Parameter(torch.zeros(1, 1, 1, height, width))
        self.w1_f = nn.Parameter(torch.ones(1, 1, 1, height, width)); self.w2_f = nn.Parameter(torch.ones(1, 1, 1, height, width))
        self.b1_f = nn.Parameter(torch.zeros(1, 1, 1, height, width)); self.b2_f = nn.Parameter(torch.zeros(1, 1, 1, height, width))
        self.layers = nn.ModuleList([SpatioTemporalLSTMCellv2(input_dim if i==0 else hidden_dims[i-1], hidden_dims[i], height, width, 3) for i in range(len(hidden_dims))])
        self.conv_last = nn.Conv2d(hidden_dims[-1], 1, 1)
        self.horizon, self.hidden_dims = horizon, hidden_dims
    def forward(self, x, ls):
        B, T_total, C, H_dim, W_dim = x.shape
        T_encoder = T_total - self.horizon
        ls_rad = (ls * (torch.pi / 180.0)).view(B, T_total, 1, 1, 1)
        o3, u, v, temp, flux = x[:,:,0:1], x[:,:,1:2], x[:,:,2:3], x[:,:,3:4], x[:,:,4:5]
        u_sin, u_cos = u * (self.w1_u * torch.sin(ls_rad + self.b1_u)), u * (self.w2_u * torch.cos(ls_rad + self.b2_u))
        v_sin, v_cos = v * (self.w1_v * torch.sin(ls_rad + self.b1_v)), v * (self.w2_v * torch.cos(ls_rad + self.b2_v))
        t_sin, t_cos = temp * (self.w1_t * torch.sin(ls_rad + self.b1_t)), temp * (self.w2_t * torch.cos(ls_rad + self.b2_t))
        f_sin, f_cos = flux * (self.w1_f * torch.sin(ls_rad + self.b1_f)), flux * (self.w2_f * torch.cos(ls_rad + self.b2_f))
        o3_fused = o3 + (self.w1_o3 * torch.sin(ls_rad + self.b1_o3))
        x_new = torch.cat([o3_fused, u_sin, u_cos, v_sin, v_cos, t_sin, t_cos, f_sin, f_cos], dim=2)
        h = [torch.zeros(B, d, H_dim, W_dim, device=x.device) for d in self.hidden_dims]
        c = [torch.zeros_like(hi) for hi in h]; m = torch.zeros_like(h[0])
        for t in range(T_encoder):
            inp = x_new[:, t]
            for i, cell in enumerate(self.layers): h[i], c[i], m = cell(inp, h[i], c[i], m); inp = h[i]
        preds = []
        current_o3 = x[:, T_encoder-1 : T_encoder, 0:1]
        for t in range(self.horizon):
            step_idx = T_encoder + t
            real_meteo = x_new[:, step_idx, 1:]; ls_step = ls_rad[:, step_idx:step_idx+1]
            current_o3_fused = current_o3 + (self.w1_o3 * torch.sin(ls_step + self.b1_o3))
            inp = torch.cat([current_o3_fused.squeeze(1), real_meteo], dim=1)
            for i, cell in enumerate(self.layers): h[i], c[i], m = cell(inp, h[i], c[i], m); inp = h[i]
            pred_o3 = self.conv_last(h[-1]); preds.append(pred_o3); current_o3 = pred_o3.unsqueeze(1)
        return torch.stack(preds, dim=1)

# ========================================
# 4. 推理与绘图
# ========================================
print("\n[Step 4] Inference...")
model = PredRNNv2(height=H, width=W).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

preds_all, trues_all = [], []
with torch.no_grad():
    for xb, lsb, yb in test_loader:
        xb, lsb = xb.to(device), lsb.to(device)
        y_p = model(xb, lsb).cpu().numpy()
        preds_all.append(y_p)
        trues_all.append(yb.numpy())

preds = np.concatenate(preds_all, axis=0) * y_std + y_mean
trues = np.concatenate(trues_all, axis=0) * y_std + y_mean

# 扁平化数据进行散点绘图
y_true = trues.flatten()
y_pred = preds.flatten()

print("\n[Step 5] Plotting...")
plt.figure(figsize=(9, 8), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

# 使用 hexbin 代替 gaussian_kde 以提高性能和内存效率
# gridsize 控制六边形的大小，cmap 使用 viridis 体现密度
hb = ax.hexbin(y_true, y_pred, gridsize=100, cmap='viridis', mincnt=1, bins='log')

# 绘制 y=x 线
max_val = max(np.max(y_true), np.max(y_pred))
min_val = min(np.min(y_true), np.min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='1:1 Line')

# 标签与标题
plt.xlabel('真实臭氧 (DU)', fontsize=12)
plt.ylabel('预测臭氧 (DU)', fontsize=12)
plt.title('真实值 VS 预测值 (V6.1.1 H=10)', fontsize=14, fontweight='bold')

# 添加统计信息
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
plt.text(0.05, 0.95, f'RMSE: {rmse:.4f}\n$R^2$: {r2:.4f}', 
         transform=ax.transAxes, verticalalignment='top', fontsize=12,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

cb = plt.colorbar(hb, ax=ax)
cb.set_label('样本密度 (log10)', fontsize=10)

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()

save_name = "O3_Scatter_Plot.png"
plt.savefig(os.path.join(current_dir, save_name), dpi=300)
print(f"✅ 散点图已保存至: {os.path.join(current_dir, save_name)}")
