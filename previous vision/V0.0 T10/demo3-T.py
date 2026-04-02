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
sys.stdout = Logger(os.path.join(base_dir, "models", "训练过程", "T.txt"))

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


# 确保目录存在 (相对于运行根目录)
os.makedirs(os.path.join(base_dir, "models", "训练过程"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "models", "训练结果"), exist_ok=True)
sys.stdout = Logger(os.path.join(base_dir, "models", "训练过程", "UVPDST.txt"))

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
# 存入臭氧柱数据和太阳黄经数据
for f in file_list:  # 遍历所有文件
    ds = nc.Dataset(f)  # 打开文件

    data = ds.variables['o3col'][:]  # 读取臭氧柱总量数据
    o3_list.append(data)  # 存入列表

    # ★ 新增：读取并存储 OpenMars 的 Ls
    if 'Ls' in ds.variables:
        om_ls_list.append(ds.variables['Ls'][:])
    elif 'ls' in ds.variables:
        om_ls_list.append(ds.variables['ls'][:])
    else:
        raise ValueError(f"OpenMars 文件 {f} 中未找到 Ls 变量，请检查变量名！")

    ds.close()

o3col = np.concatenate(o3_list, axis=0)  # 把臭氧数据按照0维进行拼接 不可以按照经纬纬度拼接，时间维度更合适
om_ls_raw = np.concatenate(om_ls_list, axis=0)  # 把LS数据进行拼接
print(f"OpenMars 最终形状: {o3col.shape}")

# ========================================
# 2. MCD 读取 (MY27 + MY28 双文件)
# ========================================
print("\n[Step 2] Loading MCD Data (MY27 + MY28)...")

mcd_vars = ['Temperature']
short_names = ['temp']

# u 纬向风
# v 经向风
# ps 地表气压
# temp 温度
# dustq 尘埃光学厚度
# Solar_Flux_DN 太阳辐射通量

mcd_data_list = {k: [] for k in short_names}  # 为每个变量创建列表
mcd_ls_list = []  # ★ 新增：存储 MCD 的 Ls

target_files = [
    os.path.join(mcd_dir, "MCD_MY27_Lat-90-90_real.nc"),
    os.path.join(mcd_dir, "MCD_MY28_Lat-90-90_real.nc")
]


# 合并火星日和小时维度
# 目的是和OpenMars数据对齐，OpenMars三维 MCD四维
# S 火星日
# H 小时
# Y 纬度
# X 经度
def merge_sol_hour(x):
    S, H, Y, X = x.shape
    return x.reshape(S * H, Y, X)


for f_path in target_files:  # 遍历所有的MCD文件
    if not os.path.exists(f_path):
        continue  # 不存在则跳过

    print(f"正在读取: {os.path.basename(f_path)}")
    ds = nc.Dataset(f_path)

    # 提取气象变量 同时用merge_sol_hour函数将数据按时间轴合并
    mcd_data_list['temp'].append(merge_sol_hour(ds.variables['Temperature'][:]))
    # ★ 修复: 读取 MCD 的 Ls 并扩展为匹配 (Sol * Hour) 的 1D 时间轴
    if 'Ls' in ds.variables:
        ls_tmp = ds.variables['Ls'][:]
    elif 'ls' in ds.variables:
        ls_tmp = ds.variables['ls'][:]
    else:
        raise ValueError(f"MCD 文件 {f_path} 中未找到 Ls 变量！")

    # 从风场数据中提取纬度信息S_dim:Sol数量 H_dim:每天的小时数
    u_shape = ds.variables['U_Wind'].shape
    S_dim, H_dim = u_shape[0], u_shape[1]

    if ls_tmp.ndim == 1 and len(ls_tmp) == S_dim:  # 如果Ls是1D数组并且长度等于Sol数，说明是逐Sol的Ls值，需要拓展到逐小时
        ls_expanded = np.zeros(S_dim * H_dim)  # np.zeros用来创造全0数组
        for i in range(S_dim):  # 第i个Sol
            ls_start = ls_tmp[i]  # 第i个Sol的起点
            if i < S_dim - 1:  # 不是最后一天
                ls_end = ls_tmp[i + 1]  # 第i个Sol个终点
                if ls_end < ls_start:
                    ls_end += 360.0
            else:  # 处理最后一个Sol的Ls
                ls_end = ls_start + (ls_tmp[1] - ls_tmp[0] if S_dim > 1 else 0.5)

            # 第i个Sol对应小时的切片==在ls_start和ls_end之间生成H_dim个等距点 不包含终点
            ls_expanded[i * H_dim: (i + 1) * H_dim] = np.linspace(ls_start, ls_end, H_dim, endpoint=False)

        # 统一模 360 返回原始性质，后续由 unwrap_ls 统一处理
        ls_expanded = ls_expanded % 360.0
        mcd_ls_list.append(ls_expanded)
    else:
        mcd_ls_list.append(ls_tmp.flatten())  # 不需要插值时展平处理

    ds.close()

# 拼接 MY27 and MY28
vars_dict = {}
for k in short_names:
    vars_dict[k] = np.concatenate(mcd_data_list[k], axis=0)

mcd_ls_raw = np.concatenate(mcd_ls_list, axis=0)  # ★ MCD 拼接后的时间轴

# 检查形状是否匹配 只检查了空间未检查时间
if vars_dict['temp'].shape[1:] != o3col.shape[1:]:
    print(f"❌ 尺寸警告: MCD {vars_dict['temp'].shape[1:]} vs OpenMars {o3col.shape[1:]}")
    print("请检查经纬度筛选范围是否一致！")
else:
    print("✅ 空间尺寸完美匹配！")


# ========================================
# ★ FIX-A: 清理 fill_value / NaN / Inf
# ========================================
def clean_invalid(x, name):
    x = np.array(x, dtype=np.float32)
    bad = ~np.isfinite(x) | (np.abs(x) > 1e10)  # 进行非有限值或绝对值过大判断后的数据
    if np.any(bad):
        print(f"⚠️ {name}: cleaned {bad.sum()} invalid values")
        x[bad] = np.nan  # 将无效值统一为NAN
    return np.nan_to_num(x, nan=0.0)  # 将NAN替换为0


# OpenMars
y_raw = clean_invalid(o3col, "OpenMars O3")  # 对OpenMars数据进行清洗，存入y_raw

# MCD
for k in vars_dict:
    vars_dict[k] = clean_invalid(vars_dict[k], k)  # 对MCD的每个气象变量进行清洗

# ==================2026-3-4-00-33=================

# ========================================
# 3. 物理预处理
# ========================================

# Log 变换沙尘 (因为它跨度很大)
# 3. 物理预处理
if 'dustq' in vars_dict:
    vars_dict['dustq'][vars_dict['dustq'] < 0] = 0.0
    # vars_dict['dustq'] = np.log1p(vars_dict['dustq'])

if 'fluxsurf_dn_sw' in vars_dict:
    vars_dict['fluxsurf_dn_sw'] /= (np.max(vars_dict['fluxsurf_dn_sw']) + 1e-6)

# ========================================
# 4. 时间对齐 (核心修改：基于连续 Ls 插值对齐)
# ========================================
print("\n[Step 4] Time Alignment (Interpolating MCD to OpenMars based on Ls)...")


# ★ 辅助函数：解包跨年的 Ls
# 火星年结束 Ls 会从 359 突降回 0。这个函数会将 MY28 的 0~68° 转换为 360°~428°，变为单调递增
def unwrap_ls(ls_array):
    ls_unwrapped = np.copy(ls_array)
    year_offset = 0
    for i in range(1, len(ls_unwrapped)):
        if ls_array[i] < ls_array[i - 1] - 180:  # 发现跨年突降
            year_offset += 360
        ls_unwrapped[i] += year_offset
    return ls_unwrapped


# 1. 展开两个数据集的 Ls 形成连续时间轴
om_ls_continuous = unwrap_ls(om_ls_raw)
mcd_ls_continuous = unwrap_ls(mcd_ls_raw)

print(f"OpenMars 连续 Ls 范围: {om_ls_continuous.min():.2f} ~ {om_ls_continuous.max():.2f}")
print(f"MCD      连续 Ls 范围: {mcd_ls_continuous.min():.2f} ~ {mcd_ls_continuous.max():.2f}")

# 2. 以 OpenMars 的时间长度为绝对基准
y_raw = o3col

# 3. 沿着时间轴 (axis=0) 对 MCD 气象变量进行线性插值对齐
for k in vars_dict:
    # 创建插值器。fill_value="extrapolate" 允许外推，防止极微小的边界舍入误差报错
    interpolator = interp1d(
        mcd_ls_continuous,  # MCD的Ls
        vars_dict[k],  # Y轴：气象变量数据
        axis=0,  # 沿时间轴插值
        kind='linear',  # 线性插值
        bounds_error=False,
        fill_value="extrapolate"
    )
    # 将 MCD 数据精准“变形”到 OpenMars 的时间格点上
    vars_dict[k] = interpolator(om_ls_continuous)

print(f"✅ 物理对齐完成！两者时间维度现已统一为: {len(om_ls_continuous)}")

# ========================================
# 5. 构建 X, y 数据集
# ========================================
# 特征顺序: [O3_prev, temp]
X_raw = np.stack(
    [y_raw, vars_dict['temp']],
    axis=-1
)
T, H, W, C = X_raw.shape  # T：时间 H：垂直方向网格数 W：水平方向网格数 C：特征数
print(f"最终数据集 X_raw: {X_raw.shape}")

# ----------------------------------------
# ★ 修复数据泄露：按 80% 划分时间步，仅用训练集数据 fit
# ----------------------------------------
time_split = int(0.8 * T)

X_scaled = np.zeros_like(X_raw)  # 创造全0数组
for c in range(C):
    scaler = StandardScaler()
    # 关键修改：仅使用前 80% 的时间步数据进行 fit（学习均值和方差）
    scaler.fit(X_raw[:time_split, ..., c].reshape(time_split, -1))

    # 将学到的规则应用 (transform) 到所有数据上
    X_scaled[..., c] = scaler.transform(
        X_raw[..., c].reshape(T, -1)
    ).reshape(T, H, W)

# 关键修改：y_mean 和 y_std 也必须只从训练集计算
y_mean = y_raw[:time_split].mean()
y_std = y_raw[:time_split].std()

# 对全局 y 进行标准化
y_scaled = (y_raw - y_mean) / y_std
# ----------------------------------------

# 制作滑窗序列
X_seq, y_seq = [], []
for i in range(T - window - horizon + 1):
    X_seq.append(X_scaled[i: i + window])
    y_seq.append(y_scaled[i + window: i + window + horizon])

# 转为 Tensor 并移至 GPU (如果显存够大，可以直接在这里 to(device))
# 为了防止显存爆掉，我们在 DataLoader 循环里再 to(device)
X_torch = torch.tensor(np.array(X_seq)).permute(0, 1, 4, 2, 3).float()
y_torch = torch.tensor(np.array(y_seq)).unsqueeze(2).float()

# 划分训练/测试集
split = int(0.8 * len(X_torch))  # 80% 训练
train_dataset = TensorDataset(X_torch[:split], y_torch[:split])
test_dataset = TensorDataset(X_torch[split:], y_torch[split:])

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
    def __init__(self, input_dim=2, hidden_dims=[64, 64, 64], height=H, width=W, horizon=horizon):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            in_ch = input_dim if i == 0 else hidden_dims[i - 1]
            self.layers.append(
                SpatioTemporalLSTMCellv2(in_ch, hidden_dims[i], height, width, 3)
            )
        self.conv_last = nn.Conv2d(hidden_dims[-1], 1, 1)
        self.horizon = horizon
        self.hidden_dims = hidden_dims

    def forward(self, x):
        B, T, C, H, W = x.shape
        # 初始化状态
        h = [torch.zeros(B, d, H, W, device=x.device) for d in self.hidden_dims]
        c = [torch.zeros_like(h[i]) for i in range(len(h))]
        m = torch.zeros_like(h[0])

        # Encoder
        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.layers):
                h[i], c[i], m = cell(inp, h[i], c[i], m)
                inp = h[i]

        # Decoder
        preds = []
        dec_inp = x[:, -1]  # 使用最后一个输入作为起始
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
    for xb, yb in train_loader:
        # 关键：移动到 GPU
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
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
    for xb, yb in test_loader:
        xb = xb.to(device)
        y_pred = model(xb).cpu().numpy()  # 预测完放回 CPU
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
torch.save(model.state_dict(), os.path.join(base_dir, "models", "训练结果", "predrnn_highlat_gpu_T.pth"))
print("\n模型已保存: predrnn_highlat_gpu.pth")

# ========================================
# 10. 变量–臭氧相关矩阵
# ========================================
print("\n[Analysis] Calculating Correlation Matrix...")
var_names = ['O3', 'temp']

# 空间平均 (Time, )
series = {
    'O3': y_raw.mean(axis=(1, 2)),
    'temp': vars_dict['temp'].mean(axis=(1, 2)),
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