# AAOzone

项目概述

AAOzone 是用于行星（Mars）臭氧列（O3 column）时空预测的研究/实验代码库。核心脚本基于 PredRNNv2 风格的时空循环单元，融合 OpenMars（观测/重分析）与 MCD（气象场）数据，进行数据对齐、训练、评估与特征重要性（PFI）分析。

主要文件与目录

- Dataset/
  - OpenMars/      : OpenMars 数据集（*.nc），脚本会读取 o3col 与 Ls 变量。
  - MCDALL/        : MCD 气象数据（MY27/MY28 等 *.nc）。
  - 注意：数据目录被 .gitignore 忽略，不要将大文件提交。

- models/
  - 训练模型/      : demo/脚本（e.g. models\训练模型\demo3-UVST.py）。
  - 训练过程/      : 日志（models\训练过程\UVST.txt）
  - 训练结果/      : 输出权重与图（models\训练结果\predrnn_highlat_gpu_UVST.pth, PFI_Analysis_Fixed.png）

- previous vision/  : 历史实验与 checkpoints（仅作参考）。

核心脚本说明

models\训练模型\demo3-UVST.py
- 功能：
  1) 读取 OpenMars 与 MCD 数据，基于太阳体位置（Ls）对时间轴插值/对齐；
  2) 构建无时间泄露的数据序列（window, horizon）；
  3) 标准化（StandardScaler）并构造 PyTorch DataLoader；
  4) 定义 PredRNNv2（SpatioTemporalLSTMCellv2）模型，使用自回归解码；
  5) 训练（Adam, SmoothL1Loss）；
  6) 评估（RMSE/MAE/R2/SMAPE/filtered MAPE）；
  7) 保存模型与生成 PFI（Permutation Feature Importance）分析图表。

- 关键超参数（脚本顶部，可直接编辑）：
  - window = 3
  - horizon = 3
  - batch_size = 16
  - epochs = 10
  - lr = 1e-4

- 输入变量：脚本将构建包含 5 个物理量的输入通道序列：
  [O3, U_Wind, V_Wind, Temperature, Solar_Flux]
  在模型内部将这些气象量用基于 Ls 的正余弦权重扩展为 8 个通道（sin/cos）+ O3，共 9 通道输入。

运行说明

1. 环境准备（推荐）：

   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt

2. 必要依赖（脚本使用）：
   - Python 3.8+
   - numpy, scipy, netCDF4
   - torch (建议与 CUDA 匹配的版本)
   - scikit-learn, matplotlib, seaborn

3. 运行训练（在仓库根）：

   python models\训练模型\demo3-UVST.py

4. 输出：
   - 日志: models\训练过程\UVST.txt
   - 模型权重: models\训练结果\predrnn_highlat_gpu_UVST.pth
   - PFI 图: models\训练结果\PFI_Analysis_Fixed.png

注意事项与建议

- 数据文件名与变量：OpenMars 文件需包含 o3col 与 Ls/ls；MCD 文件需包含 U_Wind, V_Wind, Temperature, Solar_Flux_DN（脚本内存在变量名映射）。
- 若数据变量命名或形状不同，可能需要调整脚本中的变量提取或 merge_sol_hour 函数。
- GPU：若可用，PyTorch 会自动使用 CUDA。训练在 CPU 上可能很慢。
- 修改超参数：直接编辑脚本顶部的 window/horizon/batch_size/epochs 等参数以快速实验。
- 日志与可视化：脚本将把 stdout 重定向到 models\训练过程\UVST.txt，图像显示失败时会使用 Agg 后端并保存到文件。

贡献与许可证

欢迎以 Fork + PR 方式贡献：
- 在 PR 中说明改动、复现步骤与测试结果。
- 若希望他人复现训练结果，请提供小规模示例数据或数据下载脚本。

许可证

请在推送前添加合适的 LICENSE（例如 MIT）。

联系方式

仓库: https://github.com/ZongAn-W/AAOzone

