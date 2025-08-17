import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d

# 读取 Excel 数据
# 假设 "变化帧.xlsx" 与本脚本位于同一目录，默认读取第一个工作表
DF = pd.read_excel("变化帧.xlsx", sheet_name=0)

# 提取帧序列与温度序列
X = DF["变化帧（图片名）"].values
Y = DF["这一张温度"].values

# -----------------------------
#   1. 断点自动检索 & 数据分段
# -----------------------------
# 经验上干涉条纹在 2.7×10^4 帧附近出现明显跃迁，
# 在温度序列中表现为斜率突然增大——这里直接以 X >= 27000 的首个索引作为分段点。
split_idx = np.where(X >= 27000)[0][0]

# -----------------------------
#   2. 构建前段三次样条（自然端点）
# -----------------------------
cs = CubicSpline(X[:split_idx], Y[:split_idx], bc_type="natural")

# -----------------------------
#   3. 构建后段线性插值
# -----------------------------
lin = interp1d(X[split_idx:], Y[split_idx:], kind="linear", fill_value="extrapolate")

# -----------------------------
#   4. 生成 30 帧采样网格并分段计算温度
# -----------------------------
# 采样范围：从 0 帧到原始序列最大帧，闭区间，步长 30 帧
x_sample = np.arange(0, X.max() + 1, 30)

# 分段函数：利用向量化避免显式 for‑loop
mask = x_sample < X[split_idx]
y_sample = np.empty_like(x_sample, dtype=float)
y_sample[mask] = cs(x_sample[mask])
y_sample[~mask] = lin(x_sample[~mask])

# -----------------------------
#   5. 保存结果
# -----------------------------
result = pd.DataFrame({"帧编号": x_sample, "拟合温度": y_sample})
result.to_excel("每30帧拟合温度.xlsx", index=False)

# 控制台输出首末各 5 行，便于快速检查
print("\n===== 拟合结果预览 (前 5 条) =====")
print(result.head())
print("\n===== 拟合结果预览 (后 5 条) =====")
print(result.tail())
