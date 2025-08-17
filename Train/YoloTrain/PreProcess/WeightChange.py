# convert_to_3c.py
import torch
from ultralytics import YOLO
import torch.nn as nn

ckpt_path = r"D:\Dark-Detection\Train\YoloTrain\Model\runs\train_with_temp\weights\best.pt"
model = YOLO(ckpt_path).model            # 加载 nn.Module

conv0 = model.model[0].conv              # YOLOv8 backbone 的第 0 层
w = conv0.weight                         # [32,1,3,3]

# 构造新的 3-通道卷积层，并复制权重
new_conv = nn.Conv2d(3, w.shape[0], 3, stride=conv0.stride,
                     padding=conv0.padding, bias=False)
new_conv.weight.data = w.repeat(1, 3, 1, 1) / 3  # 简单复制+平均
model.model[0].conv = new_conv

# 保存新权重
torch.save({"model": model}, ckpt_path.replace(".pt", "_3c.pt"))
print("✅ 已保存 3-通道权重  →", ckpt_path.replace(".pt", "_3c.pt"))
