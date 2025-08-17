from custom_yolo import CustomYOLO
import torch.multiprocessing as mp

def main():
    # 加载自定义模型
    model = CustomYOLO('yolo12s.pt')  # 加载预训练的YOLOv8n模型并添加温度预测支持

    # 训练模型
    results = model.train(
        data='data.yaml',  # 数据配置文件路径
        epochs=100,                  # 训练轮数
        imgsz=704,                 # 图像尺寸
        batch=12,                   # 批次大小
        nbs=36,               # ⇨ 有效 batch=24×3=72（不增显存）
        device='0',                 # GPU设备
        workers=3,                  # 数据加载的工作进程数
        patience=30,                # 早停的耐心值
        save=True,                  # 保存模型
        project='Model/runs',       # 保存结果的项目文件夹
        name='train_with_temp',     # 实验名称
        exist_ok=True,             # 如果存在同名实验文件夹则覆盖
        pretrained=True,           # 使用预训练权重
        optimizer='AdamW',         # 使用AdamW优化器
        lr0=1e-3,                  # 初始学习率
        lrf=1e-4,                  # 最终学习率
        momentum=0.937,            # SGD动量
        weight_decay=0.0005,       # 权重衰减
        warmup_epochs=3,           # 预热轮数
        warmup_momentum=0.8,       # 预热动量
        warmup_bias_lr=0.1,        # 预热偏置学习率
        verbose=True,              # 是否打印详细信息
        seed=42,                   # 随机种子
        deterministic=True,        # 确保结果可复现
        cache=False,               # 禁用缓存以避免潜在问题
        amp=True,                  # 混合精度（减显存）
        label_smoothing=0.05,      # 标签平滑
        plots=True,                # 生成训练过程图
        save_period=50,            # 每50个epoch保存一次
        cos_lr=True,               # 使用余弦学习率调度
        multi_scale=True,          # 使用多尺度训练
        overlap_mask=True,         # 重叠掩码
        mask_ratio=4,              # 掩码比例
        degrees=0.0,               # 旋转增强
        translate=0.1,             # 平移增强
        scale=0.5,                 # 缩放增强
        shear=0.1,                # 剪切增强
        perspective=0.0,          # 透视增强
        flipud=0.1,               # 上下翻转
        fliplr=0.5,               # 左右翻转
        mixup=0.1,                # mixup增强
        copy_paste=0.0,           # 复制粘贴增强
        mosaic=0.0,
        box=10.0,  # <<< 新增
        cls=0.05,  # <<< 新增
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    )

if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()