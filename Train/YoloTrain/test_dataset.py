#!/usr/bin/env python3
"""
测试数据集加载 - 诊断双任务训练问题
"""

import torch
from custom_yolo import CustomYOLO, CustomYOLODataset
from pathlib import Path
import yaml

def test_dataset_loading():
    """测试数据集加载功能"""
    print("=== 测试数据集加载 ===")
    
    # 1. 检查数据配置文件
    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    print(f"数据配置: {data_config}")
    
    # 2. 检查路径
    dataset_path = Path(data_config['path'])
    train_images = dataset_path / data_config['train']
    train_labels = dataset_path / 'labels_6' / 'train'
    
    print(f"数据集路径: {dataset_path.absolute()}")
    print(f"训练图像路径: {train_images.absolute()}")
    print(f"训练标签路径: {train_labels.absolute()}")
    
    print(f"图像路径存在: {train_images.exists()}")
    print(f"标签路径存在: {train_labels.exists()}")
    
    # 3. 检查文件数量
    if train_images.exists():
        image_files = list(train_images.glob('*.jpg'))
        print(f"图像文件数量: {len(image_files)}")
        if image_files:
            print(f"示例图像: {image_files[0].name}")
    
    if train_labels.exists():
        label_files = list(train_labels.glob('*.txt'))
        print(f"标签文件数量: {len(label_files)}")
        if label_files:
            print(f"示例标签: {label_files[0].name}")
            
            # 检查标签格式
            with open(label_files[0], 'r') as f:
                first_line = f.readline().strip()
                parts = first_line.split()
                print(f"标签格式: {len(parts)}列")
                print(f"示例标签行: {first_line}")

def test_custom_dataset():
    """测试自定义数据集类"""
    print("\n=== 测试自定义数据集类 ===")
    
    try:
        # 创建数据集实例
        dataset = CustomYOLODataset(
            img_path='dataset/images/train',
            imgsz=704,
            batch_size=1,
            augment=False,
            hyp={},
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix='test: '
        )
        
        print(f"数据集创建成功")
        print(f"数据集长度: {len(dataset)}")
        
        # 测试获取一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本键: {sample.keys()}")
            if 'temperature' in sample:
                print(f"温度信息: {sample['temperature']}")
            else:
                print("⚠ 样本中没有温度信息")
                
    except Exception as e:
        print(f"数据集创建失败: {e}")
        import traceback
        traceback.print_exc()

def test_model_creation():
    """测试模型创建"""
    print("\n=== 测试模型创建 ===")
    
    try:
        model = CustomYOLO('yolo12s.pt')
        print(f"模型创建成功")
        print(f"热回归头: {model.has_thermal_head()}")
        print(f"多任务损失: {model.has_multi_task_loss()}")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 3, 704, 704)
        with torch.no_grad():
            yolo_out, temp_out = model.forward_with_temperature(dummy_input)
            print(f"YOLO输出形状: {[x.shape for x in yolo_out] if isinstance(yolo_out, (list, tuple)) else yolo_out.shape}")
            print(f"温度输出: {temp_out.item() if temp_out is not None else 'None'}")
            
    except Exception as e:
        print(f"模型测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    test_dataset_loading()
    test_custom_dataset()
    test_model_creation()

if __name__ == "__main__":
    main()