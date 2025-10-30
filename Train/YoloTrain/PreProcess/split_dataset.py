import os
import shutil
from pathlib import Path
import numpy as np

def split_dataset(dataset_path, train_ratio=0.8):
    # 转换为绝对路径
    dataset_path = Path(dataset_path).resolve()
    images_path = dataset_path / 'images'
    labels_path = dataset_path / 'labels_6'
    
    # 创建所需的目录结构
    for path in [images_path / 'train', images_path / 'val',
                labels_path / 'train', labels_path / 'val']:
        path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    all_images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        if list(images_path.glob(f'*{ext}')):
            all_images.extend(list(images_path.glob(f'*{ext}')))
    
    # 如果images目录下直接有图片，移动到train子目录
    if all_images:
        print("发现图片在images目录下，移动到train子目录...")
        for img_path in all_images:
            shutil.move(str(img_path), str(images_path / 'train' / img_path.name))
    
    # 同样处理标注文件
    all_labels = list(labels_path.glob('*.txt'))
    if all_labels:
        print("发现标注文件在labels目录下，移动到train子目录...")
        for label_path in all_labels:
            shutil.move(str(label_path), str(labels_path / 'train' / label_path.name))
    
    # 获取所有训练图像文件
    image_files = sorted([f.name for f in (images_path / 'train').glob('*.jpg')])
    if not image_files:
        image_files = sorted([f.name for f in (images_path / 'train').glob('*.jpeg')])
    if not image_files:
        image_files = sorted([f.name for f in (images_path / 'train').glob('*.png')])
    
    if not image_files:
        raise FileNotFoundError("未找到任何图像文件！")
    
    total_images = len(image_files)
    
    # 计算分割点
    val_size = int(total_images * (1 - train_ratio) / 2)  # 前后各10%
    train_start = val_size
    train_end = total_images - val_size
    
    print(f"\n数据集统计信息:")
    print(f"总图像数量: {total_images}")
    print(f"验证集(前): {val_size}张")
    print(f"训练集: {train_end - train_start}张")
    print(f"验证集(后): {val_size}张")
    
    # 按顺序处理文件
    print("\n开始处理数据集...")
    
    # 1. 首先处理前10%为验证集
    print("\n处理前10%验证集...")
    for i in range(val_size):
        img_file = image_files[i]
        label_file = os.path.splitext(img_file)[0] + '.txt'
        
        # 移动图片
        img_src = images_path / 'train' / img_file
        img_dst = images_path / 'val' / img_file
        if img_src.exists():
            shutil.copy2(str(img_src), str(img_dst))
            img_src.unlink()
            
        # 移动标注
        label_src = labels_path / 'train' / label_file
        label_dst = labels_path / 'val' / label_file
        if label_src.exists():
            shutil.copy2(str(label_src), str(label_dst))
            label_src.unlink()
    
    # 2. 处理中间80%为训练集
    print("处理中间80%训练集...")
    # 训练集文件保持在原位置，不需要移动
    
    # 3. 最后处理后10%为验证集
    print("处理后10%验证集...")
    for i in range(train_end, total_images):
        img_file = image_files[i]
        label_file = os.path.splitext(img_file)[0] + '.txt'
        
        # 移动图片
        img_src = images_path / 'train' / img_file
        img_dst = images_path / 'val' / img_file
        if img_src.exists():
            shutil.copy2(str(img_src), str(img_dst))
            img_src.unlink()
            
        # 移动标注
        label_src = labels_path / 'train' / label_file
        label_dst = labels_path / 'val' / label_file
        if label_src.exists():
            shutil.copy2(str(label_src), str(label_dst))
            label_src.unlink()
    
    print("\n数据集划分完成！")
    
    # 验证结果
    train_images = len(list((images_path / 'train').glob('*.jpg')))
    val_images = len(list((images_path / 'val').glob('*.jpg')))
    print(f"\n最终数据集统计:")
    print(f"训练集图片数量: {train_images}")
    print(f"验证集图片数量: {val_images}")

if __name__ == '__main__':
    dataset_path = r"D:\Dark-Detection\Train\YoloTrain\dataset"  # 使用绝对路径
    split_dataset(dataset_path) 