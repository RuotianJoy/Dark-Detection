import os
import numpy as np
import json
from pathlib import Path

def process_labels(dataset_path):
    """处理标签文件，分离温度信息"""
    dataset_path = Path(dataset_path)
    temp_data = {}
    
    # 处理训练集和验证集
    for split in ['train', 'val']:
        labels_dir = dataset_path / 'labels_6' / split
        temp_data[split] = {}
        
        if not labels_dir.exists():
            continue
            
        # 创建临时目录来存储转换后的标签
        temp_dir = dataset_path / 'labels_temp' / split
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理每个标签文件
        for label_file in labels_dir.glob('*.txt'):
            temps = []  # 存储温度值
            new_labels = []  # 存储转换后的标签
            
            # 读取原始标签
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            # 处理每一行
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 6:
                    # 提取标准YOLO格式的部分
                    yolo_parts = parts[:5]
                    # 提取温度值
                    temp = float(parts[5])
                    temps.append(temp)
                    new_labels.append(' '.join(yolo_parts))
            
            # 保存转换后的标签
            new_label_file = temp_dir / label_file.name
            with open(new_label_file, 'w') as f:
                f.write('\n'.join(new_labels))
            
            # 存储温度信息
            temp_data[split][label_file.stem] = np.mean(temps)  # 使用平均温度
            
    # 保存温度数据
    temp_file = dataset_path / 'temperature_data.json'
    with open(temp_file, 'w') as f:
        json.dump(temp_data, f, indent=4)
        
    # 备份原始标签目录
    labels_dir = dataset_path / 'labels_6'
    labels_backup = dataset_path / 'labels_original'
    if not labels_backup.exists():
        os.rename(labels_dir, labels_backup)
        os.rename(dataset_path / 'labels_temp', labels_dir)
    
    return temp_file

if __name__ == '__main__':
    dataset_path = 'dataset'
    temp_file = process_labels(dataset_path)
    print(f"处理完成。温度数据保存在: {temp_file}") 