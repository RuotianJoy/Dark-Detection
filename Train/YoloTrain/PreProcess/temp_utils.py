import pandas as pd
import numpy as np
from pathlib import Path

class TemperatureProcessor:
    def __init__(self, temp_file):
        """初始化温度处理器
        Args:
            temp_file: 温度数据文件路径
        """
        # 读取Excel文件，假设第一行是列名
        self.temp_data = pd.read_excel(temp_file)
        print("\nExcel文件列名:", list(self.temp_data.columns))
        self.process_temp_data()
        
    def normalize_frame_id(self, frame_id, reverse=False):
        """统一处理帧号格式
        Args:
            frame_id: 原始帧号（可能是数字、字符串等）
            reverse: 是否反向映射（从数据集帧号映射到温度数据帧号）
        Returns:
            str: 标准化的帧号（frame_XXXXXX格式）
        """
        # 打印原始帧号用于调试
        print(f"处理帧号: {frame_id}, 类型: {type(frame_id)}")
        
        # 移除可能的'frame_'前缀并获取数字部分
        if isinstance(frame_id, str) and frame_id.startswith('frame_'):
            frame_id = frame_id[6:]
        
        # 转换为整数
        try:
            # 如果是浮点数，先转换为整数
            if isinstance(frame_id, float):
                frame_num = int(frame_id)
            else:
                frame_num = int(str(frame_id))
            

            normalized_id = f"frame_{frame_num:06d}"
                
            print(f"标准化后的帧号: {normalized_id}")
            return normalized_id
        except (ValueError, TypeError) as e:
            print(f"警告: 无法处理的帧号格式: {frame_id}, 错误: {str(e)}")
            return None
        
    def process_temp_data(self):
        """处理温度数据，将帧号与温度对应"""
        # 创建温度字典
        self.temp_dict = {'train': {}, 'val': {}}
        
        # 将所有温度数据先加载到一个临时字典中
        temp_dict_all = {}
        
        # 获取帧号和温度的列名
        frame_col = None
        temp_col = None
        
        # 尝试找到正确的列名
        possible_frame_cols = ['frame', 'frame_id', 'frameid', 'id', '帧号', '帧编号']
        possible_temp_cols = ['temperature', 'temp', '拟合温度']
        
        # 打印所有列名用于调试
        print("\n所有列名:")
        for col in self.temp_data.columns:
            print(f"- {col} (类型: {self.temp_data[col].dtype})")
        
        for col in self.temp_data.columns:
            col_lower = str(col).lower()
            if frame_col is None:
                for possible_col in possible_frame_cols:
                    if possible_col in col_lower:
                        frame_col = col
                        break
            if temp_col is None:
                for possible_col in possible_temp_cols:
                    if possible_col in col_lower:
                        temp_col = col
                        break
        
        if frame_col is None or temp_col is None:
            raise ValueError(f"无法找到帧号或温度列。请确保Excel文件包含正确的列名。\n"
                           f"当前列名: {list(self.temp_data.columns)}\n"
                           f"可用的帧号列名: {possible_frame_cols}\n"
                           f"可用的温度列名: {possible_temp_cols}")
        
        print(f"\n使用的列名:")
        print(f"帧号列: {frame_col}")
        print(f"温度列: {temp_col}")
        
        # 打印前几行数据用于调试
        print("\n前5行数据:")
        print(self.temp_data.head())
        
        # 处理每一行数据
        for _, row in self.temp_data.iterrows():
            frame_id = self.normalize_frame_id(row[frame_col])
            if frame_id is not None:
                temp_dict_all[frame_id] = float(row[temp_col])
        
        # 打印所有处理后的帧号用于调试
        print("\n处理后的帧号列表:")
        for frame_id in sorted(temp_dict_all.keys())[:5]:
            print(f"- {frame_id}: {temp_dict_all[frame_id]}")
            
        # 扫描数据集目录，根据现有的训练集和验证集划分来组织温度数据
        dataset_path = Path('dataset')  # 使用相对路径
        for split in ['train', 'val']:
            images_dir = dataset_path / 'images' / split
            if not images_dir.exists():
                print(f"警告: {images_dir} 目录不存在")
                continue
                
            # 根据图像文件名获取对应的温度数据
            print(f"\n处理{split}集图像:")
            for img_file in images_dir.glob('*.jpg'):
                # 获取不带扩展名的文件名
                frame_id = img_file.stem
                print(f"\n处理图像: {frame_id}")
                
                # 标准化帧号格式并进行反向映射
                frame_id = self.normalize_frame_id(frame_id, reverse=True)
                if frame_id in temp_dict_all:
                    self.temp_dict[split][frame_id] = temp_dict_all[frame_id]
                    print(f"找到对应温度数据: {temp_dict_all[frame_id]}")
                else:
                    print(f"警告: 未找到图像 {img_file.name} 对应的温度数据")
        
        # 计算训练集和验证集的温度范围
        if self.temp_dict['train']:
            self.train_temp_range = {
                'min': min(self.temp_dict['train'].values()),
                'max': max(self.temp_dict['train'].values())
            }
        else:
            print("警告: 训练集温度数据为空")
            self.train_temp_range = {'min': 0, 'max': 1}
            
        if self.temp_dict['val']:
            self.val_temp_range = {
                'min': min(self.temp_dict['val'].values()),
                'max': max(self.temp_dict['val'].values())
            }
        else:
            print("警告: 验证集温度数据为空")
            self.val_temp_range = {'min': 0, 'max': 1}
        
        print("\n温度数据统计:")
        print(f"训练集温度范围: {self.train_temp_range['min']:.2f}°C - {self.train_temp_range['max']:.2f}°C")
        print(f"验证集温度范围: {self.val_temp_range['min']:.2f}°C - {self.val_temp_range['max']:.2f}°C")
        print(f"训练集样本数: {len(self.temp_dict['train'])}")
        print(f"验证集样本数: {len(self.temp_dict['val'])}")
            
    def get_temperature(self, frame_id, split='train'):
        """获取指定帧的温度
        Args:
            frame_id: 帧号（文件名，如'frame_000030'）
            split: 数据集划分（'train'或'val'）
        Returns:
            float: 温度值
        """
        # 标准化帧号格式
        frame_id = self.normalize_frame_id(frame_id)
        return self.temp_dict[split].get(frame_id, None)
    
    def normalize_temperature(self, temp, split='train'):
        """将温度归一化到[0,1]范围
        Args:
            temp: 原始温度值
            split: 数据集划分（'train'或'val'）
        Returns:
            float: 归一化后的温度值
        """
        temp_range = self.train_temp_range if split == 'train' else self.val_temp_range
        return (temp - temp_range['min']) / (temp_range['max'] - temp_range['min'])

def create_temp_labels(dataset_path, temp_processor):
    """为数据集创建温度标签文件
    Args:
        dataset_path: 数据集根目录
        temp_processor: TemperatureProcessor实例
    """
    dataset_path = Path(dataset_path)
    
    # 处理训练集和验证集
    for split in ['train', 'val']:
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels_6' / split
        
        # 确保标签目录存在
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理每个图像文件
        for img_file in images_dir.glob('*.jpg'):
            frame_id = img_file.stem  # 获取不带扩展名的文件名
            label_file = labels_dir / f"{frame_id}.txt"
            
            if label_file.exists():
                # 读取原始标签
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                # 获取温度值并归一化
                temp = temp_processor.get_temperature(frame_id, split)
                if temp is not None:
                    norm_temp = temp_processor.normalize_temperature(temp, split)
                    
                    # 修改标签：添加温度信息到每个边界框
                    new_lines = []
                    for line in lines:
                        # YOLO格式：class x y w h
                        parts = line.strip().split()
                        # 添加温度作为额外属性
                        new_line = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {norm_temp:.6f}\n"
                        new_lines.append(new_line)
                    
                    # 保存修改后的标签
                    with open(label_file, 'w') as f:
                        f.writelines(new_lines)
                else:
                    print(f"警告: 未找到帧 {frame_id} 的温度数据")
                        
    print("\n温度标签处理完成！") 