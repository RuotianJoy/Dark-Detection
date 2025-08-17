from temp_utils import TemperatureProcessor, create_temp_labels
import os

def main():
    print("开始处理温度数据...")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 切换到脚本所在目录
    os.chdir(script_dir)
    
    # 初始化温度处理器
    temp_processor = TemperatureProcessor('../../DataProcess/temperature/每30帧拟合温度.xlsx')
    
    # 数据集路径
    dataset_path = 'dataset'
    
    # 处理温度标签
    print("正在处理温度标签...")
    create_temp_labels(dataset_path, temp_processor)

if __name__ == '__main__':
    main() 