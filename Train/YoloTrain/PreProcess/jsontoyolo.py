import json
import os

# 指定包含JSON文件的文件夹路径
json_folder_path = 'D:\\Dark-Detection\\Train\\YoloTrain\\dataset\\labels_original\\json_file'  # 替换为包含JSON文件的文件夹路径
output_dir = 'D:\\Dark-Detection\\Train\\YoloTrain\\dataset\\labels\\train'  # 替换为你想要保存YOLO格式标签的路径

# 定义标签到类别ID的映射
label_to_id = {
    'udark': 0,
    'cdark': 1
}

os.makedirs(output_dir, exist_ok=True)

# 遍历文件夹中的所有JSON文件
for filename in os.listdir(json_folder_path):
    if filename.endswith('.json'):
        json_file_path = os.path.join(json_folder_path, filename)
        
        # 读取JSON文件
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # 提取图像信息
        image_width = data['imageWidth']
        image_height = data['imageHeight']
        
        # 创建输出文件路径
        output_file_path = os.path.join(output_dir, os.path.splitext(data['imagePath'])[0] + '.txt')
        
        # 打开输出文件，准备写入所有标注
        with open(output_file_path, 'w') as out_file:
            # 遍历标注的形状
            for shape in data['shapes']:
                label = shape['label']  # 标签类别
                points = shape['points']  # 矩形的四个角点
                
                # 获取类别ID
                class_id = label_to_id.get(label, 0)  # 如果标签不在映射中，默认使用0
                
                # 计算矩形边界框
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min = min(x_coords)
                x_max = max(x_coords)
                y_min = min(y_coords)
                y_max = max(y_coords)
                
                # 计算中心点、宽度和高度（YOLO格式）
                x_center = (x_min + x_max) / 2.0 / image_width
                y_center = (y_min + y_max) / 2.0 / image_height
                bbox_width = (x_max - x_min) / image_width
                bbox_height = (y_max - y_min) / image_height
                
                # 确保值在0-1范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                bbox_width = max(0, min(1, bbox_width))
                bbox_height = max(0, min(1, bbox_height))
                
                # YOLO格式：<class_id> <x_center> <y_center> <width> <height>
                yolo_format = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
                out_file.write(yolo_format)
        
        print(f"已成功转换JSON文件: {filename}")

print(f"所有JSON文件已成功转换为YOLO格式，并保存到: {output_dir}")