import os
import shutil

# 指定需要分离文件的文件夹路径
source_folder = 'D:\\Dark-Detection\\Processed_extracted_frames\\Processed_extracted_frames'  # 修改为你的文件夹路径

# 创建目标文件夹
jpg_folder = os.path.join(source_folder, 'D:\\Dark-Detection\\Train\\YoloTrain\\dataset\\images\\train')
json_folder = os.path.join(source_folder, 'D:\\Dark-Detection\\Train\\YoloTrain\\dataset\\labels\\json_file')

os.makedirs(jpg_folder, exist_ok=True)
os.makedirs(json_folder, exist_ok=True)

# 遍历文件夹中的文件
for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)

    if os.path.isfile(file_path):
        if filename.endswith('.jpg'):
            shutil.move(file_path, jpg_folder)  # 移动jpg文件
        elif filename.endswith('.json'):
            shutil.move(file_path, json_folder)  # 移动json文件

print("文件已成功分类并移动到各自的文件夹中。")