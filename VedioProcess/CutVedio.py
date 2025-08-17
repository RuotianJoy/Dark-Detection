import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=30):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()  # 读取一帧

        # 如果视频读取完毕，退出循环
        if not ret:
            break

        # 每隔 frame_interval 帧保存一次图片
        if frame_count % frame_interval == 0:
            # 生成帧图片文件名（只包含帧号）
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)  # 保存当前帧为图片
            print(f"保存 {frame_filename}")

        frame_count += 1

    # 释放视频捕获对象
    cap.release()
    print("视频处理完成")

# 示例调用
video_path = "../Vedio/Processed2.mp4"         # 替换为你的视频文件路径
output_folder = "Processed_extracted_frames"  # 替换为你想保存帧图片的文件夹路径

extract_frames(video_path, output_folder, frame_interval=30)
