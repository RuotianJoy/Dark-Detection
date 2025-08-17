import cv2, pandas as pd, numpy as np
from ultralytics import YOLO
import math
from collections import defaultdict

def analyze_stripe_motion(history, window_size):
    """分析条纹运动状态和趋势"""
    if len(history) < 2:
        return "静止"
    
    # 取最近的几个位置点
    recent_positions = history[-min(window_size, len(history)):]
    
    if len(recent_positions) < 2:
        return "静止"
    
    # 计算位移和速度
    displacements_x = []
    displacements_y = []
    velocities = []
    
    for i in range(1, len(recent_positions)):
        prev_pos = recent_positions[i-1]
        curr_pos = recent_positions[i]
        
        dx = curr_pos['x'] - prev_pos['x']
        dy = curr_pos['y'] - prev_pos['y']
        dt = curr_pos['frame'] - prev_pos['frame']
        
        if dt > 0:
            displacement = math.sqrt(dx**2 + dy**2)
            velocity = displacement / dt
            
            displacements_x.append(dx)
            displacements_y.append(dy)
            velocities.append(velocity)
    
    if not velocities:
        return "静止"
    
    # 分析运动趋势
    avg_velocity = np.mean(velocities)
    avg_dx = np.mean(displacements_x)
    avg_dy = np.mean(displacements_y)
    
    # 判断运动状态
    if avg_velocity < 0.5:  # 速度阈值
        motion_state = "静止"
    elif abs(avg_dx) > abs(avg_dy) * 2:  # 主要水平运动
        if avg_dx > 0:
            motion_state = f"右移({avg_velocity:.1f}px/帧)"
        else:
            motion_state = f"左移({avg_velocity:.1f}px/帧)"
    elif abs(avg_dy) > abs(avg_dx) * 2:  # 主要垂直运动
        if avg_dy > 0:
            motion_state = f"下移({avg_velocity:.1f}px/帧)"
        else:
            motion_state = f"上移({avg_velocity:.1f}px/帧)"
    else:  # 斜向运动
        direction = math.atan2(avg_dy, avg_dx) * 180 / math.pi
        motion_state = f"斜移({avg_velocity:.1f}px/帧,{direction:.0f}°)"
    
    return motion_state

def generate_motion_analysis_report(records, stripe_history):
    """生成条纹运动趋势分析报告"""
    print("\n=== 条纹运动状态分析报告 ===")
    
    # 统计各种运动状态
    motion_stats = {}
    total_frames = len(records)
    
    for record in records:
        motion_states = record.get('motion_states', '')
        if motion_states:
            states = motion_states.split('; ')
            for state in states:
                if ':' in state:
                    stripe_id, motion = state.split(':', 1)
                    if stripe_id not in motion_stats:
                        motion_stats[stripe_id] = []
                    motion_stats[stripe_id].append(motion)
    
    # 分析每个条纹的运动趋势
    for stripe_id, motions in motion_stats.items():
        print(f"\n条纹 {stripe_id}:")
        
        # 统计运动状态分布
        motion_counts = {}
        for motion in motions:
            motion_type = motion.split('(')[0]  # 提取运动类型
            motion_counts[motion_type] = motion_counts.get(motion_type, 0) + 1
        
        print(f"  总检测帧数: {len(motions)}")
        for motion_type, count in motion_counts.items():
            percentage = (count / len(motions)) * 100
            print(f"  {motion_type}: {count}帧 ({percentage:.1f}%)")
        
        # 分析位置变化趋势
        if stripe_id in stripe_history:
            positions = stripe_history[stripe_id]
            if len(positions) >= 2:
                start_pos = positions[0]
                end_pos = positions[-1]
                total_dx = end_pos['x'] - start_pos['x']
                total_dy = end_pos['y'] - start_pos['y']
                total_displacement = math.sqrt(total_dx**2 + total_dy**2)
                
                print(f"  总位移: {total_displacement:.1f}像素")
                print(f"  水平位移: {total_dx:.1f}像素")
                print(f"  垂直位移: {total_dy:.1f}像素")
                
                if len(positions) > 1:
                    avg_velocity = total_displacement / (end_pos['frame'] - start_pos['frame'])
                    print(f"  平均速度: {avg_velocity:.2f}像素/帧")
    
    print("\n=== 整体运动趋势 ===")
    print(f"检测到的条纹数量: {len(motion_stats)}")
    print(f"总分析帧数: {total_frames}")
    
    # 计算活跃度（有运动的帧数比例）
    active_frames = sum(1 for record in records if record.get('motion_states') and '静止' not in record.get('motion_states', ''))
    activity_rate = (active_frames / total_frames) * 100 if total_frames > 0 else 0
    print(f"运动活跃度: {activity_rate:.1f}% ({active_frames}/{total_frames}帧)")

def video_detection(video_path, model_path):
    model = YOLO(model_path)
    model.conf = 0.15
    model.iou = 0.7

    # 读取温度表
    df_temp = pd.read_excel("D:\\Dark-Detection\\DataProcess\\temperature\\每30帧拟合温度.xlsx")
    frames = df_temp["帧编号"].values
    temps = df_temp["拟合温度"].values

    cap = cv2.VideoCapture(video_path)
    records = []  # 用于收集输出数据
    frame_id = 0
    
    # 用于跟踪条纹运动状态
    stripe_history = defaultdict(list)  # 存储每个条纹的历史位置
    motion_window = 5  # 用于计算运动趋势的帧数窗口

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 获取拟合温度并格式化
        temp_val = float(np.interp(frame_id, frames, temps))
        temp_text = f"{temp_val:.6f}"

        # 执行YOLOv8检测
        results = model(frame, conf=0.2, iou=0.15)
        res = results[0]

        # 提取检测信息
        boxes = res.boxes
        motion_states = []  # 存储运动状态信息
        
        if boxes:  # 如果有检测到条纹
            class_ids = boxes.cls.cpu().numpy().astype(int)
            class_names = [model.names[c] for c in class_ids]
            types = ", ".join(sorted(set(class_names)))  # 所有检测到的条纹类型
            count = len(boxes)
            
            # 计算每个检测框的位置和运动状态
            xyxy = boxes.xyxy.cpu().numpy()
            positions = []
            
            for i, [x1, y1, x2, y2] in enumerate(xyxy):
                w = int(x2 - x1); h = int(y2 - y1)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 条纹标识（基于类型和大致位置）
                stripe_id = f"{class_names[i]}_{int(center_y//50)}"
                
                # 记录当前位置
                current_pos = {'frame': frame_id, 'x': center_x, 'y': center_y, 'w': w, 'h': h}
                stripe_history[stripe_id].append(current_pos)
                
                # 计算运动状态
                motion_info = analyze_stripe_motion(stripe_history[stripe_id], motion_window)
                motion_states.append(f"{stripe_id}:{motion_info}")
                
                positions.append(f"{w}x{h}@({int(center_x)},{int(center_y)})")
            
            positions_str = "; ".join(positions)
            motion_str = "; ".join(motion_states)
        else:
            types = ""
            count = 0
            positions_str = ""
            motion_str = ""

        # 在帧图像上绘制检测框和温度文字
        annotated_frame = res.plot()               # 绘制YOLO检测框和标签
        cv2.putText(annotated_frame, temp_text,    # 先绘制黑边文字
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(annotated_frame, temp_text,    # 再绘制白色文字
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("YOLOv8 Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 收集本帧数据
        records.append({
            "frame_id": frame_id,
            "temperature": round(temp_val, 1),
            "types": types,
            "count": count,
            "positions": positions_str,
            "motion_states": motion_str
        })
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    
    # 生成运动分析报告
    generate_motion_analysis_report(records, stripe_history)
    
    # 输出Excel（包含运动状态信息）
    df_out = pd.DataFrame(records)
    output_filename = "detection_output_with_motion.xlsx"
    df_out.to_excel(output_filename, index=False)
    print(f"\n检测结果已保存到: {output_filename}")
    print(f"新增列: motion_states - 包含每个条纹的运动状态和速度信息")
    
    return records, stripe_history


def test_motion_analysis():
    """测试条纹运动分析功能"""
    print("开始条纹运动状态分析测试...")
    print("功能说明:")
    print("1. 跟踪每个条纹的位置变化")
    print("2. 计算运动速度和方向")
    print("3. 分析运动趋势和活跃度")
    print("4. 生成详细的运动分析报告")
    print("\n开始处理视频...")
    
    # 执行检测和运动分析
    records, stripe_history = video_detection(
        "D:\\Dark-Detection\\Vedio\\Processed2.mp4", 
        "D:\\Dark-Detection\\Train\\YoloTrain\\Model\\runs\\train_with_temp\\weights\\best.pt"
    )
    
    print("\n=== 测试完成 ===")
    print(f"共分析了 {len(records)} 帧")
    print(f"跟踪了 {len(stripe_history)} 个条纹")
    print("\n输出文件说明:")
    print("- detection_output_with_motion.xlsx: 包含每帧的检测结果和运动状态")
    print("- 新增motion_states列: 记录每个条纹的运动状态(静止/左移/右移/上移/下移/斜移)")
    print("- 位置信息格式: 宽x高@(中心x,中心y)")
    print("- 运动状态格式: 条纹ID:运动类型(速度px/帧)")

if __name__ == "__main__":
    # 使用摄像头进行检测
    # video_detection()

    # 测试条纹运动分析功能
    test_motion_analysis()
    
    # 如果要检测视频文件，可以传入视频路径
    # video_detection("D:\\Dark-Detection\\Vedio\\Processed2.mp4", "D:\\Dark-Detection\\Train\\YoloTrain\\Model\\runs\\train_with_temp\\weights\\best.pt")

