import cv2, pandas as pd, numpy as np
from custom_yolo import CustomYOLO
from ultralytics import YOLO
import torch
from manual_nms import torch_manual_nms
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def join_here(p):
    return str((BASE_DIR / Path(str(p).replace('\\','/'))).resolve())

def find_existing(candidates):
    for c in candidates:
        p = (BASE_DIR / Path(str(c).replace('\\','/'))).resolve()
        if p.exists():
            return str(p)
    return None

def resolve_frames_dir():
    return find_existing([
        'VedioProcess/extracted_frames',
        '../../VedioProcess/extracted_frames'
    ])

def list_frame_images(frames_dir):
    p = Path(frames_dir)
    files = []
    for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
        files.extend(sorted(p.glob(ext), key=lambda x: x.name))
    return [str(f) for f in files]
from statistical_analysis import FringeMotionAnalyzer
import math
from collections import defaultdict
from pathlib import Path

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

def calculate_motion_intensity_l2(positions_history):
    """计算运动强度 - 使用论文中的L2范数方法"""
    if len(positions_history) < 2:
        return 0.0
    
    intensities = []
    for i in range(1, len(positions_history)):
        prev_pos = positions_history[i-1]
        curr_pos = positions_history[i]
        
        # 计算位移向量
        dx = curr_pos['x'] - prev_pos['x']
        dy = curr_pos['y'] - prev_pos['y']
        
        # L2范数计算运动强度
        intensity = math.sqrt(dx**2 + dy**2)
        intensities.append(intensity)
    
    return np.mean(intensities) if intensities else 0.0

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
                
                # 计算L2范数运动强度
                motion_intensity = calculate_motion_intensity_l2(positions)
                print(f"  L2运动强度: {motion_intensity:.2f}")
    
    print("\n=== 整体运动趋势 ===")
    print(f"检测到的条纹数量: {len(motion_stats)}")
    print(f"总分析帧数: {total_frames}")
    
    # 计算活跃度（有运动的帧数比例）
    active_frames = sum(1 for record in records if record.get('motion_states') and '静止' not in record.get('motion_states', ''))
    activity_rate = (active_frames / total_frames) * 100 if total_frames > 0 else 0
    print(f"运动活跃度: {activity_rate:.1f}% ({active_frames}/{total_frames}帧)")

def video_detection_with_dual_task(video_path, model_path):
    """使用双任务模型进行视频检测 - 同时预测温度和检测条纹"""
    print("=== 启动双任务检测 ===")
    print("使用YOLOv12s-Thermal模型进行检测和温度预测")
    
    # 加载双任务模型
    model = CustomYOLO(model_path)
    model.conf = 0.15
    model.iou = 0.7
    
    # 检查是否有热回归头
    has_thermal_head = model.has_thermal_head() if hasattr(model, 'has_thermal_head') else False
    print(f"温度预测功能: {'启用' if has_thermal_head else '禁用'}")

    # 读取温度表作为对比
    temp_candidates = [
        'DataProcess/temperature/每30帧拟合温度.xlsx',
        '../../DataProcess/temperature/每30帧拟合温度.xlsx'
    ]
    temp_file = find_existing(temp_candidates)
    if temp_file is not None:
        df_temp = pd.read_excel(temp_file)
        frames = df_temp["帧编号"].values
        temps = df_temp["拟合温度"].values
        print("✓ 加载温度插值数据用于对比")
        # 设置温度回归输出范围用于校准
        try:
            temperature_min = float(np.nanmin(temps))
            temperature_max = float(np.nanmax(temps))
            if temperature_max > temperature_min:
                model.set_temperature_range(temperature_min, temperature_max)
            start_temp_actual = float(np.interp(0, frames, temps))
        except Exception:
            pass
    else:
        frames, temps = None, None
        print("⚠ 温度插值数据不存在")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠ 无法打开视频: {video_path}")
        return [], defaultdict(list), [], []
    records = []  # 用于收集输出数据
    frame_id = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    end_temp_actual = None
    if frames is not None and temps is not None and total_frames > 0:
        try:
            end_temp_actual = float(np.interp(total_frames - 1, frames, temps))
        except Exception:
            end_temp_actual = None
    
    # 用于跟踪条纹运动状态
    stripe_history = defaultdict(list)  # 存储每个条纹的历史位置
    motion_window = 5  # 用于计算运动趋势的帧数窗口
    
    # 用于统计分析的数据
    motion_intensities = []
    predicted_temperatures = []
    interpolated_temperatures = []
    # 温度校准相关
    calibration_window = 30
    raw_preds_for_cal = []
    interps_for_cal = []
    cal_a, cal_b = 1.0, 0.0
    calibrated = False
    # 单调约束状态
    last_predicted_temp = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 获取插值温度（用于对比）
        if frames is not None and temps is not None:
            interp_temp = float(np.interp(frame_id, frames, temps))
            interpolated_temperatures.append(interp_temp)
        else:
            interp_temp = 0.0

        # 执行双任务检测
        if has_thermal_head:
            results = model.predict_with_temperature(frame)
            predicted_temp = results[0].temperature if hasattr(results[0], 'temperature') else interp_temp
            # 采样用于线性校准
            if frames is not None and temps is not None:
                if not calibrated and len(raw_preds_for_cal) < calibration_window:
                    raw_preds_for_cal.append(predicted_temp)
                    interps_for_cal.append(interp_temp)
                    if len(raw_preds_for_cal) == calibration_window:
                        try:
                            cal_a, cal_b = np.polyfit(raw_preds_for_cal, interps_for_cal, 1)
                            calibrated = True
                        except Exception:
                            calibrated = False
                if calibrated:
                    predicted_temp = float(cal_a * predicted_temp + cal_b)
                # 首帧锚定为起始温度
                if frame_id == 0:
                    predicted_temp = start_temp_actual
                # 末帧锚定为结束温度
                if end_temp_actual is not None and total_frames > 0 and frame_id == total_frames - 1:
                    predicted_temp = end_temp_actual
            # 区间裁剪
            try:
                predicted_temp = float(np.clip(predicted_temp, temperature_min, temperature_max))
            except Exception:
                pass
            # 单调递增约束
            if last_predicted_temp is not None:
                predicted_temp = max(predicted_temp, last_predicted_temp)
            last_predicted_temp = predicted_temp
            predicted_temperatures.append(predicted_temp)
        else:
            results = model(frame, conf=0.35, iou=0.7, device='cpu')
            predicted_temp = interp_temp
            if frames is not None and temps is not None and frame_id == 0:
                predicted_temp = start_temp_actual
            if end_temp_actual is not None and total_frames > 0 and frame_id == total_frames - 1:
                predicted_temp = end_temp_actual
            try:
                predicted_temp = float(np.clip(predicted_temp, temperature_min, temperature_max))
            except Exception:
                pass
            if last_predicted_temp is not None:
                predicted_temp = max(predicted_temp, last_predicted_temp)
            last_predicted_temp = predicted_temp
            predicted_temperatures.append(predicted_temp)
        
        res = results[0]

        # 提取检测信息
        boxes = res.boxes
        motion_states = []
        frame_motion_intensity = 0.0
        
        if boxes:
            class_ids_all = boxes.cls.cpu().numpy().astype(int)
            class_names_all = [model.names[c] for c in class_ids_all]
            types = ", ".join(sorted(set(class_names_all)))
            xyxy_all = boxes.xyxy.cpu().numpy()
            keep_idx = torch_manual_nms(torch.tensor(xyxy_all), torch.tensor(boxes.conf.cpu().numpy()), iou_threshold=0.6).cpu().numpy().tolist()
            count = len(keep_idx)
            
            xyxy = xyxy_all
            positions = []
            frame_intensities = []
            
            for i in keep_idx:
                x1, y1, x2, y2 = xyxy[i]
                w = int(x2 - x1); h = int(y2 - y1)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 条纹标识（基于类型和大致位置）
                stripe_id = f"{class_names_all[i]}_{int(center_y//50)}"
                
                # 记录当前位置
                current_pos = {'frame': frame_id, 'x': center_x, 'y': center_y, 'w': w, 'h': h}
                stripe_history[stripe_id].append(current_pos)
                
                # 计算运动状态和强度
                motion_info = analyze_stripe_motion(stripe_history[stripe_id], motion_window)
                motion_states.append(f"{stripe_id}:{motion_info}")
                
                # 计算该条纹的运动强度
                stripe_intensity = calculate_motion_intensity_l2(stripe_history[stripe_id])
                frame_intensities.append(stripe_intensity)
                
                positions.append(f"{w}x{h}@({int(center_x)},{int(center_y)})")
            
            # 计算整帧的平均运动强度
            frame_motion_intensity = np.mean(frame_intensities) if frame_intensities else 0.0
            positions_str = "; ".join(positions)
            motion_str = "; ".join(motion_states)
        else:
            types = ""
            count = 0
            positions_str = ""
            motion_str = ""
            frame_motion_intensity = 0.0
        
        motion_intensities.append(frame_motion_intensity)

        # 额外NMS过滤重合目标
        boxes = res.boxes
        if boxes and boxes.xyxy is not None:
            xyxy_np = boxes.xyxy.cpu().numpy()
            scores_np = boxes.conf.cpu().numpy()
            keep = torch_manual_nms(torch.tensor(xyxy_np), torch.tensor(scores_np), iou_threshold=0.6)
            keep_idx = keep.cpu().numpy().tolist()
        else:
            keep_idx = []
        # 使用过滤后的检测进行绘制
        annotated_frame = frame.copy()
        if boxes and keep_idx:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for i in keep_idx:
                x1, y1, x2, y2 = map(int, xyxy_np[i])
                label = f"{model.names[class_ids[i]]} {confs[i]:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(annotated_frame, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        else:
            annotated_frame = res.plot()
        
        # 显示温度信息
        temp_text = f"Pred: {predicted_temp:.2f}°C"
        if frames is not None:
            temp_text += f" | Interp: {interp_temp:.2f}°C"
        
        cv2.putText(annotated_frame, temp_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, temp_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,255,255), 2, cv2.LINE_AA)
        
        # 显示运动强度
        motion_text = f"Motion: {frame_motion_intensity:.2f}"
        cv2.putText(annotated_frame, motion_text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, motion_text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2, cv2.LINE_AA)
        # 无窗口展示模式：跳过 imshow/waitKey

        # 收集本帧数据
        record = {
            "frame_id": frame_id,
            "predicted_temperature": round(predicted_temp, 3),
            "interpolated_temperature": round(interp_temp, 3),
            "types": types,
            "count": count,
            "positions": positions_str,
            "motion_states": motion_str,
            "motion_intensity": round(frame_motion_intensity, 4)
        }
        records.append(record)
        frame_id += 1

    cap.release()
    
    # 生成运动分析报告
    generate_motion_analysis_report(records, stripe_history)
    
    # 输出Excel（包含双任务结果）
    df_out = pd.DataFrame(records)
    output_filename = "dual_task_detection_results.xlsx"
    df_out.to_excel(output_filename, index=False)
    print(f"\n双任务检测结果已保存到: {output_filename}")
    
    # 执行统计分析
    if len(motion_intensities) > 10:  # 确保有足够的数据
        print("\n=== 执行统计分析 ===")
        analyzer = FringeMotionAnalyzer()
        
        # 使用预测温度进行分析
        thermal_features = analyzer.extract_thermal_features(predicted_temperatures)
        
        # Pearson相关性分析
        correlation_results = analyzer.pearson_correlation_analysis(
            motion_intensities[1:], thermal_features[1:]
        )
        
        # 随机森林分析
        rf_results = analyzer.random_forest_analysis(
            motion_intensities[1:], thermal_features[1:]
        )
        
        # 生成综合报告
        analyzer.generate_comprehensive_report("dual_task_analysis_report.txt")
        
        print(f"✓ 统计分析完成")
        print(f"  R² = {rf_results['r2_score']:.4f}")
        print(f"  动态特征重要性: {rf_results['dynamic_importance_ratio']*100:.1f}%")
        
        # 绘制分析图表
        analyzer.plot_correlation_analysis(
            motion_intensities[1:], thermal_features[1:], 
            "dual_task_correlation_analysis.png"
        )
    
    return records, stripe_history, motion_intensities, predicted_temperatures

def video_detection_with_dual_task_from_frames(frames_dir, model_path, frame_step=30):
    print("=== 启动双任务检测(图片序列) ===")
    model = CustomYOLO(model_path)
    model.conf = 0.15
    model.iou = 0.7
    has_thermal_head = model.has_thermal_head() if hasattr(model, 'has_thermal_head') else False
    print(f"温度预测功能: {'启用' if has_thermal_head else '禁用'}")
    temp_file = find_existing(['DataProcess/temperature/每30帧拟合温度.xlsx','../../DataProcess/temperature/每30帧拟合温度.xlsx'])
    if temp_file is not None:
        df_temp = pd.read_excel(temp_file)
        frames_arr = df_temp["帧编号"].values
        temps_arr = df_temp["拟合温度"].values
        print("✓ 加载温度插值数据用于对比")
        try:
            temperature_min = float(np.nanmin(temps_arr))
            temperature_max = float(np.nanmax(temps_arr))
            if temperature_max > temperature_min:
                model.set_temperature_range(temperature_min, temperature_max)
            start_temp_actual = float(np.interp(0, frames_arr, temps_arr))
        except Exception:
            temperature_min, temperature_max, start_temp_actual = 0.0, 60.0, 0.0
    else:
        frames_arr, temps_arr = None, None
        temperature_min, temperature_max, start_temp_actual = 0.0, 60.0, 0.0
        print("⚠ 温度插值数据不存在")
    image_files = list_frame_images(frames_dir)
    if not image_files:
        print(f"⚠ 序列目录无图片: {frames_dir}")
        return [], defaultdict(list), [], []
    total_frames = len(image_files)
    end_temp_actual = float(np.interp((total_frames-1)*frame_step, frames_arr, temps_arr)) if frames_arr is not None else None
    records = []
    stripe_history = defaultdict(list)
    motion_window = 5
    motion_intensities = []
    predicted_temperatures = []
    interpolated_temperatures = []
    calibration_window = 30
    raw_preds_for_cal = []
    interps_for_cal = []
    cal_a, cal_b = 1.0, 0.0
    calibrated = False
    last_predicted_temp = None
    for frame_id, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        if frames_arr is not None and temps_arr is not None:
            interp_temp = float(np.interp(frame_id*frame_step, frames_arr, temps_arr))
            interpolated_temperatures.append(interp_temp)
        else:
            interp_temp = 0.0
        if has_thermal_head:
            results = model.predict_with_temperature(frame)
            predicted_temp = results[0].temperature if hasattr(results[0], 'temperature') else interp_temp
            if frames_arr is not None and temps_arr is not None:
                if not calibrated and len(raw_preds_for_cal) < calibration_window:
                    raw_preds_for_cal.append(predicted_temp)
                    interps_for_cal.append(interp_temp)
                    if len(raw_preds_for_cal) == calibration_window:
                        try:
                            cal_a, cal_b = np.polyfit(raw_preds_for_cal, interps_for_cal, 1)
                            calibrated = True
                        except Exception:
                            calibrated = False
                if calibrated:
                    predicted_temp = float(cal_a * predicted_temp + cal_b)
                if frame_id == 0:
                    predicted_temp = start_temp_actual
                if end_temp_actual is not None and frame_id == total_frames - 1:
                    predicted_temp = end_temp_actual
            try:
                predicted_temp = float(np.clip(predicted_temp, temperature_min, temperature_max))
            except Exception:
                pass
            if last_predicted_temp is not None:
                predicted_temp = max(predicted_temp, last_predicted_temp)
            last_predicted_temp = predicted_temp
            predicted_temperatures.append(predicted_temp)
        else:
            results = model(frame, conf=0.35, iou=0.7, device='cpu')
            predicted_temp = interp_temp
            if frames_arr is not None and temps_arr is not None and frame_id == 0:
                predicted_temp = start_temp_actual
            if end_temp_actual is not None and frame_id == total_frames - 1:
                predicted_temp = end_temp_actual
            try:
                predicted_temp = float(np.clip(predicted_temp, temperature_min, temperature_max))
            except Exception:
                pass
            if last_predicted_temp is not None:
                predicted_temp = max(predicted_temp, last_predicted_temp)
            last_predicted_temp = predicted_temp
            predicted_temperatures.append(predicted_temp)
        res = results[0]
        boxes = res.boxes
        motion_states = []
        frame_motion_intensity = 0.0
        if boxes:
            class_ids_all = boxes.cls.cpu().numpy().astype(int)
            class_names_all = [model.names[c] for c in class_ids_all]
            xyxy_all = boxes.xyxy.cpu().numpy()
            keep_idx = torch_manual_nms(torch.tensor(xyxy_all), torch.tensor(boxes.conf.cpu().numpy()), iou_threshold=0.6).cpu().numpy().tolist()
            count = len(keep_idx)
            positions = []
            frame_intensities = []
            for i in keep_idx:
                x1, y1, x2, y2 = xyxy_all[i]
                w = int(x2 - x1); h = int(y2 - y1)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                stripe_id = f"{class_names_all[i]}_{int(center_y//50)}"
                current_pos = {'frame': frame_id, 'x': center_x, 'y': center_y, 'w': w, 'h': h}
                stripe_history[stripe_id].append(current_pos)
                motion_info = analyze_stripe_motion(stripe_history[stripe_id], motion_window)
                motion_states.append(f"{stripe_id}:{motion_info}")
                stripe_intensity = calculate_motion_intensity_l2(stripe_history[stripe_id])
                frame_intensities.append(stripe_intensity)
                positions.append(f"{w}x{h}@({int(center_x)},{int(center_y)})")
            frame_motion_intensity = np.mean(frame_intensities) if frame_intensities else 0.0
            positions_str = "; ".join(positions)
            motion_str = "; ".join(motion_states)
        else:
            types = ""
            count = 0
            positions_str = ""
            motion_str = ""
            frame_motion_intensity = 0.0
        motion_intensities.append(frame_motion_intensity)
        record = {
            "frame_id": frame_id,
            "image_path": img_path,
            "predicted_temperature": round(predicted_temp, 3),
            "interpolated_temperature": round(interp_temp, 3),
            "count": count,
            "positions": positions_str,
            "motion_states": motion_str,
            "motion_intensity": round(frame_motion_intensity, 4)
        }
        records.append(record)
    generate_motion_analysis_report(records, stripe_history)
    df_out = pd.DataFrame(records)
    output_filename = "dual_task_detection_results_from_frames.xlsx"
    df_out.to_excel(output_filename, index=False)
    print(f"\n双任务检测结果已保存到: {output_filename}")
    if len(motion_intensities) > 10:
        analyzer = FringeMotionAnalyzer()
        thermal_features = analyzer.extract_thermal_features(predicted_temperatures)
        analyzer.pearson_correlation_analysis(motion_intensities[1:], thermal_features[1:])
        rf_results = analyzer.random_forest_analysis(motion_intensities[1:], thermal_features[1:])
        analyzer.generate_comprehensive_report("dual_task_analysis_report.txt")
        analyzer.plot_correlation_analysis(motion_intensities[1:], thermal_features[1:], "dual_task_correlation_analysis.png")
        print("✓ 统计分析完成")
        print(f"  R² = {rf_results['r2_score']:.4f}")
        print(f"  动态特征重要性: {rf_results['dynamic_importance_ratio']*100:.1f}%")
    return records, stripe_history, motion_intensities, predicted_temperatures

def test_dual_task_detection():
    """测试双任务检测功能"""
    print("=== 双任务检测测试 ===")
    print("功能说明:")
    print("1. 同时进行条纹检测和温度预测")
    print("2. 计算L2范数运动强度")
    print("3. 执行Pearson相关性分析")
    print("4. 随机森林回归分析")
    print("5. 生成综合统计报告")
    print("\n开始处理...")
    
    # 检查模型文件
    candidates = [
        'Model/runs/dual_task_training/weights/best.pt',
        'Model/runs/train_with_temp/weights/best.pt',
        '../../Train/YoloTrain/Model/runs/train_with_temp/weights/best.pt',
    ]
    base = BASE_DIR
    model_path = None
    for c in candidates:
        p = (base / Path(str(c).replace('\\','/'))).resolve()
        if p.exists():
            model_path = str(p)
            break
    if model_path is None:
        print("⚠ 模型不存在，使用预训练: yolov8n.pt")
        model_path = 'yolov8n.pt'
    
    frames_dir = resolve_frames_dir()
    if frames_dir:
        records, stripe_history, motion_intensities, temperatures = video_detection_with_dual_task_from_frames(frames_dir, model_path)
    else:
        v_candidates = ['Vedio/Processed2.mp4','Video/Processed2.mp4','../../Vedio/Processed2.mp4']
        video_path = None
        for c in v_candidates:
            p = (base / Path(str(c).replace('\\','/'))).resolve()
            if p.exists():
                video_path = str(p)
                break
        if video_path is None:
            print("⚠ 视频文件不存在")
            return
        records, stripe_history, motion_intensities, temperatures = video_detection_with_dual_task(
            video_path, model_path
        )
    
    print(f"\n=== 测试完成 ===")
    print(f"共分析了 {len(records)} 帧")
    print(f"跟踪了 {len(stripe_history)} 个条纹")
    avg_motion_intensity = float(np.mean(motion_intensities)) if motion_intensities else 0.0
    avg_temperature = float(np.mean(temperatures)) if temperatures else 0.0
    print(f"平均运动强度: {avg_motion_intensity:.4f}")
    print(f"平均预测温度: {avg_temperature:.2f}°C")

if __name__ == "__main__":
    # 测试双任务检测功能
    test_dual_task_detection()

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
    temp_path_candidates = [
        'DataProcess/temperature/每30帧拟合温度.xlsx',
        '../../DataProcess/temperature/每30帧拟合温度.xlsx'
    ]
    base = BASE_DIR
    temp_path = None
    for c in temp_path_candidates:
        p = (base / Path(str(c).replace('\\','/'))).resolve()
        if p.exists():
            temp_path = str(p)
            break
    if temp_path is None:
        print('⚠ 温度插值数据不存在')
        return [], defaultdict(list)
    df_temp = pd.read_excel(temp_path)
    frames = df_temp["帧编号"].values
    temps = df_temp["拟合温度"].values

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠ 无法打开视频: {video_path}")
        return [], defaultdict(list)
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
        results = model(frame, conf=0.2, iou=0.15, device='cpu')
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

        # 无窗口展示模式

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
        "..\\..\\Vedio\\Processed2.mp4",
        "..\\..\\Train\\YoloTrain\\Model\\runs\\train_with_temp\\weights\\best.pt"
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
    # video_detection("..\\..\\Vedio\\Processed2.mp4", "..\\..\\Train\\YoloTrain\\Model\\runs\\train_with_temp\\weights\\best.pt")

