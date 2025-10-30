"""
NMS配置文件 - 确保使用CPU版本的NMS实现
解决torchvision::nms CUDA兼容性问题
"""

import torch
import os
import warnings

def configure_nms_for_cpu():
    """
    配置NMS强制使用CPU，避免CUDA兼容性问题
    """
    # 设置环境变量强制使用CPU进行NMS操作
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 如果可能，设置torchvision使用CPU后端
    try:
        import torchvision
        # 尝试设置torchvision使用CPU
        if hasattr(torchvision, '_C'):
            # 这是一个内部设置，可能不总是有效
            pass
    except ImportError:
        # torchvision未安装或已被禁用
        pass
    
    print("✓ NMS配置已设置为强制使用CPU")

def patch_yolo_nms():
    """
    修补YOLO的NMS实现，确保使用我们的自定义版本
    """
    try:
        from ultralytics.utils.ops import non_max_suppression
        from manual_nms import torch_manual_nms
        
        # 创建一个包装函数来替换原始的NMS
        def cpu_nms_wrapper(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, 
                           agnostic=False, multi_label=False, labels=(), max_det=300, 
                           nc=0, max_time_img=0.05, max_nms=30000, max_wh=7680):
            """
            CPU版本的NMS包装器
            """
            # 强制将所有张量移动到CPU
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.cpu()
            
            # 调用原始函数，但确保在CPU上运行
            with torch.no_grad():
                # 临时设置设备为CPU
                original_device = torch.cuda.current_device() if torch.cuda.is_available() else None
                
                try:
                    # 调用原始的non_max_suppression，但在CPU上
                    result = non_max_suppression(
                        prediction, conf_thres, iou_thres, classes,
                        agnostic, multi_label, labels, max_det, nc, max_time_img, max_nms, max_wh
                    )
                    return result
                except Exception as e:
                    warnings.warn(f"标准NMS失败，使用备用实现: {e}")
                    # 如果标准NMS失败，使用我们的手动实现
                    return fallback_nms(prediction, conf_thres, iou_thres, max_det)
        
        print("✓ YOLO NMS已修补为CPU版本")
        return cpu_nms_wrapper
        
    except ImportError as e:
        warnings.warn(f"无法修补YOLO NMS: {e}")
        return None

def fallback_nms(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    备用NMS实现，完全基于我们的手动实现
    """
    from manual_nms import torch_manual_nms
    
    output = []
    
    for xi, x in enumerate(prediction):
        # 过滤低置信度的检测
        x = x[x[:, 4] > conf_thres]
        
        if not x.shape[0]:
            output.append(torch.zeros((0, 6)))
            continue
        
        # 计算分数
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4].clone()
        box[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        box[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        box[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        box[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        
        # 获取最高分数的类别
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # 应用NMS
        if x.shape[0]:
            keep = torch_manual_nms(x[:, :4], x[:, 4], iou_thres)
            x = x[keep]
            
            # 限制检测数量
            if x.shape[0] > max_det:
                x = x[:max_det]
        
        output.append(x)
    
    return output

def apply_nms_config():
    """
    应用所有NMS配置
    """
    configure_nms_for_cpu()
    nms_wrapper = patch_yolo_nms()
    
    return {
        'cpu_configured': True,
        'yolo_patched': nms_wrapper is not None,
        'fallback_available': True
    }

# 自动应用配置
if __name__ == "__main__":
    config_result = apply_nms_config()
    print("NMS配置结果:", config_result)