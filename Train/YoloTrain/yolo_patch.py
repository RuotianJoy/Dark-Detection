"""
YOLO补丁 - 彻底解决torchvision::nms CUDA兼容性问题
通过monkey patching替换所有可能的NMS调用
"""

import torch
import warnings
import os
import sys

def patch_torchvision_nms():
    """
    完全替换torchvision.ops.nms以避免CUDA问题
    """
    try:
        import torchvision.ops
        from manual_nms import torch_manual_nms
        
        def cpu_nms_replacement(boxes, scores, iou_threshold):
            """
            torchvision.ops.nms的CPU替代实现
            """
            # 强制转换到CPU
            boxes_cpu = boxes.cpu() if boxes.is_cuda else boxes
            scores_cpu = scores.cpu() if scores.is_cuda else scores
            
            # 使用我们的手动实现
            keep = torch_manual_nms(boxes_cpu, scores_cpu, iou_threshold)
            
            # 如果原始输入在CUDA上，将结果转回CUDA
            if boxes.is_cuda:
                keep = keep.cuda()
            
            return keep
        
        # 替换torchvision.ops.nms
        torchvision.ops.nms = cpu_nms_replacement
        print("✓ 已替换torchvision.ops.nms为CPU版本")
        
        # 如果存在batched_nms，也替换它
        if hasattr(torchvision.ops, 'batched_nms'):
            def cpu_batched_nms_replacement(boxes, scores, idxs, iou_threshold):
                """
                batched_nms的CPU替代实现
                """
                boxes_cpu = boxes.cpu() if boxes.is_cuda else boxes
                scores_cpu = scores.cpu() if scores.is_cuda else scores
                idxs_cpu = idxs.cpu() if idxs.is_cuda else idxs
                
                # 简化的batched NMS实现
                keep = []
                unique_idxs = torch.unique(idxs_cpu)
                
                for class_id in unique_idxs:
                    mask = idxs_cpu == class_id
                    class_boxes = boxes_cpu[mask]
                    class_scores = scores_cpu[mask]
                    
                    if len(class_boxes) > 0:
                        class_keep = torch_manual_nms(class_boxes, class_scores, iou_threshold)
                        # 转换回原始索引
                        original_indices = torch.where(mask)[0]
                        keep.extend(original_indices[class_keep].tolist())
                
                result = torch.tensor(keep, dtype=torch.long)
                if boxes.is_cuda:
                    result = result.cuda()
                
                return result
            
            torchvision.ops.batched_nms = cpu_batched_nms_replacement
            print("✓ 已替换torchvision.ops.batched_nms为CPU版本")
        
        return True
        
    except ImportError:
        print("⚠ torchvision未安装，跳过NMS替换")
        return False
    except Exception as e:
        print(f"⚠ NMS替换失败: {e}")
        return False

def patch_ultralytics_nms():
    """
    替换ultralytics内部的NMS调用
    """
    try:
        # 尝试导入ultralytics的NMS相关模块
        from ultralytics.utils import ops
        from manual_nms import torch_manual_nms
        
        # 保存原始函数
        if hasattr(ops, 'non_max_suppression'):
            original_nms = ops.non_max_suppression
            
            def patched_non_max_suppression(prediction, *args, **kwargs):
                """
                修补后的non_max_suppression，强制使用CPU，支持所有参数
                """
                # 强制将预测结果移到CPU
                if isinstance(prediction, torch.Tensor) and prediction.is_cuda:
                    prediction_cpu = prediction.cpu()
                    # 调用原始函数，但在CPU上
                    result = original_nms(prediction_cpu, *args, **kwargs)
                    # 将结果转回CUDA（如果需要）
                    if prediction.is_cuda and isinstance(result, (list, tuple)):
                        result = [r.cuda() if isinstance(r, torch.Tensor) else r for r in result]
                    elif prediction.is_cuda and isinstance(result, torch.Tensor):
                        result = result.cuda()
                    return result
                else:
                    return original_nms(prediction, *args, **kwargs)
            
            # 替换函数
            ops.non_max_suppression = patched_non_max_suppression
            print("✓ 已修补ultralytics.utils.ops.non_max_suppression")
            return True
            
    except ImportError as e:
        print(f"⚠ 无法导入ultralytics模块: {e}")
        return False
    except Exception as e:
        print(f"⚠ ultralytics NMS修补失败: {e}")
        return False

def set_environment_variables():
    """
    设置环境变量以强制使用CPU进行某些操作
    """
    # 强制CUDA操作同步，便于调试
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 尝试禁用CUDA优化的NMS
    os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1'
    
    # 设置PyTorch使用确定性算法
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    print("✓ 已设置环境变量以优化CUDA兼容性")

def apply_comprehensive_nms_patch():
    """
    应用全面的NMS补丁
    """
    print("=== 应用全面NMS补丁 ===")
    
    # 设置环境变量
    set_environment_variables()
    
    # 替换torchvision NMS
    torchvision_patched = patch_torchvision_nms()
    
    # 替换ultralytics NMS
    ultralytics_patched = patch_ultralytics_nms()
    
    # 设置PyTorch确定性模式
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("✓ 已启用PyTorch确定性算法模式")
    except Exception as e:
        print(f"⚠ 无法启用确定性算法: {e}")
    
    # 禁用CUDA的某些优化
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("✓ 已配置CUDNN为确定性模式")
    except Exception as e:
        print(f"⚠ CUDNN配置失败: {e}")
    
    result = {
        'torchvision_patched': torchvision_patched,
        'ultralytics_patched': ultralytics_patched,
        'environment_configured': True
    }
    
    print(f"补丁应用结果: {result}")
    return result

# 自动应用补丁
if __name__ == "__main__":
    apply_comprehensive_nms_patch()