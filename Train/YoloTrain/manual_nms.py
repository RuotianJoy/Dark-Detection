"""
手动NMS实现 - 解决torchvision::nms CUDA兼容性问题
基于用户提供的代码进行优化，支持PyTorch张量和NumPy数组
"""

import numpy as np
import torch


def manual_nms(detections, threshold=0.5, score_threshold=0.0):
    """
    手动实现的非极大值抑制 (NMS)
    
    Args:
        detections: 检测结果，格式为 [x1, y1, x2, y2, score] 或 [x, y, w, h, score]
                   可以是torch.Tensor或numpy.ndarray
        threshold: IoU阈值，默认0.5
        score_threshold: 分数阈值，默认0.0
    
    Returns:
        keep: 保留的检测框索引列表
    """
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()
    
    if len(detections) == 0:
        return []
    
    # 过滤低分数的检测
    scores = detections[:, 4]
    valid_indices = np.where(scores > score_threshold)[0]
    
    if len(valid_indices) == 0:
        return []
    
    detections = detections[valid_indices]
    scores = detections[:, 4]
    
    # 按分数降序排序
    ordered_indices = np.argsort(scores)[::-1]
    
    keep = []
    while ordered_indices.size > 0:
        # 选择分数最高的框
        idx = ordered_indices[0]
        keep.append(valid_indices[idx])
        
        if ordered_indices.size == 1:
            break
        
        # 计算与其他框的IoU
        current_box = detections[idx]
        remaining_boxes = detections[ordered_indices[1:]]
        
        ious = calculate_iou(current_box, remaining_boxes)
        
        # 保留IoU小于阈值的框
        filtered_indices = np.where(ious <= threshold)[0]
        ordered_indices = ordered_indices[filtered_indices + 1]
    
    return keep


def calculate_iou(box, boxes):
    """
    计算一个框与多个框的IoU
    
    Args:
        box: 单个框 [x1, y1, x2, y2, score]
        boxes: 多个框 [[x1, y1, x2, y2, score], ...]
    
    Returns:
        iou: IoU值数组
    """
    if len(boxes) == 0:
        return np.array([])
    
    # 提取坐标
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    # 计算交集面积
    intersection_width = np.maximum(x2 - x1, 0)
    intersection_height = np.maximum(y2 - y1, 0)
    intersection = intersection_width * intersection_height
    
    # 计算各框面积
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 计算并集面积
    union = area_box + area_boxes - intersection
    
    # 避免除零
    union = np.maximum(union, 1e-8)
    
    # 计算IoU
    iou = intersection / union
    
    return iou


def convert_xywh_to_xyxy(boxes):
    """
    将中心点格式 [x_center, y_center, width, height] 转换为 [x1, y1, x2, y2]
    
    Args:
        boxes: 框坐标，格式为 [x_center, y_center, width, height, score]
    
    Returns:
        converted_boxes: 转换后的框坐标 [x1, y1, x2, y2, score]
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.clone()
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = x_center - width/2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = y_center - height/2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]      # x2 = x1 + width
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]      # y2 = y1 + height
    else:
        boxes = boxes.copy()
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    
    return boxes


def torch_manual_nms(boxes, scores, iou_threshold=0.5, score_threshold=0.0):
    """
    PyTorch版本的手动NMS实现，完全在CPU上运行
    
    Args:
        boxes: 框坐标 [N, 4] (x1, y1, x2, y2)
        scores: 分数 [N]
        iou_threshold: IoU阈值
        score_threshold: 分数阈值
    
    Returns:
        keep: 保留的索引
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.int64, device=boxes.device)
    
    # 强制转换到CPU
    boxes = boxes.cpu()
    scores = scores.cpu()
    
    # 过滤低分数
    valid_mask = scores > score_threshold
    if not valid_mask.any():
        return torch.zeros(0, dtype=torch.int64, device=boxes.device)
    
    valid_indices = torch.where(valid_mask)[0]
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    
    # 计算面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 按分数排序
    _, order = scores.sort(0, descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(valid_indices[order.item()].item())
            break
            
        i = order[0]
        keep.append(valid_indices[i].item())
        
        # 计算IoU
        xx1 = torch.max(boxes[order[1:], 0], boxes[i, 0])
        yy1 = torch.max(boxes[order[1:], 1], boxes[i, 1])
        xx2 = torch.min(boxes[order[1:], 2], boxes[i, 2])
        yy2 = torch.min(boxes[order[1:], 3], boxes[i, 3])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        intersection = w * h
        
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / torch.clamp(union, min=1e-8)
        
        # 保留IoU小于阈值的框
        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return torch.tensor(keep, dtype=torch.int64)


# 测试代码
if __name__ == "__main__":
    # 测试示例
    print("测试手动NMS实现...")
    
    # 使用用户提供的示例数据
    detections = torch.Tensor([[10, 10, 50, 50, 0.9], 
                              [20, 20, 60, 60, 0.8], 
                              [30, 30, 70, 70, 0.95]])
    threshold = 0.5
    
    # 测试numpy版本
    keep_numpy = manual_nms(detections.numpy(), threshold)
    print(f"NumPy版本结果: {keep_numpy}")
    
    # 测试torch版本
    boxes = detections[:, :4]
    scores = detections[:, 4]
    keep_torch = torch_manual_nms(boxes, scores, threshold)
    print(f"PyTorch版本结果: {keep_torch.tolist()}")
    
    print("✓ 手动NMS实现测试完成")