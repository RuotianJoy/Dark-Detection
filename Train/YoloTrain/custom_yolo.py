from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
# 注释掉torchvision导入以避免CUDA NMS问题
# import torchvision
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils.instance import Instances
import json
from pathlib import Path
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import DEFAULT_CFG
import math
from copy import copy
import numpy as np
# 导入手动NMS实现
from manual_nms import torch_manual_nms, manual_nms

class ThermalRegressionHead(nn.Module):
    """热回归头 - 实现论文中的温度预测模块"""
    def __init__(self, input_channels=1024, hidden_dim=256, out_min=0.0, out_max=60.0):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.hidden_layer = nn.Linear(input_channels, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.out_min = out_min
        self.out_max = out_max
        
    def set_output_range(self, out_min, out_max):
        self.out_min = float(out_min)
        self.out_max = float(out_max)
        
    def forward(self, features):
        pooled = self.global_avg_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        hidden = self.relu(self.hidden_layer(pooled))
        hidden = self.dropout(hidden)
        raw = self.output_layer(hidden)
        scaled = torch.sigmoid(raw) * (self.out_max - self.out_min) + self.out_min
        return scaled.squeeze(-1)

class CustomYOLODataset(YOLODataset):
    """自定义数据集类 - 处理6列标签格式"""
    def __init__(self, *args, **kwargs):
        # 确保data参数存在
        if 'data' not in kwargs:
            kwargs['data'] = {'channels': 3}  # 提供默认值
        super().__init__(*args, **kwargs)
        self.temperature_data = {}
        self._load_temperature_data()
    
    def _load_temperature_data(self):
        """加载温度数据"""
        if not hasattr(self, 'label_files') or self.label_files is None:
            return
            
        for i, label_file in enumerate(self.label_files):
            if Path(label_file).exists():
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        temperatures = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) == 6:  # 6列格式
                                temperatures.append(float(parts[5]))
                            elif len(parts) == 5:  # 标准5列格式，使用默认温度
                                temperatures.append(0.0)
                        if temperatures:
                            # 使用平均温度作为该图像的温度标签
                            self.temperature_data[i] = np.mean(temperatures)
                        else:
                            self.temperature_data[i] = 0.0
                except Exception as e:
                    print(f"加载温度数据失败 {label_file}: {e}")
                    self.temperature_data[i] = 0.0
            else:
                self.temperature_data[i] = 0.0
    
    def __getitem__(self, index):
        """获取数据项，包含温度信息"""
        item = super().__getitem__(index)
        
        # 添加温度标签
        temperature = self.temperature_data.get(index, 0.0)
        item['temperature'] = torch.tensor(temperature, dtype=torch.float32)
        
        return item

class MultiTaskLoss(nn.Module):
    """
    多任务损失函数 - 严格按照tex文件中的数学公式实现
    
    实现公式：
    L_total(Θ) = L_YOLO(Θ; D_det) + λ_temp * L_temp(Θ; D_therm)
    
    其中：
    L_temp(Θ) = (1/|B|) * Σ(n=1 to |B|) w_n * (T̂_n(Θ) - T_n^gt)² + λ_reg * ||Θ_temp||₂²
    """
    
    def __init__(self, lambda_temp=1.0, lambda_reg=0.001, use_adaptive_weights=True):
        super().__init__()
        self.lambda_temp = lambda_temp  # 温度损失权重，tex中设为1.0
        self.lambda_reg = lambda_reg    # L2正则化系数
        self.use_adaptive_weights = use_adaptive_weights
        
    def compute_instance_weights(self, pred_temp, gt_temp):
        """
        计算实例特定权重 w_n
        根据预测误差动态调整权重，误差大的样本权重更高
        """
        if not self.use_adaptive_weights:
            return torch.ones_like(pred_temp)
        
        # 计算每个样本的绝对误差
        abs_errors = torch.abs(pred_temp - gt_temp)
        
        # 使用softmax归一化权重，误差大的样本权重更高
        weights = F.softmax(abs_errors, dim=0)
        
        # 缩放权重使其均值为1，保持损失的数值稳定性
        weights = weights * len(weights)
        
        return weights.detach()  # 不参与梯度计算
    
    def thermal_regression_loss(self, pred_temp, gt_temp, temp_head_params, batch_size):
        """
        计算热回归损失，严格按照tex公式实现：
        L_temp(Θ) = (1/|B|) * Σ(n=1 to |B|) w_n * (T̂_n(Θ) - T_n^gt)² + λ_reg * ||Θ_temp||₂²
        """
        # 计算实例权重
        weights = self.compute_instance_weights(pred_temp, gt_temp)
        
        # 加权均方误差
        squared_errors = (pred_temp - gt_temp) ** 2
        weighted_mse = torch.sum(weights * squared_errors) / batch_size
        
        # L2正则化项
        reg_loss = 0
        for param in temp_head_params:
            reg_loss += torch.norm(param, 2) ** 2  # L2范数的平方
        
        # 总的温度回归损失
        temp_loss = weighted_mse + self.lambda_reg * reg_loss
        
        return temp_loss, weighted_mse, reg_loss
        
    def forward(self, yolo_loss, pred_temp, gt_temp, temp_head_params):
        """
        计算总损失函数
        
        Args:
            yolo_loss: YOLO检测损失 L_YOLO
            pred_temp: 预测温度 T̂_n(Θ)
            gt_temp: 真实温度 T_n^gt
            temp_head_params: 温度预测头参数 Θ_temp
            
        Returns:
            dict: 包含各项损失的字典
        """
        batch_size = pred_temp.size(0)
        
        # 计算热回归损失
        temp_loss, weighted_mse, reg_loss = self.thermal_regression_loss(
            pred_temp, gt_temp, temp_head_params, batch_size
        )
        
        # 总损失：L_total = L_YOLO + λ_temp * L_temp
        total_loss = yolo_loss + self.lambda_temp * temp_loss
        
        return {
            'total_loss': total_loss,
            'yolo_loss': yolo_loss,
            'temp_loss': temp_loss,
            'weighted_mse': weighted_mse,
            'reg_loss': reg_loss,
            'lambda_temp': self.lambda_temp,
            'lambda_reg': self.lambda_reg
        }
    
    def get_loss_components_info(self):
        """返回损失函数组件信息，用于调试和分析"""
        return {
            'formula': 'L_total = L_YOLO + λ_temp * (weighted_MSE + λ_reg * ||Θ_temp||₂²)',
            'lambda_temp': self.lambda_temp,
            'lambda_reg': self.lambda_reg,
            'adaptive_weights': self.use_adaptive_weights,
            'description': 'Hierarchical Multi-Objective Loss Function as described in methodology.tex'
        }

class CustomValidator(DetectionValidator):
    """自定义验证器"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp_predictions = []
        self.temp_targets = []
    
    def postprocess(self, preds):
        """后处理预测结果"""
        return preds

class CustomYOLO(YOLO):
    """自定义YOLO模型 - 实现双任务学习架构"""
    def __init__(self, model='yolo12s.pt'):
        super().__init__(model)
        self._patch_torchvision_nms()
        self._feature_hook_output = None
        self._feature_layer = None
        self._register_feature_hooks_all()
        # 使用字典存储自定义组件，避免属性访问问题
        self._custom_components = {
            'thermal_head': None,
            'multi_task_loss': MultiTaskLoss()
        }
        self._add_thermal_head()
    
    def _patch_torchvision_nms(self):
        try:
            import torchvision
            def _custom_nms(boxes, scores, iou_threshold):
                return torch_manual_nms(boxes, scores, iou_threshold)
            torchvision.ops.nms = _custom_nms
            print("✓ 使用自定义CPU NMS替代torchvision::nms")
        except Exception as e:
            print(f"⚠ 自定义NMS替换失败: {e}")
    
    def get_thermal_head(self):
        """获取热回归头"""
        return self._custom_components['thermal_head']
    
    def get_multi_task_loss(self):
        """获取多任务损失函数"""
        return self._custom_components['multi_task_loss']
    
    def has_thermal_head(self):
        """检查是否有热回归头"""
        return self._custom_components['thermal_head'] is not None
    
    def has_multi_task_loss(self):
        """检查是否有多任务损失函数"""
        return self._custom_components['multi_task_loss'] is not None
        
    def _add_thermal_head(self):
        """添加热回归头到模型 - 使用默认通道并在首次前向动态适配"""
        input_channels = 1024
        
        # 创建热回归头
        try:
            self._custom_components['thermal_head'] = ThermalRegressionHead(input_channels)
            device = next(self.model.parameters()).device
            self._custom_components['thermal_head'].to(device)
            print(f"✓ 成功创建热回归头，输入通道数: {input_channels} (动态适配)")
            return True
        except Exception as e:
            print(f"✗ 热回归头创建失败: {e}")
            return False
    
    def _smart_load(self, key):
        """重写以返回自定义组件"""
        if key == 'validator':
            return CustomValidator
        elif key == 'dataset':
            return CustomYOLODataset
        return super()._smart_load(key)
    
    def forward_with_temperature(self, x):
        """前向传播，同时预测检测和温度"""
        x_resized = self._resize_to_stride(x)
        self._feature_hook_output = None
        yolo_output = self.model(x_resized)
        
        # 获取backbone特征用于温度预测
        thermal_head = self.get_thermal_head()
        if thermal_head is not None:
            try:
                features = self._feature_hook_output
                if features is None or len(features.shape) != 4:
                    features = self._extract_backbone_features(x_resized)
                if features is not None and isinstance(features, torch.Tensor):
                    c = features.shape[1]
                    expected = thermal_head.hidden_layer.in_features
                    if c != expected:
                        device = next(self.model.parameters()).device
                        self._custom_components['thermal_head'] = ThermalRegressionHead(c).to(device)
                        thermal_head = self.get_thermal_head()
                temp_output = thermal_head(features)
                return yolo_output, temp_output
            except Exception as e:
                print(f"温度预测失败: {e}")
                return yolo_output, None
        
        return yolo_output, None

    def _resize_to_stride(self, x, stride=32):
        b, c, h, w = x.shape
        nh = math.ceil(h / stride) * stride
        nw = math.ceil(w / stride) * stride
        if nh == h and nw == w:
            return x
        return F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)

    def _register_feature_hook(self, layer):
        try:
            def _hook(module, inputs, output):
                self._feature_hook_output = output if isinstance(output, torch.Tensor) else output[0]
            layer.register_forward_hook(_hook)
        except Exception:
            pass
    
    def _register_feature_hooks_all(self):
        try:
            modules = getattr(self.model, 'model', None)
            if modules is None:
                return
            def _hook(module, inputs, output):
                out = output[0] if isinstance(output, (tuple, list)) else output
                if isinstance(out, torch.Tensor) and out.ndim == 4:
                    self._feature_hook_output = out
            for m in modules:
                try:
                    m.register_forward_hook(_hook)
                except Exception:
                    pass
        except Exception:
            pass

    def _extract_backbone_features(self, x):
        try:
            out = self._feature_hook_output
            if isinstance(out, torch.Tensor) and out.ndim == 4:
                return out
            return None
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def train_step(self, batch):
        """训练步骤 - 实现多任务学习"""
        images = batch['img']
        targets = batch.get('bboxes', None)
        temperatures = batch.get('temperature', None)
        
        # 前向传播
        yolo_pred, temp_pred = self.forward_with_temperature(images)
        
        # 计算YOLO损失
        yolo_loss = self.model.loss(yolo_pred, targets) if targets is not None else 0
        
        # 计算多任务损失
        multi_task_loss = self.get_multi_task_loss()
        thermal_head = self.get_thermal_head()
        
        if temp_pred is not None and temperatures is not None and multi_task_loss is not None:
            try:
                temperatures = temperatures.to(images.device).float()
            except Exception:
                pass
            loss_dict = multi_task_loss(
                yolo_loss, 
                temp_pred, 
                temperatures,
                thermal_head.parameters() if thermal_head else []
            )
            return loss_dict
        
        return {'total_loss': yolo_loss, 'yolo_loss': yolo_loss, 'temp_loss': 0, 'reg_loss': 0}
    
    def predict_with_temperature(self, source):
        """预测，包含温度信息"""
        results = super().predict(source, device='cpu', conf=0.25, iou=0.7)
        
        # 如果有热回归头，添加温度预测
        thermal_head = self.get_thermal_head()
        if thermal_head is not None:
            for result in results:
                if hasattr(result, 'orig_img'):
                    # 对单张图像进行温度预测
                    img_np = result.orig_img[:, :, ::-1].copy()
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
                    img_tensor = img_tensor / 255.0  # 归一化
                    device = next(self.model.parameters()).device
                    img_tensor = img_tensor.to(device)
                    
                    with torch.no_grad():
                        _, temp_pred = self.forward_with_temperature(img_tensor)
                        if temp_pred is not None:
                            result.temperature = temp_pred.item()
        
        return results

    def set_temperature_range(self, tmin, tmax):
        head = self.get_thermal_head()
        if head is not None:
            head.set_output_range(tmin, tmax)
    
    @staticmethod
    def box_nms(boxes, scores, iou_threshold):
        """
        改进的自定义NMS实现 - 完全避免CUDA兼容性问题
        强制在CPU上运行，使用手动实现的NMS算法
        """
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.int64, device=boxes.device)
        
        # 强制转换到CPU以避免CUDA问题
        original_device = boxes.device
        boxes_cpu = boxes.cpu()
        scores_cpu = scores.cpu()
        
        # 转换为xyxy格式 (如果输入是xywh格式)
        if boxes_cpu.shape[1] >= 4:
            # 假设输入格式是 [x_center, y_center, width, height]
            x1 = boxes_cpu[:, 0] - boxes_cpu[:, 2] / 2
            y1 = boxes_cpu[:, 1] - boxes_cpu[:, 3] / 2
            x2 = boxes_cpu[:, 0] + boxes_cpu[:, 2] / 2
            y2 = boxes_cpu[:, 1] + boxes_cpu[:, 3] / 2
            xyxy_boxes = torch.stack([x1, y1, x2, y2], dim=1)
        else:
            xyxy_boxes = boxes_cpu
        
        # 使用手动NMS实现
        try:
            keep_indices = torch_manual_nms(xyxy_boxes, scores_cpu, iou_threshold)
            # 将结果转换回原始设备
            return keep_indices.to(original_device)
        except Exception as e:
            print(f"手动NMS失败，使用备用实现: {e}")
            # 备用实现：简化的NMS
            return CustomYOLO._fallback_nms(xyxy_boxes, scores_cpu, iou_threshold).to(original_device)
    
    @staticmethod
    def _fallback_nms(boxes, scores, iou_threshold):
        """备用NMS实现，确保在任何情况下都能工作"""
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.int64)
        
        # 计算面积
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 按分数排序
        _, order = scores.sort(0, descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
                
            i = order[0]
            keep.append(i.item())
            
            # 计算IoU
            xx1 = torch.max(boxes[order[1:], 0], boxes[i, 0])
            yy1 = torch.max(boxes[order[1:], 1], boxes[i, 1])
            xx2 = torch.min(boxes[order[1:], 2], boxes[i, 2])
            yy2 = torch.min(boxes[order[1:], 3], boxes[i, 3])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            intersection = w * h
            
            union = areas[i] + areas[order[1:]] - intersection
            # 避免除零
            union = torch.clamp(union, min=1e-8)
            iou = intersection / union
            
            # 保留IoU小于阈值的框
            inds = torch.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return torch.tensor(keep, dtype=torch.int64)
    
    def build_dataset(self, img_path, mode="train", batch=None):
        """构建自定义数据集"""
        return CustomYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == 'train',
            hyp=self.model.args,
            rect=mode == 'val',
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix=f'{mode}: '
        )
