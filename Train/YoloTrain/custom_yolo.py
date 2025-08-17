from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils.instance import Instances
import json
from pathlib import Path
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import DEFAULT_CFG
from copy import copy

class CustomYOLODataset(YOLODataset):
    def get_labels(self):
        """Custom label loading to handle 6-column format."""
        cache_path = self.label_path.parent / f"{self.label_path.stem}.cache"
        try:
            cache = torch.load(str(cache_path))
            if cache.get("labels", None) is None:
                cache = self.cache_labels_custom(cache_path)
        except Exception:
            cache = self.cache_labels_custom(cache_path)

        return cache

    def cache_labels_custom(self, path):
        """Custom label caching to handle 6-column format."""
        x = {}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = "Scanning custom labels..."
        with open(path.parent / self.label_path.name, "r") as f:
            pbar = enumerate(f)
            for i, line in pbar:
                try:
                    # 验证标签格式
                    parts = line.strip().split()
                    if len(parts) == 6:  # 确保是6列格式
                        label_line = " ".join(parts[:5])  # 只使用前5列作为标准YOLO格式
                        temperature = float(parts[5])  # 保存温度信息
                        x[i] = dict(label=label_line, temperature=temperature)
                        nm += 1  # 计数有效标签
                    else:
                        ne += 1  # 计数错误标签
                        msgs.append(f"{self.im_files[i]}: ignoring invalid label format")
                except Exception as e:
                    ne += 1
                    msgs.append(f"{self.im_files[i]}: {e}")

        # 保存缓存
        x["hash"] = self.get_hash(self.label_path.parent)
        x["results"] = nf, nm, ne, nc, len(msgs)
        x["msgs"] = msgs
        x["version"] = self.cache_version
        torch.save(x, str(path))
        return x

class CustomValidator(DetectionValidator):
    def postprocess(self, preds):
        """Skip NMS completely by returning predictions as is."""
        return preds

class CustomYOLO(YOLO):
    def __init__(self, model='yolo11s.pt'):
        super().__init__(model)
        self.temp_data = None
        
    def _smart_load(self, key):
        """Override to return our custom validator."""
        if key == 'validator':
            return CustomValidator
        return super()._smart_load(key)

    def train(self, **kwargs):
        """Training with temperature data."""
        if 'data' in kwargs:
            dataset_path = Path(kwargs['data']).parent / 'dataset'
            self.load_temp_data(dataset_path)
        return super().train(**kwargs)

    def load_temp_data(self, dataset_path):
        """Load temperature data."""
        temp_file = Path(dataset_path) / 'temperature_data.json'
        if temp_file.exists():
            with open(temp_file, 'r') as f:
                self.temp_data = json.load(f)

    def _load_and_process_data(self, data):
        """Process data batch."""
        if self.temp_data is not None:
            img_files = data.get('im_file', [])
            if isinstance(img_files, (list, tuple)):
                temps = []
                for img_file in img_files:
                    img_name = Path(img_file).stem
                    split = 'train' if '/train/' in img_file else 'val'
                    temp = self.temp_data[split].get(img_name, 0.0)
                    temps.append(temp)
                data['temperatures'] = torch.tensor(temps, device=data['img'].device)
        return data
        
    def loss(self, batch):
        """Calculate loss."""
        loss_dict = super().loss(batch)
        if 'temperatures' in batch:
            pred_temp = self.model.output_temp
            temp_loss = nn.MSELoss()(pred_temp, batch['temperatures'])
            loss_dict['temp_loss'] = temp_loss
            loss_dict['loss'] += temp_loss
        return loss_dict

    def predict(self, source=None, stream=False, **kwargs):
        """Predict with temperature information."""
        results = super().predict(source=source, stream=stream, **kwargs)
        for result in results:
            if hasattr(result, 'boxes'):
                result.boxes.temp = self.model.output_temp
        return results

    @staticmethod
    def box_nms(boxes, scores, iou_threshold):
        """自定义NMS实现"""
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.int64, device=boxes.device)
        
        # 获取框的坐标
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # 计算框的面积
        areas = (x2 - x1) * (y2 - y1)
        
        # 按照分数排序
        _, order = scores.sort(0, descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            i = order[0]
            keep.append(i)
            
            # 计算IoU
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= iou_threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
        
        return torch.tensor(keep, dtype=torch.int64, device=boxes.device)

    def _apply_nms(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
        """使用自定义NMS处理预测结果"""
        bs = prediction.shape[0]
        max_det = 300

        # 对每个批次进行处理
        output = [torch.zeros((0, 6), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):
            # 应用置信度阈值
            x = x[x[:, 4] > conf_thres]

            # 如果没有检测框，继续下一个批次
            if not x.shape[0]:
                continue

            # 计算分数
            scores = x[:, 4]

            # 应用NMS
            i = self.box_nms(x[:, :4], scores, iou_thres)
            if i.shape[0] > max_det:
                i = i[:max_det]

            output[xi] = x[i]

        return output

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset with custom dataset class."""
        return build_yolo_dataset(
            self.args, 
            img_path, 
            batch, 
            self.data, 
            mode=mode, 
            rect=mode == "val", 
            stride=32
        ) 