import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedMultiTaskLoss(nn.Module):
    """
    改进的多任务损失函数 - 完全符合tex文件中的数学描述
    
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


class StandardMultiTaskLoss(nn.Module):
    """
    标准多任务损失函数 - 不使用自适应权重的简化版本
    适用于对tex描述的基本实现
    """
    
    def __init__(self, lambda_temp=1.0, lambda_reg=0.001):
        super().__init__()
        self.lambda_temp = lambda_temp
        self.lambda_reg = lambda_reg
        
    def forward(self, yolo_loss, pred_temp, gt_temp, temp_head_params):
        """标准实现，使用均匀权重"""
        batch_size = pred_temp.size(0)
        
        # 标准MSE损失（相当于w_n = 1的情况）
        mse_loss = F.mse_loss(pred_temp, gt_temp)
        
        # L2正则化
        reg_loss = 0
        for param in temp_head_params:
            reg_loss += torch.norm(param, 2) ** 2
        
        # 温度损失
        temp_loss = mse_loss + self.lambda_reg * reg_loss
        
        # 总损失
        total_loss = yolo_loss + self.lambda_temp * temp_loss
        
        return {
            'total_loss': total_loss,
            'yolo_loss': yolo_loss,
            'temp_loss': temp_loss,
            'mse_loss': mse_loss,
            'reg_loss': reg_loss
        }


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建测试数据
    batch_size = 12  # 按照tex文件中的批次大小
    yolo_loss = torch.tensor(2.5, requires_grad=True)
    pred_temp = torch.randn(batch_size, requires_grad=True)
    gt_temp = torch.randn(batch_size)
    
    # 模拟温度预测头参数
    temp_head_params = [
        torch.randn(256, 1024, requires_grad=True),  # 权重矩阵
        torch.randn(256, requires_grad=True),        # 偏置
        torch.randn(1, 256, requires_grad=True),     # 输出层权重
        torch.randn(1, requires_grad=True)           # 输出层偏置
    ]
    
    # 测试改进的损失函数
    print("=== 改进的多任务损失函数测试 ===")
    improved_loss = ImprovedMultiTaskLoss(lambda_temp=1.0, lambda_reg=0.001)
    result = improved_loss(yolo_loss, pred_temp, gt_temp, temp_head_params)
    
    print(f"总损失: {result['total_loss'].item():.4f}")
    print(f"YOLO损失: {result['yolo_loss'].item():.4f}")
    print(f"温度损失: {result['temp_loss'].item():.4f}")
    print(f"加权MSE: {result['weighted_mse'].item():.4f}")
    print(f"正则化损失: {result['reg_loss'].item():.4f}")
    
    # 测试标准损失函数
    print("\n=== 标准多任务损失函数测试 ===")
    standard_loss = StandardMultiTaskLoss(lambda_temp=1.0, lambda_reg=0.001)
    result2 = standard_loss(yolo_loss, pred_temp, gt_temp, temp_head_params)
    
    print(f"总损失: {result2['total_loss'].item():.4f}")
    print(f"YOLO损失: {result2['yolo_loss'].item():.4f}")
    print(f"温度损失: {result2['temp_loss'].item():.4f}")
    print(f"MSE损失: {result2['mse_loss'].item():.4f}")
    print(f"正则化损失: {result2['reg_loss'].item():.4f}")
    
    # 显示损失函数信息
    print("\n=== 损失函数组件信息 ===")
    info = improved_loss.get_loss_components_info()
    for key, value in info.items():
        print(f"{key}: {value}")