"""
损失函数测试脚本 - 检查MultiTaskLoss实现
"""

import torch
import torch.nn as nn
import numpy as np
from custom_yolo import MultiTaskLoss, ThermalRegressionHead

def test_multi_task_loss():
    """测试多任务损失函数"""
    print("=== 测试多任务损失函数 ===")
    
    # 初始化损失函数
    loss_fn = MultiTaskLoss(lambda_temp=1.0, lambda_reg=0.001)
    print(f"✓ 损失函数初始化成功")
    print(f"  - 温度损失权重: {loss_fn.lambda_temp}")
    print(f"  - 正则化权重: {loss_fn.lambda_reg}")
    
    # 创建模拟数据
    batch_size = 4
    
    # 模拟YOLO损失
    yolo_loss = torch.tensor(2.5, requires_grad=True)
    
    # 模拟温度预测和真实值
    pred_temp = torch.randn(batch_size, requires_grad=True)
    gt_temp = torch.randn(batch_size)
    
    # 创建模拟的温度预测头参数
    thermal_head = ThermalRegressionHead(input_channels=1024, hidden_dim=256)
    temp_head_params = list(thermal_head.parameters())
    
    print(f"\n模拟数据:")
    print(f"  - YOLO损失: {yolo_loss.item():.4f}")
    print(f"  - 预测温度: {pred_temp.detach().numpy()}")
    print(f"  - 真实温度: {gt_temp.numpy()}")
    print(f"  - 温度头参数数量: {len(temp_head_params)}")
    
    # 计算损失
    try:
        loss_dict = loss_fn(yolo_loss, pred_temp, gt_temp, temp_head_params)
        
        print(f"\n✓ 损失计算成功:")
        print(f"  - 总损失: {loss_dict['total_loss'].item():.4f}")
        print(f"  - YOLO损失: {loss_dict['yolo_loss'].item():.4f}")
        print(f"  - 温度损失: {loss_dict['temp_loss'].item():.4f}")
        print(f"  - 正则化损失: {loss_dict['reg_loss'].item():.4f}")
        
        # 检查梯度
        loss_dict['total_loss'].backward()
        print(f"\n✓ 反向传播成功")
        print(f"  - YOLO损失梯度: {yolo_loss.grad}")
        print(f"  - 预测温度梯度: {pred_temp.grad is not None}")
        
        return True
        
    except Exception as e:
        print(f"✗ 损失计算失败: {e}")
        return False

def test_loss_components():
    """测试损失函数各个组件"""
    print("\n=== 测试损失函数组件 ===")
    
    # 测试MSE损失
    mse_loss = nn.MSELoss()
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.1, 2.1, 2.9])
    mse_result = mse_loss(pred, target)
    print(f"✓ MSE损失测试: {mse_result.item():.4f}")
    
    # 测试L2正则化
    params = [torch.randn(10, 5), torch.randn(5)]
    reg_loss = sum(torch.norm(param, 2) for param in params)
    print(f"✓ L2正则化测试: {reg_loss.item():.4f}")
    
    # 测试损失权重
    lambda_temp = 1.0
    lambda_reg = 0.001
    yolo_loss = 2.0
    temp_loss = 0.5
    reg_loss = 10.0
    
    total_loss = yolo_loss + lambda_temp * temp_loss + lambda_reg * reg_loss
    print(f"✓ 加权损失测试: {total_loss:.4f}")
    print(f"  - YOLO: {yolo_loss}")
    print(f"  - 温度: {lambda_temp} × {temp_loss} = {lambda_temp * temp_loss}")
    print(f"  - 正则: {lambda_reg} × {reg_loss} = {lambda_reg * reg_loss}")

def test_loss_function_improvements():
    """测试损失函数的改进建议"""
    print("\n=== 损失函数改进建议 ===")
    
    improvements = [
        "1. 添加Focal Loss用于处理类别不平衡",
        "2. 添加SmoothL1Loss用于边界框回归",
        "3. 实现自适应权重调整机制",
        "4. 添加温度损失的鲁棒性处理",
        "5. 实现损失函数的可视化监控"
    ]
    
    for improvement in improvements:
        print(f"  {improvement}")
    
    # 示例：Focal Loss实现
    print(f"\n示例 - Focal Loss实现:")
    print("""
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            
        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            return focal_loss.mean()
    """)
    
    # 示例：SmoothL1Loss
    print(f"\n示例 - SmoothL1Loss用于边界框:")
    print("""
    smooth_l1_loss = nn.SmoothL1Loss()
    bbox_loss = smooth_l1_loss(pred_boxes, gt_boxes)
    """)

def analyze_current_implementation():
    """分析当前损失函数实现的优缺点"""
    print("\n=== 当前实现分析 ===")
    
    print("✅ 优点:")
    print("  1. 基本的多任务损失框架已建立")
    print("  2. 包含温度回归损失和L2正则化")
    print("  3. 支持梯度反向传播")
    print("  4. 返回详细的损失分解信息")
    
    print("\n❌ 不足:")
    print("  1. 缺少Focal Loss处理类别不平衡")
    print("  2. 没有SmoothL1Loss用于边界框回归")
    print("  3. 权重是固定的，缺少自适应调整")
    print("  4. 没有处理温度数据缺失的情况")
    print("  5. 缺少损失函数的验证和监控")
    
    print("\n🔧 建议改进:")
    print("  1. 实现更完整的YOLO损失函数")
    print("  2. 添加温度损失的鲁棒性处理")
    print("  3. 实现动态权重调整机制")
    print("  4. 添加损失函数的可视化")
    print("  5. 增加异常处理和边界检查")

def main():
    """主测试函数"""
    print("🔍 损失函数全面检查")
    print("=" * 50)
    
    # 测试多任务损失函数
    success = test_multi_task_loss()
    
    # 测试损失组件
    test_loss_components()
    
    # 测试改进建议
    test_loss_function_improvements()
    
    # 分析当前实现
    analyze_current_implementation()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 损失函数基本功能正常")
        print("💡 建议按照改进建议完善实现")
    else:
        print("❌ 损失函数存在问题，需要修复")
    
    return success

if __name__ == "__main__":
    main()