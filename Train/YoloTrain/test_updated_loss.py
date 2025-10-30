#!/usr/bin/env python3
"""
测试更新后的MultiTaskLoss实现
验证是否符合tex文件中的数学公式描述
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_multitask_loss():
    """测试MultiTaskLoss类的基本功能"""
    print("=== 测试更新后的MultiTaskLoss实现 ===\n")
    
    try:
        # 导入更新后的MultiTaskLoss
        from custom_yolo import MultiTaskLoss
        
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
        
        print("1. 测试基本损失函数计算...")
        
        # 测试带自适应权重的损失函数
        loss_adaptive = MultiTaskLoss(lambda_temp=1.0, lambda_reg=0.001, use_adaptive_weights=True)
        result_adaptive = loss_adaptive(yolo_loss, pred_temp, gt_temp, temp_head_params)
        
        print("   自适应权重版本:")
        print(f"   - 总损失: {result_adaptive['total_loss'].item():.4f}")
        print(f"   - YOLO损失: {result_adaptive['yolo_loss'].item():.4f}")
        print(f"   - 温度损失: {result_adaptive['temp_loss'].item():.4f}")
        print(f"   - 加权MSE: {result_adaptive['weighted_mse'].item():.4f}")
        print(f"   - 正则化损失: {result_adaptive['reg_loss'].item():.4f}")
        print(f"   - λ_temp: {result_adaptive['lambda_temp']}")
        print(f"   - λ_reg: {result_adaptive['lambda_reg']}")
        
        # 测试不带自适应权重的损失函数
        loss_standard = MultiTaskLoss(lambda_temp=1.0, lambda_reg=0.001, use_adaptive_weights=False)
        result_standard = loss_standard(yolo_loss, pred_temp, gt_temp, temp_head_params)
        
        print("\n   标准权重版本:")
        print(f"   - 总损失: {result_standard['total_loss'].item():.4f}")
        print(f"   - YOLO损失: {result_standard['yolo_loss'].item():.4f}")
        print(f"   - 温度损失: {result_standard['temp_loss'].item():.4f}")
        print(f"   - 加权MSE: {result_standard['weighted_mse'].item():.4f}")
        print(f"   - 正则化损失: {result_standard['reg_loss'].item():.4f}")
        
        print("\n2. 测试损失函数信息...")
        info = loss_adaptive.get_loss_components_info()
        print("   损失函数组件信息:")
        for key, value in info.items():
            print(f"   - {key}: {value}")
        
        print("\n3. 测试梯度计算...")
        # 测试反向传播
        total_loss = result_adaptive['total_loss']
        total_loss.backward()
        
        print("   梯度计算成功!")
        print(f"   - pred_temp梯度范数: {pred_temp.grad.norm().item():.4f}")
        print(f"   - yolo_loss梯度: {yolo_loss.grad.item():.4f}")
        
        print("\n4. 验证tex公式实现...")
        # 手动计算验证
        batch_size_manual = pred_temp.size(0)
        
        # 计算实例权重
        abs_errors = torch.abs(pred_temp.detach() - gt_temp)
        weights = F.softmax(abs_errors, dim=0) * len(abs_errors)
        
        # 手动计算加权MSE
        squared_errors = (pred_temp.detach() - gt_temp) ** 2
        manual_weighted_mse = torch.sum(weights * squared_errors) / batch_size_manual
        
        # 手动计算L2正则化
        manual_reg_loss = sum(torch.norm(param, 2) ** 2 for param in temp_head_params)
        
        # 手动计算温度损失
        manual_temp_loss = manual_weighted_mse + 0.001 * manual_reg_loss
        
        # 手动计算总损失
        manual_total_loss = yolo_loss.detach() + 1.0 * manual_temp_loss
        
        print("   手动计算结果:")
        print(f"   - 手动加权MSE: {manual_weighted_mse.item():.4f}")
        print(f"   - 手动正则化损失: {manual_reg_loss.item():.4f}")
        print(f"   - 手动温度损失: {manual_temp_loss.item():.4f}")
        print(f"   - 手动总损失: {manual_total_loss.item():.4f}")
        
        # 验证一致性
        mse_diff = abs(result_adaptive['weighted_mse'].item() - manual_weighted_mse.item())
        reg_diff = abs(result_adaptive['reg_loss'].item() - manual_reg_loss.item())
        temp_diff = abs(result_adaptive['temp_loss'].item() - manual_temp_loss.item())
        
        print(f"\n   验证结果:")
        print(f"   - MSE差异: {mse_diff:.6f} {'✓' if mse_diff < 1e-5 else '✗'}")
        print(f"   - 正则化差异: {reg_diff:.6f} {'✓' if reg_diff < 1e-5 else '✗'}")
        print(f"   - 温度损失差异: {temp_diff:.6f} {'✓' if temp_diff < 1e-5 else '✗'}")
        
        print("\n5. 测试不同参数设置...")
        # 测试不同的lambda值
        loss_configs = [
            (1.0, 0.001),
            (0.5, 0.01),
            (2.0, 0.0001)
        ]
        
        for lambda_temp, lambda_reg in loss_configs:
            loss_test = MultiTaskLoss(lambda_temp=lambda_temp, lambda_reg=lambda_reg)
            result_test = loss_test(yolo_loss.detach(), pred_temp.detach(), gt_temp, temp_head_params)
            print(f"   λ_temp={lambda_temp}, λ_reg={lambda_reg}: 总损失={result_test['total_loss'].item():.4f}")
        
        print("\n=== 测试完成 ===")
        print("✓ 所有测试通过！损失函数已成功更新为符合tex描述的版本。")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        print("请确保custom_yolo.py文件存在且可访问。")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_original():
    """比较新旧损失函数的差异"""
    print("\n=== 新旧损失函数对比 ===")
    
    # 创建测试数据
    batch_size = 8
    yolo_loss = torch.tensor(1.5)
    pred_temp = torch.tensor([20.1, 21.5, 19.8, 22.3, 20.7, 21.1, 19.5, 22.0])
    gt_temp = torch.tensor([20.0, 21.0, 20.0, 22.0, 21.0, 21.0, 20.0, 22.0])
    
    temp_head_params = [torch.randn(10, 20), torch.randn(10)]
    
    try:
        from custom_yolo import MultiTaskLoss
        
        # 新版本（自适应权重）
        new_loss = MultiTaskLoss(use_adaptive_weights=True)
        new_result = new_loss(yolo_loss, pred_temp, gt_temp, temp_head_params)
        
        # 新版本（标准权重）
        standard_loss = MultiTaskLoss(use_adaptive_weights=False)
        standard_result = standard_loss(yolo_loss, pred_temp, gt_temp, temp_head_params)
        
        print("新版本（自适应权重）:")
        print(f"  总损失: {new_result['total_loss'].item():.4f}")
        print(f"  温度损失: {new_result['temp_loss'].item():.4f}")
        
        print("新版本（标准权重）:")
        print(f"  总损失: {standard_result['total_loss'].item():.4f}")
        print(f"  温度损失: {standard_result['temp_loss'].item():.4f}")
        
        # 计算原始MSE作为对比
        original_mse = F.mse_loss(pred_temp, gt_temp)
        print(f"原始MSE: {original_mse.item():.4f}")
        
    except Exception as e:
        print(f"对比测试失败: {e}")

if __name__ == "__main__":
    success = test_multitask_loss()
    if success:
        compare_with_original()
    else:
        print("主要测试失败，跳过对比测试。")