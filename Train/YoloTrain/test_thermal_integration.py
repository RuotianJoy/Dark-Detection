#!/usr/bin/env python3
"""
测试CustomYOLO中热回归头的集成情况
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_thermal_integration():
    """测试热回归头集成"""
    print("=" * 60)
    print("CustomYOLO 热回归头集成测试")
    print("=" * 60)
    
    try:
        # 导入必要的模块
        from custom_yolo import CustomYOLO, ThermalRegressionHead
        import torch
        
        print("✓ 成功导入模块")
        
        # 1. 测试CustomYOLO初始化
        print("\n1. 测试CustomYOLO初始化...")
        try:
            # 使用较小的模型进行测试
            model = CustomYOLO('yolo11n.pt')  # 使用nano版本
            print("✓ CustomYOLO初始化成功")
        except Exception as e:
            print(f"✗ CustomYOLO初始化失败: {e}")
            return False
        
        # 2. 检查热回归头是否存在
        print("\n2. 检查热回归头...")
        has_thermal = model.has_thermal_head()
        thermal_head = model.get_thermal_head()
        
        print(f"   - 是否有热回归头: {has_thermal}")
        print(f"   - 热回归头对象: {type(thermal_head).__name__ if thermal_head else 'None'}")
        
        if thermal_head:
            print(f"   - 热回归头参数数量: {sum(p.numel() for p in thermal_head.parameters())}")
            print("✓ 热回归头存在且可访问")
        else:
            print("✗ 热回归头不存在")
            return False
        
        # 3. 测试前向传播
        print("\n3. 测试前向传播...")
        try:
            # 创建测试输入
            batch_size = 2
            test_input = torch.randn(batch_size, 3, 640, 640)
            
            # 测试forward_with_temperature
            yolo_output, temp_output = model.forward_with_temperature(test_input)
            
            print(f"   - YOLO输出类型: {type(yolo_output)}")
            print(f"   - 温度输出类型: {type(temp_output)}")
            
            if temp_output is not None:
                print(f"   - 温度输出形状: {temp_output.shape}")
                print(f"   - 温度输出范围: [{temp_output.min().item():.3f}, {temp_output.max().item():.3f}]")
                print("✓ 前向传播成功，热回归头正常工作")
            else:
                print("✗ 温度输出为None，热回归头未正常工作")
                return False
                
        except Exception as e:
            print(f"✗ 前向传播失败: {e}")
            return False
        
        # 4. 测试特征提取
        print("\n4. 测试特征提取...")
        try:
            features = model._extract_backbone_features(test_input)
            print(f"   - 提取的特征形状: {features.shape}")
            print(f"   - 特征维度: {len(features.shape)}D")
            
            if len(features.shape) == 4:  # [B, C, H, W]
                print(f"   - 批次大小: {features.shape[0]}")
                print(f"   - 通道数: {features.shape[1]}")
                print(f"   - 空间尺寸: {features.shape[2]}x{features.shape[3]}")
                print("✓ 特征提取成功")
            else:
                print("⚠ 特征维度不符合预期")
                
        except Exception as e:
            print(f"✗ 特征提取失败: {e}")
            return False
        
        # 5. 测试多任务损失
        print("\n5. 测试多任务损失...")
        try:
            multi_task_loss = model.get_multi_task_loss()
            print(f"   - 多任务损失对象: {type(multi_task_loss).__name__}")
            
            if multi_task_loss:
                # 模拟损失计算
                yolo_loss = torch.tensor(1.0)
                pred_temp = torch.randn(batch_size, 1)
                gt_temp = torch.randn(batch_size, 1)
                
                loss_dict = multi_task_loss(
                    yolo_loss, pred_temp, gt_temp, thermal_head.parameters()
                )
                
                print(f"   - 损失字典键: {list(loss_dict.keys())}")
                for key, value in loss_dict.items():
                    print(f"   - {key}: {value.item():.4f}")
                print("✓ 多任务损失计算成功")
            else:
                print("✗ 多任务损失对象不存在")
                return False
                
        except Exception as e:
            print(f"✗ 多任务损失测试失败: {e}")
            return False
        
        # 6. 集成度评估
        print("\n6. 集成度评估...")
        integration_score = 0
        max_score = 5
        
        # 检查各个组件
        if model.has_thermal_head():
            integration_score += 1
            print("   ✓ 热回归头已集成")
        
        if model.has_multi_task_loss():
            integration_score += 1
            print("   ✓ 多任务损失已集成")
        
        if temp_output is not None:
            integration_score += 1
            print("   ✓ 温度预测功能正常")
        
        if features.shape[1] > 0:  # 有效特征通道
            integration_score += 1
            print("   ✓ 特征提取功能正常")
        
        if 'total_loss' in loss_dict:
            integration_score += 1
            print("   ✓ 损失计算功能正常")
        
        integration_percentage = (integration_score / max_score) * 100
        print(f"\n   集成完成度: {integration_score}/{max_score} ({integration_percentage:.1f}%)")
        
        if integration_percentage >= 80:
            print("   🎉 热回归头集成良好！")
            return True
        elif integration_percentage >= 60:
            print("   ⚠ 热回归头基本集成，但需要改进")
            return True
        else:
            print("   ❌ 热回归头集成不完整")
            return False
            
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("请确保所有依赖模块都已正确安装")
        return False
    except Exception as e:
        print(f"✗ 测试过程中发生错误: {e}")
        return False

def analyze_integration_architecture():
    """分析集成架构"""
    print("\n" + "=" * 60)
    print("热回归头集成架构分析")
    print("=" * 60)
    
    analysis = {
        "集成方式": "外部附加式集成",
        "优点": [
            "不修改原始YOLO架构",
            "保持YOLO检测性能",
            "模块化设计，易于维护",
            "支持独立的温度预测"
        ],
        "缺点": [
            "特征提取可能不够深度融合",
            "需要额外的前向传播步骤",
            "可能存在特征不匹配问题"
        ],
        "改进建议": [
            "考虑在YOLO neck层集成温度分支",
            "使用特征金字塔网络(FPN)共享特征",
            "添加注意力机制增强特征融合",
            "优化特征提取的层级选择"
        ]
    }
    
    for key, value in analysis.items():
        print(f"\n{key}:")
        if isinstance(value, list):
            for item in value:
                print(f"  • {item}")
        else:
            print(f"  {value}")

if __name__ == "__main__":
    print("开始测试CustomYOLO热回归头集成...")
    
    success = test_thermal_integration()
    analyze_integration_architecture()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 测试完成：热回归头已成功集成到CustomYOLO中")
    else:
        print("❌ 测试完成：热回归头集成存在问题")
    print("=" * 60)