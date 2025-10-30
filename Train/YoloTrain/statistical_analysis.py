import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class FringeMotionAnalyzer:
    """条纹运动与温度相关性分析器 - 实现论文中的统计分析方法"""
    
    def __init__(self):
        self.motion_data = []
        self.temperature_data = []
        self.correlation_results = {}
        self.rf_model = None
        self.feature_importance = {}
        
    def load_detection_data(self, detection_file):
        """加载检测结果数据"""
        if Path(detection_file).exists():
            df = pd.read_excel(detection_file)
            return df
        else:
            raise FileNotFoundError(f"检测结果文件不存在: {detection_file}")
    
    def calculate_motion_intensity(self, positions_history):
        """计算运动强度 - 实现论文中的L2范数公式"""
        motion_intensities = []
        
        for i in range(1, len(positions_history)):
            prev_pos = positions_history[i-1]
            curr_pos = positions_history[i]
            
            # 计算位移向量
            dx = curr_pos['x'] - prev_pos['x']
            dy = curr_pos['y'] - prev_pos['y']
            
            # L2范数计算运动强度
            motion_intensity = np.sqrt(dx**2 + dy**2)
            motion_intensities.append(motion_intensity)
            
        return motion_intensities
    
    def extract_thermal_features(self, temperatures, window_size=5):
        """提取热特征 - 包括静态和动态特征"""
        thermal_features = []
        
        for i in range(len(temperatures)):
            features = {}
            
            # 静态特征：绝对温度
            features['temperature'] = temperatures[i]
            features['temp_squared'] = temperatures[i] ** 2
            
            # 动态特征：温度梯度
            if i > 0:
                features['temp_gradient'] = temperatures[i] - temperatures[i-1]
            else:
                features['temp_gradient'] = 0
            
            # 二阶导数（加速度）
            if i > 1:
                features['temp_acceleration'] = (temperatures[i] - temperatures[i-1]) - (temperatures[i-1] - temperatures[i-2])
            else:
                features['temp_acceleration'] = 0
            
            # 滑动窗口统计特征
            start_idx = max(0, i - window_size + 1)
            window_temps = temperatures[start_idx:i+1]
            
            features['temp_mean'] = np.mean(window_temps)
            features['temp_std'] = np.std(window_temps) if len(window_temps) > 1 else 0
            features['temp_range'] = np.max(window_temps) - np.min(window_temps)
            
            thermal_features.append(features)
            
        return thermal_features
    
    def pearson_correlation_analysis(self, motion_intensities, thermal_features):
        """Pearson相关性分析 - 实现论文中的相关性计算"""
        results = {}
        
        # 提取特征向量
        temperatures = [f['temperature'] for f in thermal_features]
        temp_gradients = [f['temp_gradient'] for f in thermal_features]
        temp_accelerations = [f['temp_acceleration'] for f in thermal_features]
        
        # 确保数据长度一致
        min_len = min(len(motion_intensities), len(temperatures))
        motion_intensities = motion_intensities[:min_len]
        temperatures = temperatures[:min_len]
        temp_gradients = temp_gradients[:min_len]
        temp_accelerations = temp_accelerations[:min_len]
        
        # 计算Pearson相关系数
        # 温度梯度与运动强度的相关性
        corr_gradient, p_gradient = stats.pearsonr(temp_gradients, motion_intensities)
        results['gradient_motion'] = {
            'correlation': corr_gradient,
            'p_value': p_gradient,
            'significant': p_gradient < 0.05
        }
        
        # 绝对温度与运动强度的相关性
        corr_temp, p_temp = stats.pearsonr(temperatures, motion_intensities)
        results['temperature_motion'] = {
            'correlation': corr_temp,
            'p_value': p_temp,
            'significant': p_temp < 0.05
        }
        
        # 温度加速度与运动强度的相关性
        corr_accel, p_accel = stats.pearsonr(temp_accelerations, motion_intensities)
        results['acceleration_motion'] = {
            'correlation': corr_accel,
            'p_value': p_accel,
            'significant': p_accel < 0.05
        }
        
        self.correlation_results = results
        return results
    
    def random_forest_analysis(self, motion_intensities, thermal_features, test_size=0.2):
        """随机森林回归分析 - 实现论文中的集成学习方法"""
        # 准备特征矩阵
        feature_names = list(thermal_features[0].keys())
        X = np.array([[f[name] for name in feature_names] for f in thermal_features])
        y = np.array(motion_intensities)
        
        # 确保数据长度一致
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        # 分割训练和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 训练随机森林模型
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        self.rf_model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = self.rf_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # 特征重要性分析
        importance_scores = self.rf_model.feature_importances_
        self.feature_importance = dict(zip(feature_names, importance_scores))
        
        # 分类特征重要性
        dynamic_features = ['temp_gradient', 'temp_acceleration']
        static_features = ['temperature', 'temp_squared', 'temp_mean', 'temp_std', 'temp_range']
        
        dynamic_importance = sum([self.feature_importance.get(f, 0) for f in dynamic_features])
        static_importance = sum([self.feature_importance.get(f, 0) for f in static_features])
        
        results = {
            'r2_score': r2,
            'mse': mse,
            'feature_importance': self.feature_importance,
            'dynamic_importance_ratio': dynamic_importance / (dynamic_importance + static_importance),
            'static_importance_ratio': static_importance / (dynamic_importance + static_importance)
        }
        
        return results
    
    def generate_comprehensive_report(self, output_file="motion_analysis_report.txt"):
        """生成综合分析报告"""
        report = []
        report.append("=== 条纹运动与温度相关性分析报告 ===\n")
        
        # Pearson相关性分析结果
        report.append("1. Pearson相关性分析结果:")
        if self.correlation_results:
            for analysis_type, result in self.correlation_results.items():
                significance = "显著" if result['significant'] else "不显著"
                report.append(f"   {analysis_type}:")
                report.append(f"     相关系数: {result['correlation']:.6f}")
                report.append(f"     p值: {result['p_value']:.6f}")
                report.append(f"     统计显著性: {significance}")
                report.append("")
        
        # 随机森林分析结果
        if self.rf_model is not None:
            report.append("2. 随机森林回归分析结果:")
            rf_results = self.random_forest_analysis(self.motion_data, 
                                                   self.extract_thermal_features(self.temperature_data))
            report.append(f"   决定系数 (R²): {rf_results['r2_score']:.4f}")
            report.append(f"   均方误差 (MSE): {rf_results['mse']:.6f}")
            report.append(f"   动态特征重要性: {rf_results['dynamic_importance_ratio']*100:.2f}%")
            report.append(f"   静态特征重要性: {rf_results['static_importance_ratio']*100:.2f}%")
            report.append("")
            
            report.append("   特征重要性排序:")
            sorted_features = sorted(self.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features:
                report.append(f"     {feature}: {importance:.4f}")
            report.append("")
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"分析报告已保存到: {output_file}")
        return '\n'.join(report)
    
    def plot_correlation_analysis(self, motion_intensities, thermal_features, save_path="correlation_plots.png"):
        """绘制相关性分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 提取特征
        temperatures = [f['temperature'] for f in thermal_features]
        temp_gradients = [f['temp_gradient'] for f in thermal_features]
        
        # 确保数据长度一致
        min_len = min(len(motion_intensities), len(temperatures))
        motion_intensities = motion_intensities[:min_len]
        temperatures = temperatures[:min_len]
        temp_gradients = temp_gradients[:min_len]
        
        # 温度vs运动强度
        axes[0, 0].scatter(temperatures, motion_intensities, alpha=0.6)
        axes[0, 0].set_xlabel('Temperature (°C)')
        axes[0, 0].set_ylabel('Motion Intensity')
        axes[0, 0].set_title('Temperature vs Motion Intensity')
        
        # 温度梯度vs运动强度
        axes[0, 1].scatter(temp_gradients, motion_intensities, alpha=0.6)
        axes[0, 1].set_xlabel('Temperature Gradient (°C/frame)')
        axes[0, 1].set_ylabel('Motion Intensity')
        axes[0, 1].set_title('Temperature Gradient vs Motion Intensity')
        
        # 时间序列图
        time_indices = range(len(motion_intensities))
        axes[1, 0].plot(time_indices, motion_intensities, label='Motion Intensity', alpha=0.7)
        ax_temp = axes[1, 0].twinx()
        ax_temp.plot(time_indices, temperatures, 'r-', label='Temperature', alpha=0.7)
        axes[1, 0].set_xlabel('Frame Index')
        axes[1, 0].set_ylabel('Motion Intensity')
        ax_temp.set_ylabel('Temperature (°C)')
        axes[1, 0].set_title('Time Series: Motion vs Temperature')
        
        # 特征重要性图
        if self.feature_importance:
            features = list(self.feature_importance.keys())
            importances = list(self.feature_importance.values())
            axes[1, 1].barh(features, importances)
            axes[1, 1].set_xlabel('Feature Importance')
            axes[1, 1].set_title('Random Forest Feature Importance')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"相关性分析图表已保存到: {save_path}")

def main():
    """主函数 - 执行完整的统计分析流程"""
    analyzer = FringeMotionAnalyzer()
    
    # 这里需要根据实际的检测结果文件进行分析
    # 示例数据生成（实际使用时应该从检测结果中加载）
    np.random.seed(42)
    n_frames = 1000
    
    # 模拟温度数据（基于论文中的温度变化模式）
    temperatures = []
    for i in range(n_frames):
        if i < 270:  # 前27000帧的模拟
            temp = 30 + 10 * np.sin(i * 0.01) + np.random.normal(0, 0.5)
        else:  # 后续帧的线性增长
            temp = 43.8 + 0.1 * (i - 270) + np.random.normal(0, 0.2)
        temperatures.append(temp)
    
    # 模拟运动强度数据（与温度梯度相关）
    motion_intensities = []
    for i in range(1, len(temperatures)):
        temp_gradient = temperatures[i] - temperatures[i-1]
        # 运动强度与温度梯度正相关，与绝对温度负相关
        motion = abs(temp_gradient) * 2 + np.random.normal(0, 0.5) - temperatures[i] * 0.01
        motion = max(0, motion)  # 确保非负
        motion_intensities.append(motion)
    
    # 提取热特征
    thermal_features = analyzer.extract_thermal_features(temperatures)
    
    # 执行Pearson相关性分析
    correlation_results = analyzer.pearson_correlation_analysis(motion_intensities, thermal_features[1:])
    
    # 执行随机森林分析
    rf_results = analyzer.random_forest_analysis(motion_intensities, thermal_features[1:])
    
    # 生成报告
    report = analyzer.generate_comprehensive_report()
    print(report)
    
    # 绘制分析图表
    analyzer.plot_correlation_analysis(motion_intensities, thermal_features[1:])

if __name__ == "__main__":
    main()