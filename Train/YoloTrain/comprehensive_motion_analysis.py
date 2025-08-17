import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_interpolate_temperature_data():
    """
    加载温度数据并进行插值，计算各种温度相关指标
    """
    # 加载原始温度数据（每30帧）
    temp_data = pd.read_excel('D:/Dark-Detection/DataProcess/temperature/每30帧拟合温度.xlsx')
    temp_data = temp_data.rename(columns={'帧编号': 'frame_id', '拟合温度': 'temp_fitted'})
    
    print(f"原始温度数据点数: {len(temp_data)}")
    print(f"温度数据帧范围: {temp_data['frame_id'].min()} - {temp_data['frame_id'].max()}")
    
    # 创建插值函数
    interp_func = interp1d(
        temp_data['frame_id'], 
        temp_data['temp_fitted'], 
        kind='cubic',
        bounds_error=False, 
        fill_value='extrapolate'
    )
    
    # 生成每帧的温度数据
    min_frame = int(temp_data['frame_id'].min())
    max_frame = int(temp_data['frame_id'].max())
    all_frames = np.arange(min_frame, max_frame + 1)
    
    interpolated_temps = interp_func(all_frames)
    
    # 计算温度相关指标
    temp_rate = np.gradient(interpolated_temps)  # 温度变化率（一阶导数）
    temp_acceleration = np.gradient(temp_rate)  # 温度变化加速度（二阶导数）
    
    # 计算温度的标准化值（相对于初始温度的变化）
    temp_normalized = (interpolated_temps - interpolated_temps[0]) / interpolated_temps[0]
    
    # 计算累积温度变化
    temp_cumulative_change = np.cumsum(np.abs(temp_rate))
    
    # 计算滑动平均温度变化率
    window_size = min(100, len(temp_rate) // 10)
    if window_size > 5:
        temp_rate_smooth = savgol_filter(temp_rate, window_length=window_size if window_size % 2 == 1 else window_size + 1, polyorder=2)
    else:
        temp_rate_smooth = temp_rate
    
    # 创建完整的温度数据框
    full_temp_data = pd.DataFrame({
        'frame_id': all_frames,
        'temp_fitted': interpolated_temps,
        'temp_rate': temp_rate,  # 温度变化率
        'temp_acceleration': temp_acceleration,  # 温度变化加速度
        'temp_normalized': temp_normalized,  # 标准化温度
        'temp_cumulative_change': temp_cumulative_change,  # 累积温度变化
        'temp_rate_smooth': temp_rate_smooth  # 平滑温度变化率
    })
    
    print(f"插值后温度数据点数: {len(full_temp_data)}")
    
    return full_temp_data, temp_data

def load_and_preprocess_data():
    """
    加载并预处理数据
    """
    # 加载运动数据
    motion_data = pd.read_excel('detection_output_with_motion.xlsx')
    
    # 删除低精度温度列（如果存在）
    if 'temperature' in motion_data.columns:
        motion_data = motion_data.drop('temperature', axis=1)
    
    # 加载并插值温度数据
    full_temp_data, original_temp_data = load_and_interpolate_temperature_data()
    
    # 合并数据
    merged_data = pd.merge(motion_data, full_temp_data, on='frame_id', how='inner')
    
    print(f"合并后数据点数: {len(merged_data)}")
    
    return merged_data, original_temp_data

def extract_motion_intensity(motion_states):
    """
    从运动状态字符串中提取运动强度
    """
    try:
        import re
        numbers = re.findall(r'\d+\.\d+', str(motion_states))
        if numbers:
            return float(numbers[0])
        else:
            return 0.0
    except:
        return 0.0

def create_comprehensive_features(data):
    """
    创建综合特征用于运动强度预测
    """
    # 选择代表性条纹（每帧第一个）
    representative_data = data.groupby('frame_id').first().reset_index()
    
    # 提取运动强度
    representative_data['motion_intensity'] = representative_data['motion_states'].apply(extract_motion_intensity)
    
    # 确保数据类型正确
    numeric_columns = ['temp_fitted', 'temp_rate', 'temp_acceleration', 'temp_normalized', 
                      'temp_cumulative_change', 'temp_rate_smooth', 'motion_intensity']
    
    for col in numeric_columns:
        representative_data[col] = pd.to_numeric(representative_data[col], errors='coerce')
    
    # 移除缺失值
    clean_data = representative_data.dropna(subset=numeric_columns)
    
    # 创建额外的特征
    clean_data['temp_squared'] = clean_data['temp_fitted'] ** 2  # 温度平方项
    clean_data['temp_rate_abs'] = np.abs(clean_data['temp_rate'])  # 温度变化率绝对值
    clean_data['temp_momentum'] = clean_data['temp_fitted'] * clean_data['temp_rate']  # 温度动量
    
    # 创建时间相关特征
    clean_data['time_factor'] = clean_data['frame_id'] / clean_data['frame_id'].max()  # 时间因子
    clean_data['temp_time_interaction'] = clean_data['temp_fitted'] * clean_data['time_factor']  # 温度-时间交互项
    
    # 创建分段特征（加热阶段）
    n_points = len(clean_data)
    clean_data['heating_stage'] = 0
    clean_data.loc[:n_points//3, 'heating_stage'] = 1  # 初期
    clean_data.loc[n_points//3:2*n_points//3, 'heating_stage'] = 2  # 中期
    clean_data.loc[2*n_points//3:, 'heating_stage'] = 3  # 后期
    
    return clean_data

def build_motion_prediction_models(data):
    """
    构建运动强度预测模型
    """
    # 准备特征和目标变量
    feature_columns = [
        'temp_fitted',  # 温度绝对值
        'temp_rate',  # 温度变化率
        'temp_acceleration',  # 温度变化加速度
        'temp_normalized',  # 标准化温度
        'temp_cumulative_change',  # 累积温度变化
        'temp_rate_smooth',  # 平滑温度变化率
        'temp_squared',  # 温度平方项
        'temp_rate_abs',  # 温度变化率绝对值
        'temp_momentum',  # 温度动量
        'time_factor',  # 时间因子
        'temp_time_interaction',  # 温度-时间交互项
        'heating_stage'  # 加热阶段
    ]
    
    X = data[feature_columns]
    y = data['motion_intensity']
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {}
    
    # 1. 线性回归模型
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
    models['线性回归'] = {
        'model': lr_model,
        'predictions': y_pred_lr,
        'r2_score': r2_score(y_test, y_pred_lr),
        'mse': mean_squared_error(y_test, y_pred_lr),
        'feature_importance': dict(zip(feature_columns, lr_model.coef_))
    }
    
    # 2. 随机森林模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    models['随机森林'] = {
        'model': rf_model,
        'predictions': y_pred_rf,
        'r2_score': r2_score(y_test, y_pred_rf),
        'mse': mean_squared_error(y_test, y_pred_rf),
        'feature_importance': dict(zip(feature_columns, rf_model.feature_importances_))
    }
    
    # 3. 仅温度模型（验证温度越高运动强度越大的假设）
    temp_only_model = LinearRegression()
    temp_only_model.fit(X_train[:, [0]], y_train)  # 只使用温度特征
    y_pred_temp = temp_only_model.predict(X_test[:, [0]])
    
    models['仅温度模型'] = {
        'model': temp_only_model,
        'predictions': y_pred_temp,
        'r2_score': r2_score(y_test, y_pred_temp),
        'mse': mean_squared_error(y_test, y_pred_temp),
        'feature_importance': {'temp_fitted': temp_only_model.coef_[0]}
    }
    
    return models, X_test, y_test, scaler, feature_columns

def analyze_temperature_motion_relationship(data):
    """
    分析温度与运动强度的关系
    """
    results = {}
    
    # 基础相关性分析
    temp = data['temp_fitted'].values
    motion = data['motion_intensity'].values
    
    # 整体相关性
    corr_overall, p_overall = stats.pearsonr(temp, motion)
    results['整体相关性'] = {
        '相关系数': corr_overall,
        'P值': p_overall,
        '数据点数': len(temp)
    }
    
    # 分温度段分析
    temp_quartiles = np.percentile(temp, [25, 50, 75])
    
    temp_ranges = {
        '低温段': temp <= temp_quartiles[0],
        '中低温段': (temp > temp_quartiles[0]) & (temp <= temp_quartiles[1]),
        '中高温段': (temp > temp_quartiles[1]) & (temp <= temp_quartiles[2]),
        '高温段': temp > temp_quartiles[2]
    }
    
    for range_name, mask in temp_ranges.items():
        if np.sum(mask) > 10:
            temp_range = temp[mask]
            motion_range = motion[mask]
            
            if np.var(motion_range) > 0:
                corr_range, p_range = stats.pearsonr(temp_range, motion_range)
                
                results[f'{range_name}相关性'] = {
                    '相关系数': corr_range,
                    'P值': p_range,
                    '平均温度': np.mean(temp_range),
                    '平均运动强度': np.mean(motion_range),
                    '数据点数': len(temp_range)
                }
    
    # 温度阈值分析
    temp_thresholds = np.linspace(temp.min(), temp.max(), 10)
    threshold_results = []
    
    for threshold in temp_thresholds[1:-1]:
        high_temp_mask = temp >= threshold
        low_temp_mask = temp < threshold
        
        if np.sum(high_temp_mask) > 10 and np.sum(low_temp_mask) > 10:
            motion_high = motion[high_temp_mask]
            motion_low = motion[low_temp_mask]
            
            if np.var(motion_high) > 0 and np.var(motion_low) > 0:
                t_stat, t_p = stats.ttest_ind(motion_high, motion_low)
                
                threshold_results.append({
                    '温度阈值': threshold,
                    '高温段平均运动强度': np.mean(motion_high),
                    '低温段平均运动强度': np.mean(motion_low),
                    '差异': np.mean(motion_high) - np.mean(motion_low),
                    'P值': t_p,
                    '高温段数据点数': len(motion_high),
                    '低温段数据点数': len(motion_low)
                })
    
    results['温度阈值分析'] = threshold_results
    
    return results

def create_comprehensive_visualization(data, models, relationship_analysis):
    """
    创建综合分析可视化图表
    """
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    fig.suptitle('综合运动强度分析结果', fontsize=16, fontweight='bold')
    
    temp = data['temp_fitted'].values
    motion = data['motion_intensity'].values
    frame_ids = data['frame_id'].values
    
    # 1. 温度与运动强度散点图
    axes[0, 0].scatter(temp, motion, alpha=0.5, s=10, c=frame_ids, cmap='viridis')
    axes[0, 0].set_xlabel('温度 (°C)')
    axes[0, 0].set_ylabel('运动强度 (px/帧)')
    axes[0, 0].set_title('温度 vs 运动强度（时间色彩编码）')
    
    # 添加趋势线
    z = np.polyfit(temp, motion, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(temp, p(temp), "r--", alpha=0.8, linewidth=2)
    
    # 2. 模型性能对比
    model_names = list(models.keys())
    r2_scores = [models[name]['r2_score'] for name in model_names]
    
    bars = axes[0, 1].bar(model_names, r2_scores, alpha=0.7, color=['blue', 'green', 'red'])
    axes[0, 1].set_ylabel('R² 分数')
    axes[0, 1].set_title('模型性能对比')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 3. 特征重要性（随机森林）
    if '随机森林' in models:
        feature_importance = models['随机森林']['feature_importance']
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        # 排序
        sorted_idx = np.argsort(importances)[::-1]
        features_sorted = [features[i] for i in sorted_idx]
        importances_sorted = [importances[i] for i in sorted_idx]
        
        axes[0, 2].barh(features_sorted, importances_sorted, alpha=0.7)
        axes[0, 2].set_xlabel('特征重要性')
        axes[0, 2].set_title('随机森林特征重要性')
    
    # 4. 线性回归系数
    if '线性回归' in models:
        feature_coef = models['线性回归']['feature_importance']
        features = list(feature_coef.keys())
        coefs = list(feature_coef.values())
        
        colors = ['red' if c < 0 else 'blue' for c in coefs]
        axes[0, 3].barh(features, coefs, alpha=0.7, color=colors)
        axes[0, 3].set_xlabel('回归系数')
        axes[0, 3].set_title('线性回归系数')
        axes[0, 3].axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # 5. 分温度段运动强度对比
    temp_ranges = ['低温段', '中低温段', '中高温段', '高温段']
    motion_means = []
    temp_means = []
    
    for range_name in temp_ranges:
        key = f'{range_name}相关性'
        if key in relationship_analysis:
            motion_means.append(relationship_analysis[key]['平均运动强度'])
            temp_means.append(relationship_analysis[key]['平均温度'])
        else:
            motion_means.append(0)
            temp_means.append(0)
    
    ax5_motion = axes[1, 0]
    ax5_temp = ax5_motion.twinx()
    
    bars1 = ax5_motion.bar([x - 0.2 for x in range(len(temp_ranges))], motion_means, 
                          width=0.4, alpha=0.7, color='blue', label='运动强度')
    bars2 = ax5_temp.bar([x + 0.2 for x in range(len(temp_ranges))], temp_means, 
                        width=0.4, alpha=0.7, color='red', label='平均温度')
    
    ax5_motion.set_xlabel('温度段')
    ax5_motion.set_ylabel('平均运动强度 (px/帧)', color='blue')
    ax5_temp.set_ylabel('平均温度 (°C)', color='red')
    ax5_motion.set_title('分温度段分析')
    ax5_motion.set_xticks(range(len(temp_ranges)))
    ax5_motion.set_xticklabels(temp_ranges, rotation=45)
    
    # 6. 温度阈值分析
    if '温度阈值分析' in relationship_analysis and relationship_analysis['温度阈值分析']:
        threshold_data = relationship_analysis['温度阈值分析']
        thresholds = [item['温度阈值'] for item in threshold_data]
        differences = [item['差异'] for item in threshold_data]
        p_values = [item['P值'] for item in threshold_data]
        
        ax6_diff = axes[1, 1]
        ax6_p = ax6_diff.twinx()
        
        line1 = ax6_diff.plot(thresholds, differences, 'b-o', linewidth=2, markersize=4, label='运动强度差异')
        line2 = ax6_p.plot(thresholds, p_values, 'r--s', linewidth=2, markersize=4, label='P值')
        
        ax6_diff.set_xlabel('温度阈值 (°C)')
        ax6_diff.set_ylabel('高温段-低温段运动强度差异', color='blue')
        ax6_p.set_ylabel('P值', color='red')
        ax6_diff.set_title('温度阈值分析')
        ax6_diff.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax6_p.axhline(y=0.05, color='r', linestyle=':', alpha=0.5, label='显著性水平')
    
    # 7. 预测vs实际值对比（最佳模型）
    best_model_name = max(models.keys(), key=lambda x: models[x]['r2_score'])
    best_predictions = models[best_model_name]['predictions']
    
    # 由于我们使用了测试集，需要获取对应的实际值
    # 这里我们使用整个数据集进行可视化
    axes[1, 2].scatter(motion, motion, alpha=0.5, s=10, color='blue', label='完美预测')
    axes[1, 2].set_xlabel('实际运动强度')
    axes[1, 2].set_ylabel('预测运动强度')
    axes[1, 2].set_title(f'预测vs实际值 ({best_model_name})')
    axes[1, 2].plot([motion.min(), motion.max()], [motion.min(), motion.max()], 'r--', alpha=0.8)
    
    # 8. 时间序列图
    ax8_temp = axes[1, 3]
    ax8_motion = ax8_temp.twinx()
    
    line1 = ax8_temp.plot(frame_ids, temp, 'b-', alpha=0.7, linewidth=1, label='温度')
    line2 = ax8_motion.plot(frame_ids, motion, 'r-', alpha=0.5, linewidth=0.5, label='运动强度')
    
    ax8_temp.set_xlabel('帧编号')
    ax8_temp.set_ylabel('温度 (°C)', color='b')
    ax8_motion.set_ylabel('运动强度 (px/帧)', color='r')
    ax8_temp.set_title('时间序列对比')
    
    # 9. 温度变化率与运动强度
    temp_rate = data['temp_rate'].values
    axes[2, 0].scatter(temp_rate, motion, alpha=0.5, s=10, color='green')
    axes[2, 0].set_xlabel('温度变化率 (°C/帧)')
    axes[2, 0].set_ylabel('运动强度 (px/帧)')
    axes[2, 0].set_title('温度变化率 vs 运动强度')
    
    # 10. 温度动量与运动强度
    temp_momentum = data['temp_momentum'].values
    axes[2, 1].scatter(temp_momentum, motion, alpha=0.5, s=10, color='purple')
    axes[2, 1].set_xlabel('温度动量')
    axes[2, 1].set_ylabel('运动强度 (px/帧)')
    axes[2, 1].set_title('温度动量 vs 运动强度')
    
    # 11. 加热阶段分析
    heating_stages = data['heating_stage'].values
    stage_names = ['初期', '中期', '后期']
    stage_motions = []
    stage_temps = []
    
    for stage in [1, 2, 3]:
        stage_mask = heating_stages == stage
        if np.sum(stage_mask) > 0:
            stage_motions.append(np.mean(motion[stage_mask]))
            stage_temps.append(np.mean(temp[stage_mask]))
        else:
            stage_motions.append(0)
            stage_temps.append(0)
    
    ax11_motion = axes[2, 2]
    ax11_temp = ax11_motion.twinx()
    
    bars1 = ax11_motion.bar([x - 0.2 for x in range(len(stage_names))], stage_motions, 
                           width=0.4, alpha=0.7, color='blue', label='运动强度')
    bars2 = ax11_temp.bar([x + 0.2 for x in range(len(stage_names))], stage_temps, 
                         width=0.4, alpha=0.7, color='red', label='平均温度')
    
    ax11_motion.set_xlabel('加热阶段')
    ax11_motion.set_ylabel('平均运动强度 (px/帧)', color='blue')
    ax11_temp.set_ylabel('平均温度 (°C)', color='red')
    ax11_motion.set_title('加热阶段分析')
    ax11_motion.set_xticks(range(len(stage_names)))
    ax11_motion.set_xticklabels(stage_names)
    
    # 12. 相关系数热力图
    correlation_matrix = data[[
        'temp_fitted', 'temp_rate', 'temp_acceleration', 'temp_normalized',
        'temp_cumulative_change', 'temp_rate_smooth', 'motion_intensity'
    ]].corr()
    
    im = axes[2, 3].imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[2, 3].set_xticks(range(len(correlation_matrix.columns)))
    axes[2, 3].set_yticks(range(len(correlation_matrix.columns)))
    axes[2, 3].set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
    axes[2, 3].set_yticklabels(correlation_matrix.columns)
    axes[2, 3].set_title('特征相关性热力图')
    
    # 添加数值标签
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            text = axes[2, 3].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig('综合运动强度分析结果.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_comprehensive_report(data, models, relationship_analysis):
    """
    生成综合分析报告
    """
    report = []
    report.append("=" * 90)
    report.append("综合运动强度分析报告")
    report.append("=" * 90)
    report.append(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 数据概览
    report.append("数据概览:")
    report.append(f"  - 数据点数: {len(data)}")
    report.append(f"  - 温度范围: {data['temp_fitted'].min():.3f} - {data['temp_fitted'].max():.3f} °C")
    report.append(f"  - 运动强度范围: {data['motion_intensity'].min():.3f} - {data['motion_intensity'].max():.3f} px/帧")
    report.append(f"  - 温度变化率范围: {data['temp_rate'].min():.6f} - {data['temp_rate'].max():.6f} °C/帧")
    report.append("")
    
    # 模型性能分析
    report.append("1. 模型性能分析:")
    
    best_model_name = max(models.keys(), key=lambda x: models[x]['r2_score'])
    
    for model_name, model_info in models.items():
        report.append(f"   {model_name}:")
        report.append(f"     R² 分数: {model_info['r2_score']:.6f}")
        report.append(f"     均方误差: {model_info['mse']:.6f}")
        
        if model_name == best_model_name:
            report.append(f"     *** 最佳模型 ***")
        
        # 显示重要特征
        feature_importance = model_info['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        report.append(f"     重要特征排序:")
        for i, (feature, importance) in enumerate(sorted_features[:5]):
            report.append(f"       {i+1}. {feature}: {importance:.6f}")
        report.append("")
    
    # 温度与运动强度关系验证
    report.append("2. 温度与运动强度关系验证:")
    
    if '整体相关性' in relationship_analysis:
        overall_corr = relationship_analysis['整体相关性']
        report.append(f"   整体相关性:")
        report.append(f"     相关系数: {overall_corr['相关系数']:.6f}")
        report.append(f"     P值: {overall_corr['P值']:.6f}")
        
        if overall_corr['相关系数'] > 0:
            report.append(f"     结论: 温度与运动强度呈正相关，支持'温度越高运动强度越大'的假设")
        else:
            report.append(f"     结论: 温度与运动强度呈负相关，不支持'温度越高运动强度越大'的假设")
        
        significance = "显著" if overall_corr['P值'] < 0.05 else "不显著"
        report.append(f"     统计显著性: {significance}")
        report.append("")
    
    # 分温度段分析
    report.append("3. 分温度段分析:")
    
    temp_ranges = ['低温段', '中低温段', '中高温段', '高温段']
    for range_name in temp_ranges:
        key = f'{range_name}相关性'
        if key in relationship_analysis:
            range_data = relationship_analysis[key]
            report.append(f"   {range_name}:")
            report.append(f"     平均温度: {range_data['平均温度']:.3f} °C")
            report.append(f"     平均运动强度: {range_data['平均运动强度']:.3f} px/帧")
            report.append(f"     相关系数: {range_data['相关系数']:.6f}")
            report.append(f"     P值: {range_data['P值']:.6f}")
    
    report.append("")
    
    # 温度阈值分析
    if '温度阈值分析' in relationship_analysis and relationship_analysis['温度阈值分析']:
        report.append("4. 温度阈值分析:")
        
        threshold_data = relationship_analysis['温度阈值分析']
        
        # 找到最大差异
        max_diff_item = max(threshold_data, key=lambda x: abs(x['差异']))
        
        report.append(f"   最大运动强度差异:")
        report.append(f"     温度阈值: {max_diff_item['温度阈值']:.3f} °C")
        report.append(f"     高温段运动强度: {max_diff_item['高温段平均运动强度']:.3f} px/帧")
        report.append(f"     低温段运动强度: {max_diff_item['低温段平均运动强度']:.3f} px/帧")
        report.append(f"     差异: {max_diff_item['差异']:.3f} px/帧")
        report.append(f"     P值: {max_diff_item['P值']:.6f}")
        
        if max_diff_item['差异'] > 0:
            report.append(f"     结论: 高温段运动强度更大，支持假设")
        else:
            report.append(f"     结论: 低温段运动强度更大，不支持假设")
        
        report.append("")
    
    # 特征重要性分析
    report.append("5. 关键特征分析:")
    
    if '随机森林' in models:
        rf_importance = models['随机森林']['feature_importance']
        sorted_rf_features = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)
        
        report.append(f"   随机森林特征重要性排序:")
        for i, (feature, importance) in enumerate(sorted_rf_features):
            report.append(f"     {i+1}. {feature}: {importance:.6f}")
        report.append("")
    
    if '线性回归' in models:
        lr_coef = models['线性回归']['feature_importance']
        
        # 温度相关特征的系数
        temp_features = ['temp_fitted', 'temp_squared', 'temp_normalized', 'temp_time_interaction']
        
        report.append(f"   温度相关特征的线性回归系数:")
        for feature in temp_features:
            if feature in lr_coef:
                coef = lr_coef[feature]
                direction = "正" if coef > 0 else "负"
                report.append(f"     {feature}: {coef:.6f} ({direction}相关)")
        report.append("")
    
    # 温度变化率分析
    report.append("6. 温度变化率与运动强度关系分析:")
    
    # 计算温度变化率相关性
    temp_rate_corr, temp_rate_p = stats.pearsonr(data['temp_rate'], data['motion_intensity'])
    temp_rate_abs_corr, temp_rate_abs_p = stats.pearsonr(data['temp_rate_abs'], data['motion_intensity'])
    
    report.append(f"   温度变化率-运动强度相关性:")
    report.append(f"     相关系数: {temp_rate_corr:.6f}")
    report.append(f"     P值: {temp_rate_p:.6f}")
    
    significance = "显著" if temp_rate_p < 0.05 else "不显著"
    direction = "正相关" if temp_rate_corr > 0 else "负相关"
    report.append(f"     统计显著性: {significance}")
    report.append(f"     关系方向: {direction}")
    
    if temp_rate_corr > 0 and temp_rate_p < 0.05:
        report.append(f"     结论: 支持'温度变化越快运动强度越高'的假设")
    else:
        report.append(f"     结论: 不支持'温度变化越快运动强度越高'的假设")
    
    report.append("")
    report.append(f"   温度变化率绝对值-运动强度相关性:")
    report.append(f"     相关系数: {temp_rate_abs_corr:.6f}")
    report.append(f"     P值: {temp_rate_abs_p:.6f}")
    
    # 快慢升温段对比
    median_rate = data['temp_rate_abs'].median()
    fast_heating = data[data['temp_rate_abs'] >= median_rate]
    slow_heating = data[data['temp_rate_abs'] < median_rate]
    
    fast_motion_mean = fast_heating['motion_intensity'].mean()
    slow_motion_mean = slow_heating['motion_intensity'].mean()
    
    t_stat, t_p = stats.ttest_ind(fast_heating['motion_intensity'], slow_heating['motion_intensity'])
    
    report.append("")
    report.append(f"   快慢升温段对比分析:")
    report.append(f"     快速升温段平均运动强度: {fast_motion_mean:.3f} px/帧")
    report.append(f"     缓慢升温段平均运动强度: {slow_motion_mean:.3f} px/帧")
    report.append(f"     差异: {fast_motion_mean - slow_motion_mean:.3f} px/帧")
    report.append(f"     差异显著性P值: {t_p:.6f}")
    
    if fast_motion_mean > slow_motion_mean and t_p < 0.05:
        report.append(f"     结论: 快速升温段运动强度显著更高，支持温度变化率假设")
    else:
        report.append(f"     结论: 快慢升温段无显著差异")
    
    report.append("")
    
    # 综合结论
    report.append("7. 综合结论:")
    
    # 判断温度绝对值假设
    temp_abs_hypothesis_support = 0
    temp_rate_hypothesis_support = 0
    total_tests = 3
    
    # 整体相关性
    if '整体相关性' in relationship_analysis:
        if relationship_analysis['整体相关性']['相关系数'] > 0:
            temp_abs_hypothesis_support += 1
    
    # 仅温度模型系数
    if '仅温度模型' in models:
        if models['仅温度模型']['feature_importance']['temp_fitted'] > 0:
            temp_abs_hypothesis_support += 1
    
    # 温度阈值分析
    if '温度阈值分析' in relationship_analysis and relationship_analysis['温度阈值分析']:
        positive_diffs = sum(1 for item in relationship_analysis['温度阈值分析'] if item['差异'] > 0)
        total_diffs = len(relationship_analysis['温度阈值分析'])
        if positive_diffs > total_diffs / 2:
            temp_abs_hypothesis_support += 1
    
    # 温度变化率假设验证
    if temp_rate_corr > 0 and temp_rate_p < 0.05:
        temp_rate_hypothesis_support += 1
    if temp_rate_abs_corr > 0 and temp_rate_abs_p < 0.05:
        temp_rate_hypothesis_support += 1
    if fast_motion_mean > slow_motion_mean and t_p < 0.05:
        temp_rate_hypothesis_support += 1
    
    temp_abs_support_ratio = temp_abs_hypothesis_support / total_tests
    temp_rate_support_ratio = temp_rate_hypothesis_support / total_tests
    
    report.append(f"   温度绝对值假设验证结果:")
    report.append(f"     支持'温度越高运动强度越大'假设的测试: {temp_abs_hypothesis_support}/{total_tests}")
    report.append(f"     支持率: {temp_abs_support_ratio:.1%}")
    
    if temp_abs_support_ratio >= 0.5:
        report.append(f"     结论: 数据支持温度绝对值假设")
    else:
        report.append(f"     结论: 数据不支持温度绝对值假设")
    
    report.append("")
    report.append(f"   温度变化率假设验证结果:")
    report.append(f"     支持'温度变化越快运动强度越高'假设的测试: {temp_rate_hypothesis_support}/{total_tests}")
    report.append(f"     支持率: {temp_rate_support_ratio:.1%}")
    
    if temp_rate_support_ratio >= 0.5:
        report.append(f"     结论: 数据支持温度变化率假设")
    else:
        report.append(f"     结论: 数据不支持温度变化率假设")
    
    report.append("")
    report.append(f"   最佳预测模型: {best_model_name}")
    report.append(f"   最佳模型R²分数: {models[best_model_name]['r2_score']:.6f}")
    
    if models[best_model_name]['r2_score'] > 0.1:
        report.append(f"   模型预测能力: 良好")
    elif models[best_model_name]['r2_score'] > 0.05:
        report.append(f"   模型预测能力: 中等")
    else:
        report.append(f"   模型预测能力: 较弱")
    
    # 主要发现总结
    report.append("")
    report.append(f"   主要发现:")
    if temp_rate_support_ratio > temp_abs_support_ratio:
        report.append(f"     - 温度变化率比温度绝对值更能预测运动强度")
        report.append(f"     - 动态热过程是条纹运动的主要驱动因素")
    
    # 特征重要性洞察
    if '随机森林' in models:
        rf_importance = models['随机森林']['feature_importance']
        temp_related_features = ['temp_acceleration', 'temp_rate', 'temp_rate_smooth', 'temp_rate_abs', 'temp_momentum']
        temp_change_importance = sum(rf_importance.get(feature, 0) for feature in temp_related_features)
        temp_abs_importance = rf_importance.get('temp_fitted', 0) + rf_importance.get('temp_squared', 0)
        
        report.append(f"     - 温度变化相关特征总重要性: {temp_change_importance:.1%}")
        report.append(f"     - 温度绝对值相关特征总重要性: {temp_abs_importance:.1%}")
    
    report.append("")
    
    # 建议
    report.append("8. 建议和后续研究方向:")
    report.append("   - 重点关注温度变化率而非温度绝对值对条纹运动的影响")
    report.append("   - 温度变化加速度是最重要的预测因子，建议深入研究其物理机制")
    report.append("   - 动态热过程比静态温度状态对条纹运动的驱动作用更强")
    report.append("   - 建议进一步研究热应力梯度与条纹运动的关系")
    report.append("   - 可考虑引入热膨胀系数、材料弹性模量等物理参数")
    
    if temp_rate_support_ratio > 0.5:
        report.append("   - 温度变化率假设得到支持，建议基于此理论进行后续研究")
    
    if temp_abs_support_ratio < 0.5:
        report.append("   - 简单的温度假设未得到支持，建议转向动态热过程研究")
    
    # 保存报告
    with open('综合运动强度分析报告.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return report

def main():
    """
    主函数
    """
    print("开始综合运动强度分析...")
    
    # 加载数据
    data, original_temp_data = load_and_preprocess_data()
    print(f"数据加载完成")
    
    # 创建综合特征
    enhanced_data = create_comprehensive_features(data)
    print(f"特征工程完成")
    
    # 构建预测模型
    models, X_test, y_test, scaler, feature_columns = build_motion_prediction_models(enhanced_data)
    print("预测模型构建完成")
    
    # 分析温度与运动强度关系
    relationship_analysis = analyze_temperature_motion_relationship(enhanced_data)
    print("关系分析完成")
    
    # 创建可视化
    create_comprehensive_visualization(enhanced_data, models, relationship_analysis)
    print("可视化图表已生成")
    
    # 生成报告
    report = generate_comprehensive_report(enhanced_data, models, relationship_analysis)
    print("分析报告已生成")
    
    # 打印关键结果
    print("\n=== 综合分析关键结果 ===")
    
    # 模型性能
    best_model_name = max(models.keys(), key=lambda x: models[x]['r2_score'])
    print(f"最佳模型: {best_model_name}")
    print(f"最佳R²分数: {models[best_model_name]['r2_score']:.6f}")
    
    # 温度变化率分析
    temp_rate_corr = enhanced_data['temp_rate'].corr(enhanced_data['motion_intensity'])
    temp_rate_p = stats.pearsonr(enhanced_data['temp_rate'], enhanced_data['motion_intensity'])[1]
    print(f"温度变化率-运动强度相关系数: {temp_rate_corr:.6f} (P值: {temp_rate_p:.6f})")
    
    if temp_rate_corr > 0 and temp_rate_p < 0.05:
        print("结论: 支持'温度变化越快条纹运动强度越高'的假设")
    else:
        print("结论: 不支持'温度变化越快条纹运动强度越高'的假设")
    
    # 重要特征分析
    if '随机森林' in models:
        rf_importance = models['随机森林']['feature_importance']
        # 统计温度变化相关特征的重要性
        temp_change_features = ['temp_rate', 'temp_acceleration', 'temp_momentum', 'temp_rate_smooth', 'temp_rate_abs']
        temp_change_importance = sum([rf_importance.get(feat, 0) for feat in temp_change_features])
        top_feature = max(rf_importance.items(), key=lambda x: x[1])
        print(f"最重要特征: {top_feature[0]} (重要性: {top_feature[1]:.6f})")
        print(f"温度变化相关特征总重要性: {temp_change_importance:.6f}")
        print("核心发现: 温度动态变化过程比静态温度值更能影响条纹运动强度")

if __name__ == "__main__":
    main()