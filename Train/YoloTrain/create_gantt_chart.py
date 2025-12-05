#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目甘特图生成脚本
基于迈克耳孙干涉条纹运动强度分析研究项目
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_gantt_chart():
    """创建项目甘特图"""
    
    # Define project phases and timeline
    tasks = [
        {
            'task': 'Project Preparation',
            'start': '2025-11-01',
            'duration': 14,
            'color': '#FF6B6B',
            'subtasks': [
                {'name': 'Literature Review', 'duration': 7},
                {'name': 'Experimental Design', 'duration': 5},
                {'name': 'Equipment Preparation', 'duration': 2}
            ]
        },
        {
            'task': 'Data Collection',
            'start': '2025-11-15',
            'duration': 7,
            'color': '#45B7D1',
            'subtasks': [
                {'name': 'Parameter Adjustment', 'duration': 2},
                {'name': 'Fringe Image Acquisition', 'duration': 3},
                {'name': 'Temperature Data Recording', 'duration': 2}
            ]
        },
        {
            'task': 'Data Preprocessing',
            'start': '2025-11-22',
            'duration': 14,
            'color': '#96CEB4',
            'subtasks': [
                {'name': 'Image Preprocessing', 'duration': 5},
                {'name': 'Temperature Data Interpolation', 'duration': 4},
                {'name': 'Dataset Construction', 'duration': 5}
            ]
        },
        {
            'task': 'Experiment Iteration',
            'start': '2025-12-01',
            'duration': 14,
            'color': '#8FBC8F',
            'subtasks': [
                {'name': 'Hyperparameter Tuning', 'duration': 7},
                {'name': 'Baseline Training', 'duration': 7}
            ]
        },
        {
            'task': 'Data Annotation',
            'start': '2025-12-06',
            'duration': 25,
            'color': '#FFEAA7',
            'subtasks': [
                {'name': 'Fringe Region Annotation', 'duration': 18},
                {'name': 'Annotation Quality Check', 'duration': 4},
                {'name': 'Format Conversion', 'duration': 3}
            ]
        },
        {
            'task': 'YOLO Model Development',
            'start': '2025-12-31',
            'duration': 25,
            'color': '#DDA0DD',
            'subtasks': [
                {'name': 'Model Architecture Design', 'duration': 5},
                {'name': 'Training Environment Setup', 'duration': 3},
                {'name': 'Model Training', 'duration': 12},
                {'name': 'Model Optimization', 'duration': 5}
            ]
        },
        {
            'task': 'Model Validation & Testing',
            'start': '2026-01-25',
            'duration': 12,
            'color': '#FFB6C1',
            'subtasks': [
                {'name': 'Performance Evaluation', 'duration': 4},
                {'name': 'Detection Result Validation', 'duration': 4},
                {'name': 'Error Analysis', 'duration': 4}
            ]
        },
        {
            'task': 'Motion Intensity Analysis',
            'start': '2026-02-06',
            'duration': 16,
            'color': '#87CEEB',
            'subtasks': [
                {'name': 'Feature Engineering', 'duration': 5},
                {'name': 'Statistical Analysis', 'duration': 6},
                {'name': 'Machine Learning Modeling', 'duration': 5}
            ]
        },
        {
            'task': 'Result Analysis & Visualization',
            'start': '2026-02-22',
            'duration': 10,
            'color': '#F0E68C',
            'subtasks': [
                {'name': 'Data Visualization', 'duration': 4},
                {'name': 'Result Interpretation', 'duration': 3},
                {'name': 'Chart Creation', 'duration': 3}
            ]
        },
        {
            'task': 'Paper Writing',
            'start': '2026-03-04',
            'duration': 20,
            'color': '#FFA07A',
            'subtasks': [
                {'name': 'First Draft', 'duration': 12},
                {'name': 'Revision', 'duration': 5},
                {'name': 'Final Version', 'duration': 3}
            ]
        }
    ]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # 转换日期格式
    start_date = datetime.strptime('2025-11-01', '%Y-%m-%d')
    today_date = datetime.strptime('2025-12-01', '%Y-%m-%d')
    excel_data = {}
    try:
        df_actual = pd.read_excel('Project_Timeline_Details.xlsx')
        name_col = 'phase' if 'phase' in df_actual.columns else 'Phase'
        s_col = 'Start_date' if 'Start_date' in df_actual.columns else 'start_date'
        e_col = 'Estimated_End_date' if 'Estimated_End_date' in df_actual.columns else 'estimated_end_date'
        a_col = None
        if 'Actual_End_date' in df_actual.columns:
            a_col = 'Actual_End_date'
        elif 'actual_end_date' in df_actual.columns:
            a_col = 'actual_end_date'
        d_col = 'duration' if 'duration' in df_actual.columns else None
        st_col = 'status' if 'status' in df_actual.columns else None
        for _, row in df_actual.iterrows():
            name = str(row.get(name_col, '')).strip()
            if not name:
                continue
            excel_data[name] = {
                'start': str(row.get(s_col, '')).strip(),
                'estimated_end': str(row.get(e_col, '')).strip(),
                'actual_end': str(row.get(a_col, '')).strip() if a_col else '',
                'duration': int(row.get(d_col)) if d_col and pd.notnull(row.get(d_col)) else None,
                'status': str(row.get(st_col, '')).strip() if st_col else ''
            }
    except Exception:
        excel_data = {}
    
    # 绘制甘特图
    y_pos = np.arange(len(tasks))
    
    for i, task in enumerate(tasks):
        ed = excel_data.get(task['task'])
        start_str = ed['start'] if ed and ed.get('start') else task['start']
        if '/' in start_str:
            task_start = datetime.strptime(start_str, '%Y/%m/%d')
        else:
            task_start = datetime.strptime(start_str, '%Y-%m-%d')
        planned_duration = ed['duration'] if ed and ed.get('duration') else task['duration']
        task_end = task_start + timedelta(days=planned_duration)
        
        ax.barh(i, planned_duration, left=(task_start - start_date).days, 
                height=0.7, color=task['color'], alpha=0.3, 
                edgecolor='black', linewidth=0.5)
        
        completed_days = max(0, min((today_date - task_start).days, planned_duration))
        if completed_days > 0:
            ax.barh(i, completed_days, left=(task_start - start_date).days,
                    height=0.7, color=task['color'], alpha=0.9,
                    edgecolor='black', linewidth=0.5)
        
        ax.text((task_start - start_date).days + planned_duration/2, i, 
                f"{task['task']}\n({planned_duration} days)", 
                ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        if ed and ed.get('status'):
            s_text = ed.get('status')
            if s_text.lower() in ['completed', '已完成']:
                status_text = 'Completed'
                status_color = '#2E7D32'
            elif s_text.lower() in ['in progress', '进行中']:
                status_text = 'In progress'
                status_color = '#FB8C00'
            else:
                status_text = 'Unfinished'
                status_color = '#757575'
        else:
            if today_date >= task_end:
                status_text = 'Completed'
                status_color = '#2E7D32'
            elif today_date < task_start:
                status_text = 'Not started'
                status_color = '#757575'
            else:
                status_text = 'In progress'
                status_color = '#FB8C00'
        status_x = (task_start - start_date).days + planned_duration - 0.5
        ax.text(status_x, i, status_text, ha='right', va='center', fontsize=8, color=status_color, fontweight='bold')

        actual_str = ed.get('actual_end') if ed else None
        if actual_str:
            try:
                if '/' in actual_str:
                    actual_date = datetime.strptime(actual_str, '%Y/%m/%d')
                else:
                    actual_date = datetime.strptime(actual_str, '%Y-%m-%d')
                actual_duration = max(0, (actual_date - task_start).days)
                ax.barh(i-0.18, actual_duration, left=(task_start - start_date).days,
                        height=0.35, color='none', edgecolor='black', linewidth=1.2, hatch='////')
            except Exception:
                pass
    
    # 设置y轴
    ax.set_yticks(y_pos)
    ax.set_yticklabels([task['task'] for task in tasks])
    ax.invert_yaxis()
    
    # 设置x轴（日期）
    ax.set_xlim(0, 150)  # 5个月的项目周期
    
    # 创建日期标签
    date_labels = []
    date_positions = []
    for i in range(0, 151, 15):  # 每15天一个标签
        date = start_date + timedelta(days=i)
        date_labels.append(date.strftime('%m/%d'))
        date_positions.append(i)
    
    ax.set_xticks(date_positions)
    ax.set_xticklabels(date_labels, rotation=45)
    
    # 添加网格
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # 设置标题和标签
    ax.set_title('Michelson Interferometer Fringe Motion Intensity Analysis\nProject Gantt Chart', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Project Timeline (2025-2026)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Project Phases', fontsize=11, fontweight='bold')
    
    # 添加里程碑标记
    milestones = [
        {'name': 'Data Collection Start', 'date': '2025-11-15', 'color': 'red'},
        {'name': 'Data Collection Complete', 'date': '2025-11-22', 'color': 'blue'},
        {'name': 'Model Training Complete', 'date': '2026-01-25', 'color': 'green'},
        {'name': 'Paper Complete', 'date': '2026-03-24', 'color': 'purple'}
    ]
    
    for milestone in milestones:
        milestone_date = datetime.strptime(milestone['date'], '%Y-%m-%d')
        milestone_pos = (milestone_date - start_date).days
        ax.axvline(x=milestone_pos, color=milestone['color'], 
                  linestyle=':', linewidth=2, alpha=0.7)
        ax.text(milestone_pos, len(tasks), milestone['name'], 
               rotation=90, ha='right', va='bottom', 
               color=milestone['color'], fontweight='bold', fontsize=9)

    today_pos = (today_date - start_date).days
    ax.axvline(x=today_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(today_pos, len(tasks), 'Today', rotation=90, ha='right', va='bottom', color='black', fontweight='bold', fontsize=9)
    
    # 添加图例
    legend_elements = []
    for task in tasks:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=task['color'], 
                                           alpha=0.8, label=task['task']))
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='white', edgecolor='black', hatch='////', label='Actual completion'))
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
             fontsize=9, title='Project Phases', title_fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def create_detailed_timeline():
    """创建详细的项目时间线表"""
    
    tasks_data = []
    today_date = datetime.strptime('2025-12-01', '%Y-%m-%d')
    
    base = [
        {
            'phase': 'Project Preparation',
            'start_date': '2025-11-01',
            'estimated_end_date': '2025-11-14',
            'deliverables': 'Literature Review, Experimental Design, Equipment List'
        },
        {
            'phase': 'Data Collection',
            'start_date': '2025-11-15',
            'estimated_end_date': '2025-11-21',
            'deliverables': '49,984 Image Frames, Temperature Data'
        },
        {
            'phase': 'Data Preprocessing',
            'start_date': '2025-11-22',
            'estimated_end_date': '2025-12-05',
            'deliverables': 'Preprocessed Images, Interpolated Temperature Data'
        },
        {
            'phase': 'Experiment Iteration',
            'start_date': '2025-12-01',
            'estimated_end_date': '2025-12-14',
            'deliverables': 'Experiment logs, preliminary results'
        },
        {
            'phase': 'Data Annotation',
            'start_date': '2025-12-06',
            'estimated_end_date': '2025-12-30',
            'deliverables': 'Annotated Dataset (158 Training, 333 Validation)'
        },
        {
            'phase': 'YOLO Model Development',
            'start_date': '2025-12-31',
            'estimated_end_date': '2026-01-24',
            'deliverables': 'Trained YOLOv12s Model'
        },
        {
            'phase': 'Model Validation & Testing',
            'start_date': '2026-01-25',
            'estimated_end_date': '2026-02-05',
            'deliverables': 'Performance Report, Detection Results'
        },
        {
            'phase': 'Motion Intensity Analysis',
            'start_date': '2026-02-06',
            'estimated_end_date': '2026-02-21',
            'deliverables': 'Motion Analysis Results, Prediction Model'
        },
        {
            'phase': 'Result Analysis & Visualization',
            'start_date': '2026-02-22',
            'estimated_end_date': '2026-03-03',
            'deliverables': 'Analysis Report, Visualization Charts'
        },
        {
            'phase': 'Paper Writing',
            'start_date': '2026-03-04',
            'estimated_end_date': '2026-03-23',
            'deliverables': 'Academic Paper'
        }
    ]
    
    for item in base:
        s = datetime.strptime(item['start_date'], '%Y-%m-%d')
        e = datetime.strptime(item['estimated_end_date'], '%Y-%m-%d')
        duration = (e - s).days
        if today_date >= e:
            status = 'Completed'
            actual_end_date = e.strftime('%Y-%m-%d')
        elif s <= today_date < e:
            status = 'In progress'
            actual_end_date = ''
        else:
            status = 'Unfinished'
            actual_end_date = ''
        tasks_data.append({
            'phase': item['phase'],
            'start_date': item['start_date'],
            'estimated_end_date': item['estimated_end_date'],
            'actual_end_date': actual_end_date,
            'duration': duration,
            'deliverables': item['deliverables'],
            'status': status
        })
    
    return tasks_data

if __name__ == "__main__":
    # 创建甘特图
    fig = create_gantt_chart()
    
    # 保存图片
    plt.savefig('Project_Gantt_Chart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('Project_Gantt_Chart.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("Gantt chart saved as 'Project_Gantt_Chart.png' and 'Project_Gantt_Chart.pdf'")
    
    # 创建项目时间线表
    timeline = create_detailed_timeline()
    
    # 保存为Excel文件
    df = pd.DataFrame(timeline)
    try:
        df.to_excel('Project_Timeline_Details.xlsx', index=False, engine='openpyxl')
    except Exception:
        df.to_csv('Project_Timeline_Details.csv', index=False, encoding='utf-8-sig')
    
    print("Project timeline details saved as 'Project_Timeline_Details.xlsx'")
    
    # 显示图表
    plt.show()