#!/usr/bin/env python3
"""
生成广告点击率预测数据集，适合 Wide & Deep 和 DeepFM 模型
包含类别特征和数值特征
"""

import pandas as pd
import numpy as np

np.random.seed(42)

def generate_ad_click_dataset(n_samples=10000):
    """生成广告点击数据集"""
    
    # 类别特征
    user_genders = ['Male', 'Female', 'Unknown']
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
    cities = ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Chengdu', 'Wuhan', 'Hangzhou', 'Nanjing']
    device_types = ['Mobile', 'Desktop', 'Tablet']
    os_types = ['iOS', 'Android', 'Windows', 'MacOS']
    ad_categories = ['Game', 'Finance', 'E-commerce', 'Education', 'Travel', 'Food']
    time_slots = ['Morning', 'Afternoon', 'Evening', 'Night']
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # 生成数据
    data = {
        # 用户特征
        'user_gender': np.random.choice(user_genders, n_samples),
        'age_group': np.random.choice(age_groups, n_samples),
        'city': np.random.choice(cities, n_samples),
        'device_type': np.random.choice(device_types, n_samples),
        'os_type': np.random.choice(os_types, n_samples),
        
        # 广告特征
        'ad_category': np.random.choice(ad_categories, n_samples),
        'ad_price': np.random.uniform(1, 100, n_samples),
        'ad_quality_score': np.random.uniform(0, 10, n_samples),
        
        # 上下文特征
        'time_slot': np.random.choice(time_slots, n_samples),
        'weekday': np.random.choice(weekdays, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        
        # 数值特征 - 用户行为统计
        'user_history_clicks': np.random.poisson(5, n_samples),
        'user_history_impressions': np.random.poisson(20, n_samples),
        'user_ctr': np.random.beta(2, 8, n_samples),
        'user_avg_session_duration': np.random.exponential(300, n_samples),
        
        # 数值特征 - 广告相关
        'ad_position': np.random.randint(1, 10, n_samples),
        'ad_text_length': np.random.randint(20, 200, n_samples),
        'ad_image_count': np.random.randint(0, 5, n_samples),
        'ad_has_video': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    }
    
    df = pd.DataFrame(data)
    
    # 计算 CTR 并生成标签
    # Wide 部分特征（记忆能力）
    wide_score = (
        (df['user_gender'] == 'Female').astype(int) * 0.1 +
        (df['age_group'].isin(['25-34', '35-44'])).astype(int) * 0.15 +
        (df['device_type'] == 'Mobile').astype(int) * 0.2 +
        (df['ad_category'] == 'Game').astype(int) * 0.1 +
        (df['ad_category'] == 'Finance').astype(int) * 0.05 +
        (df['time_slot'] == 'Evening').astype(int) * 0.1 +
        df['is_weekend'] * 0.05 +
        df['ad_has_video'] * 0.15
    )
    
    # Deep 部分特征（泛化能力）- 数值特征的非线性组合
    deep_score = (
        np.log1p(df['user_history_clicks']) * 0.1 +
        df['user_ctr'] * 0.2 +
        np.log1p(df['user_avg_session_duration']) * 0.05 +
        (10 - df['ad_position']) / 10 * 0.15 +
        df['ad_quality_score'] / 10 * 0.2 +
        np.log1p(df['ad_price']) * 0.05
    )
    
    # 添加噪声
    noise = np.random.normal(0, 0.1, n_samples)
    
    # 最终得分
    total_score = wide_score + deep_score + noise
    
    # 生成二分类标签 (点击/未点击)
    click_prob = 1 / (1 + np.exp(-total_score))
    df['target'] = (np.random.random(n_samples) < click_prob).astype(int)
    
    return df

# 生成不同规模的数据集
print("生成广告点击数据集...")

# 训练集
df_train = generate_ad_click_dataset(10000)
df_train.to_csv('data/ad_click_train.csv', index=False)
print(f"训练集: {len(df_train)} 条")

# 测试集
df_test = generate_ad_click_dataset(2000)
df_test.to_csv('data/ad_click_test.csv', index=False)
print(f"测试集: {len(df_test)} 条")

# 查看数据信息
print(f"\n特征列表: {list(df_train.columns[:-1])}")
print(f"类别特征: {[c for c in df_train.columns if df_train[c].dtype == 'object']}")
print(f"数值特征: {[c for c in df_train.columns if df_train[c].dtype != 'object' and c != 'target']}")
print(f"\n类别分布: {df_train['target'].value_counts().to_dict()}")
print(f"正样本比例: {df_train['target'].mean():.2%}")

print("\n数据集生成完成!")
