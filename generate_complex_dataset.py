#!/usr/bin/env python3
"""生成复杂数据集用于模型测试"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# 创建一个复杂的多分类数据集
print("生成复杂数据集...")

# 1. 复杂多分类数据 - 10个特征，5个类别，有噪声
X1, y1 = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_repeated=2,
    n_classes=5,
    n_clusters_per_class=2,
    flip_y=0.1,  # 10% 标签噪声
    class_sep=0.8,  # 类别分离度较低
    random_state=42
)

# 添加特征名称
columns = [f'feature_{i+1}' for i in range(10)]
df_complex = pd.DataFrame(X1, columns=columns)
df_complex['target'] = y1

# 保存复杂数据集
df_complex.to_csv('data/dataset_complex.csv', index=False)
print(f"复杂数据集已保存: {df_complex.shape[0]} 条, {df_complex.shape[1]-1} 个特征, {df_complex['target'].nunique()} 个类别")
print(f"类别分布: {dict(df_complex['target'].value_counts().sort_index())}")

# 2. 创建一个更难的二分类数据集（非线性可分）
X2, y2 = make_moons(n_samples=1000, noise=0.3, random_state=42)
df_moons = pd.DataFrame(X2, columns=['feature_1', 'feature_2'])
df_moons['target'] = y2

# 添加额外噪声特征
for i in range(3, 8):
    df_moons[f'feature_{i}'] = np.random.randn(1000) * 0.5

df_moons.to_csv('data/dataset_moons.csv', index=False)
print(f"\nMoons数据集已保存: {df_moons.shape[0]} 条, {df_moons.shape[1]-1} 个特征")

# 3. 创建一个包含缺失值和异常值的真实场景数据集
n_samples = 1500
n_features = 8

X3 = np.random.randn(n_samples, n_features)
# 创建非线性关系
y3 = (X3[:, 0] ** 2 + X3[:, 1] ** 2 + np.sin(X3[:, 2] * 3) > 1).astype(int)
# 添加多类别
y3 = np.where(X3[:, 3] > 1, 2, y3)
y3 = np.where((X3[:, 0] + X3[:, 1]) < -2, 3, y3)

df_real = pd.DataFrame(X3, columns=[f'feature_{i+1}' for i in range(n_features)])
df_real['target'] = y3

# 添加缺失值 (5%)
for col in df_real.columns[:-1]:
    mask = np.random.random(n_samples) < 0.05
    df_real.loc[mask, col] = np.nan

# 添加异常值
outlier_idx = np.random.choice(n_samples, size=50, replace=False)
df_real.loc[outlier_idx, 'feature_1'] = np.random.choice([-10, 10], 50)
df_real.loc[outlier_idx, 'feature_2'] = np.random.choice([-10, 10], 50)

df_real.to_csv('data/dataset_real.csv', index=False)
print(f"\n真实场景数据集已保存: {df_real.shape[0]} 条, 含缺失值和异常值")
print(f"类别分布: {dict(df_real['target'].value_counts().sort_index())}")

print("\n所有数据集生成完成!")
