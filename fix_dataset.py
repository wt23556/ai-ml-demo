import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# === 1. 改进小数据集 (500行，3分类更有挑战性) ===
np.random.seed(42)
n_samples = 500
X, y = make_classification(
    n_samples=n_samples,
    n_features=5,
    n_informative=4,
    n_redundant=1,
    n_classes=3,
    n_clusters_per_class=2,
    weights=[0.33, 0.33, 0.34],
    flip_y=0.15,
    class_sep=0.6,
    random_state=42
)

df_small = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(5)])
df_small['target'] = y
df_small.to_csv('data/dataset.csv', index=False)
print('=== 小数据集 (dataset.csv) ===')
print(f'样本数: {n_samples} (从90增加到500)')
print(f'目标分布:\n{df_small["target"].value_counts().sort_index()}')
print(f'各类占比: {np.bincount(y) / len(y) * 100}')
print()

# === 2. 改进大数据集 (解决类别不平衡) ===
np.random.seed(42)
n_samples = 50000

numerical_features = {
    'numerical_1': np.random.randn(n_samples) * 2 + 5,
    'numerical_2': np.random.randn(n_samples) + 3,
    'numerical_3': np.random.randn(n_samples) * 3,
    'numerical_4': np.random.exponential(2, n_samples),
    'numerical_5': np.random.uniform(0, 10, n_samples),
}

categorical_features = {
    'category_1': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    'category_2': np.random.choice(['X', 'Y', 'Z'], n_samples, p=[0.5, 0.3, 0.2]),
    'category_3': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.4, 0.4, 0.2]),
    'category_4': np.random.randint(0, 10, n_samples).astype(str),
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

lin_score = 0.3 * numerical_features['numerical_1'] + \
            0.2 * numerical_features['numerical_2'] - \
            0.1 * numerical_features['numerical_3']

cat1_map = {'A': 1, 'B': 0.5, 'C': 0, 'D': -0.5, 'E': -1}
cat2_map = {'X': 0.8, 'Y': 0, 'Z': -0.8}
cat1_effect = np.array([cat1_map[c] for c in categorical_features['category_1']])
cat2_effect = np.array([cat2_map[c] for c in categorical_features['category_2']])

cross_effect = np.where(
    (categorical_features['category_1'] == 'A') & (categorical_features['category_2'] == 'X'), 
    1.5,
    np.where(
        (categorical_features['category_1'] == 'E') & (categorical_features['category_2'] == 'Z'), 
        -1.5, 
        0
    )
)

# 降低阈值，使类别更均衡
score = lin_score + cat1_effect + cat2_effect + cross_effect + np.random.randn(n_samples) * 2.0
target = (sigmoid(score) > 0.35).astype(int)

df_large = pd.DataFrame({**numerical_features, **categorical_features})
df_large['target'] = target
df_large.to_csv('data/large_dataset.csv', index=False)

print('=== 大数据集 (large_dataset.csv) ===')
print(f'样本数: {n_samples}')
print(f'目标分布:\n{df_large["target"].value_counts().sort_index()}')
print(f'各类占比: {np.bincount(target) / len(target) * 100}')
print()
print('✅ 数据集更新完成！训练效果和混淆矩阵变化会更明显！')
