import numpy as np
import pandas as pd

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

df = pd.DataFrame({**numerical_features, **categorical_features})

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

score = lin_score + cat1_effect + cat2_effect + cross_effect + np.random.randn(n_samples) * 1.5
df['target'] = (sigmoid(score) > 0.5).astype(int)

df.to_csv('data/large_dataset.csv', index=False)
print(f'生成数据集完成: {df.shape}')
print(f'目标分布:')
print(df['target'].value_counts())
print('\n数据集前5行:')
print(df.head())
print('\n特征类型:')
print(df.dtypes)
