import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 2000

feature_1 = np.random.uniform(4.0, 8.0, n_samples)
feature_2 = np.random.uniform(2.0, 5.0, n_samples)
feature_3 = np.random.uniform(1.0, 7.0, n_samples)
feature_4 = np.random.uniform(0.1, 3.0, n_samples)
feature_5 = np.random.uniform(0.1, 3.0, n_samples)

feature_6 = np.random.uniform(0.5, 2.5, n_samples)
feature_7 = np.random.uniform(1.0, 4.0, n_samples)
feature_8 = np.random.uniform(0.2, 1.5, n_samples)

categories = ['A', 'B', 'C', 'D']
cat_feature_1 = np.random.choice(categories, n_samples, p=[0.3, 0.3, 0.25, 0.15])
cat_feature_2 = np.random.choice(['X', 'Y', 'Z'], n_samples, p=[0.4, 0.35, 0.25])
cat_feature_3 = np.random.choice(['High', 'Medium', 'Low'], n_samples, p=[0.3, 0.4, 0.3])

regions = ['North', 'South', 'East', 'West']
region = np.random.choice(regions, n_samples, p=[0.25, 0.25, 0.25, 0.25])

seasons = ['Spring', 'Summer', 'Fall', 'Winter']
season = np.random.choice(seasons, n_samples, p=[0.25, 0.25, 0.25, 0.25])

target = []
for i in range(n_samples):
    score = 0
    score += (feature_1[i] - 6) * 0.3
    score += (feature_2[i] - 3.5) * 0.2
    score += (feature_3[i] - 4) * 0.4
    score += (feature_4[i] - 1.5) * 0.5
    score += (feature_5[i] - 1.5) * 0.3
    score += (feature_6[i] - 1.5) * 0.2
    score += (feature_7[i] - 2.5) * 0.15
    score += (feature_8[i] - 0.8) * 0.1
    
    if cat_feature_1[i] == 'A':
        score += 0.5
    elif cat_feature_1[i] == 'D':
        score -= 0.3
    
    if cat_feature_2[i] == 'X':
        score += 0.3
    elif cat_feature_2[i] == 'Z':
        score -= 0.2
    
    if cat_feature_3[i] == 'High':
        score += 0.4
    elif cat_feature_3[i] == 'Low':
        score -= 0.3
    
    if region[i] in ['North', 'East']:
        score += 0.2
    
    if season[i] in ['Spring', 'Summer']:
        score += 0.15
    
    score += np.random.normal(0, 0.3)
    
    if score < -0.5:
        target.append(0)
    elif score < 0.5:
        target.append(1)
    else:
        target.append(2)

target = np.array(target)

df = pd.DataFrame({
    'feature_1': np.round(feature_1, 2),
    'feature_2': np.round(feature_2, 2),
    'feature_3': np.round(feature_3, 2),
    'feature_4': np.round(feature_4, 2),
    'feature_5': np.round(feature_5, 2),
    'feature_6': np.round(feature_6, 2),
    'feature_7': np.round(feature_7, 2),
    'feature_8': np.round(feature_8, 2),
    'cat_feature_1': cat_feature_1,
    'cat_feature_2': cat_feature_2,
    'cat_feature_3': cat_feature_3,
    'region': region,
    'season': season,
    'target': target
})

print(f"Dataset shape: {df.shape}")
print(f"\nTarget distribution:")
print(df['target'].value_counts().sort_index())
print(f"\nFeature types:")
print(df.dtypes)

df.to_csv('dataset_extended.csv', index=False)
print(f"\nDataset saved to dataset_extended.csv")
