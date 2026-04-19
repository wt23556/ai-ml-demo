#!/usr/bin/env python3
"""模型参数优化测试脚本"""

import sys
sys.path.insert(0, '.')
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

# 加载数据
dp = DataProcessor('data/dataset.csv')
df = dp.load_data()
X_train, X_test, y_train, y_test = dp.preprocess(df)

print('=' * 60)
print('随机森林参数优化测试')
print('=' * 60)
print(f'数据集大小: {df.shape[0]} 条, 特征数: {df.shape[1]-1}')
print(f'训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条')
print(f'类别分布: {dict(df["target"].value_counts().sort_index())}')
print()

# 测试不同参数组合
configs = [
    {
        'name': '1. 默认配置',
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt'
    },
    {
        'name': '2. 保守配置(防过拟合)',
        'n_estimators': 50,
        'max_depth': 3,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt'
    },
    {
        'name': '3. 深度限制',
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'max_features': 'sqrt'
    },
    {
        'name': '4. 更多树+叶子限制',
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 3,
        'max_features': 'sqrt'
    },
    {
        'name': '5. 特征限制',
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'log2'
    },
    {
        'name': '6. 超保守配置',
        'n_estimators': 100,
        'max_depth': 4,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 0.5
    }
]

print(f'{"配置名称":<25} {"训练准确率":<12} {"测试准确率":<12} {"交叉验证":<12}')
print('-' * 65)

results = []
for cfg in configs:
    name = cfg.pop('name')
    mt = ModelTrainer(model_dir='model')

    # 训练
    result = mt.train(X_train, y_train, **cfg)

    # 测试集评估
    y_pred = mt.predict(X_test)
    test_acc = (y_pred == y_test.values).mean()

    # 5折交叉验证
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=cfg.get('n_estimators', 100),
        max_depth=cfg.get('max_depth'),
        min_samples_split=cfg.get('min_samples_split', 2),
        min_samples_leaf=cfg.get('min_samples_leaf', 1),
        max_features=cfg.get('max_features', 'sqrt'),
        random_state=42
    )
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()

    train_acc = result['final_accuracy']

    print(f'{name:<25} {train_acc:<12.4f} {test_acc:<12.4f} {cv_mean:<12.4f}')

    results.append({
        'name': name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_mean,
        'config': cfg
    })

print()
print('=' * 60)
print('优化建议:')
print('=' * 60)

# 找出最佳配置
best = max(results, key=lambda x: x['test_acc'])
print(f'最佳测试准确率: {best["test_acc"]:.4f} - {best["name"]}')

# 检查过拟合
for r in results:
    gap = r['train_acc'] - r['test_acc']
    if gap > 0.05:
        print(f'⚠️  {r["name"]} 存在过拟合 (差距: {gap:.4f})')
    elif r['train_acc'] == 1.0 and r['test_acc'] == 1.0:
        print(f'✅ {r["name"]} 完美拟合 (无过拟合)')

print()
print('推荐参数组合:')
print(f'  n_estimators: {best["config"]["n_estimators"]}')
print(f'  max_depth: {best["config"]["max_depth"]}')
print(f'  min_samples_split: {best["config"]["min_samples_split"]}')
print(f'  min_samples_leaf: {best["config"]["min_samples_leaf"]}')
print(f'  max_features: {best["config"]["max_features"]}')
