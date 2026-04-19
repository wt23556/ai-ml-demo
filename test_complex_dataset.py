#!/usr/bin/env python3
"""复杂数据集模型测试脚本"""

import sys
sys.path.insert(0, '.')
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 数据集列表
datasets = [
    ('dataset.csv', '原始鸢尾花'),
    ('dataset_complex.csv', '复杂多分类'),
    ('dataset_moons.csv', 'Moons非线性'),
    ('dataset_real.csv', '真实场景(含噪声)')
]

print('=' * 80)
print('复杂数据集模型测试')
print('=' * 80)

results = []

for dataset_file, dataset_name in datasets:
    print(f'\n{"="*80}')
    print(f'数据集: {dataset_name} ({dataset_file})')
    print('=' * 80)
    
    try:
        # 加载数据
        dp = DataProcessor(f'data/{dataset_file}')
        df = dp.load_data()
        X_train, X_test, y_train, y_test = dp.preprocess(df)
        
        n_classes = df['target'].nunique()
        print(f'样本数: {len(df)}, 特征数: {len(dp.feature_names)}, 类别数: {n_classes}')
        print(f'类别分布: {dict(df["target"].value_counts().sort_index())}')
        
        # 测试不同参数配置
        configs = [
            {
                'name': '默认配置',
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            },
            {
                'name': '防过拟合',
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 3
            },
            {
                'name': '深度限制',
                'n_estimators': 200,
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 5
            },
            {
                'name': '更多树',
                'n_estimators': 300,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 2
            }
        ]
        
        print(f'\n{"配置":<15} {"训练准确率":<12} {"测试准确率":<12} {"过拟合差距":<12}')
        print('-' * 55)
        
        dataset_results = []
        for cfg in configs:
            cfg_name = cfg.pop('name')
            mt = ModelTrainer(model_dir='model')
            
            # 训练
            result = mt.train(X_train, y_train, **cfg)
            
            # 测试
            y_pred = mt.predict(X_test)
            test_acc = (y_pred == y_test.values).mean()
            train_acc = result['final_accuracy']
            gap = train_acc - test_acc
            
            print(f'{cfg_name:<15} {train_acc:<12.4f} {test_acc:<12.4f} {gap:<12.4f}')
            
            dataset_results.append({
                'config_name': cfg_name,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'gap': gap,
                'config': cfg
            })
        
        # 找出最佳配置
        best = max(dataset_results, key=lambda x: x['test_acc'])
        print(f'\n✅ 最佳配置: {best["config_name"]} (测试准确率: {best["test_acc"]:.4f})')
        
        results.append({
            'dataset': dataset_name,
            'best_config': best['config_name'],
            'best_test_acc': best['test_acc'],
            'best_train_acc': best['train_acc'],
            'gap': best['gap']
        })
        
    except Exception as e:
        print(f'❌ 错误: {e}')

# 汇总
print('\n' + '=' * 80)
print('测试结果汇总')
print('=' * 80)
print(f'{"数据集":<20} {"最佳配置":<15} {"训练准确率":<12} {"测试准确率":<12} {"过拟合":<10}')
print('-' * 75)
for r in results:
    overfit = "是" if r['gap'] > 0.05 else "否"
    print(f'{r["dataset"]:<20} {r["best_config"]:<15} {r["best_train_acc"]:<12.4f} {r["best_test_acc"]:<12.4f} {overfit:<10}')

print('\n' + '=' * 80)
print('优化建议:')
print('=' * 80)
for r in results:
    if r['gap'] > 0.1:
        print(f'• {r["dataset"]}: 严重过拟合，建议使用更保守的参数')
    elif r['gap'] > 0.05:
        print(f'• {r["dataset"]}: 轻微过拟合，可适当增加正则化')
    else:
        print(f'• {r["dataset"]}: 拟合良好')
