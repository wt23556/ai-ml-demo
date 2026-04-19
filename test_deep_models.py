import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.wide_deep_model import WideDeepTrainer
from src.deepfm_model import DeepFMTrainer
from src.model_trainer import ModelTrainer
from src.data_processor import DataProcessor
from src.model_evaluator import ModelEvaluator

def evaluate_model(y_true, y_pred, model_name):
    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average='weighted'), 4),
        "recall": round(recall_score(y_true, y_pred, average='weighted'), 4),
        "f1_score": round(f1_score(y_true, y_pred, average='weighted'), 4)
    }

def main():
    print("=" * 70)
    print("🚀 Wide & Deep 和 DeepFM 模型测试")
    print("=" * 70)
    
    data_path = os.path.join(os.path.dirname(__file__), "data", "dataset_extended.csv")
    df = pd.read_csv(data_path)
    print(f"\n📊 数据集信息:")
    print(f"   样本数: {len(df)}")
    print(f"   特征数: {len(df.columns) - 1}")
    print(f"   类别分布: {dict(df['target'].value_counts().sort_index())}")
    
    results = []
    
    print("\n" + "=" * 70)
    print("🌲 测试 RandomForest 模型 (基准)")
    print("=" * 70)
    
    processor = DataProcessor(data_path=data_path)
    df_loaded = processor.load_data()
    X_train, X_test, y_train, y_test = processor.preprocess(df_loaded)
    
    rf_trainer = ModelTrainer(model_dir="model")
    start_time = time.time()
    rf_result = rf_trainer.train(X_train, y_train, n_estimators=100)
    rf_train_time = time.time() - start_time
    
    start_time = time.time()
    rf_pred = rf_trainer.predict(X_test)
    rf_pred_time = time.time() - start_time
    
    rf_metrics = evaluate_model(y_test, rf_pred, "RandomForest")
    rf_metrics["train_time"] = round(rf_train_time, 2)
    rf_metrics["pred_time"] = round(rf_pred_time, 4)
    results.append(rf_metrics)
    
    print(f"\n✅ RandomForest 结果:")
    print(f"   准确率: {rf_metrics['accuracy']:.4f}")
    print(f"   精确率: {rf_metrics['precision']:.4f}")
    print(f"   召回率: {rf_metrics['recall']:.4f}")
    print(f"   F1分数: {rf_metrics['f1_score']:.4f}")
    print(f"   训练时间: {rf_train_time:.2f}s")
    
    print("\n" + "=" * 70)
    print("🧠 测试 Wide & Deep 模型")
    print("=" * 70)
    
    wd_trainer = WideDeepTrainer(model_dir="model")
    wide_input, deep_input, y = wd_trainer.preprocess(df)
    
    split_idx = int(len(df) * 0.8)
    wide_train, wide_test = wide_input[:split_idx], wide_input[split_idx:]
    deep_train, deep_test = deep_input[:split_idx], deep_input[split_idx:]
    y_train_wd, y_test_wd = y[:split_idx], y[split_idx:]
    
    start_time = time.time()
    wd_result = wd_trainer.train(
        wide_train, deep_train, y_train_wd,
        epochs=50, batch_size=64, learning_rate=0.001,
        hidden_units=[256, 128, 64]
    )
    wd_train_time = time.time() - start_time
    
    start_time = time.time()
    wd_pred = wd_trainer.predict(wide_test, deep_test)
    wd_pred_time = time.time() - start_time
    
    wd_metrics = evaluate_model(y_test_wd, wd_pred, "Wide&Deep")
    wd_metrics["train_time"] = round(wd_train_time, 2)
    wd_metrics["pred_time"] = round(wd_pred_time, 4)
    wd_metrics["final_train_acc"] = round(wd_result["final_accuracy"], 4)
    results.append(wd_metrics)
    
    print(f"\n✅ Wide & Deep 结果:")
    print(f"   准确率: {wd_metrics['accuracy']:.4f}")
    print(f"   精确率: {wd_metrics['precision']:.4f}")
    print(f"   召回率: {wd_metrics['recall']:.4f}")
    print(f"   F1分数: {wd_metrics['f1_score']:.4f}")
    print(f"   训练时间: {wd_train_time:.2f}s")
    
    wd_trainer.save_model()
    print("   模型已保存")
    
    print("\n" + "=" * 70)
    print("🎯 测试 DeepFM 模型")
    print("=" * 70)
    
    dfm_trainer = DeepFMTrainer(model_dir="model")
    X, y_dfm = dfm_trainer.preprocess(df)
    
    X_train_dfm, X_test_dfm = X[:split_idx], X[split_idx:]
    y_train_dfm, y_test_dfm = y_dfm[:split_idx], y_dfm[split_idx:]
    
    start_time = time.time()
    dfm_result = dfm_trainer.train(
        X_train_dfm, y_train_dfm,
        epochs=50, batch_size=64, learning_rate=0.001,
        embedding_dim=10, hidden_units=[256, 128, 64]
    )
    dfm_train_time = time.time() - start_time
    
    start_time = time.time()
    dfm_pred = dfm_trainer.predict(X_test_dfm)
    dfm_pred_time = time.time() - start_time
    
    dfm_metrics = evaluate_model(y_test_dfm, dfm_pred, "DeepFM")
    dfm_metrics["train_time"] = round(dfm_train_time, 2)
    dfm_metrics["pred_time"] = round(dfm_pred_time, 4)
    dfm_metrics["final_train_acc"] = round(dfm_result["final_accuracy"], 4)
    results.append(dfm_metrics)
    
    print(f"\n✅ DeepFM 结果:")
    print(f"   准确率: {dfm_metrics['accuracy']:.4f}")
    print(f"   精确率: {dfm_metrics['precision']:.4f}")
    print(f"   召回率: {dfm_metrics['recall']:.4f}")
    print(f"   F1分数: {dfm_metrics['f1_score']:.4f}")
    print(f"   训练时间: {dfm_train_time:.2f}s")
    
    dfm_trainer.save_model()
    print("   模型已保存")
    
    print("\n" + "=" * 70)
    print("📊 模型对比总结")
    print("=" * 70)
    
    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))
    
    best_model = max(results, key=lambda x: x['accuracy'])
    print(f"\n🏆 最佳模型: {best_model['model']} (准确率: {best_model['accuracy']:.4f})")
    
    print("\n📈 详细分类报告:")
    print("\n--- Wide & Deep ---")
    print(classification_report(y_test_wd, wd_pred, target_names=['Class 0', 'Class 1', 'Class 2']))
    
    print("\n--- DeepFM ---")
    print(classification_report(y_test_dfm, dfm_pred, target_names=['Class 0', 'Class 1', 'Class 2']))
    
    with open("model_comparison_results.json", "w") as f:
        json.dump({
            "dataset": {
                "samples": len(df),
                "features": len(df.columns) - 1,
                "class_distribution": {int(k): int(v) for k, v in df['target'].value_counts().sort_index().items()}
            },
            "results": results,
            "best_model": best_model['model']
        }, f, indent=2, ensure_ascii=False)
    
    print("\n📄 结果已保存到 model_comparison_results.json")
    print("=" * 70)

if __name__ == "__main__":
    main()
