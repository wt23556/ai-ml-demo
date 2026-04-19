import pandas as pd
from sklearn.model_selection import train_test_split
from src.deep_models import WideDeepModel, DeepFMModel

df = pd.read_csv('data/large_dataset.csv')
print(f'数据集大小: {df.shape}')

df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
print(f'训练集: {df_train.shape}, 验证集: {df_val.shape}')

print('\n=== 测试 Wide & Deep 模型 ===')
wide_deep = WideDeepModel(model_dir='model')
result = wide_deep.train(df_train, df_val, epochs=5, batch_size=1024)
print(f'训练结果: Acc={result["final_train_acc"]:.4f}, Val Acc={result["final_val_acc"]:.4f}')
wide_deep.save_model()

print('\n=== 测试 DeepFM 模型 ===')
deepfm = DeepFMModel(model_dir='model')
result = deepfm.train(df_train, df_val, epochs=5, batch_size=1024)
print(f'训练结果: Acc={result["final_train_acc"]:.4f}, Val Acc={result["final_val_acc"]:.4f}')
deepfm.save_model()
