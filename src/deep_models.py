#!/usr/bin/env python3
"""
深度学习模型: Wide & Deep 和 DeepFM
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入 TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    keras.config.enable_unsafe_deserialization()
    TF_AVAILABLE = True
    logger.info("TensorFlow 已加载")
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow 未安装，将使用模拟模式")


class WideDeepModel:
    """Wide & Deep Learning Model"""
    
    def __init__(self, model_dir: str = "model", embedding_dim: int = 8):
        self.model_dir = model_dir
        self.embedding_dim = embedding_dim
        self.model = None
        self.history = []
        self.categorical_features = []
        self.numerical_features = []
        self.vocabularies = {}
        os.makedirs(model_dir, exist_ok=True)
        
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'target'):
        """准备特征列表"""
        feature_cols = [c for c in df.columns if c != target_col]
        
        self.categorical_features = []
        self.numerical_features = []
        
        for col in feature_cols:
            dtype_name = df[col].dtype.name.lower()
            unique_count = len(df[col].unique())
            unique_ratio = unique_count / len(df)
            
            # 判断是否为类别特征：字符串类型 或 唯一值少的数值列
            is_string_type = dtype_name in ['object', 'string', 'str']
            is_category_type = dtype_name == 'category'
            is_low_cardinality = unique_count < 50 and unique_ratio < 0.01
            
            if is_string_type or is_category_type or (is_low_cardinality and col.startswith('category')):
                self.categorical_features.append(col)
            elif dtype_name in ['int64', 'float64', 'int32', 'float32']:
                self.numerical_features.append(col)
            elif col.startswith('numerical'):
                self.numerical_features.append(col)
            else:
                self.categorical_features.append(col)
        
        # 构建类别特征的词汇表
        for col in self.categorical_features:
            self.vocabularies[col] = sorted(df[col].unique().tolist())
            
        logger.info(f"类别特征: {self.categorical_features}")
        logger.info(f"数值特征: {self.numerical_features}")
        
    def build_model(self, wide_feature_dim: int, deep_feature_dim: int):
        """构建 Wide & Deep 模型"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow 未安装")
            
        # Wide 部分 - 线性模型
        wide_input = layers.Input(shape=(wide_feature_dim,), name='wide_input')
        wide_output = layers.Dense(1, activation='linear', name='wide_output')(wide_input)
        
        # Deep 部分 - DNN
        deep_input = layers.Input(shape=(deep_feature_dim,), name='deep_input')
        
        x = deep_input
        for i, units in enumerate([128, 64, 32]):
            x = layers.Dense(units, activation='relu', name=f'deep_dense_{i}')(x)
            x = layers.BatchNormalization(name=f'deep_bn_{i}')(x)
            x = layers.Dropout(0.3, name=f'deep_dropout_{i}')(x)
        
        deep_output = layers.Dense(1, activation='linear', name='deep_output')(x)
        
        # 合并 Wide 和 Deep
        combined = layers.Add(name='combine')([wide_output, deep_output])
        output = layers.Activation('sigmoid', name='output')(combined)
        
        model = Model(inputs=[wide_input, deep_input], outputs=output, name='WideDeep')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def preprocess_data(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """预处理数据"""
        if fit:
            self.prepare_features(df)
        
        # Wide 特征: 原始类别特征的 One-Hot + 数值特征
        wide_parts = []
        
        # 类别特征 One-Hot
        for col in self.categorical_features:
            vocab = self.vocabularies[col]
            encoded = pd.Categorical(df[col], categories=vocab).codes
            one_hot = np.eye(len(vocab))[encoded]
            wide_parts.append(one_hot)
        
        # 数值特征直接加入 Wide
        if self.numerical_features:
            wide_parts.append(df[self.numerical_features].values)
        
        X_wide = np.hstack(wide_parts) if len(wide_parts) > 1 else wide_parts[0]
        
        # Deep 特征: 类别特征的 Embedding + 数值特征
        deep_parts = []
        
        # 类别特征用整数编码（用于 Embedding）
        for col in self.categorical_features:
            vocab = self.vocabularies[col]
            encoded = pd.Categorical(df[col], categories=vocab).codes
            deep_parts.append(encoded.reshape(-1, 1))
        
        # 数值特征标准化后加入 Deep
        if self.numerical_features:
            num_values = df[self.numerical_features].values
            if fit:
                self.num_means = num_values.mean(axis=0)
                self.num_stds = num_values.std(axis=0) + 1e-8
            num_normalized = (num_values - self.num_means) / self.num_stds
            deep_parts.append(num_normalized)
        
        X_deep = np.hstack(deep_parts)
        
        y = df['target'].values if 'target' in df.columns else None
        
        return X_wide, X_deep, y
    
    def train(self, df_train: pd.DataFrame, df_val: Optional[pd.DataFrame] = None,
              epochs: int = 20, batch_size: int = 256) -> Dict:
        """训练模型"""
        logger.info("=" * 50)
        logger.info("Training Wide & Deep Model...")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # 预处理数据
        X_wide_train, X_deep_train, y_train = self.preprocess_data(df_train, fit=True)
        
        if df_val is not None:
            X_wide_val, X_deep_val, y_val = self.preprocess_data(df_val, fit=False)
            validation_data = ([X_wide_val, X_deep_val], y_val)
        else:
            validation_data = None
        
        # 构建模型
        self.model = self.build_model(X_wide_train.shape[1], X_deep_train.shape[1])
        logger.info(self.model.summary())
        
        # 训练
        history = self.model.fit(
            [X_wide_train, X_deep_train], y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        self.history = history.history
        
        # 获取最终指标
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history.get('val_accuracy', [0])[-1]
        final_train_auc = history.history['auc'][-1]
        final_val_auc = history.history.get('val_auc', [0])[-1]
        
        logger.info(f"Training completed!")
        logger.info(f"Train Accuracy: {final_train_acc:.4f}, AUC: {final_train_auc:.4f}")
        if validation_data:
            logger.info(f"Val Accuracy: {final_val_acc:.4f}, AUC: {final_val_auc:.4f}")
        
        return {
            'model_type': 'WideDeep',
            'epochs': epochs,
            'batch_size': batch_size,
            'final_train_acc': float(final_train_acc),
            'final_val_acc': float(final_val_acc),
            'final_train_auc': float(final_train_auc),
            'final_val_auc': float(final_val_auc),
            'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """预测"""
        X_wide, X_deep, _ = self.preprocess_data(df, fit=False)
        predictions = self.model.predict([X_wide, X_deep], verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        X_wide, X_deep, _ = self.preprocess_data(df, fit=False)
        return self.model.predict([X_wide, X_deep], verbose=0).flatten()
    
    def save_model(self, filename: str = "wide_deep_model"):
        """保存模型"""
        model_path = os.path.join(self.model_dir, f"{filename}.keras")
        config_path = os.path.join(self.model_dir, f"{filename}_config.json")
        
        self.model.save(model_path)
        
        config = {
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'vocabularies': self.vocabularies,
            'num_means': self.num_means.tolist() if hasattr(self, 'num_means') else None,
            'num_stds': self.num_stds.tolist() if hasattr(self, 'num_stds') else None,
            'embedding_dim': self.embedding_dim
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, filename: str = "wide_deep_model"):
        """加载模型"""
        model_path = os.path.join(self.model_dir, f"{filename}.keras")
        config_path = os.path.join(self.model_dir, f"{filename}_config.json")
        
        self.model = keras.models.load_model(model_path)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        self.categorical_features = config['categorical_features']
        self.numerical_features = config['numerical_features']
        self.vocabularies = config['vocabularies']
        self.num_means = np.array(config['num_means']) if config['num_means'] else None
        self.num_stds = np.array(config['num_stds']) if config['num_stds'] else None
        self.embedding_dim = config['embedding_dim']
        
        logger.info(f"Model loaded from {model_path}")


class DeepFMModel:
    """DeepFM Model - 简化版本，使用多层网络模拟特征交叉"""
    
    def __init__(self, model_dir: str = "model", embedding_dim: int = 8):
        self.model_dir = model_dir
        self.embedding_dim = embedding_dim
        self.model = None
        self.history = []
        self.categorical_features = []
        self.numerical_features = []
        self.vocabularies = {}
        os.makedirs(model_dir, exist_ok=True)
        
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'target'):
        """准备特征"""
        feature_cols = [c for c in df.columns if c != target_col]
        
        self.categorical_features = []
        self.numerical_features = []
        
        for col in feature_cols:
            dtype_name = df[col].dtype.name.lower()
            unique_count = len(df[col].unique())
            unique_ratio = unique_count / len(df)
            
            is_string_type = dtype_name in ['object', 'string', 'str']
            is_category_type = dtype_name == 'category'
            is_low_cardinality = unique_count < 50 and unique_ratio < 0.01
            
            if is_string_type or is_category_type or (is_low_cardinality and col.startswith('category')):
                self.categorical_features.append(col)
            elif dtype_name in ['int64', 'float64', 'int32', 'float32']:
                self.numerical_features.append(col)
            elif col.startswith('numerical'):
                self.numerical_features.append(col)
            else:
                self.categorical_features.append(col)
        
        for col in self.categorical_features:
            self.vocabularies[col] = sorted(df[col].unique().tolist())
            
        logger.info(f"类别特征: {self.categorical_features}")
        logger.info(f"数值特征: {self.numerical_features}")
        
    def build_model(self, n_categories: int, n_numerical: int):
        """构建简化的 DeepFM 模型"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow 未安装")
        
        total_features = n_categories + n_numerical
        
        # 输入层
        input_layer = layers.Input(shape=(total_features,), name='input')
        
        # FM 线性部分 (Wide)
        linear = layers.Dense(64, activation='relu', name='linear_1')(input_layer)
        
        # 特征交叉层 - 使用 Dense 模拟 FM 二阶交互
        cross1 = layers.Dense(128, activation='relu', name='cross_1')(input_layer)
        cross2 = layers.Dense(64, activation='relu', name='cross_2')(cross1)
        cross = layers.Multiply(name='fm_interaction')([linear, cross2])
        
        # Deep 部分 (DNN)
        deep = layers.Dense(128, activation='relu', name='deep_dense_1')(input_layer)
        deep = layers.BatchNormalization(name='deep_bn_1')(deep)
        deep = layers.Dropout(0.3, name='deep_dropout_1')(deep)
        
        deep = layers.Dense(64, activation='relu', name='deep_dense_2')(deep)
        deep = layers.BatchNormalization(name='deep_bn_2')(deep)
        deep = layers.Dropout(0.3, name='deep_dropout_2')(deep)
        
        deep = layers.Dense(32, activation='relu', name='deep_dense_3')(deep)
        
        # 合并 Wide + Cross + Deep
        concat = layers.Concatenate(name='concat')([linear, cross, deep])
        final = layers.Dense(16, activation='relu', name='final_dense')(concat)
        output = layers.Dense(1, activation='sigmoid', name='output')(final)
        
        model = Model(inputs=input_layer, outputs=output, name='DeepFM')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def preprocess_data(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """预处理数据"""
        if fit:
            self.prepare_features(df)
        
        feature_parts = []
        
        # 类别特征 One-Hot
        for col in self.categorical_features:
            vocab = self.vocabularies[col]
            encoded = pd.Categorical(df[col], categories=vocab).codes
            one_hot = np.eye(len(vocab))[encoded]
            feature_parts.append(one_hot)
        
        # 数值特征
        if self.numerical_features:
            num_values = df[self.numerical_features].values
            if fit:
                self.num_means = num_values.mean(axis=0)
                self.num_stds = num_values.std(axis=0) + 1e-8
            num_normalized = (num_values - self.num_means) / self.num_stds
            feature_parts.append(num_normalized)
        
        X = np.hstack(feature_parts)
        y = df['target'].values if 'target' in df.columns else None
        
        return X, y
    
    def train(self, df_train: pd.DataFrame, df_val: Optional[pd.DataFrame] = None,
              epochs: int = 20, batch_size: int = 256) -> Dict:
        """训练模型"""
        logger.info("=" * 50)
        logger.info("Training DeepFM Model...")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        X_train, y_train = self.preprocess_data(df_train, fit=True)
        
        if df_val is not None:
            X_val, y_val = self.preprocess_data(df_val, fit=False)
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        n_categories = sum(len(v) for v in self.vocabularies.values())
        n_numerical = len(self.numerical_features)
        
        self.model = self.build_model(n_categories, n_numerical)
        logger.info(self.model.summary())
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        self.history = history.history
        
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history.get('val_accuracy', [0])[-1]
        final_train_auc = history.history['auc'][-1]
        final_val_auc = history.history.get('val_auc', [0])[-1]
        
        logger.info(f"Training completed!")
        logger.info(f"Train Accuracy: {final_train_acc:.4f}, AUC: {final_train_auc:.4f}")
        if validation_data:
            logger.info(f"Val Accuracy: {final_val_acc:.4f}, AUC: {final_val_auc:.4f}")
        
        return {
            'model_type': 'DeepFM',
            'epochs': epochs,
            'batch_size': batch_size,
            'final_train_acc': float(final_train_acc),
            'final_val_acc': float(final_val_acc),
            'final_train_auc': float(final_train_auc),
            'final_val_auc': float(final_val_auc),
            'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """预测"""
        X, _ = self.preprocess_data(df, fit=False)
        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        X, _ = self.preprocess_data(df, fit=False)
        return self.model.predict(X, verbose=0).flatten()
    
    def save_model(self, filename: str = "deepfm_model"):
        """保存模型"""
        model_path = os.path.join(self.model_dir, f"{filename}.keras")
        config_path = os.path.join(self.model_dir, f"{filename}_config.json")
        
        self.model.save(model_path)
        
        config = {
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'vocabularies': self.vocabularies,
            'num_means': self.num_means.tolist() if hasattr(self, 'num_means') else None,
            'num_stds': self.num_stds.tolist() if hasattr(self, 'num_stds') else None,
            'embedding_dim': self.embedding_dim
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, filename: str = "deepfm_model"):
        """加载模型"""
        model_path = os.path.join(self.model_dir, f"{filename}.keras")
        config_path = os.path.join(self.model_dir, f"{filename}_config.json")
        
        self.model = keras.models.load_model(model_path)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        self.categorical_features = config['categorical_features']
        self.numerical_features = config['numerical_features']
        self.vocabularies = config['vocabularies']
        self.num_means = np.array(config['num_means']) if config['num_means'] else None
        self.num_stds = np.array(config['num_stds']) if config['num_stds'] else None
        self.embedding_dim = config['embedding_dim']
        
        logger.info(f"Model loaded from {model_path}")
