import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WideComponent(nn.Module):
    def __init__(self, input_dim: int):
        super(WideComponent, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)


class DeepComponent(nn.Module):
    def __init__(self, input_dim: int, hidden_units: List[int] = [256, 128, 64]):
        super(DeepComponent, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        self.deep = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_units[-1], 1)
    
    def forward(self, x):
        x = self.deep(x)
        return self.output(x)


class WideAndDeep(nn.Module):
    def __init__(self, wide_input_dim: int, deep_input_dim: int, 
                 hidden_units: List[int] = [256, 128, 64], num_classes: int = 3):
        super(WideAndDeep, self).__init__()
        self.wide = WideComponent(wide_input_dim)
        self.deep = DeepComponent(deep_input_dim, hidden_units)
        self.final = nn.Linear(2, num_classes)
    
    def forward(self, wide_input, deep_input):
        wide_out = self.wide(wide_input)
        deep_out = self.deep(deep_input)
        combined = torch.cat([wide_out, deep_out], dim=1)
        return self.final(combined)


class WideDeepTrainer:
    def __init__(self, model_dir: str = "model"):
        self.model_dir = model_dir
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_features = []
        self.categorical_features = []
        self.feature_info = {}
        self.training_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Using device: {self.device}")
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        wide_features = []
        deep_features = []
        
        for col in self.numeric_features:
            deep_features.append(df[col].values.reshape(-1, 1))
        
        for col in self.categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
            
            n_categories = len(self.label_encoders[col].classes_)
            one_hot = np.eye(n_categories)[df[col].values]
            wide_features.append(one_hot)
            deep_features.append(one_hot)
        
        wide_input = np.hstack(wide_features) if wide_features else np.zeros((len(df), 1))
        deep_input = np.hstack(deep_features) if deep_features else np.zeros((len(df), 1))
        
        return wide_input, deep_input
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        logger.info("Preprocessing data for Wide & Deep model...")
        
        df_processed = df.copy()
        
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].dtype in ['int64', 'float64']:
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                else:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        if 'target' not in df_processed.columns:
            raise ValueError("Target column 'target' not found")
        
        y = df_processed['target'].values
        df_features = df_processed.drop('target', axis=1)
        
        self.numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df_features.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Numeric features: {self.numeric_features}")
        logger.info(f"Categorical features: {self.categorical_features}")
        
        if len(self.numeric_features) > 0:
            numeric_data = df_features[self.numeric_features].values
            numeric_data = self.scaler.fit_transform(numeric_data)
        else:
            numeric_data = np.zeros((len(df_features), 0))
        
        wide_input, deep_input = self._prepare_features(df_features)
        
        if len(self.numeric_features) > 0:
            deep_input = np.hstack([numeric_data, deep_input])
        
        self.feature_info = {
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'wide_input_dim': wide_input.shape[1],
            'deep_input_dim': deep_input.shape[1],
            'label_encoder_classes': {k: v.classes_.tolist() for k, v in self.label_encoders.items()}
        }
        
        logger.info(f"Wide input dim: {wide_input.shape[1]}")
        logger.info(f"Deep input dim: {deep_input.shape[1]}")
        
        return (torch.FloatTensor(wide_input), 
                torch.FloatTensor(deep_input), 
                y)
    
    def train(self, wide_input: torch.Tensor, deep_input: torch.Tensor, 
              y: np.ndarray, epochs: int = 50, batch_size: int = 64, 
              learning_rate: float = 0.001, hidden_units: List[int] = [256, 128, 64]) -> Dict:
        logger.info("=" * 50)
        logger.info("Training Wide & Deep Model...")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        logger.info(f"Hidden units: {hidden_units}")
        logger.info("=" * 50)
        
        num_classes = len(np.unique(y))
        
        self.model = WideAndDeep(
            wide_input_dim=wide_input.shape[1],
            deep_input_dim=deep_input.shape[1],
            hidden_units=hidden_units,
            num_classes=num_classes
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(wide_input, deep_input, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.training_history = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_wide, batch_deep, batch_y in dataloader:
                batch_wide = batch_wide.to(self.device)
                batch_deep = batch_deep.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_wide, batch_deep)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            scheduler.step()
            accuracy = correct / total
            avg_loss = total_loss / len(dataloader)
            self.training_history.append(accuracy)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        logger.info(f"Training completed! Final accuracy: {self.training_history[-1]:.4f}")
        
        return {
            "status": "success",
            "model_type": "WideAndDeep",
            "epochs": epochs,
            "hidden_units": hidden_units,
            "final_accuracy": self.training_history[-1],
            "training_history": self.training_history
        }
    
    def predict(self, wide_input: torch.Tensor, deep_input: torch.Tensor) -> np.ndarray:
        if self.model is None:
            raise ValueError("No model available. Train a model first.")
        
        self.model.eval()
        with torch.no_grad():
            wide_input = wide_input.to(self.device)
            deep_input = deep_input.to(self.device)
            outputs = self.model(wide_input, deep_input)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, wide_input: torch.Tensor, deep_input: torch.Tensor) -> np.ndarray:
        if self.model is None:
            raise ValueError("No model available. Train a model first.")
        
        self.model.eval()
        with torch.no_grad():
            wide_input = wide_input.to(self.device)
            deep_input = deep_input.to(self.device)
            outputs = self.model(wide_input, deep_input)
            proba = torch.softmax(outputs, dim=1)
            return proba.cpu().numpy()
    
    def save_model(self, filename: str = "wide_deep_model.pt") -> str:
        if self.model is None:
            raise ValueError("No model to save.")
        
        model_path = os.path.join(self.model_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_info': self.feature_info,
            'training_history': self.training_history,
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'label_encoders': {k: v.classes_.tolist() for k, v in self.label_encoders.items()}
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, filename: str = "wide_deep_model.pt"):
        model_path = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.feature_info = checkpoint['feature_info']
        self.training_history = checkpoint.get('training_history', [])
        
        if checkpoint.get('scaler_mean') is not None:
            self.scaler.mean_ = np.array(checkpoint['scaler_mean'])
            self.scaler.scale_ = np.array(checkpoint['scaler_scale'])
        
        for col, classes in checkpoint.get('label_encoders', {}).items():
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].classes_ = np.array(classes)
        
        self.numeric_features = self.feature_info.get('numeric_features', [])
        self.categorical_features = self.feature_info.get('categorical_features', [])
        
        self.model = WideAndDeep(
            wide_input_dim=self.feature_info['wide_input_dim'],
            deep_input_dim=self.feature_info['deep_input_dim'],
            num_classes=3
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {model_path}")
        return self.model
