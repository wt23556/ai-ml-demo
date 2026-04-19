import os
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model_dir: str = "model"):
        self.model_dir = model_dir
        self.model = None
        self.training_history = []
        
        os.makedirs(model_dir, exist_ok=True)
        
    def train(self, X_train, y_train, n_estimators: int = 100, max_depth: int = None,
              min_samples_split: int = 2, min_samples_leaf: int = 1,
              max_features: str = "sqrt", bootstrap: bool = True,
              class_weight: str = None) -> dict:
        logger.info("=" * 50)
        logger.info("Starting model training...")
        logger.info(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
        logger.info(f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
        logger.info(f"max_features={max_features}, bootstrap={bootstrap}, class_weight={class_weight}")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info("=" * 50)
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            warm_start=True
        )
        
        self.training_history = []
        
        for i in range(1, n_estimators + 1):
            self.model.n_estimators = i
            self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_train)
            acc = accuracy_score(y_train, y_pred)
            self.training_history.append(acc)
            
            if i % 20 == 0 or i == n_estimators:
                logger.info(f"Trees: {i}/{n_estimators} - Training Accuracy: {acc:.4f}")
        
        logger.info("Training completed!")
        logger.info(f"Final training accuracy: {self.training_history[-1]:.4f}")
        
        return {
            "status": "success",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "class_weight": class_weight,
            "final_accuracy": self.training_history[-1],
            "training_history": self.training_history
        }
    
    def save_model(self, filename: str = "random_forest_model.joblib") -> str:
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_path = os.path.join(self.model_dir, filename)
        joblib.dump({
            'model': self.model,
            'training_history': self.training_history
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, filename: str = "random_forest_model.joblib"):
        model_path = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        data = joblib.load(model_path)
        self.model = data['model']
        self.training_history = data.get('training_history', [])
        
        logger.info(f"Model loaded from {model_path}")
        return self.model
    
    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise ValueError("No model available. Train or load a model first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise ValueError("No model available. Train or load a model first.")
        
        return self.model.predict_proba(X)
