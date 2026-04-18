import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    confusion_matrix,
    classification_report
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, output_dir: str = "static"):
        self.output_dir = output_dir
        self.metrics = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate(self, y_true, y_pred, training_history: list = None) -> dict:
        logger.info("=" * 50)
        logger.info("Starting model evaluation...")
        logger.info("=" * 50)
        
        self.metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"Precision: {self.metrics['precision']:.4f}")
        logger.info(f"Recall: {self.metrics['recall']:.4f}")
        
        self._plot_confusion_matrix(y_true, y_pred)
        
        if training_history:
            self._plot_accuracy_curve(training_history)
        
        logger.info("Evaluation completed!")
        
        return self.metrics
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            cbar=True,
            square=True
        )
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
        
    def _plot_accuracy_curve(self, training_history: list):
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(training_history) + 1)
        plt.plot(epochs, training_history, 'b-', linewidth=2, label='Training Accuracy')
        
        plt.title('Training Accuracy Curve', fontsize=16)
        plt.xlabel('Number of Trees (n_estimators)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'accuracy_curve.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Accuracy curve saved to {save_path}")
    
    def get_classification_report(self, y_true, y_pred) -> str:
        return classification_report(y_true, y_pred, zero_division=0)
