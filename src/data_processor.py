import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, data_path: str = "data/dataset.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.numeric_features = []
        self.categorical_features = []
        
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def preprocess(self, df: pd.DataFrame) -> tuple:
        logger.info("Starting data preprocessing...")
        
        df_processed = df.copy()
        
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].dtype in ['int64', 'float64']:
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                    logger.info(f"Filled missing values in {col} with mean")
                else:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                    logger.info(f"Filled missing values in {col} with mode")
        
        if 'target' not in df_processed.columns:
            raise ValueError("Target column 'target' not found in dataset")
        
        X = df_processed.drop('target', axis=1)
        y = df_processed['target']
        
        self.feature_names = X.columns.tolist()
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Feature names: {self.feature_names}")
        logger.info(f"Numeric features: {self.numeric_features}")
        logger.info(f"Categorical features: {self.categorical_features}")
        
        for col in self.categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        logger.info("Feature normalization completed using StandardScaler")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def transform_input(self, data: dict) -> np.ndarray:
        df = pd.DataFrame([data])
        
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.feature_names]
        
        for col in self.categorical_features:
            if col in df.columns and col in self.label_encoders:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    df[col] = 0
        
        df_scaled = self.scaler.transform(df)
        
        return df_scaled
