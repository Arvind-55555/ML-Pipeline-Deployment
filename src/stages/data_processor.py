import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import mlflow

class DataProcessor:
    """Handles data processing and feature engineering"""
    
    def __init__(self):
        self.scaler = None
        self.encoder = None
        
    def build_sklearn_processor(self, config: Dict[str, Any]):
        """Build data processor for sklearn pipelines"""
        
        def process_data(data_path: str):
            # Extract data_processing config
            data_config = config.get('data_processing', config)
            
            # Load data
            df = pd.read_csv(data_path)
            
            # Apply transformations based on config
            processed_data = self._apply_transformations(df, data_config)
            
            # Split data
            X_train, X_test, y_train, y_test = self._split_data(
                processed_data, data_config
            )
            
            # Log with MLflow
            try:
                with mlflow.start_run():
                    mlflow.log_params(data_config)
            except Exception:
                # MLflow logging is optional, continue if it fails
                pass
                
            return {
                'X_train': X_train,
                'X_test': X_test, 
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': list(X_train.columns)
            }
        
        return process_data
    
    def _apply_transformations(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Apply data transformations"""
        
        # Handle missing values
        if 'missing_values' in config:
            strategy = config['missing_values']['strategy']
            if strategy == 'mean':
                df = df.fillna(df.mean())
            elif strategy == 'median':
                df = df.fillna(df.median())
        
        # Scale numerical features
        if 'scaling' in config and config['scaling']['enabled']:
            numerical_features = config['scaling']['features']
            self.scaler = StandardScaler()
            df[numerical_features] = self.scaler.fit_transform(
                df[numerical_features]
            )
        
        # Encode categorical features
        if 'encoding' in config and config['encoding']['enabled']:
            categorical_features = config['encoding']['features']
            self.encoder = LabelEncoder()
            for feature in categorical_features:
                df[feature] = self.encoder.fit_transform(df[feature])
        
        return df
    
    def _split_data(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Split data into train/test sets"""
        target = config['target_column']
        test_size = config.get('test_size', 0.2)
        random_state = config.get('random_state', 42)
        
        X = df.drop(columns=[target])
        y = df[target]
        
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )