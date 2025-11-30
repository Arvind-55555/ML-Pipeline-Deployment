import mlflow
import mlflow.sklearn
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Optional imports
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def build_sklearn_trainer(self, config: Dict[str, Any]):
        """Build sklearn model trainer"""
        
        def train_model(data: Dict[str, Any]):
            X_train, X_test = data['X_train'], data['X_test']
            y_train, y_test = data['y_train'], data['y_test']
            
            # Get model configuration
            model_config = config['training']['model']
            model_type = model_config['type']
            
            # Initialize model
            if model_type == 'random_forest':
                model = RandomForestClassifier(**model_config['params'])
            elif model_type == 'xgboost':
                if xgb is None:
                    raise ImportError("xgboost is not installed. Install it with: pip install xgboost")
                model = xgb.XGBClassifier(**model_config['params'])
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model with MLflow tracking
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(model_config['params'])
                mlflow.log_param("model_type", model_type)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Save model artifacts
                model_info = {
                    'model': model,
                    'metrics': {
                        'accuracy': accuracy,
                        'f1_score': f1
                    },
                    'run_id': mlflow.active_run().info.run_id
                }
                
                return model_info
        
        return train_model