"""
Customer Churn Prediction Pipeline
Based on IBM Telco Customer Churn Dataset
Repository: https://github.com/IBM/telco-customer-churn-prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import requests
import io

logger = logging.getLogger(__name__)

class CustomerChurnPipeline:
    """Customer churn prediction pipeline with full ML lifecycle"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.config = {
            'data_url': 'https://raw.githubusercontent.com/IBM/telco-customer-churn-prediction/master/WA_Fn-UseC_-Telco-Customer-Churn.csv',
            'target_column': 'Churn',
            'test_size': 0.2,
            'random_state': 42
        }
    
    def ingest_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Data Ingestion - CSV or Database"""
        try:
            if data_path:
                df = pd.read_csv(data_path)
            else:
                # Download from GitHub
                logger.info("Downloading Telco Customer Churn dataset...")
                response = requests.get(self.config['data_url'], timeout=30)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text))
            
            logger.info(f"✅ Data ingested: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature Engineering"""
        df = df.copy()
        
        # Handle missing values
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Create new features
        df['MonthlyChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        df['TotalChargesPerTenure'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Encode categorical variables
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                          'MultipleLines', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = pd.Categorical(df[col]).codes
        
        # Encode target
        df['Churn'] = (df['Churn'] == 'Yes').astype(int)
        
        logger.info("✅ Feature engineering completed")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training"""
        from sklearn.preprocessing import StandardScaler
        
        # Select features
        feature_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                       'MonthlyChargesPerTenure', 'TotalChargesPerTenure',
                       'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                       'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod']
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols]
        y = df[self.config['target_column']]
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, available_cols
    
    def train_model(self, X: np.ndarray, y: pd.Series, algorithm: str = 'random_forest') -> Dict[str, Any]:
        """Model Training with Multiple Algorithms"""
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=y
        )
        
        # Select algorithm
        if algorithm == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        elif algorithm == 'gradient_boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif algorithm == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train
        model.fit(X_train, y_train)
        self.model = model
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"✅ Model trained - Algorithm: {algorithm}")
        logger.info(f"   Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'algorithm': algorithm
        }
    
    def hyperparameter_tuning(self, X: np.ndarray, y: pd.Series) -> Dict[str, Any]:
        """Hyperparameter Tuning"""
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
        
        base_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, 
            scoring='f1', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        
        logger.info(f"✅ Hyperparameter tuning completed")
        logger.info(f"   Best params: {grid_search.best_params_}")
        logger.info(f"   Best score: {grid_search.best_score_:.3f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'model': self.model
        }
    
    def predict(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a customer"""
        if self.model is None:
            # Return mock prediction if model not trained
            return {
                'churn_prediction': False,
                'churn_probability': 0.3,
                'no_churn_probability': 0.7,
                'warning': 'Model not trained. Using mock prediction. Train model first for accurate results.'
            }
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([customer_data])
            
            # Apply feature engineering
            df = self.feature_engineering(df)
            
            # Prepare features
            X, _, feature_cols = self.prepare_features(df)
            
            # Scale if scaler available
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Predict
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            return {
                'churn_prediction': bool(prediction),
                'churn_probability': float(probability[1]),
                'no_churn_probability': float(probability[0])
            }
        except Exception as e:
            # Fallback to mock prediction on error
            logger.warning(f"Prediction error: {e}, using fallback")
            return {
                'churn_prediction': False,
                'churn_probability': 0.3,
                'no_churn_probability': 0.7,
                'error': str(e)
            }
    
    def evaluate_model(self, X: np.ndarray, y: pd.Series) -> Dict[str, Any]:
        """Model Evaluation"""
        from sklearn.metrics import classification_report, confusion_matrix
        
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist()
        }

