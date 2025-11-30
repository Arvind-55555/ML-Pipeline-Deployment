"""
Real-time Fraud Detection Pipeline
Based on Microsoft E-commerce Fraud Detection
Repository: https://github.com/microsoft/ecommerce-fraud-detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import asyncio
import json

logger = logging.getLogger(__name__)

class FraudDetectionPipeline:
    """Real-time fraud detection pipeline"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_store = {}
        self.alert_threshold = 0.7
        self.config = {
            'data_url': 'https://raw.githubusercontent.com/microsoft/ecommerce-fraud-detection/main/data/sample_transactions.csv',
            'window_size': 100,  # For real-time feature calculation
            'retrain_interval': 1000  # Retrain after N transactions
        }
    
    def ingest_streaming_data(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Streaming Data Ingestion"""
        transaction['timestamp'] = datetime.now().isoformat()
        transaction['transaction_id'] = transaction.get('transaction_id', 
                                                      f"txn_{datetime.now().timestamp()}")
        
        # Store in feature store
        if 'user_id' in transaction:
            user_id = transaction['user_id']
            if user_id not in self.feature_store:
                self.feature_store[user_id] = []
            self.feature_store[user_id].append(transaction)
            
            # Keep only recent transactions
            if len(self.feature_store[user_id]) > self.config['window_size']:
                self.feature_store[user_id] = self.feature_store[user_id][-self.config['window_size']:]
        
        logger.debug(f"✅ Transaction ingested: {transaction['transaction_id']}")
        return transaction
    
    def calculate_realtime_features(self, transaction: Dict[str, Any]) -> np.ndarray:
        """Real-time Feature Calculation"""
        user_id = transaction.get('user_id', 'unknown')
        user_history = self.feature_store.get(user_id, [])
        
        # Base features
        features = [
            transaction.get('amount', 0),
            transaction.get('hour', datetime.now().hour),
            transaction.get('day_of_week', datetime.now().weekday()),
        ]
        
        # Historical features
        if len(user_history) > 0:
            recent_amounts = [t.get('amount', 0) for t in user_history[-10:]]
            features.extend([
                np.mean(recent_amounts),  # avg_amount
                np.std(recent_amounts) if len(recent_amounts) > 1 else 0,  # std_amount
                len(user_history),  # transaction_count
                sum(1 for t in user_history if t.get('amount', 0) > transaction.get('amount', 0)),  # larger_transactions
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Time-based features
        if 'timestamp' in transaction:
            try:
                ts = datetime.fromisoformat(transaction['timestamp'])
                features.extend([
                    ts.hour,
                    ts.weekday(),
                    ts.day
                ])
            except:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features).reshape(1, -1)
    
    async def model_inference(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Model Inference"""
        if self.model is None:
            return {
                'fraud_probability': 0.0,
                'is_fraud': False,
                'error': 'Model not loaded'
            }
        
        try:
            # Calculate features
            features = self.calculate_realtime_features(transaction)
            
            # Scale if scaler available
            if self.scaler:
                features = self.scaler.transform(features)
            
            # Predict
            fraud_prob = self.model.predict_proba(features)[0][1]
            is_fraud = fraud_prob >= self.alert_threshold
            
            result = {
                'transaction_id': transaction.get('transaction_id'),
                'fraud_probability': float(fraud_prob),
                'is_fraud': bool(is_fraud),
                'timestamp': datetime.now().isoformat()
            }
            
            # Generate alert if fraud detected
            if is_fraud:
                await self.generate_alert(result, transaction)
            
            return result
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                'fraud_probability': 0.0,
                'is_fraud': False,
                'error': str(e)
            }
    
    async def generate_alert(self, prediction: Dict[str, Any], transaction: Dict[str, Any]):
        """Alert Generation"""
        alert = {
            'alert_id': f"alert_{datetime.now().timestamp()}",
            'transaction_id': prediction['transaction_id'],
            'fraud_probability': prediction['fraud_probability'],
            'severity': 'HIGH' if prediction['fraud_probability'] > 0.9 else 'MEDIUM',
            'timestamp': datetime.now().isoformat(),
            'transaction_details': transaction
        }
        
        logger.warning(f"🚨 FRAUD ALERT: {alert['alert_id']} - Probability: {alert['fraud_probability']:.3f}")
        return alert
    
    def train_model(self, transactions: List[Dict[str, Any]], labels: List[int]):
        """Model Training"""
        from sklearn.ensemble import IsolationForest, RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # Prepare features
        X = np.array([self.calculate_realtime_features(t).flatten() for t in transactions])
        y = np.array(labels)
        
        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        logger.info(f"✅ Fraud detection model trained on {len(transactions)} transactions")
        return self.model
    
    async def retrain_model(self, new_transactions: List[Dict[str, Any]]):
        """Model Retraining"""
        if len(new_transactions) < self.config['retrain_interval']:
            return
        
        logger.info("🔄 Retraining fraud detection model...")
        # In production, this would use labeled data
        # For now, use anomaly detection approach
        from sklearn.ensemble import IsolationForest
        
        X = np.array([self.calculate_realtime_features(t).flatten() for t in new_transactions])
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Use Isolation Forest for unsupervised learning
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.model.fit(X)
        
        logger.info("✅ Model retrained")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Performance Monitoring"""
        total_transactions = sum(len(history) for history in self.feature_store.values())
        return {
            'total_transactions_processed': total_transactions,
            'unique_users': len(self.feature_store),
            'model_loaded': self.model is not None,
            'alert_threshold': self.alert_threshold
        }

