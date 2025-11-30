import schedule
import time
import threading
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import joblib

class ContinuousTrainingPipeline:
    """Continuous training and model refresh pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        self.thread = None
        self.current_model = None
        self.model_version = 1
        self.performance_history = []
        
    def start_continuous_training(self, data_fetcher: Callable, validation_data: tuple):
        """Start continuous training in background thread"""
        self.is_running = True
        self.thread = threading.Thread(
            target=self._training_loop,
            args=(data_fetcher, validation_data),
            daemon=True
        )
        self.thread.start()
        print("🔄 Continuous training pipeline started")
    
    def _training_loop(self, data_fetcher: Callable, validation_data: tuple):
        """Main training loop running in background"""
        
        # Setup training schedule
        schedule.every().day.at("02:00").do(self._retrain_model, data_fetcher, validation_data)  # Nightly
        schedule.every(1).hours.do(self._check_model_drift, validation_data)  # Drift monitoring
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _retrain_model(self, data_fetcher: Callable, validation_data: Tuple):
        """Retrain model with latest data"""
        print(f"🔄 [{datetime.now()}] Starting model retraining...")
        
        try:
            # Fetch latest data
            X_new, y_new = data_fetcher()
            
            if len(X_new) < 100:  # Minimum data check
                print("⚠️  Not enough data for retraining")
                return
            
            # Train new model
            from sklearn.ensemble import RandomForestClassifier
            new_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            new_model.fit(X_new, y_new)
            
            # Validate new model
            X_val, y_val = validation_data
            new_accuracy = accuracy_score(y_val, new_model.predict(X_val))
            
            # Compare with current model
            current_accuracy = self._evaluate_current_model(X_val, y_val)
            
            # Deploy if better or significant drift
            improvement = new_accuracy - current_accuracy
            if improvement > 0.02 or new_accuracy > 0.85:  # 2% improvement or good absolute performance
                self._deploy_new_model(new_model, new_accuracy)
                print(f"✅ New model deployed (v{self.model_version}), accuracy: {new_accuracy:.3f}")
            else:
                print(f"ℹ️  Model not deployed. Improvement: {improvement:.3f}")
                
        except Exception as e:
            print(f"❌ Retraining failed: {str(e)}")
    
    def _check_model_drift(self, validation_data: Tuple):
        """Check for model performance drift"""
        X_val, y_val = validation_data
        
        if self.current_model is None:
            return
        
        current_accuracy = accuracy_score(y_val, self.current_model.predict(X_val))
        
        # Store performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'accuracy': current_accuracy
        })
        
        # Check for significant drift (last 10 measurements)
        if len(self.performance_history) >= 10:
            recent_accuracies = [p['accuracy'] for p in self.performance_history[-10:]]
            baseline = np.mean(recent_accuracies[:5])  # First half as baseline
            current_avg = np.mean(recent_accuracies[-5:])  # Recent performance
            
            drift = baseline - current_avg
            if drift > 0.05:  # 5% performance drop
                print(f"🚨 Model drift detected: {drift:.3f}")
                # Trigger emergency retraining
                self._trigger_emergency_retraining(validation_data)
    
    def _evaluate_current_model(self, X_val, y_val) -> float:
        """Evaluate current model performance"""
        if self.current_model is None:
            return 0.0
        return accuracy_score(y_val, self.current_model.predict(X_val))
    
    def _trigger_emergency_retraining(self, validation_data: Tuple):
        """Trigger emergency model retraining due to drift"""
        print("🚨 Triggering emergency retraining due to model drift...")
        # In a real implementation, this would fetch fresh data and retrain immediately
        # For now, just log the event
        pass
    
    def _deploy_new_model(self, new_model: Any, accuracy: float):
        """Deploy new model version"""
        self.current_model = new_model
        self.model_version += 1
        
        # Save model with versioning
        model_path = f"workspace/models/model_v{self.model_version}.pkl"
        joblib.dump(new_model, model_path)
        
        # Update current model symlink
        current_path = "workspace/models/current_model.pkl"
        if os.path.exists(current_path):
            os.remove(current_path)
        os.symlink(model_path, current_path)
        
        # Log deployment
        with mlflow.start_run(run_name=f"deployment_v{self.model_version}"):
            mlflow.log_metric("deployment_accuracy", accuracy)
            mlflow.log_param("model_version", self.model_version)
            mlflow.sklearn.log_model(new_model, "model")
        
        # Update API servers if running
        self._refresh_servers()
    
    def _refresh_servers(self):
        """Refresh running API servers with new model"""
        # Implementation to update servers
        print("🔄 Refreshing model servers...")
    
    def stop_continuous_training(self):
        """Stop continuous training pipeline"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=10)
        print("🛑 Continuous training stopped")