import asyncio
import json
import pandas as pd
from datetime import datetime
from typing import AsyncGenerator, Dict, Any
import threading

# Optional imports for real-time streaming
try:
    import websockets
except ImportError:
    websockets = None

try:
    from kafka import KafkaConsumer, KafkaProducer
except ImportError:
    KafkaConsumer = None
    KafkaProducer = None

class RealTimeDataProcessor:
    """Process real-time data streams for ML pipelines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_store = {}
        self.model = None
        self.is_training = False
        
    async def start_websocket_stream(self, uri: str):
        """Start processing WebSocket data stream"""
        if websockets is None:
            raise ImportError("websockets is required. Install with: pip install websockets")
        async with websockets.connect(uri) as websocket:
            async for message in websocket:
                data = json.loads(message)
                processed_features = self._process_realtime_features(data)
                
                # Make real-time prediction
                if self.model:
                    prediction = self.model.predict([processed_features])
                    yield {
                        'timestamp': datetime.now().isoformat(),
                        'features': processed_features,
                        'prediction': prediction[0],
                        'raw_data': data
                    }
                
                # Store for continuous training
                await self._update_feature_store(data, processed_features)
    
    def start_kafka_consumer(self, topic: str, bootstrap_servers: str):
        """Start Kafka consumer for real-time data"""
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='ml-pipeline-consumer'
        )
        
        def consume_messages():
            for message in consumer:
                data = json.loads(message.value.decode('utf-8'))
                self._process_kafka_message(data)
        
        thread = threading.Thread(target=consume_messages, daemon=True)
        thread.start()
        return consumer
    
    def _process_realtime_features(self, data: Dict) -> list:
        """Extract and process features from real-time data"""
        features = []
        
        # Example: Financial data features
        if 'price' in data and 'volume' in data:
            features.extend([
                data.get('price', 0),
                data.get('volume', 0),
                data.get('price', 0) * data.get('volume', 0),  # price_volume
                self._calculate_moving_average(data),
                self._calculate_volatility(data)
            ])
        
        # Example: IoT sensor features
        elif 'sensor_readings' in data:
            readings = data['sensor_readings']
            features.extend([
                readings.get('temperature', 0),
                readings.get('humidity', 0),
                readings.get('pressure', 0),
                self._calculate_sensor_trend(readings)
            ])
        
        return features
    
    def _calculate_moving_average(self, data: Dict) -> float:
        """Calculate simple moving average"""
        price_key = f"price_{data.get('symbol', 'default')}"
        if price_key not in self.feature_store:
            self.feature_store[price_key] = []
        
        prices = self.feature_store[price_key]
        prices.append(data.get('price', 0))
        
        # Keep only last 50 values
        if len(prices) > 50:
            prices.pop(0)
        
        return sum(prices) / len(prices) if prices else 0
    
    async def _update_feature_store(self, raw_data: Dict, features: list):
        """Update feature store with new data"""
        symbol = raw_data.get('symbol', 'default')
        if symbol not in self.feature_store:
            self.feature_store[symbol] = []
        
        self.feature_store[symbol].append({
            'timestamp': datetime.now(),
            'features': features,
            'raw_data': raw_data
        })
        
        # Keep only recent data
        if len(self.feature_store[symbol]) > 1000:
            self.feature_store[symbol] = self.feature_store[symbol][-1000:]
        
        # Trigger retraining if enough new data
        if len(self.feature_store[symbol]) % 100 == 0 and not self.is_training:
            await self._trigger_retraining(symbol)
    
    async def _trigger_retraining(self, symbol: str):
        """Trigger model retraining with new data"""
        self.is_training = True
        try:
            # Get recent data for training
            training_data = self.feature_store[symbol][-500:]
            
            # Prepare features and labels (simplified)
            X = [item['features'] for item in training_data]
            y = [self._create_labels(item) for item in training_data]
            
            # Retrain model (in background)
            await self._retrain_model_async(X, y)
            
        finally:
            self.is_training = False
    
    def _create_labels(self, item: Dict) -> int:
        """Create labels from data item for training"""
        # This is a simplified label creation
        # In a real implementation, this would extract the target from the data
        # For now, create a simple binary label based on some feature
        features = item.get('features', [])
        if not features:
            return 0
        
        # Simple heuristic: label based on first feature value
        # Replace with actual label extraction logic
        return 1 if features[0] > 0.5 else 0
    
    async def _retrain_model_async(self, X, y):
        """Retrain model asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._retrain_model, X, y)
    
    def _retrain_model(self, X, y):
        """Retrain the model with new data"""
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        from pathlib import Path
        
        if len(X) < 10:  # Minimum data requirement
            return
        
        # Retrain model
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(X, y)
        
        # Save updated model
        model_path = Path('workspace/models/current_model.pkl')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"✅ Model retrained with {len(X)} samples")
        print(f"✅ Model retrained with {len(X)} samples")