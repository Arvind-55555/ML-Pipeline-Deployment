#!/usr/bin/env python3
"""
Unified ML Pipeline Server - Enterprise Examples
Based on real GitHub repositories
"""

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import logging
import numpy as np

# Import all pipelines
from examples.customer_churn.churn_pipeline import CustomerChurnPipeline
from examples.fraud_detection.fraud_pipeline import FraudDetectionPipeline
from examples.medical_image.medical_pipeline import MedicalImagePipeline
from examples.recommendation.recommendation_pipeline import RecommendationPipeline
from examples.autonomous_vehicle.vehicle_pipeline import AutonomousVehiclePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise ML Pipeline Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipelines
churn_pipeline = CustomerChurnPipeline()
fraud_pipeline = FraudDetectionPipeline()
medical_pipeline = MedicalImagePipeline()
recommendation_pipeline = RecommendationPipeline()
vehicle_pipeline = AutonomousVehiclePipeline()

# Read landing page
landing_path = Path(__file__).parent / "ui" / "landing.html"
landing_html = ""
if landing_path.exists():
    landing_html = landing_path.read_text()
    logger.info("✅ Landing page loaded")


# Request models
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"


class TransactionData(BaseModel):
    user_id: str
    amount: float
    transaction_id: Optional[str] = None
    hour: Optional[int] = None
    day_of_week: Optional[int] = None


class RecommendationRequest(BaseModel):
    user_id: str
    top_k: int = 10


class VehicleSensorData(BaseModel):
    camera: Optional[list] = None
    lidar: Optional[list] = None
    radar: Optional[dict] = None
    imu: Optional[dict] = None
    gps: Optional[dict] = None


# Landing Page
@app.get("/", response_class=HTMLResponse)
async def root():
    return landing_html


# Customer Churn Endpoints
@app.get("/churn", response_class=HTMLResponse)
async def churn_page():
    churn_ui_path = Path(__file__).parent / "ui" / "churn.html"
    if churn_ui_path.exists():
        return churn_ui_path.read_text()
    return "<h1>Customer Churn Prediction</h1><p>UI file not found</p>"


@app.post("/churn/predict")
async def predict_churn(customer: CustomerData):
    """Predict customer churn"""
    try:
        customer_dict = customer.dict()
        result = churn_pipeline.predict(customer_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/churn/train")
async def train_churn_model(algorithm: str = "random_forest"):
    """Train churn prediction model"""
    try:
        # Ingest data
        df = churn_pipeline.ingest_data()

        # Feature engineering
        df = churn_pipeline.feature_engineering(df)

        # Prepare features
        X, y, feature_cols = churn_pipeline.prepare_features(df)

        # Train
        result = churn_pipeline.train_model(X, y, algorithm)

        return {
            "status": "success",
            "metrics": result["metrics"],
            "algorithm": algorithm,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Fraud Detection Endpoints
@app.get("/fraud", response_class=HTMLResponse)
async def fraud_page():
    fraud_ui_path = Path(__file__).parent / "ui" / "fraud.html"
    if fraud_ui_path.exists():
        return fraud_ui_path.read_text()
    return "<h1>Real-time Fraud Detection</h1><p>UI file not found</p>"


@app.post("/fraud/analyze")
async def analyze_fraud(transaction: TransactionData):
    """Analyze transaction for fraud"""
    try:
        transaction_dict = transaction.dict()

        # Ingest
        fraud_pipeline.ingest_streaming_data(transaction_dict)

        # Infer
        result = await fraud_pipeline.model_inference(transaction_dict)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fraud/metrics")
async def fraud_metrics():
    """Get fraud detection performance metrics"""
    return fraud_pipeline.get_performance_metrics()


# Medical Image Endpoints
@app.get("/medical", response_class=HTMLResponse)
async def medical_page():
    medical_ui_path = Path(__file__).parent / "ui" / "medical.html"
    if medical_ui_path.exists():
        return medical_ui_path.read_text()
    return "<h1>Medical Image Analysis</h1><p>UI file not found</p>"


@app.post("/medical/predict")
async def predict_medical_image(file: UploadFile = File(None)):
    """Predict medical image"""
    try:
        if file is None:
            return {
                "prediction": "No image provided",
                "confidence": 0.0,
                "error": "Please upload an image file",
            }

        # Save uploaded file temporarily
        contents = await file.read()
        temp_path = Path(f"workspace/temp_{file.filename}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(contents)

        # Predict
        result = medical_pipeline.predict(str(temp_path))

        # Cleanup
        temp_path.unlink()

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Recommendation Endpoints
@app.get("/recommendation", response_class=HTMLResponse)
async def recommendation_page():
    rec_ui_path = Path(__file__).parent / "ui" / "recommendation.html"
    if rec_ui_path.exists():
        return rec_ui_path.read_text()
    return "<h1>E-commerce Recommendation</h1><p>UI file not found</p>"


@app.post("/recommendation/recommend")
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations for user"""
    try:
        # Initialize with sample behaviors if empty
        if len(recommendation_pipeline.feature_store["user_behaviors"]) == 0:
            # Add sample behaviors for demo
            sample_users = [f"user_{i}" for i in range(1, 6)]
            sample_items = [f"item_{i}" for i in range(1, 21)]
            behaviors = ["view", "click", "purchase", "rating"]

            for user in sample_users:
                for _ in range(10):
                    item = np.random.choice(sample_items)
                    behavior = np.random.choice(behaviors)
                    metadata = (
                        {"rating": np.random.randint(1, 6)}
                        if behavior == "rating"
                        else {}
                    )
                    recommendation_pipeline.track_user_behavior(
                        user, item, behavior, metadata
                    )

            logger.info("✅ Initialized with sample user behaviors")

        # Update feature store
        recommendation_pipeline.update_feature_store()

        # Train model if not trained
        if recommendation_pipeline.model is None:
            recommendation_pipeline.train_model()

        # Get recommendations
        recommendations = recommendation_pipeline.recommend(
            request.user_id, request.top_k
        )

        return {"user_id": request.user_id, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommendation/track")
async def track_behavior(user_id: str, item_id: str, behavior_type: str):
    """Track user behavior"""
    recommendation_pipeline.track_user_behavior(user_id, item_id, behavior_type)
    return {"status": "tracked"}


@app.get("/recommendation/analytics")
async def recommendation_analytics():
    """Get recommendation analytics"""
    return recommendation_pipeline.get_performance_analytics()


# Autonomous Vehicle Endpoints
@app.get("/vehicle", response_class=HTMLResponse)
async def vehicle_page():
    vehicle_ui_path = Path(__file__).parent / "ui" / "vehicle.html"
    if vehicle_ui_path.exists():
        return vehicle_ui_path.read_text()
    return "<h1>Autonomous Vehicle Perception</h1><p>UI file not found</p>"


@app.post("/vehicle/inference")
async def vehicle_inference(sensor_data: VehicleSensorData):
    """Real-time vehicle perception inference"""
    try:
        sensor_dict = sensor_data.dict()
        result = await vehicle_pipeline.realtime_inference(sensor_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vehicle/monitor")
async def vehicle_monitor():
    """Get vehicle performance metrics"""
    return vehicle_pipeline.monitor_performance()


# Health Check
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "services": {
            "customer_churn": True,
            "fraud_detection": True,
            "medical_image": True,
            "recommendation": True,
            "autonomous_vehicle": True,
        },
    }


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Enterprise ML Pipeline Server")
    print("=" * 70)
    print("🌐 Landing Page: http://122.0.0.1:8000")
    print("📊 Customer Churn: http://127.0.0.1:8000/churn")
    print("🚨 Fraud Detection: http://127.0.0.1:8000/fraud")
    print("🏥 Medical Image: http://127.0.0.1:8000/medical")
    print("🛒 Recommendation: http://127.0.0.1:8000/recommendation")
    print("🚗 Autonomous Vehicle: http://127.0.0.1:8000/vehicle")
    print("❤️  Health: http://127.0.0.1:8000/health")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000)
