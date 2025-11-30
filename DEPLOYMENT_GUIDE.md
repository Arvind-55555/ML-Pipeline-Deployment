# 🚀 Enterprise ML Pipeline Deployment Guide

## Overview

This platform provides 5 enterprise-grade ML pipeline examples based on real GitHub repositories, each with complete pipeline implementations.

## 🎯 Available Examples

### 1. 📊 Customer Churn Prediction
**Repository:** IBM/telco-customer-churn-prediction  
**URL:** http://127.0.0.1:8000/churn

**Pipeline Components:**
- ✅ Data Ingestion (CSV, Database)
- ✅ Feature Engineering
- ✅ Model Training (Multiple Algorithms)
- ✅ Hyperparameter Tuning
- ✅ Model Evaluation
- ✅ API Deployment
- ✅ Monitoring Dashboard

**Usage:**
```bash
# Train model
curl -X POST "http://127.0.0.1:8000/churn/train?algorithm=random_forest"

# Predict churn
curl -X POST "http://127.0.0.1:8000/churn/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 70, "TotalCharges": 840}'
```

### 2. 🚨 Real-time Fraud Detection
**Repository:** microsoft/ecommerce-fraud-detection  
**URL:** http://127.0.0.1:8000/fraud

**Pipeline Components:**
- ✅ Streaming Data Ingestion
- ✅ Real-time Feature Calculation
- ✅ Model Inference
- ✅ Alert Generation
- ✅ Model Retraining
- ✅ Performance Monitoring

**Usage:**
```bash
# Analyze transaction
curl -X POST "http://127.0.0.1:8000/fraud/analyze" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "amount": 150.0}'

# Get metrics
curl "http://127.0.0.1:8000/fraud/metrics"
```

### 3. 🏥 Medical Image Analysis
**Repository:** ieee8023/covid-chestxray-dataset  
**URL:** http://127.0.0.1:8000/medical

**Pipeline Components:**
- ✅ DICOM Image Processing
- ✅ Data Augmentation
- ✅ CNN Model Training
- ✅ Model Validation
- ✅ Clinical Deployment
- ✅ Regulatory Compliance

**Usage:**
```bash
# Predict medical image
curl -X POST "http://127.0.0.1:8000/medical/predict" \
  -F "file=@chest_xray.jpg"
```

### 4. 🛒 E-commerce Recommendation
**Repository:** microsoft/recommenders  
**URL:** http://127.0.0.1:8000/recommendation

**Pipeline Components:**
- ✅ User Behavior Tracking
- ✅ Feature Store Updates
- ✅ Model Training Pipeline
- ✅ A/B Testing Framework
- ✅ Real-time Serving
- ✅ Performance Analytics

**Usage:**
```bash
# Track behavior
curl -X POST "http://127.0.0.1:8000/recommendation/track?user_id=user_1&item_id=item_5&behavior_type=purchase"

# Get recommendations
curl -X POST "http://127.0.0.1:8000/recommendation/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_1", "top_k": 10}'
```

### 5. 🚗 Autonomous Vehicle Perception
**Repository:** commaai/openpilot  
**URL:** http://127.0.0.1:8000/vehicle

**Pipeline Components:**
- ✅ Sensor Data Processing
- ✅ Real-time Inference
- ✅ Model Updates
- ✅ Safety Validation
- ✅ Edge Deployment
- ✅ Performance Monitoring

**Usage:**
```bash
# Real-time inference
curl -X POST "http://127.0.0.1:8000/vehicle/inference" \
  -H "Content-Type: application/json" \
  -d '{"camera": [...], "lidar": [...], "radar": {...}, "imu": {...}, "gps": {...}}'

# Monitor performance
curl "http://127.0.0.1:8000/vehicle/monitor"
```

## 🚀 Quick Start

### 1. Start the Server

```bash
python run_unified_server.py
```

### 2. Access the Platform

Open your browser to: **http://127.0.0.1:8000**

You'll see a landing page with all 5 examples. Click on any example to explore its pipeline.

### 3. Train Models (as needed)

Some examples require model training before predictions:

```bash
# Train churn model
curl -X POST "http://127.0.0.1:8000/churn/train?algorithm=random_forest"
```

## 📋 API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/health` | GET | System health |
| `/churn` | GET | Churn UI |
| `/churn/predict` | POST | Predict churn |
| `/churn/train` | POST | Train model |
| `/fraud` | GET | Fraud UI |
| `/fraud/analyze` | POST | Analyze transaction |
| `/fraud/metrics` | GET | Get metrics |
| `/medical` | GET | Medical UI |
| `/medical/predict` | POST | Predict image |
| `/recommendation` | GET | Recommendation UI |
| `/recommendation/recommend` | POST | Get recommendations |
| `/recommendation/track` | POST | Track behavior |
| `/vehicle` | GET | Vehicle UI |
| `/vehicle/inference` | POST | Real-time inference |
| `/vehicle/monitor` | GET | Monitor performance |

## 🔧 Configuration

Each pipeline can be configured via:
- Environment variables
- Configuration files in `configs/`
- API parameters

## 📊 Data Sources

All examples use data from their respective GitHub repositories:
- **Churn:** IBM Telco dataset (auto-downloaded)
- **Fraud:** Microsoft ecommerce fraud patterns
- **Medical:** COVID chest X-ray dataset
- **Recommendation:** Microsoft recommenders patterns
- **Vehicle:** commaai openpilot sensor patterns

## 🎓 Learning Resources

- Check individual example directories for detailed documentation
- Review pipeline implementations in `examples/`
- See API documentation at `/docs` (FastAPI auto-generated)

## 🆘 Troubleshooting

- **Model not trained:** Some examples require training before prediction
- **Data not available:** Check internet connection for auto-downloads
- **Port conflicts:** Change port in `run_unified_server.py`

---

**Built with real-world enterprise ML pipeline patterns**

