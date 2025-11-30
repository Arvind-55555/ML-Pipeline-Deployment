# Enterprise ML Pipeline Deployment Platform

A comprehensive, production-ready ML pipeline deployment system with 5 enterprise examples based on real GitHub repositories.

## Overview

This platform provides complete ML pipeline implementations for 5 enterprise use cases, each with:
- Full pipeline lifecycle (data ingestion → training → deployment → monitoring)
- Real-time inference capabilities
- Web-based UI for interaction
- RESTful API endpoints
- Data/model management from GitHub repositories

## Available Examples

### 1. Customer Churn Prediction
**Repository:** [IBM/telco-customer-churn-prediction](https://github.com/IBM/telco-customer-churn-prediction)  
**URL:** http://127.0.0.1:8000/churn

**Complete Pipeline:**
- ✅ Data Ingestion (CSV, Database)
- ✅ Feature Engineering
- ✅ Model Training (Random Forest, Gradient Boosting, Logistic Regression)
- ✅ Hyperparameter Tuning
- ✅ Model Evaluation
- ✅ API Deployment
- ✅ Monitoring Dashboard

**Features:**
- Automatic dataset download from GitHub
- Multiple algorithm support
- Comprehensive evaluation metrics
- Real-time prediction API

### 2. Real-time Fraud Detection
**Repository:** [microsoft/ecommerce-fraud-detection](https://github.com/microsoft/ecommerce-fraud-detection)  
**URL:** http://127.0.0.1:8000/fraud

**Complete Pipeline:**
- ✅ Streaming Data Ingestion
- ✅ Real-time Feature Calculation
- ✅ Model Inference
- ✅ Alert Generation
- ✅ Model Retraining
- ✅ Performance Monitoring

**Features:**
- Real-time transaction processing
- Automatic fraud alerts
- Continuous learning
- Performance metrics tracking

### 3. Medical Image Analysis
**Repository:** [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)  
**URL:** http://127.0.0.1:8000/medical

**Complete Pipeline:**
- ✅ DICOM Image Processing
- ✅ Data Augmentation
- ✅ CNN Model Training
- ✅ Model Validation
- ✅ Clinical Deployment
- ✅ Regulatory Compliance

**Features:**
- Medical image classification
- CNN-based deep learning
- Clinical deployment checklist
- HIPAA compliance considerations

### 4. E-commerce Recommendation
**Repository:** [microsoft/recommenders](https://github.com/microsoft/recommenders)  
**URL:** http://127.0.0.1:8000/recommendation

**Complete Pipeline:**
- ✅ User Behavior Tracking
- ✅ Feature Store Updates
- ✅ Model Training Pipeline
- ✅ A/B Testing Framework
- ✅ Real-time Serving
- ✅ Performance Analytics

**Features:**
- Collaborative filtering
- Matrix factorization
- Real-time recommendations
- A/B testing support

### 5. Autonomous Vehicle Perception
**Repository:** [commaai/openpilot](https://github.com/commaai/openpilot)  
**URL:** http://127.0.0.1:8000/vehicle

**Complete Pipeline:**
- ✅ Sensor Data Processing (Camera, LiDAR, Radar, IMU, GPS)
- ✅ Real-time Inference
- ✅ Model Updates
- ✅ Safety Validation
- ✅ Edge Deployment
- ✅ Performance Monitoring

**Features:**
- Multi-sensor fusion
- Real-time perception
- Safety validation
- Edge deployment support

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Start the Server

```bash
python run_unified_server.py
```

### Access the Platform

Open your browser to: **http://127.0.0.1:8000**

You'll see a beautiful landing page with all 5 examples. Click on any example to explore its complete pipeline.

## 📖 Usage Examples

### Customer Churn Prediction

```python
# Train model
POST http://127.0.0.1:8000/churn/train?algorithm=random_forest

# Predict churn
POST http://127.0.0.1:8000/churn/predict
{
  "tenure": 12,
  "MonthlyCharges": 70.0,
  "TotalCharges": 840.0
}
```

### Real-time Fraud Detection

```python
# Analyze transaction
POST http://127.0.0.1:8000/fraud/analyze
{
  "user_id": "user_123",
  "amount": 150.0
}
```

### Medical Image Analysis

```python
# Predict medical image
POST http://127.0.0.1:8000/medical/predict
# Upload image file via form-data
```

### E-commerce Recommendation

```python
# Track user behavior
POST http://127.0.0.1:8000/recommendation/track?user_id=user_1&item_id=item_5&behavior_type=purchase

# Get recommendations
POST http://127.0.0.1:8000/recommendation/recommend
{
  "user_id": "user_1",
  "top_k": 10
}
```

### Autonomous Vehicle Perception

```python
# Real-time inference
POST http://127.0.0.1:8000/vehicle/inference
{
  "camera": [...],
  "lidar": [...],
  "radar": {...},
  "imu": {...},
  "gps": {...}
}
```

## Architecture

### Directory Structure

```
ml-pipe-deploy/
├── examples/
│   ├── customer_churn/      # IBM Telco Churn
│   ├── fraud_detection/      # Microsoft Fraud Detection
│   ├── medical_image/        # COVID Chest X-ray
│   ├── recommendation/       # Microsoft Recommenders
│   └── autonomous_vehicle/  # commaai openpilot
├── src/
│   ├── core/                 # Orchestration
│   ├── stages/               # Pipeline stages
│   ├── utils/                # Utilities
│   └── deployment/           # Deployment tools
├── ui/                       # Web UIs
├── configs/                  # Configuration files
└── run_unified_server.py     # Main server
```

### Key Components

- **Pipeline Orchestrator** - Manages complete ML lifecycle
- **Model Registry** - Dynamic model management from GitHub
- **Data Manager** - Dataset downloading and management
- **Real-time Processors** - Streaming data handling
- **API Server** - FastAPI-based RESTful APIs
- **Web UIs** - Interactive dashboards for each example

## 📋 API Documentation

All APIs are documented via FastAPI's automatic documentation:
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

## Configuration

Each pipeline can be configured via:
- YAML configuration files in `configs/`
- Environment variables
- API parameters

## Data Sources

All examples automatically download data from their respective GitHub repositories:
- **Churn:** IBM Telco Customer Churn dataset
- **Fraud:** Microsoft ecommerce fraud patterns
- **Medical:** COVID-19 chest X-ray dataset
- **Recommendation:** Microsoft recommenders patterns
- **Vehicle:** commaai openpilot sensor patterns

## 🎓 Learning Resources

- See `DEPLOYMENT_GUIDE.md` for detailed deployment instructions
- Check individual example directories for implementation details
- Review pipeline code in `examples/` for learning patterns

## Development

### Adding New Examples

1. Create pipeline class in `examples/your_example/`
2. Add API endpoints in `run_unified_server.py`
3. Create UI page in `ui/your_example.html`
4. Add to landing page menu

### Extending Pipelines

All pipelines follow a consistent interface:
- `ingest_data()` - Data ingestion
- `feature_engineering()` - Feature creation
- `train_model()` - Model training
- `predict()` - Inference
- `evaluate()` - Evaluation

## Requirements

See `requirements.txt` for complete dependency list. Key packages:
- scikit-learn, pandas, numpy
- fastapi, uvicorn
- transformers, tensorflow (for some examples)
- mlflow (for tracking)

## Troubleshooting

- **Model not trained:** Some examples require training before prediction (use `/train` endpoint)
- **Data download fails:** Check internet connection and GitHub repository availability
- **Port conflicts:** Change port in `run_unified_server.py`

## License

MIT License - See LICENSE file for details

## Acknowledgments

This platform uses data and patterns from:
- IBM Telco Customer Churn Prediction
- Microsoft E-commerce Fraud Detection
- IEEE COVID Chest X-ray Dataset
- Microsoft Recommenders
- commaai openpilot

---

**Built for Enterprise ML Pipeline Deployment** 🚀
