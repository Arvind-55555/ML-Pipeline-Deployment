# 🌺 ML Pipeline UI Guide

## Quick Start

### Option 1: Complete Pipeline (Train + Serve with UI)
```bash
python run_local.py full-deploy
```

This will:
1. Train the model on the iris dataset
2. Start the API server with web UI
3. Open your browser to `http://127.0.0.1:8000`

### Option 2: Step by Step

1. **Train the model:**
   ```bash
   python run_local.py train
   ```

2. **Serve the model with UI:**
   ```bash
   python run_local.py serve --model workspace/models/trained_model.pkl
   ```

3. **Open your browser:**
   Navigate to `http://127.0.0.1:8000`

## Using the Web UI

The web interface allows you to:

1. **Input Features:**
   - Sepal Length (cm)
   - Sepal Width (cm)
   - Petal Length (cm)
   - Petal Width (cm)

2. **Get Predictions:**
   - Click "🔮 Predict Species" button
   - The model will predict one of three iris species:
     - **Setosa** (ID: 0)
     - **Versicolor** (ID: 1)
     - **Virginica** (ID: 2)

3. **Example Values:**
   - Setosa: 5.1, 3.5, 1.4, 0.2
   - Versicolor: 7.0, 3.2, 4.7, 1.4
   - Virginica: 6.3, 3.3, 6.0, 2.5

## API Endpoints

- **Web UI:**** `http://127.0.0.1:8000/`
- **Predict:** `POST http://127.0.0.1:8000/predict`
- **Health:** `GET http://127.0.0.1:8000/health`

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running, or use:
```bash
python run_local.py stop
```

## Troubleshooting

- **UI not loading?** Make sure the server is running and check the port (default: 8000)
- **Prediction errors?** Verify the model file exists at `workspace/models/trained_model.pkl`
- **Connection refused?** Ensure no other service is using port 8000

