#!/usr/bin/env python3
"""
IoT Sensor Analytics Server
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import asyncio
from examples.iot_sensors.sensor_pipeline import IoTSensorPipeline

app = FastAPI(title="IoT Sensor Analytics Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = IoTSensorPipeline()

class SensorReading(BaseModel):
    temperature: float
    humidity: float
    pressure: float
    vibration: float

# Read UI HTML
ui_path = Path(__file__).parent / "ui" / "iot_sensors.html"
ui_html = ""
if ui_path.exists():
    ui_html = ui_path.read_text()

@app.get("/", response_class=HTMLResponse)
async def root():
    return ui_html

@app.post("/iot/analyze")
async def analyze_sensor(reading: SensorReading):
    """Analyze sensor reading for anomalies"""
    try:
        reading_dict = {
            'sensor_id': 'manual_input',
            'timestamp': None,  # Will be set by detect_anomaly
            'temperature': reading.temperature,
            'humidity': reading.humidity,
            'pressure': reading.pressure,
            'vibration': reading.vibration
        }
        
        # Detect anomalies
        is_anomaly = await pipeline._detect_anomaly(reading_dict)
        
        # Determine which metrics are anomalous
        anomalies = []
        if reading.temperature > 24.0 or reading.temperature < 19.0:
            anomalies.append("temperature")
        if reading.humidity > 55.0 or reading.humidity < 45.0:
            anomalies.append("humidity")
        if reading.vibration > 8.0:
            anomalies.append("vibration")
        
        return {
            "is_anomaly": is_anomaly,
            "anomalies": anomalies,
            "temperature": reading.temperature,
            "humidity": reading.humidity,
            "pressure": reading.pressure,
            "vibration": reading.vibration
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/iot/health")
async def health():
    return {"status": "healthy", "service": "iot_sensor_analytics"}

if __name__ == "__main__":
    print("🏭 Starting IoT Sensor Analytics Server...")
    print("🌐 UI available at: http://127.0.0.1:8001")
    print("🔗 API available at: http://127.0.0.1:8001/iot/analyze")
    uvicorn.run(app, host="127.0.0.1", port=8001)

