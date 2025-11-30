import subprocess
import threading
import time
import signal
import sys
from pathlib import Path
import logging

class LocalDeployer:
    """Deploys ML pipelines locally using subprocesses"""
    
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        self.processes = {}
        self.logger = logging.getLogger(__name__)
    
    def deploy_api_server(self, model_path: str, port: int = 8000, host: str = "127.0.0.1"):
        """Deploy a model as a local FastAPI server"""
        
        # Create API server script dynamically
        api_script = self._create_api_script(model_path, port, host)
        script_path = self.workspace_dir / f"api_server_{port}.py"
        script_path.write_text(api_script)
        
        # Start the server process
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        deployment_id = f"api_server_{port}"
        self.processes[deployment_id] = process
        
        # Wait for server to start
        self._wait_for_server(host, port, timeout=10)
        
        self.logger.info(f"API server deployed on http://{host}:{port}")
        return deployment_id
    
    def _create_api_script(self, model_path: str, port: int, host: str) -> str:
        """Create a standalone FastAPI server script"""
        # Read UI HTML file
        ui_path = Path(__file__).parent.parent.parent / "ui" / "index.html"
        ui_content = ""
        if ui_path.exists():
            ui_content = ui_path.read_text()
            # For triple-quoted strings, we need to escape triple quotes and backslashes
            # But preserve newlines (they work in triple quotes)
            ui_content = ui_content.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
        
        return f'''
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# Load the model
model = joblib.load("{model_path}")

app = FastAPI(title="Local ML Model Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    features: list

# Serve UI
ui_html = """{ui_content}"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return ui_html

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        features_array = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features_array)
        return {{"prediction": prediction.tolist()[0]}}
    except Exception as e:
        return JSONResponse(status_code=500, content={{"error": str(e)}})

@app.get("/health")
async def health():
    return {{"status": "healthy", "model_loaded": True}}

if __name__ == "__main__":
    uvicorn.run(app, host="{host}", port={port})
'''
    
    def _wait_for_server(self, host: str, port: int, timeout: int = 10):
        """Wait for server to become available"""
        import socket
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex((host, port))
                    if result == 0:
                        return True
            except:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"Server didn't start within {timeout} seconds")
    
    def stop_deployment(self, deployment_id: str):
        """Stop a running deployment"""
        if deployment_id in self.processes:
            self.processes[deployment_id].terminate()
            self.processes[deployment_id].wait()
            del self.processes[deployment_id]
            self.logger.info(f"Stopped deployment: {deployment_id}")
    
    def stop_all(self):
        """Stop all running deployments"""
        for deployment_id in list(self.processes.keys()):
            self.stop_deployment(deployment_id)
    
    def get_status(self):
        """Get status of all deployments"""
        status = {}
        for deployment_id, process in self.processes.items():
            status[deployment_id] = {
                "running": process.poll() is None,
                "returncode": process.poll()
            }
        return status