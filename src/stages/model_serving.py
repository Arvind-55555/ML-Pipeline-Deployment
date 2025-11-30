from fastapi import FastAPI, HTTPException
import uvicorn
from typing import List, Dict, Any
import mlflow.pyfunc


class ModelServer:
    """Handles model serving and API creation"""

    def build_sklearn_server(self, config: Dict[str, Any]):
        """Build FastAPI server for sklearn models"""

        class SklearnModelServer:
            def __init__(self, model_path: str):
                self.model = mlflow.pyfunc.load_model(model_path)
                self.app = FastAPI(title="ML Model Server")
                self._setup_routes()

            def _setup_routes(self):
                @self.app.post("/predict")
                async def predict(features: List[List[float]]):
                    try:
                        predictions = self.model.predict(features)
                        return {"predictions": predictions.tolist()}
                    except Exception as e:
                        raise HTTPException(500, str(e))

                @self.app.get("/health")
                async def health():
                    return {"status": "healthy"}

                @self.app.get("/metadata")
                async def metadata():
                    return {
                        "model_type": "sklearn",
                        "input_shape": "variable",
                        "version": "1.0.0",
                    }

            def serve(self, host: str = "0.0.0.0", port: int = 8000):
                uvicorn.run(self.app, host=host, port=port)

        return SklearnModelServer
