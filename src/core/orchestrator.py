import yaml
from typing import Dict, Any
from pathlib import Path
import mlflow
from .pipeline_builder import PipelineBuilder
from ..utils.local_deployer import LocalDeployer
from ..utils.logger import setup_logging


class MLLocalOrchestrator:
    """
    Local development orchestrator for ML pipelines
    """

    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_config() if config_path else {}
        self.logger = setup_logging(__name__)
        self.pipeline_builder = PipelineBuilder()
        self.deployer = LocalDeployer()

        # Set MLflow tracking to local directory
        mlflow.set_tracking_uri("file:./mlruns")

    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration from YAML"""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def create_pipeline(self, pipeline_config: Dict[str, Any]) -> str:
        """Create and validate a pipeline from configuration"""
        pipeline_type = pipeline_config.get("pipeline", {}).get("type", "sklearn")

        self.pipeline = self.pipeline_builder.build_pipeline(
            pipeline_type, pipeline_config
        )

        self._validate_pipeline_config(pipeline_config)
        return f"pipeline_{pipeline_type}_{id(self.pipeline)}"

    def run_pipeline(self, data_path: str = None) -> Dict[str, Any]:
        """Execute the complete ML pipeline locally"""
        self.logger.info("Starting local ML pipeline execution...")

        results = {}

        try:
            # 1. Data Processing
            self.logger.info("🔄 Stage 1: Data Processing")
            processed_data = self.pipeline.build_data_processing()(
                data_path or self.config["data_processing"]["input_path"]
            )
            results["data_processing"] = processed_data

            # 2. Model Training
            self.logger.info("🎯 Stage 2: Model Training")
            training_results = self.pipeline.build_training()(processed_data)
            results["training"] = training_results

            # 3. Save model locally
            model_path = self._save_model_locally(training_results["model"])
            results["model_path"] = model_path

            self.logger.info("✅ Pipeline completed successfully!")
            accuracy = training_results['metrics'].get('accuracy', 'N/A')
            self.logger.info(f"📊 Model accuracy: {accuracy}")
            self.logger.info(f"💾 Model saved to: {model_path}")

            return results

        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}")
            raise

    def deploy_model(self, model_path: str, port: int = 8000) -> str:
        """Deploy trained model as local API server"""
        self.logger.info(f"🚀 Deploying model API on port {port}...")

        deployment_id = self.deployer.deploy_api_server(
            model_path=model_path, port=port
        )

        self.logger.info(f"✅ Model API deployed: http://127.0.0.1:{port}")
        self.logger.info(f"   - Web UI: http://127.0.0.1:{port}")
        self.logger.info(f"   - Predict: POST http://127.0.0.1:{port}/predict")
        self.logger.info(f"   - Health: GET http://127.0.0.1:{port}/health")

        return deployment_id

    def _save_model_locally(self, model) -> str:
        """Save model to local workspace"""
        import joblib

        model_path = Path("workspace/models/trained_model.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        return str(model_path)

    def _validate_pipeline_config(self, config: Dict[str, Any]):
        """Validate pipeline configuration"""
        required_sections = ["data_processing", "training"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")

    def stop_servers(self):
        """Stop all running API servers"""
        self.deployer.stop_all()
        self.logger.info("All servers stopped")

    def get_deployment_status(self):
        """Get status of all deployments"""
        return self.deployer.get_status()
