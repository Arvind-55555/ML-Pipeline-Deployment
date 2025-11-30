from abc import ABC, abstractmethod
from typing import Dict, Any
from ..stages.data_processor import DataProcessor
from ..stages.model_trainer import ModelTrainer
from ..stages.model_serving import ModelServer

class Pipeline(ABC):
    """Abstract base class for ML pipelines"""
    
    @abstractmethod
    def build_data_processing(self, config: Dict[str, Any]):
        pass
    
    @abstractmethod
    def build_training(self, config: Dict[str, Any]):
        pass
    
    @abstractmethod
    def build_serving(self, config: Dict[str, Any]):
        pass

class SklearnPipeline(Pipeline):
    """Scikit-learn based pipeline"""
    
    def __init__(self):
        self.config = None
    
    def build_data_processing(self, config: Dict[str, Any] = None):
        if config is None:
            config = self.config
        return DataProcessor().build_sklearn_processor(config)
    
    def build_training(self, config: Dict[str, Any] = None):
        if config is None:
            config = self.config
        return ModelTrainer().build_sklearn_trainer(config)
    
    def build_serving(self, config: Dict[str, Any] = None):
        if config is None:
            config = self.config
        return ModelServer().build_sklearn_server(config)

class PyTorchPipeline(Pipeline):
    """PyTorch based pipeline"""
    
    def __init__(self):
        self.config = None
    
    def build_data_processing(self, config: Dict[str, Any] = None):
        if config is None:
            config = self.config
        return DataProcessor().build_pytorch_processor(config)
    
    def build_training(self, config: Dict[str, Any] = None):
        if config is None:
            config = self.config
        return ModelTrainer().build_pytorch_trainer(config)
    
    def build_serving(self, config: Dict[str, Any] = None):
        if config is None:
            config = self.config
        return ModelServer().build_pytorch_server(config)

class PipelineBuilder:
    """Factory for building different types of ML pipelines"""
    
    def build_pipeline(self, pipeline_type: str, config: Dict[str, Any]):
        pipeline_classes = {
            "sklearn": SklearnPipeline,
            "pytorch": PyTorchPipeline,
        }
        
        if pipeline_type not in pipeline_classes:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}. Supported types: {list(pipeline_classes.keys())}")
        
        pipeline_class = pipeline_classes[pipeline_type]
        pipeline_instance = pipeline_class()
        # Store config in the pipeline instance for later use
        pipeline_instance.config = config
        return pipeline_instance