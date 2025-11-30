"""
Dynamic Model Registry - Downloads and manages models from GitHub and other sources
"""

import os
import json
import requests
import joblib
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from urllib.parse import urlparse
import zipfile
import shutil

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Manages ML models from various sources including GitHub"""
    
    def __init__(self, registry_dir: str = "workspace/models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
        self.registry = self._load_registry()
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from file"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save model registry to file"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_id: str, model_info: Dict[str, Any]):
        """Register a model in the registry"""
        self.registry[model_id] = {
            **model_info,
            'registered_at': str(Path().cwd()),
        }
        self._save_registry()
        logger.info(f"Registered model: {model_id}")
    
    def download_from_github(self, repo_url: str, file_path: str, model_id: str) -> Optional[str]:
        """Download model file from GitHub repository"""
        try:
            # Convert GitHub URL to raw content URL
            if 'github.com' in repo_url:
                repo_url = repo_url.replace('github.com', 'raw.githubusercontent.com')
                if '/blob/' in repo_url:
                    repo_url = repo_url.replace('/blob/', '/')
            
            logger.info(f"Downloading model from: {repo_url}")
            response = requests.get(repo_url, timeout=30)
            response.raise_for_status()
            
            # Save model
            model_path = self.registry_dir / f"{model_id}.pkl"
            with open(model_path, 'wb') as f:
                f.write(response.content)
            
            self.register_model(model_id, {
                'source': 'github',
                'url': repo_url,
                'path': str(model_path),
                'status': 'downloaded'
            })
            
            return str(model_path)
        except Exception as e:
            logger.error(f"Failed to download model from GitHub: {e}")
            return None
    
    def download_dataset(self, dataset_url: str, dataset_name: str) -> Optional[str]:
        """Download dataset from URL"""
        try:
            datasets_dir = Path("workspace/data")
            datasets_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading dataset: {dataset_name}")
            response = requests.get(dataset_url, timeout=60, stream=True)
            response.raise_for_status()
            
            dataset_path = datasets_dir / dataset_name
            
            # Handle zip files
            if dataset_url.endswith('.zip'):
                zip_path = datasets_dir / f"{dataset_name}.zip"
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract zip
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(datasets_dir)
                zip_path.unlink()
                
                return str(datasets_dir)
            else:
                with open(dataset_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return str(dataset_path)
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return None
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """Load a registered model"""
        if model_id not in self.registry:
            logger.error(f"Model {model_id} not found in registry")
            return None
        
        model_info = self.registry[model_id]
        model_path = model_info.get('path')
        
        if not model_path or not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            # Try joblib first (sklearn models)
            try:
                return joblib.load(model_path)
            except:
                # Fallback to pickle
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def list_models(self) -> Dict[str, Any]:
        """List all registered models"""
        return self.registry
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        return self.registry.get(model_id)

