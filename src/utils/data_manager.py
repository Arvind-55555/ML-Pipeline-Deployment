"""
Data Manager - Downloads and manages datasets from various sources
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import zipfile
import json

logger = logging.getLogger(__name__)

class DataManager:
    """Manages datasets from GitHub, Kaggle, and other sources"""
    
    # Popular datasets from GitHub and other sources
    DATASET_REGISTRY = {
        "iris": {
            "name": "Iris Dataset",
            "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
            "type": "csv",
            "description": "Classic iris flower dataset"
        },
        "titanic": {
            "name": "Titanic Dataset",
            "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            "type": "csv",
            "description": "Titanic passenger survival dataset"
        },
        "wine": {
            "name": "Wine Quality Dataset",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "type": "csv",
            "description": "Wine quality prediction dataset"
        },
        "house_prices": {
            "name": "House Prices",
            "url": "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
            "type": "csv",
            "description": "California housing prices dataset"
        }
    }
    
    def __init__(self, data_dir: str = "workspace/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.data_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save dataset metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def download_dataset(self, dataset_id: str) -> Optional[str]:
        """Download a dataset by ID"""
        if dataset_id not in self.DATASET_REGISTRY:
            logger.error(f"Dataset {dataset_id} not found in registry")
            return None
        
        dataset_info = self.DATASET_REGISTRY[dataset_id]
        url = dataset_info['url']
        
        try:
            logger.info(f"Downloading dataset: {dataset_info['name']}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Save dataset
            file_path = self.data_dir / f"{dataset_id}.csv"
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Load and get basic info
            df = pd.read_csv(file_path)
            
            self.metadata[dataset_id] = {
                "name": dataset_info['name'],
                "path": str(file_path),
                "rows": len(df),
                "columns": list(df.columns),
                "description": dataset_info.get('description', ''),
                "downloaded_at": str(Path().cwd())
            }
            self._save_metadata()
            
            logger.info(f"✅ Dataset downloaded: {file_path} ({len(df)} rows)")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return None
    
    def load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load a downloaded dataset"""
        if dataset_id not in self.metadata:
            # Try to download if not found
            if dataset_id in self.DATASET_REGISTRY:
                self.download_dataset(dataset_id)
            else:
                logger.error(f"Dataset {dataset_id} not found")
                return None
        
        dataset_path = self.metadata[dataset_id]['path']
        if not Path(dataset_path).exists():
            logger.error(f"Dataset file not found: {dataset_path}")
            return None
        
        try:
            return pd.read_csv(dataset_path)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None
    
    def list_datasets(self) -> Dict[str, Any]:
        """List all available datasets"""
        return {
            "available": self.DATASET_REGISTRY,
            "downloaded": self.metadata
        }
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a dataset"""
        if dataset_id in self.metadata:
            return self.metadata[dataset_id]
        elif dataset_id in self.DATASET_REGISTRY:
            return self.DATASET_REGISTRY[dataset_id]
        return None

