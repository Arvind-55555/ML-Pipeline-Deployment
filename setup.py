"""
Setup script for ML Pipeline Deployment Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = (
    readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
)

setup(
    name="ml-pipeline-deploy",
    version="1.0.0",
    description="Enterprise ML Pipeline Deployment Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ML Pipeline Team",
    author_email="your-email@example.com",
    url="https://github.com/Arvind-55555/ML-Pipeline-Deployment",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.html", "*.js", "*.css"],
    },
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "click>=8.1.0",
        "pyyaml>=6.0",
        "mlflow>=2.8.0",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "full": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "xgboost>=1.7.0",
            "lightgbm>=3.3.0",
            "tensorflow>=2.13.0",
            "pydicom>=2.4.0",
            "Pillow>=10.0.0",
            "yfinance>=0.2.0",
            "schedule>=1.2.0",
            "kafka-python>=2.0.2",
            "websockets>=11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-pipeline=src.cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine-learning ml-pipeline mlops deployment fastapi",
)
