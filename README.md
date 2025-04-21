# ML Model Training and Serving Project

This repository contains a refactored machine learning pipeline that demonstrates best practices for machine learning engineering. The project is organized to support the full ML lifecycle from data processing to model training and serving via an API.

## Architecture Overview

The project follows a modular architecture with clear separation of concerns:

```
refactored/
├── blob_store_stub/          # Simulated blob storage (would be replaced with actual storage in production)
│   ├── models/               # Trained model artifacts
│   ├── parameters/           # Best hyperparameters
│   └── reports/              # Model evaluation reports and visualizations
├── config/                   # Configuration files
├── data/                     # Input data
├── docker/                   # Docker configurations for training and serving
├── notebooks/                # Jupyter notebooks (original and modified)
├── source/                   # Core source code
│   ├── data_processing.py    # Data loading and preprocessing
│   ├── models.py             # Model definitions
│   ├── serve.py              # API server for model serving
│   ├── settings.py           # Application settings and configurations
│   ├── train_run.py          # Entry point for model training
│   └── trainer.py            # Model training logic
└── test/                     # Unit and integration tests
```

### Key Components:

1. **Data Processing**: Handles data loading, cleaning, and feature engineering
2. **Model Training**: Implements model training, hyperparameter tuning, and evaluation
3. **Model Serving**: Provides an API to serve model predictions
4. **Configuration**: Manages settings and hyperparameters
5. **Storage**: Uses a blob store (simulated in this demo) for model artifacts

## Setup and Installation

### Prerequisites

- Python 3.12
-  uv package manager

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd refactored
   ```

2. Install dependencies using uv:
   ```bash
   uv sync --all-groups
   ```

## Usage Examples

### Training a Model

To train a model with default parameters:

```bash
python source/train_run.py
```

### Serving the Model

Start the prediction API server:

```bash
uvicorn source.serve:app --host 0.0.0.0 --port 8080
```

### Making Predictions

Once the server is running, you can request predictions by going to swagger http://localhost:8080/#swagger and making an adequate post request, you pass the row like the one of the data table in the request body.

### Running Tests

Execute the test suite:

```bash
pytest
```

## Project Guide

### Data Flow

1. **Data Ingestion**: Raw data is loaded from `data/pg15training.csv`
2. **Preprocessing**: Data is cleaned and features are engineered in `data_processing.py`
3. **Training**: Models are trained and evaluated in `trainer.py`
4. **Storage**: Trained models and artifacts are saved to the blob store (simulated in `blob_store_stub/`)
5. **Serving**: The API in `serve.py` loads models from storage and serves predictions

## Docker Support

The project includes Docker configurations for both training and serving:

### Build the Training Container

```bash
docker build -f docker/Dockerfile.train -t refactored-train .
```

### Run the Training Container

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/blob_store_stub:/app/blob_store_stub refactored-train
```

### Build the Serving Container

```bash
docker build -f docker/Dockerfile.serve -t refactored-serve .
```

### Run the Serving Container

```bash
docker run -p 8000:8000 -v $(pwd)/blob_store_stub:/app/blob_store_stub refactored-serve
```

### Application parameters
Application parameters are defined in source/settings.py, it automatically loads data from the .env file in order to make a change.
