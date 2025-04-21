import pickle
from typing import Dict, List

import fastapi
import pandas as pd
from fastapi import FastAPI

from source.data_processing import preprocess_data_serving
from source.models import schema
from source.settings import settings

app = FastAPI()
model = pickle.load(open(settings.MODEL_PATH, "rb"))


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/reload")
def reload_model():
    """
    Reload the model from the specified path.
    """
    global model
    with open(settings.MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return {"message": "Model reloaded successfully"}


@app.post("/predict")
def predict(data: List[Dict[str, str]]):
    """
    Predict the target variable using the trained model.

    Args:
        data (List[Dict[str, str]]): The input data for prediction.

    Returns:
        List[float]: The predicted probabilities.
    """
    global model

    df = pd.DataFrame(data)
    try:
        df = schema.validate(df)
    except Exception as e:
        return {"schema validation failed": str(e)}

    df = preprocess_data_serving(data=df)

    model.predict(df)
    return model.predict_proba(df)[:, 1].tolist()
