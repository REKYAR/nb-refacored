import fastapi
from fastapi import FastAPI
from settings import settings
import pickle
import pandas as pd
from typing import List, Dict

app = FastAPI()
model = pickle.load(open(settings.MODEL_PATH, 'rb'))

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/reload")
def reload_model():
    """
    Reload the model from the specified path.
    """
    global model
    with open(settings.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return {"message": "Model reloaded successfully"}

@app.get("/predict")
def predict(data: List[Dict[str, str]]):
    """
    Predict the target variable using the trained model.
    
    Args:
        data (List[Dict[str, str]]): The input data for prediction.
        
    Returns:
        List[float]: The predicted probabilities.
    """
    # Load the model
    with open(settings.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Convert the input data to a DataFrame
    df = pd.DataFrame(data)
    
    # Preprocess the data
    categorical_columns = ['CalYear