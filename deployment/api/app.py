# Loading a pre-trained model and setting up a FastAPI endpoint for predictions
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Define the input schema (Validation)
class CultureData(BaseModel):
    culture_values: float
    belonging_score: float
    career_opp: float

# Loading the model
MODEL_PATH = "models/best_model.bin"
model = joblib.load(MODEL_PATH)

# Initializing the App
app = FastAPI(title="Voluntās Culture Intelligence API")
 

@app.post("/predict")
def predict(data: CultureData):
    if data is None:
        raise ValueError("Input data cannot be null")
    
    # Convert incoming data to the format the model expects
    input_data = np.array([[data.culture_values, data.belonging_score, data.career_opp]])
    
    # Make prediction
    try:
        prediction = model.predict(input_data)
    except Exception as e:
        raise RuntimeError(f"An error occurred when making a prediction: {str(e)}") from e
    
    return {
        "overall_rating_prediction": float(prediction[0]),
        "status": "success"
    }

# To run this: uvicorn predict:app --reload