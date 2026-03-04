# Loading a pre-trained model and setting up a FastAPI endpoint for predictions
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Payload size guard (bytes)
MAX_REQUEST_BYTES = 10_000

# Define the input schema (Validation)
class CultureData(BaseModel):
    culture_values: float = Field(..., ge=1.0, le=5.0)
    belonging_score: float = Field(..., ge=1.0, le=5.0)
    career_opp: float = Field(..., ge=1.0, le=5.0)

# Loading the model
MODEL_PATH = "models/best_model.bin"
model = joblib.load(MODEL_PATH)

# Initializing the App
app = FastAPI(title="Voluntās Culture Intelligence API")
 
@app.middleware("http")
async def request_size_limit(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_BYTES:
        return JSONResponse({"detail": "Request too large"}, status_code=413)
    return await call_next(request)


@app.post("/predict")
def predict(data: CultureData, request: Request):
    if data is None:
        raise ValueError("Input data cannot be null")

    payload = data.model_dump()
    logger.info("input schema fields=%s", list(payload.keys()))
    
    # Convert incoming data to the format the model expects
    input_data = np.array([[data.culture_values, data.belonging_score, data.career_opp]])
    
    # Make prediction
    try:
        start = time.time()
        prediction = model.predict(input_data)
        duration_ms = (time.time() - start) * 1000
        logger.info("prediction ok duration_ms=%.2f", duration_ms)
    except Exception as e:
        raise RuntimeError(f"An error occurred when making a prediction: {str(e)}") from e
    
    return {
        "overall_rating_prediction": float(prediction[0]),
        "status": "success"
    }

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    """Expose model metadata if available."""
    card_path = Path("models/model_card.txt")
    eval_dir = Path("artifacts")
    metadata_payload = {
        "model_path": MODEL_PATH,
    }

    if card_path.exists():
        metadata_payload["model_card"] = card_path.read_text()

    # Surface latest evaluation report if present
    if eval_dir.exists():
        reports = sorted(eval_dir.glob("model_eval_*.json"))
        if reports:
            latest = reports[-1]
            metadata_payload["latest_evaluation"] = json.loads(latest.read_text())

    return metadata_payload

# To run this: uvicorn predict:app --reload
