from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import json

app = FastAPI(title="Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to "*" to allow all origins
    allow_credentials=False,  
    allow_methods=["*"],
    allow_headers=["*"],
)
    
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.joblib")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "feature_names.json")


class PredictRequest(BaseModel):
    # Either provide an input_vector of numeric features (0/1) in correct order,
    # or provide a list of symptom names present (strings) if feature names known.
    input_vector: Optional[List[int]] = None
    symptoms: Optional[List[str]] = None


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def load_label_encoder():
    if not os.path.exists(LABEL_ENCODER_PATH):
        return None
    return joblib.load(LABEL_ENCODER_PATH)


def load_feature_names():
    if not os.path.exists(FEATURES_PATH):
        return None
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def get_features():
    features = load_feature_names()
    if features is None:
        raise HTTPException(status_code=404, detail="feature_names.json not found")
    return {"features": features}


@app.post("/predict")
def predict(req: PredictRequest):
    model = load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model file model.joblib not found. Run training and save the model to backend/model.joblib")

    feature_names = load_feature_names()

    # Build input vector
    if req.input_vector is not None:
        x = req.input_vector
    elif req.symptoms is not None:
        if feature_names is None:
            raise HTTPException(status_code=400, detail="Feature names not available on server; provide input_vector instead")
        # Map symptom names to vector
        vec = [1 if fname in req.symptoms else 0 for fname in feature_names]
        x = vec
    else:
        raise HTTPException(status_code=400, detail="Provide either input_vector or symptoms")

    # Validate vector length
    if hasattr(model, "predict"):
        # ensure 2D
        try:
            pred = model.predict([x])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    else:
        raise HTTPException(status_code=500, detail="Loaded object is not a predictor")

    # Try to load label encoder to inverse transform
    le = load_label_encoder()
    if le is not None:
        try:
            label = le.inverse_transform(pred)[0]
        except Exception:
            label = str(pred[0])
    else:
        label = str(pred[0])

    return {"prediction": label, "raw_prediction": int(pred[0])}
