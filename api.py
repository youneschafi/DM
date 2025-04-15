from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load('heart_disease_model.pkl')

class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.post("/predict")
async def predict(data: PatientData):
    try:
        input_data = pd.DataFrame([data.dict()])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        return {
            "has_heart_disease": bool(prediction),
            "probability": float(probability),
            "risk_level": "high" if probability > 0.7 else "medium" if probability > 0.3 else "low"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model_info")
async def model_info():
    try:
        return {
            "model_type": type(model.named_steps['classifier']).__name__,
            "features": list(model.named_steps['preprocessor'].transformers_[0][2]) + 
                       list(model.named_steps['preprocessor'].transformers_[1][1]
                              .named_steps['onehot']
                              .get_feature_names_out())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))