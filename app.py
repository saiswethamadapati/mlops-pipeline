# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model once
model = joblib.load('model.pkl')

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width:  float
    petal_length: float
    petal_width:  float

app = FastAPI()

@app.post('/predict')
def predict(req: IrisRequest):
    features = [[
        req.sepal_length,
        req.sepal_width,
        req.petal_length,
        req.petal_width
    ]]
    pred = model.predict(features)[0]
    return {'prediction': int(pred)}
