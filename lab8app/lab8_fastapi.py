from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
import numpy as np

app = FastAPI(
    title="Lab 8",
    version="0.1",
)

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'This is a model for classifying mlruns model'}

class request_body(BaseModel):
    input : str

@app.on_event('startup')
def load_artifacts():
    global model_pipeline
    #model_path = "mlruns/6/0711c7b97b3649aa8d186adb9e8a71bc/artifacts/better_models/model.pkl"

    model_pipeline = joblib.load('model.pkl')


# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data : request_body):
    inputstr = data.input
    print(inputstr)
    features = np.array([float(x) for x in inputstr.split(',')]).reshape(1, -1)

    prediction = model_pipeline.predict(features)
    return {"prediction": int(prediction[0])}