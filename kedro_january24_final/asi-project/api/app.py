import os
import numpy as np
import pickle
from fastapi import FastAPI
from pathlib import Path
from kedro.framework.startup import bootstrap_project
from kedro.framework.context import KedroContext
import subprocess
from kedro.framework.session import KedroSession

app = FastAPI()


@app.get("/models")
def list_models():
    models_dir = "/home/kali/Desktop/kedro/asi-project/data/06_models"
    models = os.listdir(models_dir)
    return {"models": models}




@app.post("/predict")
def make_prediction(model_name: str, age: int, rating: float, distance: float):
    models_dir = "/home/kali/Desktop/kedro/asi-project/data/06_models"

    model_filenames = {
        "rf_model": "rf_trained_model.pkl",
        "gb_model": "gb_trained_model.pkl",
        "lstm_model": "lstm_model.pickle"
    }

    if model_name not in model_filenames:
        return {"error": f"Model '{model_name}' not found"}

    model_path = os.path.join(models_dir, model_filenames[model_name])

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    features = np.array([[age, rating, distance]])
    prediction = model.predict(features)
    prediction = prediction.tolist()    
    return {"prediction": prediction[0]}


@app.post("/train/{model_name}")
def train_model(model_name: str):
    model_node_map = {
        "rf_model": "incremental_train_rf_node",
        "gb_model": "incremental_train_gb_node",
    }

    if model_name not in model_node_map:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    kedro_project_path = "/home/kali/Desktop/kedro/asi-project"

    os.chdir(kedro_project_path)

    node_name = model_node_map[model_name]

    subprocess.run(["python", "-m", "kedro", "run", "--pipeline", "incremental_training"])
    return {"message": f"Further training started for model: {model_name}"}
