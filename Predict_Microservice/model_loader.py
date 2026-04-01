import mlflow
# from Predict_Microservice.select_best_model import select_best_model
from select_best_model import select_best_model
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..','src'))

def load_model():
    best_model_uri = select_best_model()
    model = mlflow.pytorch.load_model(best_model_uri)
    return model