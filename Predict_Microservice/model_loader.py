import mlflow
from Predict_Microservice.select_best_model import select_best_model

def load_model():
    best_model_uri = select_best_model()
    model = mlflow.pytorch.load_model(best_model_uri)
    return model