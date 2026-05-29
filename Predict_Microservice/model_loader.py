import mlflow
# from Predict_Microservice.select_best_model import select_best_model
from select_best_model import select_best_model
import sys
import os
import torch

# sys.path.append(os.path.join(os.path.dirname(__file__), '..','src'))

def load_model():
    best_model_uri = select_best_model()
    print(f"Loading model from URI: {best_model_uri}")
    device=None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for model loading.")
    else:
        device = torch.device("cpu")
        print("Using CPU for model loading.")
    model = mlflow.pytorch.load_model(best_model_uri, map_location=device)
    # model.to(device)
    return model