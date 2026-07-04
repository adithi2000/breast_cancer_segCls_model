import logging
import os
import sys

import mlflow
# from Predict_Microservice.select_best_model import select_best_model
from select_best_model import select_best_model
import torch

logger = logging.getLogger(__name__)

# sys.path.append(os.path.join(os.path.dirname(__file__), '..','src'))

def load_model():
    best_model_uri = select_best_model()
    logger.info("Loading model from URI: %s", best_model_uri)
    device=None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU for model loading")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for model loading")
    model = mlflow.pytorch.load_model(best_model_uri, map_location=device)
    # model.to(device)
    logger.info("Model loaded successfully")
    return model
