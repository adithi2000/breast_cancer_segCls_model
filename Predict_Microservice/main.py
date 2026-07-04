import logging
import os

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from inference import predict_pipeline
from model_loader import load_model

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

logger.info("Loading prediction model at API startup")
model = load_model()
logger.info("Prediction model loaded successfully")


def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        logger.warning("Rejected request with invalid authorization header format")
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ")[1]
    try:
        id_info = id_token.verify_oauth2_token(token, google_requests.Request())
        logger.debug("Verified Google token for user: %s", id_info.get("email", "unknown"))
        return id_info
    except ValueError:
        logger.warning("Rejected request with invalid Google token")
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/predict/")
async def predict(file: UploadFile = File(...), user: dict = Depends(verify_token)):
    logger.info("Received prediction request: filename=%s, user=%s", file.filename, user.get("email", "unknown"))
    contents = await file.read()
    predicted_class, confidence, img_buffer = predict_pipeline(model, contents)
    logger.info(
        "Prediction completed: filename=%s, class=%s, confidence=%.4f",
        file.filename,
        predicted_class,
        confidence,
    )
    return StreamingResponse(
        img_buffer,
        media_type="image/png",
        headers={
            "X-Predicted-Class": str(predicted_class),
            "X-Confidence": str(round(confidence * 100, 2)),
        },
    )


@app.get("/")
def read_root():
    logger.debug("Root endpoint requested")
    return {"message": "Welcome to the Mask Classification API! Use POST /predict/ with an image file."}


@app.get("/health")
def health_check():
    logger.debug("Health check requested: model_loaded=%s", model is not None)
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model_info")
def model_info():
    logger.debug("Model info endpoint requested")
    return {
        "model": "segmentation + classification",
        "input_size": "256x256",
        "outputs": ["segmentation mask", "classification"],
    }
