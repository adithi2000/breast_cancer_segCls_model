from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
# from Predict_Microservice.model_loader import load_model
# from Predict_Microservice.inference import predict_pipeline
from model_loader import load_model
from inference import predict_pipeline

from fastapi import Header, HTTPException, Depends
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests



# import sys
# import os

# sys.path.append(os.path.abspath(".."))

app = FastAPI()

# 🔥 Load once at startup
model = load_model()

def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    try:
        id_info = id_token.verify_oauth2_token(token, google_requests.Request())
        return id_info
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/predict/")
async def predict(file: UploadFile = File(...), user: dict = Depends(verify_token)):
    contents = await file.read()
    predicted_class,confidence,img_buffer = predict_pipeline(model, contents)
    return StreamingResponse(
        img_buffer,
        media_type="image/png",
        headers={"X-Predicted-Class": str(predicted_class), "X-Confidence": str(round(confidence, 2))}
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mask Classification API! Use POST /predict/ with an image file."}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/model_info")
def model_info():
    return {
        "model": "segmentation + classification",
        "input_size": "256x256",
        "outputs": ["segmentation mask", "classification"]
    }

