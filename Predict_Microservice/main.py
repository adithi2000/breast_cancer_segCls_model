from fastapi import FastAPI, UploadFile, File
from Predict_Microservice.model_loader import load_model
from Predict_Microservice.inference import predict_pipeline

app = FastAPI()

# 🔥 Load once at startup
model = load_model()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_pipeline(model, contents)
    return result

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

