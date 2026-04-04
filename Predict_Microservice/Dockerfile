FROM python:3.12-slim

WORKDIR /app

COPY ./Predict_Microservice/requirements.txt .
# RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libxcb1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY Predict_Microservice/ /app/
ENV PYTHONPATH=/app:/app/src
ENV MLFLOW_TRACKING_URI=http://52.195.181.124:5000/
ENV EXPERIMENT_NAME=breast_cancer_experiment-1
ENV AWS_ACCESS_KEY_ID=AKIAXFXMIXLX2RKGQ3UV
ENV AWS_SECRET_ACCESS_KEY=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

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

