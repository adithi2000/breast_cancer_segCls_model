# Breast Cancer Segmentation + Classification MLOps Project

This repository contains an end-to-end workflow for breast cancer image segmentation and classification, including model training, MLflow tracking, a FastAPI prediction service, and a Streamlit frontend with Google OAuth login.

## Overview

The project is organized around three main parts:

- `src/` contains the training pipeline, dataset helpers, and model components.
- `Predict_Microservice/` exposes the trained model through a FastAPI prediction API.
- `streamlit_app/` provides the user-facing web app for authenticated image uploads and predictions.

The repository also includes Ansible deployment playbooks and Dockerfiles for building container images.

## Project Report

Detailed write-up:

[An End-to-End MLOps Pipeline Report](https://github.com/adithi2000/breast_cancer_segCls_model/blob/e38fb018a4b1399c149694f1445d2d00860cc3b8/An%20End-to-End%20MLOps%20Pipeline%20Report.pdf)

## Features

- Dual-purpose model for segmentation and classification.
- Training pipeline with MLflow experiment tracking.
- FastAPI inference service with Google token verification.
- Streamlit frontend with Google OAuth sign-in.
- Dockerized services and deployment automation support.
- DVC-managed dataset layout for reproducible experiments.

## Tech Stack

- Python
- PyTorch
- MONAI
- FastAPI
- Streamlit
- MLflow
- Docker
- DVC
- Scikit-learn
- Google OAuth

## Repository Layout

```text
data.dvc
deploy_frontend.yml
deploy_microservice.yml
ansible/
base_image_creation/
data/
Predict_Microservice/
src/
streamlit_app/
README.md
```

Key folders:

- `src/train.py` trains the model and logs metrics and artifacts to MLflow.
- `src/modules/` contains dataset, engine, and model utilities.
- `Predict_Microservice/main.py` serves the `/predict/`, `/health`, and `/model_info` endpoints.
- `streamlit_app/app.py` handles Google OAuth and sends images to the prediction API.

## Data Layout

The dataset is expected to follow this structure:

```text
data/
	original/
		train/
			benign/
			malignant/
			normal/
		val/
			benign/
			malignant/
			normal/
		test/
			benign/
			malignant/
			normal/
	augmented/
		benign/
		malignant/
		normal/
```

The training script combines the original training data with augmented samples before fitting the model.

## Prerequisites

- Python 3.10+ recommended.
- Docker for containerized deployment.
- MLflow tracking server access.
- Google OAuth credentials for the frontend and API authentication flow.

## Environment Variables

The main components expect the following variables:

- `MLFLOW_TRACKING_URI`
- `EXPERIMENT_NAME`
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI`
- `API_URL`

## Local Setup

Install the training dependencies:

```bash
pip install -r src/requirements.txt
```

If you want to run the services in separate environments, check the Dockerfiles and the component-specific requirements files in `base_image_creation/`.

## Training

The training entrypoint is `src/train.py`.

```bash
python src/train.py
```

What it does:

- Loads training and validation data from `data/original/` and `data/augmented/`.
- Builds the model using `src/modules/model.py`.
- Trains with segmentation and classification losses.
- Logs parameters and metrics to MLflow.
- Stores the MLflow run ID in `src/RUN_ID.txt`.
- Saves the best model artifact under the MLflow run.

## Prediction API

The FastAPI service lives in `Predict_Microservice/main.py`.

Run it with Uvicorn:

```bash
uvicorn Predict_Microservice.main:app --host 0.0.0.0 --port 8000
```

Available endpoints:

- `GET /` returns a basic welcome message.
- `GET /health` reports service and model status.
- `GET /model_info` returns model metadata.
- `POST /predict/` accepts an image file and a Google Bearer token.

## Streamlit App

The frontend lives in `streamlit_app/app.py`.

Run it with:

```bash
streamlit run streamlit_app/app.py
```

The app:

- Prompts the user to log in with Google.
- Accepts an uploaded image.
- Sends the image to the prediction API.
- Displays the predicted class, confidence, and returned output image.

## Deployment Notes

If you are exposing the Streamlit app from a VM or Kubernetes cluster, the repo’s deployment flow expects `kubectl port-forward` plus `ngrok`.

Recommended flow:

```bash
kubectl port-forward --address 0.0.0.0 -n mlops-project svc/streamlit-service 30001:8501
ngrok http 30001
```

Important notes:

- Keep the `kubectl port-forward` process running while ngrok is active.
- Use the public ngrok HTTPS URL as the Google OAuth redirect URI.
- Open port `30001` in the VM firewall if needed.

## Deployment Assets

- `deploy_frontend.yml` deploys the Streamlit frontend.
- `deploy_microservice.yml` deploys the prediction microservice.
- `ansible/` contains the playbook and templates used for Kubernetes and secret/config deployment.
- `base_image_creation/` contains Docker base image definitions for FastAPI and Streamlit.

## Notes

- The project currently uses Google OAuth verification in both the Streamlit app and the prediction API.
- The best model is selected from the MLflow run based on the combined validation score used in `src/train.py`.
- If you update the data layout or environment variables, make sure the training script and the frontend/API configs are updated together.



