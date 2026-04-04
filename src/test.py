# test.py

from flask.cli import load_dotenv
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import os
# from download_from_s3 import download_from_s3
from engine import validation
from model import get_model
from dataset import create_data_list, create_val_transforms, get_loader
from dotenv import load_dotenv

def test():

    # -------------------------
    # 1. Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # 2. Data
    # -------------------------
    current_file=os.path.abspath(__file__)
    src_dir=os.path.dirname(current_file)
    root=os.path.dirname(src_dir)
    test_root = f"{root}/data/original/test"
    # download_from_s3("original/test/", test_root)

    test_data = create_data_list(test_root)

    # ✅ same as validation
    test_transforms = create_val_transforms()

    test_loader = get_loader(test_data, test_transforms, batch_size=4, shuffle=False)

    # -------------------------
    # 3. Model
    # -------------------------
    essential = get_model(in_channels=3, num_classes=3)
    model = essential['model'].to(device)

    # -------------------------
    # 4. Load BEST Model
    # -------------------------

    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(os.getenv("EXPERIMENT_NAME"))
    # model_path = "./models/best_model.pth"
    # checkpoint = torch.load(model_path, map_location=device)

    # model.load_state_dict(checkpoint["model_state_dict"])
    # model.train()

    print("✅ Loaded best model")

    # -------------------------
    # 5. Metrics
    # -------------------------
    cls_loss_fn = nn.CrossEntropyLoss()
    seg_loss_fn = nn.BCEWithLogitsLoss()
    
    
    # run_id=os.getenv("RUN_ID")
    with open("RUN_ID.txt", "r") as f:
        run_id = f.read().strip()
    
   
    
    if run_id:
        with mlflow.start_run(run_id=run_id):
            model=mlflow.pytorch.load_model(f"runs:/{run_id}/best_model")
            print(f"✅ Loaded model from MLflow run: {run_id}")
            model.to(device)
            model.train()
            total_test_loss,test_dice,test_accuracy=validation(model, test_loader, device, cls_loss_fn, seg_loss_fn)
            print("Test Loss:", total_test_loss)
            print("Test Dice:", test_dice)
            print("Test Accuracy:", test_accuracy)
            mlflow.log_metric("test_loss", total_test_loss)
            mlflow.log_metric("test_dice", test_dice)
            mlflow.log_metric("test_accuracy", test_accuracy)
            score=0.7*test_dice+0.3*test_accuracy
            mlflow.log_metric("test_score", score)
    else:
        print("⚠️ RUN_ID not found in environment variables. Skipping MLflow logging.")
        # total_test_loss=validation(model, test_loader, device, cls_loss_fn, seg_loss_fn)
        # print("Test Loss:", total_test_loss)
# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    test()