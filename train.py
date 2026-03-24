import torch.nn as nn
from monai.losses import DiceLoss
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
# from IPython.display import FileLink

from model import get_model
from dataset import create_data_list, create_train_transforms, create_val_transforms, get_loader
from engine import train_one_epoch, validation

# train.py

import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

from model import MaskClassifyModel
from dataset import create_data_list, create_train_transforms, create_val_transforms, get_loader
from engine import train_one_epoch, validation
from monai.losses import DiceLoss

from download_from_s3 import download_from_s3, get_latest_augmented_prefix
import os


def train():

    # -------------------------
    # 1. Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # 2. Data
    # -------------------------

    # train_root = "/kaggle/input/datasets/adithip2000/breast-cancer-data-train-test-split/original/train"
    # val_root = "/kaggle/input/datasets/adithip2000/breast-cancer-data-train-test-split/original/val"
   
    os.makedirs("data/original/train", exist_ok=True)
    os.makedirs("data/original/val", exist_ok=True)
    #os.makedirs("data/augmented/", exist_ok=True)


    train_root = "./data/original/train"
    download_from_s3("original/train/", train_root)

    # aug_root="./data/augmented/"
    # aug_prefix=get_latest_augmented_prefix()
    # download_from_s3(aug_prefix, aug_root)

    val_root = "./data/original/val"
    download_from_s3("original/val/", val_root)

    

    print("DATA PATH SET")

    train_data = create_data_list(train_root)
    # aug_data=create_data_list(aug_root)
    val_data = create_data_list(val_root)
    

    train_transforms = create_train_transforms()
    val_transforms = create_val_transforms()

    # train_data=train_data+aug_data

    train_loader = get_loader(train_data, train_transforms, batch_size=4, shuffle=True)
    val_loader = get_loader(val_data, val_transforms, batch_size=4, shuffle=False)
    # test_loader = get_loader(test_data, val_transforms, batch_size=4, shuffle=False)

    print("DATASET LOADER COMPLETED...")
    # -------------------------
    # 3. Model
    # -------------------------
    essential = get_model(in_channels=3,num_classes=3)
    model=essential['model'].to(device)

    # -------------------------
    # 4. Optimizer
    # -------------------------
    
    optimizer=essential['optimizer']
    # -------------------------
    # 5. Loss Functions
    # -------------------------
    seg_loss_fn = DiceLoss(sigmoid=True)
    cls_loss_fn = nn.CrossEntropyLoss()

    # -------------------------
    # 6. MLflow start
    # -------------------------
    print("MLFLOW STARTED")
    mlflow.set_tracking_uri("file:/mlruns")
    mlflow.set_experiment("breast_cancer_model")
    mlflow.start_run()

    mlflow.log_param("lr", 1e-4)
    mlflow.log_param("epochs", 250)
    mlflow.log_param("seg_loss", "DiceLoss")
    mlflow.log_param("cls_loss", "CrossEntropy")

    # -------------------------
    # 7. Training Loop
    # -------------------------
    best_loss=float("inf")
    best_model_state=None
    patience=3
    epochs=250
    count=0
    min_delta=0.001
    for e in range(epochs):
        print(f"epoch {e+1}")
        train_loss=train_one_epoch(model,optimizer,train_loader,device,cls_loss_fn,seg_loss_fn)
        print(f"train_loss is {train_loss}")
        val_loss=validation(model,val_loader,device,cls_loss_fn,seg_loss_fn)
        print(f"val_loss {val_loss}")
        # 🔥 MLflow logging
        mlflow.log_metric("train_loss", train_loss, step=e)
        mlflow.log_metric("val_loss", val_loss, step=e)
        if(val_loss < best_loss and (best_loss-val_loss > min_delta)):
            print("Saving model....")
            best_loss=val_loss
            count=0
            best_model_state=model.state_dict()
            torch.save(
            {
                "epoch":e+1,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "best_loss":best_loss
            },f"models/best_model.pth"
        )
            # f"/kaggle/working/models/best_model.pth"
        else:
            print("Not saving.. under patience")
            count+=1
        if count >= patience:
            print("Early stopping triggered")

            break
        print("=================================================")
    
    


        

    # -------------------------
    # 8. Save Model
    # -------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    mlflow.pytorch.log_model(model, "best_model")

    mlflow.end_run()

    # FileLink('/kaggle/working/models/best_model.pth')

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    train()


