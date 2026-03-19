import torch.nn as nn
from monai.losses import DiceLoss
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

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


def train():

    # -------------------------
    # 1. Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # 2. Data
    # -------------------------

    # make a download function from s3 to a path data/train,etc
    # if not os.path.exists("data/train/"):
    # download_from_s3(...)
    train_root = "path_to_train_data"
    aug_root='path'
    val_root = "path_to_val_data"
    # test_root="path_to_test_data"


    train_data = create_data_list(train_root)
    aug_data=create_data_list(aug_root)
    val_data = create_data_list(val_root)
    # test_data = create_data_list(test_root)

    train_transforms = create_train_transforms()
    val_transforms = create_val_transforms()

    train_data=train_data+aug_data

    train_loader = get_loader(train_data, train_transforms, batch_size=4, shuffle=True)
    val_loader = get_loader(val_data, val_transforms, batch_size=4, shuffle=False)
    # test_loader = get_loader(test_data, val_transforms, batch_size=4, shuffle=False)
    # -------------------------
    # 3. Model
    # -------------------------
    essential = get_model(in_channels=1, num_classes=3).to(device)
    model=essential['model']

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
        train_loss=train_one_epoch(model,train_loader,device,cls_loss_fn,seg_loss_fn)
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
            },f"/models/best_model.pth"
        )
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


# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    train()


