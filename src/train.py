import torch.nn as nn
from monai.losses import DiceLoss
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

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

# from download_from_s3 import download_from_s3, get_latest_augmented_prefix
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
   
    # os.makedirs("data/original/train", exist_ok=True)
    # os.makedirs("data/original/val", exist_ok=True)
    # os.makedirs("data/augmented/", exist_ok=True)

    current_file=os.path.abspath(__file__)
    src_dir=os.path.dirname(current_file)
    root=os.path.dirname(src_dir)

    train_root = f"{root}/data/original/train"
    # download_from_s3("original/train/", train_root)

    aug_root=f"{root}/data/augmented/"
    # aug_prefix=get_latest_augmented_prefix()
    # download_from_s3(aug_prefix, aug_root)

    val_root = f"{root}/data/original/val"
    # download_from_s3("original/val/", val_root)

    

    print("DATA PATH SET")

    train_data = create_data_list(train_root)
    aug_data=create_data_list(aug_root)
    val_data = create_data_list(val_root)
    

    train_transforms = create_train_transforms()
    val_transforms = create_val_transforms()

    train_data=train_data+aug_data

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
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(os.getenv("EXPERIMENT_NAME"))

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"RUN_ID={run_id}")
        with open("RUN_ID.txt", "w") as f:
            f.write(run_id)

        best_score=0
        best_model_state=None
        patience=3
        epochs=2
        count=0
        # min_delta=0.001
        mlflow.log_param("lr", 1e-4)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("seg_loss", "DiceLoss")
        mlflow.log_param("cls_loss", "CrossEntropy")
        mlflow.log_param("accuracy","Batch wise average accuracy")
        mlflow.log_param("Dice metric",'basis of model selection')
        mlflow.log_param("model_architecture", "SegResNet50 with dual heads")
        mlflow.log_param("data_augmentation", "Included augmented data from S3 with latest prefix")
        mlflow.log_param("early_stopping", f"Based on combined score of 0.7*val_dice + 0.3*val_accuracy with patience {patience}")

    # -------------------------
    # 7. Training Loop
    # -------------------------
    
        for e in range(epochs):
            print(f"epoch {e+1}")
            train_loss,train_dice,train_accuracy=train_one_epoch(model,optimizer,train_loader,device,cls_loss_fn,seg_loss_fn)
            print(f"train_loss is {train_loss}")
            print(f"train_dice is {train_dice}")
            val_loss,val_dice,val_accuracy=validation(model,val_loader,device,cls_loss_fn,seg_loss_fn)
            print(f"val_loss {val_loss}")
            print(f"val_dice {val_dice}")
            print(f"val_accuracy {val_accuracy}")
        # 🔥 MLflow logging
            mlflow.log_metric("train_loss", train_loss, step=e)
            mlflow.log_metric("val_loss", val_loss, step=e)
            mlflow.log_metric("train_dice", train_dice, step=e)
            mlflow.log_metric("val_dice", val_dice, step=e)
            mlflow.log_metric("train_accuracy", train_accuracy, step=e)
            mlflow.log_metric("val_accuracy", val_accuracy, step=e)
            score=0.7*val_dice+0.3*val_accuracy
            if(score > best_score + 1e-4):  # 🔥 check both dice and accuracy with a small margin
                print("Saving model....")
                best_score=score
                count=0
                best_model_state=model.state_dict()
            #     torch.save(
            # {
            #     "epoch":e+1,
            #     "model_state_dict":model.state_dict(),
            #     "optimizer_state_dict":optimizer.state_dict(),
            #     "best_score":best_score
            # },f"models/best_model.pth"
            # )
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
        mlflow.log_metric("best_val_score", best_score)

        mlflow.end_run()

    # FileLink('/kaggle/working/models/best_model.pth')

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    train()


