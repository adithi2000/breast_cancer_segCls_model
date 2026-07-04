import logging
import os

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from dotenv import load_dotenv
from monai.losses import DiceLoss

from modules.dataset import (
    create_data_list,
    create_train_transforms,
    create_val_transforms,
    get_loader,
)
from modules.engine import train_one_epoch, validation
from modules.model import get_model

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    root = os.path.dirname(src_dir)

    train_root = f"{root}/data/original/train"
    aug_root = f"{root}/data/augmented/"
    val_root = f"{root}/data/original/val"
    logger.info("Data paths set: train=%s, augmented=%s, val=%s", train_root, aug_root, val_root)

    train_data = create_data_list(train_root)
    aug_data = create_data_list(aug_root)
    val_data = create_data_list(val_root)
    logger.info(
        "Loaded data records: train=%d, augmented=%d, validation=%d",
        len(train_data),
        len(aug_data),
        len(val_data),
    )

    train_transforms = create_train_transforms()
    val_transforms = create_val_transforms()

    train_data = train_data + aug_data

    train_loader = get_loader(train_data, train_transforms, batch_size=4, shuffle=True)
    val_loader = get_loader(val_data, val_transforms, batch_size=4, shuffle=False)
    logger.info("Dataset loaders created successfully")

    essential = get_model(in_channels=3, num_classes=3)
    model = essential["model"].to(device)
    optimizer = essential["optimizer"]

    seg_loss_fn = DiceLoss(sigmoid=True)
    cls_loss_fn = nn.CrossEntropyLoss()

    logger.info("Configuring MLflow tracking")
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(os.getenv("EXPERIMENT_NAME"))
    logger.info(
        "MLflow configured: tracking_uri=%s, experiment=%s",
        mlflow.get_tracking_uri(),
        os.getenv("EXPERIMENT_NAME"),
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("Started MLflow run: %s", run_id)
        with open("RUN_ID.txt", "w") as f:
            f.write(run_id)
        logger.info("Persisted MLflow run id to RUN_ID.txt")

        best_score = 0
        best_model_state = None
        patience = 3
        epochs = 2
        count = 0

        mlflow.log_param("lr", 1e-4)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("seg_loss", "DiceLoss")
        mlflow.log_param("cls_loss", "CrossEntropy")
        mlflow.log_param("accuracy", "Batch wise average accuracy")
        mlflow.log_param("Dice metric", "basis of model selection")
        mlflow.log_param("model_architecture", "SegResNet50 with dual heads")
        mlflow.log_param("data_augmentation", "Included augmented data from S3 with latest prefix")
        mlflow.log_param(
            "early_stopping",
            f"Based on combined score of 0.7*val_dice + 0.3*val_f1 with patience {patience}",
        )
        mlflow.log_param("Device :", device)

        for e in range(epochs):
            logger.info("Starting epoch %d/%d", e + 1, epochs)
            train_loss, train_dice, train_accuracy = train_one_epoch(
                model,
                optimizer,
                train_loader,
                device,
                cls_loss_fn,
                seg_loss_fn,
            )
            logger.info(
                "Epoch %d train metrics: loss=%.6f, dice=%.6f, accuracy=%.6f",
                e + 1,
                train_loss,
                train_dice,
                train_accuracy,
            )

            val_loss, val_dice, val_accuracy, f1 = validation(
                model,
                val_loader,
                device,
                cls_loss_fn,
                seg_loss_fn,
            )
            logger.info(
                "Epoch %d validation metrics: loss=%.6f, dice=%.6f, accuracy=%.6f, f1=%.6f",
                e + 1,
                val_loss,
                val_dice,
                val_accuracy,
                f1,
            )

            mlflow.log_metric("train_loss", train_loss, step=e)
            mlflow.log_metric("val_loss", val_loss, step=e)
            mlflow.log_metric("train_dice", train_dice, step=e)
            mlflow.log_metric("val_dice", val_dice, step=e)
            mlflow.log_metric("train_accuracy", train_accuracy, step=e)
            mlflow.log_metric("val_accuracy", val_accuracy, step=e)
            mlflow.log_metric("val_f1", f1, step=e)
            score = 0.7 * val_dice + 0.3 * f1
            mlflow.log_metric("score", score, step=e)

            if score > best_score + 1e-4:
                logger.info("New best model found at epoch %d with score %.6f", e + 1, score)
                best_score = score
                count = 0
                best_model_state = model.state_dict()
            else:
                count += 1
                logger.info("No improvement at epoch %d; patience counter=%d/%d", e + 1, count, patience)

            if count >= patience:
                logger.info("Early stopping triggered at epoch %d", e + 1)
                break

            logger.info("Completed epoch %d/%d", e + 1, epochs)

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info("Loaded best model state before MLflow artifact logging")
        else:
            logger.warning("No best model state was captured; logging current model state")

        mlflow.pytorch.log_model(
            model,
            artifact_path="best_model",
            code_paths=["modules"],
            pip_requirements=["torch", "monai", "scikit-learn"],
        )
        mlflow.log_metric("best_val_score", best_score)
        logger.info("Logged best model artifact and best_val_score=%.6f", best_score)

        mlflow.end_run()
        logger.info("Ended MLflow run: %s", run_id)


if __name__ == "__main__":
    train()
