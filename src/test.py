import logging
import os

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from dotenv import load_dotenv

from modules.dataset import create_data_list, create_val_transforms, get_loader
from modules.engine import validation
from modules.model import get_model

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    root = os.path.dirname(src_dir)
    test_root = f"{root}/data/original/test"
    logger.info("Test data path set: %s", test_root)

    test_data = create_data_list(test_root)
    logger.info("Loaded test records: %d", len(test_data))

    test_transforms = create_val_transforms()
    test_loader = get_loader(test_data, test_transforms, batch_size=4, shuffle=False)

    essential = get_model(in_channels=3, num_classes=3)
    model = essential["model"].to(device)

    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(os.getenv("EXPERIMENT_NAME"))
    logger.info(
        "MLflow configured: tracking_uri=%s, experiment=%s",
        mlflow.get_tracking_uri(),
        os.getenv("EXPERIMENT_NAME"),
    )

    cls_loss_fn = nn.CrossEntropyLoss()
    seg_loss_fn = nn.BCEWithLogitsLoss()

    with open("RUN_ID.txt", "r") as f:
        run_id = f.read().strip()
    logger.info("Read MLflow run id from RUN_ID.txt: %s", run_id or "missing")

    if run_id:
        with mlflow.start_run(run_id=run_id):
            model = mlflow.pytorch.load_model(f"runs:/{run_id}/best_model")
            logger.info("Loaded model from MLflow run: %s", run_id)
            model.to(device)
            model.train()
            total_test_loss, test_dice, test_accuracy, f1 = validation(
                model,
                test_loader,
                device,
                cls_loss_fn,
                seg_loss_fn,
            )
            logger.info(
                "Test metrics: loss=%.6f, dice=%.6f, accuracy=%.6f, f1=%.6f",
                total_test_loss,
                test_dice,
                test_accuracy,
                f1,
            )
            mlflow.log_metric("test_loss", total_test_loss)
            mlflow.log_metric("test_dice", test_dice)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_f1_macro", f1)
            score = 0.7 * test_dice + 0.3 * f1
            mlflow.log_metric("test_score", score)
            logger.info("Logged test_score=%.6f to MLflow", score)
    else:
        logger.warning("RUN_ID not found. Skipping MLflow test logging.")


if __name__ == "__main__":
    test()
