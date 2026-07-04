import logging
import os

from dotenv import load_dotenv
import mlflow

logger = logging.getLogger(__name__)

# mlflow.set_experiment(os.getenv("EXPERIMENT_NAME"))

def select_best_model():
    # Fetch all runs for the experiment
    # load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    exp=mlflow.search_experiments()
    for e in exp:
        logger.debug("Available MLflow experiment: %s", e.name)
    experiment = mlflow.get_experiment_by_name(os.getenv("EXPERIMENT_NAME"))
    if experiment is None:
        logger.error("MLflow experiment not found: %s", os.getenv("EXPERIMENT_NAME"))
        raise ValueError(f"MLflow experiment not found: {os.getenv('EXPERIMENT_NAME')}")
    logger.info("Selected MLflow experiment: id=%s, name=%s", experiment.experiment_id, experiment.name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    runs=runs.dropna(subset=['metrics.best_val_score'])
    if runs.empty:
        logger.error("No MLflow runs found with metrics.best_val_score in experiment %s", experiment.name)
        raise ValueError(f"No MLflow runs found with metrics.best_val_score in experiment {experiment.name}")

    # Find the run with the best validation accuracy
    best_run = runs.loc[runs['metrics.best_val_score'].idxmax()]

    logger.info("Best run id: %s", best_run['run_id'])
    logger.info("Best validation score: %s", best_run['metrics.best_val_score'])

    # Load the best model artifact
    run_id = best_run['run_id']
    best_model_uri = f"runs:/{run_id}/best_model"
    # best_model = mlflow.pytorch.load_model(best_model_uri)
    logger.info("Best model URI: %s", best_model_uri)

    return best_model_uri
