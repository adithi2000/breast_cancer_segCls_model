from dotenv import load_dotenv
import mlflow
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# mlflow.set_experiment(os.getenv("EXPERIMENT_NAME"))

def select_best_model():
    # Fetch all runs for the experiment
    experiment = mlflow.get_experiment_by_name(os.getenv("EXPERIMENT_NAME"))
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    runs=runs.dropna(subset=['metrics.best_score'])

    # Find the run with the best validation accuracy
    best_run = runs.loc[runs['metrics.best_val_score'].idxmax()]

    print(f"Best Run ID: {best_run['run_id']}")
    print(f"Best Validation Score: {best_run['metrics.best_val_score']}")

    # Load the best model artifact
    best_model_uri = f"runs:/{best_run['run_id']}/model"
    best_model = mlflow.pytorch.load_model(best_model_uri)

    return best_model