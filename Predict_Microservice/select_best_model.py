from dotenv import load_dotenv
import mlflow
import os


# mlflow.set_experiment(os.getenv("EXPERIMENT_NAME"))

def select_best_model():
    # Fetch all runs for the experiment
    # load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    exp=mlflow.search_experiments()
    for e in exp:
        print(e.name)
    experiment = mlflow.get_experiment_by_name(os.getenv("EXPERIMENT_NAME"))
    print(f"Experiment ID: {experiment.experiment_id}, Name: {experiment.name}")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    runs=runs.dropna(subset=['metrics.best_val_score'])

    # Find the run with the best validation accuracy
    best_run = runs.loc[runs['metrics.best_val_score'].idxmax()]

    print(f"Best Run ID: {best_run['run_id']}/")
    print(f"Best Validation Score: {best_run['metrics.best_val_score']}")

    # Load the best model artifact
    run_id = best_run['run_id']
    best_model_uri = f"runs:/{run_id}/best_model"
    # print(f"Best Model URI: {best_model_uri}")
    # best_model = mlflow.pytorch.load_model(best_model_uri)
    print(f"Best Model URI: {best_model_uri}")

    return best_model_uri