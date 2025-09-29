"""
MLflow utilities for experiment tracking and model management.
"""

import os
import mlflow
import mlflow.pytorch
import torch
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path


class MLflowTracker:
    """MLflow experiment tracker for recommendation models."""

    def __init__(self, experiment_name: str = "movie_recommendation", tracking_uri: Optional[str] = None):
        """Initialize MLflow tracker."""
        self.experiment_name = experiment_name

        # Set tracking URI (defaults to local ./mlruns)
        if tracking_uri is None:
            # Use relative path for Windows compatibility
            tracking_uri = "./mlruns"

        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            print(f"Warning: Could not get existing experiment: {e}")
            experiment_id = mlflow.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_name)
        self.experiment_id = experiment_id

    def start_run(self, run_name: str, model_name: str, config: Dict[str, Any]):
        """Start a new MLflow run."""
        run = mlflow.start_run(run_name=run_name)

        # Log model type and configuration
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("framework", "pytorch")

        # Log configuration parameters
        self._log_config(config)

        return run

    def _log_config(self, config: Dict[str, Any], prefix: str = ""):
        """Recursively log configuration parameters."""
        for key, value in config.items():
            param_name = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                self._log_config(value, prefix=f"{param_name}.")
            else:
                # Convert value to string if it's not a basic type
                if not isinstance(value, (str, int, float, bool)):
                    value = str(value)
                mlflow.log_param(param_name, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, float(value), step=step)

    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        train_loss: float,
        val_loss: float,
    ):
        """Log metrics for a specific epoch."""
        # Log losses
        mlflow.log_metric("train_loss", float(train_loss), step=epoch)
        mlflow.log_metric("val_loss", float(val_loss), step=epoch)

        # Log training metrics
        for name, value in train_metrics.items():
            mlflow.log_metric(f"train_{name}", float(value), step=epoch)

        # Log validation metrics
        for name, value in val_metrics.items():
            mlflow.log_metric(f"val_{name}", float(value), step=epoch)

    def log_model(self, model: torch.nn.Module, model_name: str, best_metrics: Dict[str, float], model_path: str):
        """Log the trained model to MLflow."""
        # Log the model
        mlflow.pytorch.log_model(
            pytorch_model=model, artifact_path="model", registered_model_name=f"recommendation_{model_name}"
        )

        # Log the model file
        mlflow.log_artifact(model_path, "checkpoints")

        # Log final metrics as tags for easy filtering
        for name, value in best_metrics.items():
            mlflow.set_tag(f"final_{name}", f"{value:.4f}")

    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log an artifact to MLflow."""
        mlflow.log_artifact(artifact_path, artifact_name)

    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()


class MLflowModelSelector:
    """MLflow model selection utilities."""

    def __init__(self, experiment_name: str = "movie_recommendation"):
        """Initialize model selector."""
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def get_best_models(self, metric_name: str = "val_rmse", top_k: int = 5) -> pd.DataFrame:
        """Get top k best models based on a metric."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            print(f"Experiment '{self.experiment_name}' not found!")
            return pd.DataFrame()

        # Search for runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} ASC"],  # ASC for lower-is-better metrics
            max_results=top_k,
        )

        if runs.empty:
            print("No runs found!")
            return pd.DataFrame()

        # Select relevant columns
        columns_of_interest = [
            "run_id",
            "tags.model_type",
            "metrics.val_rmse",
            "metrics.val_mae",
            "metrics.val_accuracy",
            "metrics.val_precision",
            "metrics.val_recall",
            "metrics.val_f1",
            "status",
            "start_time",
        ]

        available_columns = [col for col in columns_of_interest if col in runs.columns]
        return runs[available_columns].head(top_k)

    def load_best_model(self, metric_name: str = "val_rmse") -> tuple:
        """Load the best model based on a metric."""
        best_runs = self.get_best_models(metric_name=metric_name, top_k=1)

        if best_runs.empty:
            return None, None

        best_run_id = best_runs.iloc[0]["run_id"]
        model_uri = f"runs:/{best_run_id}/model"

        # Load the model
        model = mlflow.pytorch.load_model(model_uri)

        return model, best_run_id

    def compare_models(self) -> pd.DataFrame:
        """Compare all models across different metrics."""
        # Set the experiment explicitly
        mlflow.set_experiment(self.experiment_name)
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            print(f"Experiment '{self.experiment_name}' not found!")
            return pd.DataFrame()

        print(f"Found experiment: {experiment.name} (ID: {experiment.experiment_id})")
        
        # Search for all runs in this experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], 
            order_by=["start_time DESC"],
            max_results=100
        )

        if runs.empty:
            print("No runs found!")
            return pd.DataFrame()

        print(f"Found {len(runs)} runs")

        # Create comparison dataframe
        comparison_data = []
        for _, run in runs.iterrows():
            # Try multiple ways to get model type with better fallback
            model_type = (
                run.get("tags.model_type") or 
                run.get("params.model.name") or 
                run.get("params.model_name") or
                self._extract_model_from_run_name(run.get("tags.mlflow.runName", "")) or
                "Unknown"
            )
            
            # Get run status
            status = run.get("status", "UNKNOWN")
            
            # Only include completed runs with metrics
            if status == "FINISHED" and not pd.isna(run.get("metrics.val_rmse")):
                comparison_data.append(
                    {
                        "model_type": model_type,
                        "run_id": run["run_id"][:8],  # Shortened for display
                        "full_run_id": run["run_id"],  # Keep full ID for reference
                        "val_rmse": run.get("metrics.val_rmse", float("inf")),
                        "val_mae": run.get("metrics.val_mae", float("inf")),
                        "val_accuracy": run.get("metrics.val_accuracy", 0.0),
                    "val_f1": run.get("metrics.val_f1", 0.0),
                    "status": run["status"],
                    "duration": run.get("metrics.training_time", 0.0),
                }
            )

        return pd.DataFrame(comparison_data).sort_values("val_rmse")

    def _extract_model_from_run_name(self, run_name: str) -> str:
        """Extract model type from run name like 'hybrid_movie_recommendation'."""
        if not run_name:
            return ""
        
        # Split by underscore and take first part
        parts = run_name.split("_")
        if len(parts) > 0:
            model_name = parts[0].lower()
            # Map known model names
            model_mapping = {
                "collaborative": "collaborative",
                "content": "content_based", 
                "hybrid": "hybrid",
                "deep": "deep_cf"
            }
            return model_mapping.get(model_name, model_name)
        return ""

    def get_model_artifacts(self, run_id: str) -> list:
        """Get all artifacts for a specific run."""
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        return [artifact.path for artifact in artifacts]


def start_mlflow_ui(port: int = 5000):
    """Start MLflow UI server."""
    import subprocess
    import webbrowser
    import time

    print(f"Starting MLflow UI on port {port}...")

    # Start MLflow server in background
    process = subprocess.Popen(["mlflow", "ui", "--port", str(port), "--host", "127.0.0.1"])

    # Wait a moment for server to start
    time.sleep(3)

    # Open browser
    url = f"http://127.0.0.1:{port}"
    print(f"MLflow UI available at: {url}")
    webbrowser.open(url)

    return process


def create_model_comparison_report(experiment_name: str = "movie_recommendation") -> str:
    """Create a markdown report comparing all models."""
    selector = MLflowModelSelector(experiment_name)
    comparison_df = selector.compare_models()

    if comparison_df.empty:
        return "No models found in MLflow experiments."

    # Create markdown report
    report = f"""# Movie Recommendation Model Comparison Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Best Models by RMSE

| Rank | Model Type | Run ID | RMSE | MAE | Accuracy | F1-Score | Status |
|------|------------|---------|------|-----|----------|----------|--------|
"""

    for idx, row in comparison_df.head(10).iterrows():
        report += f"| {idx+1} | {row['model_type']} | {row['run_id']} | {row['val_rmse']:.4f} | {row['val_mae']:.4f} | {row['val_accuracy']:.4f} | {row['val_f1']:.4f} | {row['status']} |\n"

    report += f"""
## Summary Statistics

- **Total Models Trained**: {len(comparison_df)}
- **Best Model**: {comparison_df.iloc[0]['model_type']} (RMSE: {comparison_df.iloc[0]['val_rmse']:.4f})
- **Average RMSE**: {comparison_df['val_rmse'].mean():.4f}
- **Average Accuracy**: {comparison_df['val_accuracy'].mean():.4f}

## Model Type Performance

"""

    # Group by model type
    model_stats = (
        comparison_df.groupby("model_type")
        .agg(
            {
                "val_rmse": ["mean", "min", "max", "count"],
                "val_accuracy": ["mean", "min", "max"],
                "val_f1": ["mean", "min", "max"],
            }
        )
        .round(4)
    )

    report += model_stats.to_string()
    report += "\n"

    return report
