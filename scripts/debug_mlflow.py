#!/usr/bin/env python3
"""
Debug MLflow model names to understand what's going wrong.
"""

import sys
import os
import mlflow
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_model_names(experiment_name="movie_recommendation"):
    """Debug model names in MLflow runs."""
    
    # Initialize MLflow
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"Experiment {experiment_name} not found!")
        return
    
    print(f"Debugging experiment: {experiment.name}")
    print("="*60)
    
    # Get recent runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=10,
        order_by=["start_time DESC"]
    )
    
    for i, (_, run) in enumerate(runs.iterrows()):
        run_id = run["run_id"]
        print(f"\n--- Run {i+1}: {run_id[:8]} ---")
        print(f"Status: {run.get('status', 'Unknown')}")
        print(f"Run Name: {run.get('tags.mlflow.runName', 'None')}")
        
        # Check all tag-related columns
        tag_cols = [col for col in run.index if col.startswith('tags.')]
        for col in tag_cols:
            if not pd.isna(run[col]):
                print(f"  {col}: {run[col]}")
        
        # Check all param-related columns  
        param_cols = [col for col in run.index if col.startswith('params.')]
        model_params = [col for col in param_cols if 'model' in col.lower()]
        
        if model_params:
            print("  Model Parameters:")
            for col in model_params:
                if not pd.isna(run[col]):
                    print(f"    {col}: {run[col]}")
        
        # Check metrics
        if not pd.isna(run.get('metrics.val_rmse')):
            print(f"  Val RMSE: {run.get('metrics.val_rmse'):.4f}")
            print(f"  Val Accuracy: {run.get('metrics.val_accuracy', 0):.4f}")

if __name__ == "__main__":
    debug_model_names()