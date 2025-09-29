#!/usr/bin/env python3
"""
Fix missing model names in MLflow experiments by updating tags.
"""

import sys
import os
import mlflow
from mlflow.tracking import MlflowClient

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def fix_model_names(experiment_name="movie_recommendation"):
    """Fix missing model_type tags in MLflow runs."""
    
    # Initialize MLflow
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"Experiment {experiment_name} not found!")
        return
    
    print(f"Fixing model names in experiment: {experiment.name}")
    
    # Get all runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1000
    )
    
    client = MlflowClient()
    fixed_count = 0
    
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        
        # Check if model_type tag already exists and is meaningful
        current_model_type = run.get("tags.model_type")
        
        if current_model_type and current_model_type not in ["Unknown", ""] and not any(word in current_model_type for word in ["carp", "jay", "sheep", "dog", "cat"]):
            continue  # Already has a good model type
            
        # Try to determine model type from parameters
        model_type = None
        
        # Check various parameter sources
        if run.get("params.model.name"):
            model_type = run.get("params.model.name")
        elif run.get("params.model_name"):
            model_type = run.get("params.model_name") 
        elif run.get("params.model.type"):
            model_type = run.get("params.model.type")
        else:
            # Try to extract from run name
            run_name = run.get("tags.mlflow.runName", "")
            if "_" in run_name:
                potential_model = run_name.split("_")[0].lower()
                model_mapping = {
                    "collaborative": "collaborative",
                    "content": "content_based",
                    "hybrid": "hybrid", 
                    "deep": "deep_cf"
                }
                model_type = model_mapping.get(potential_model)
        
        # Update the tag if we found a model type
        if model_type:
            try:
                client.set_tag(run_id, "model_type", model_type)
                print(f"Fixed run {run_id[:8]}: {current_model_type or 'None'} -> {model_type}")
                fixed_count += 1
            except Exception as e:
                print(f"Error updating run {run_id[:8]}: {e}")
    
    print(f"\nFixed {fixed_count} runs with missing or incorrect model names.")
    print("Re-run the check script to see updated results!")

if __name__ == "__main__":
    fix_model_names()