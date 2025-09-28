"""
Advanced MLflow Experiment Workflows
====================================

This script shows you how to run systematic experiments with MLflow tracking
to find the best models through hyperparameter sweeps and model comparisons.
"""

import subprocess
import time
import os


def run_model_comparison_experiment():
    """Run all model types and compare their performance."""

    print("=" * 70)
    print("ADVANCED MLFLOW EXPERIMENT WORKFLOWS")
    print("=" * 70)

    print("\n1. SINGLE MODEL EXPERIMENTS")
    print("-" * 40)
    print("Run individual models:")
    print("  python train_hydra.py model=collaborative")
    print("  python train_hydra.py model=content_based")
    print("  python train_hydra.py model=hybrid")
    print("  python train_hydra.py model=deep_cf")

    print("\n2. MULTI-MODEL COMPARISON (MULTIRUN)")
    print("-" * 45)
    print("Compare all models simultaneously:")
    print("  python train_hydra.py -m model=collaborative,content_based,hybrid,deep_cf")

    print("\n3. HYPERPARAMETER SWEEPS")
    print("-" * 30)

    print("\nA. Learning Rate Sweep:")
    print("  python train_hydra.py -m model=hybrid train.learning_rate=0.001,0.01,0.1")

    print("\nB. Optimizer Comparison:")
    print("  python train_hydra.py -m model=hybrid optimizer=adam,sgd")

    print("\nC. Embedding Dimension Sweep:")
    print("  python train_hydra.py -m model=collaborative model.embedding_dim=64,128,256")

    print("\nD. Batch Size Comparison:")
    print("  python train_hydra.py -m model=hybrid train.batch_size=64,128,256")

    print("\nE. Regularization Sweep:")
    print("  python train_hydra.py -m model=collaborative model.reg_lambda=0.001,0.01,0.1")

    print("\n4. COMPREHENSIVE GRID SEARCH")
    print("-" * 35)
    print("Full hyperparameter grid for best model:")
    print("  python train_hydra.py -m model=hybrid \\")
    print("    train.learning_rate=0.001,0.01 \\")
    print("    train.batch_size=128,256 \\")
    print("    model.embedding_dim=128,256")

    print("\n5. ADVANCED EXPERIMENT PATTERNS")
    print("-" * 40)

    print("\nA. Early Stopping Comparison:")
    print("  python train_hydra.py -m model=hybrid train.patience=5,10,15")

    print("\nB. Architecture Variations (for Deep CF):")
    print("  python train_hydra.py -m model=deep_cf model.hidden_layers=[128],[256],[128,64]")

    print("\nC. Dropout Regularization:")
    print("  python train_hydra.py -m model=deep_cf model.dropout=0.1,0.3,0.5")

    print("\n6. MLflow TRACKING COMMANDS")
    print("-" * 35)

    print("\nView experiments programmatically:")
    print("  python check_mlflow.py")
    print("  python mlflow_simple_guide.py")

    print("\nMLflow UI Commands:")
    print("  python -m mlflow ui --port 5000")
    print("  python -m mlflow ui --host 0.0.0.0 --port 5000  # Access from network")

    print("\n7. MODEL SELECTION & DEPLOYMENT")
    print("-" * 40)

    print("Load best model:")
    selection_code = """
from utils.mlflow_utils import MLflowModelSelector

# Initialize selector
selector = MLflowModelSelector(experiment_name="movie_recommendation")

# Get best model by RMSE
best_model, run_id = selector.load_best_model(metric_name="val_rmse")
print(f"Loaded best model from run: {run_id}")

# Get best model by accuracy  
best_acc_model, acc_run_id = selector.load_best_model(metric_name="val_accuracy")
print(f"Loaded best accuracy model from run: {acc_run_id}")

# Compare all models
comparison = selector.compare_models()
print("\\nModel Rankings:")
print(comparison[['model_type', 'val_rmse', 'val_accuracy']].head())
"""
    print(selection_code)

    print("\n8. EXPERIMENT AUTOMATION")
    print("-" * 30)

    automation_code = """
# Create automated experiment script
import subprocess
import time

experiments = [
    "python train_hydra.py model=collaborative",
    "python train_hydra.py model=content_based", 
    "python train_hydra.py model=hybrid",
    "python train_hydra.py model=deep_cf"
]

for exp in experiments:
    print(f"Running: {exp}")
    result = subprocess.run(exp.split(), capture_output=True, text=True)
    if result.returncode == 0:
        print("Success!")
    else:
        print(f"Error: {result.stderr}")
    time.sleep(5)  # Brief pause between experiments
"""
    print(automation_code)

    print("\n" + "=" * 70)
    print("EXPERIMENT WORKFLOW SUMMARY")
    print("=" * 70)

    print("\nRecommended Workflow:")
    print("1. Start with single model test: python train_hydra.py model=hybrid")
    print("2. Compare all models: python train_hydra.py -m model=collaborative,content_based,hybrid,deep_cf")
    print("3. Tune best model hyperparameters")
    print("4. Check results: python mlflow_simple_guide.py")
    print("5. Load best model for production")

    print("\nMLflow Benefits:")
    print("- Automatic experiment tracking")
    print("- Model versioning and registry")
    print("- Performance comparison")
    print("- Hyperparameter logging")
    print("- Reproducible experiments")

    print("\nNext: Run some experiments and check http://127.0.0.1:5000 to see results!")


if __name__ == "__main__":
    run_model_comparison_experiment()
