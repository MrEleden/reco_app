"""
MLflow Experiment Tracking & Model Selection Guide
=================================================

This script demonstrates how to use MLflow for comprehensive experiment tracking
and model selection in your movie recommendation system.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.mlflow_utils import MLflowModelSelector, create_model_comparison_report
import pandas as pd


def run_experiment_comparison():
    """Demonstrate MLflow experiment tracking and model selection."""

    print("MLflow Experiment Tracking & Model Selection")
    print("=" * 60)

    # Initialize MLflow model selector
    selector = MLflowModelSelector(experiment_name="movie_recommendation")

    print("\n1. VIEWING ALL EXPERIMENTS")
    print("-" * 40)

    try:
        # Get all experiments
        all_runs = selector.compare_models()

        if not all_runs.empty:
            print(f"Found {len(all_runs)} completed experiments!")
            print(f"Experiments sorted by validation RMSE (lower is better):")
            print()

            # Display top experiments
            for idx, run in all_runs.head(10).iterrows():
                model_type = run["model_type"]
                run_id = run["run_id"]
                val_rmse = run["val_rmse"]
                val_accuracy = run["val_accuracy"]
                val_f1 = run["val_f1"]
                status = run["status"]

                print(
                    f"  {idx+1:2d}. Model: {model_type:15s} | RMSE: {val_rmse:.4f} | Acc: {val_accuracy:.3f} | F1: {val_f1:.3f} | Status: {status}"
                )

            print(f"\nBEST MODEL: {all_runs.iloc[0]['model_type']} (RMSE: {all_runs.iloc[0]['val_rmse']:.4f})")

        else:
            print("No completed experiments found.")
            print("Run some training first:")
            print("   python train_hydra.py model=collaborative")
            return

    except Exception as e:
        print(f"Error accessing experiments: {e}")
        return

    print("\n2. SELECTING BEST MODELS BY DIFFERENT METRICS")
    print("-" * 50)

    # Best by RMSE (lower is better)
    print("Best by RMSE (Prediction Accuracy):")
    best_rmse = selector.get_best_models(metric_name="val_rmse", top_k=3)
    for idx, run in best_rmse.iterrows():
        model_type = run.get("tags.model_type", "Unknown")
        rmse = run.get("metrics.val_rmse", "N/A")
        print(f"   {idx+1}. {model_type} - RMSE: {rmse:.4f}")

    # Best by Accuracy (higher is better)
    print("\nBest by Accuracy (Classification Performance):")
    try:
        best_acc = selector.get_best_models(metric_name="val_accuracy", top_k=3)
        # For accuracy, we want higher values, so reverse order
        best_acc_sorted = (
            best_acc.sort_values("metrics.val_accuracy", ascending=False) if not best_acc.empty else best_acc
        )
        for idx, run in best_acc_sorted.head(3).iterrows():
            model_type = run.get("tags.model_type", "Unknown")
            accuracy = run.get("metrics.val_accuracy", "N/A")
            print(f"   {idx+1}. {model_type} - Accuracy: {accuracy:.4f}")
    except:
        print("   No accuracy metrics available yet")

    # Best by F1-Score (higher is better)
    print("\nBest by F1-Score (Balanced Performance):")
    try:
        best_f1 = selector.get_best_models(metric_name="val_f1", top_k=3)
        best_f1_sorted = best_f1.sort_values("metrics.val_f1", ascending=False) if not best_f1.empty else best_f1
        for idx, run in best_f1_sorted.head(3).iterrows():
            model_type = run.get("tags.model_type", "Unknown")
            f1 = run.get("metrics.val_f1", "N/A")
            print(f"   {idx+1}. {model_type} - F1: {f1:.4f}")
    except:
        print("   No F1 metrics available yet")

    print("\n3. EXPERIMENT INSIGHTS")
    print("-" * 30)

    if not all_runs.empty:
        # Calculate statistics
        avg_rmse = all_runs["val_rmse"].mean()
        best_rmse = all_runs["val_rmse"].min()
        avg_acc = all_runs["val_accuracy"].mean()
        best_acc = all_runs["val_accuracy"].max()

        print(f"Performance Summary:")
        print(f"   Average RMSE: {avg_rmse:.4f} | Best RMSE: {best_rmse:.4f}")
        print(f"   Average Accuracy: {avg_acc:.4f} | Best Accuracy: {best_acc:.4f}")

        # Model type analysis
        model_performance = (
            all_runs.groupby("model_type")
            .agg({"val_rmse": ["count", "mean", "min"], "val_accuracy": ["mean", "max"]})
            .round(4)
        )

        print(f"\nPerformance by Model Type:")
        for model_type in all_runs["model_type"].unique():
            model_runs = all_runs[all_runs["model_type"] == model_type]
            print(f"   {model_type}:")
            print(
                f"     - Runs: {len(model_runs)} | Best RMSE: {model_runs['val_rmse'].min():.4f} | Best Acc: {model_runs['val_accuracy'].max():.4f}"
            )

    print("\n4. HOW TO USE BEST MODEL")
    print("-" * 35)

    print("To load and use the best model:")
    print(
        """
    from utils.mlflow_utils import MLflowModelSelector
    
    # Load best model by RMSE
    selector = MLflowModelSelector()
    model, run_id = selector.load_best_model(metric_name="val_rmse")
    
    # Use the model for predictions
    predictions = model(user_ids, movie_ids)
    """
    )

    print("\n5. GENERATING REPORTS")
    print("-" * 25)

    try:
        report = create_model_comparison_report()
        report_file = "model_comparison_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"Could not generate report: {e}")

    print("\n" + "=" * 60)
    print("MLflow Experiment Analysis Complete!")
    print("\nNext Steps:")
    print("   1. Open MLflow UI: http://127.0.0.1:5000")
    print("   2. Run more experiments: python train_hydra.py -m model=collaborative,hybrid,deep_cf")
    print("   3. Compare optimizers: python train_hydra.py -m optimizer=adam,sgd")
    print("   4. Hyperparameter sweep: python train_hydra.py -m train.learning_rate=0.001,0.01,0.1")


if __name__ == "__main__":
    run_experiment_comparison()
