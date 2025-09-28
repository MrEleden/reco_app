"""
Multirun Training Monitor
========================

Monitor your multirun experiments without interrupting them.
Shows live progress and best results.
"""

import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.mlflow_utils import MLflowModelSelector


def monitor_experiments():
    """Monitor experiments and show progress."""

    print("🔄 Monitoring MLflow Experiments")
    print("=" * 50)

    try:
        selector = MLflowModelSelector(experiment_name="movie_recommendation")
        comparison = selector.compare_models()

        if not comparison.empty:
            # Current status
            total_runs = len(comparison)
            finished_runs = len(comparison[comparison["status"] == "FINISHED"])
            running_runs = len(comparison[comparison["status"] == "RUNNING"])

            print(f"📊 Experiment Status:")
            print(f"   Total runs: {total_runs}")
            print(f"   Finished: {finished_runs}")
            print(f"   Running: {running_runs}")
            print()

            # Best results
            finished_comparison = comparison[comparison["status"] == "FINISHED"]
            if not finished_comparison.empty:
                best_run = finished_comparison.iloc[0]
                print(f"🏆 Current Best Model:")
                print(f"   Model: {best_run['model_type']}")
                print(f"   RMSE: {best_run['val_rmse']:.4f}")
                print(f"   Accuracy: {best_run['val_accuracy']:.4f}")
                print()

            # Recent runs (last 5)
            print(f"📈 Recent Experiments:")
            print("-" * 30)
            for idx, run in comparison.head(5).iterrows():
                status_icon = "✅" if run["status"] == "FINISHED" else "🔄"
                print(f"   {status_icon} {run['model_type']:12s} | RMSE: {run['val_rmse']:.4f} | {run['status']}")

            print()
            print("💡 Tips:")
            print("   • Run this script again to check progress")
            print("   • Open MLflow UI: http://127.0.0.1:5000")
            print("   • Let multirun experiments complete without interruption")

        else:
            print("❌ No experiments found")
            print("💡 Start experiments with: python train_hydra.py -m model=collaborative,hybrid")

    except Exception as e:
        print(f"❌ Error accessing experiments: {e}")
        print("💡 Make sure MLflow tracking is set up correctly")


if __name__ == "__main__":
    monitor_experiments()
