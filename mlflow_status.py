"""
MLflow Status Checker & Auto-Refresh
===================================

Use this script to:
1. Check if MLflow server is running
2. View latest experiment results
3. Automatically refresh experiment data
"""

import requests
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.mlflow_utils import MLflowModelSelector


def check_server_status():
    """Check if MLflow server is accessible."""
    try:
        response = requests.get("http://127.0.0.1:5000", timeout=5)
        return response.status_code == 200
    except:
        return False


def show_latest_experiments():
    """Show latest experiment results."""
    try:
        selector = MLflowModelSelector(experiment_name="movie_recommendation")
        comparison = selector.compare_models()

        if not comparison.empty:
            print(f"\n📊 Latest Experiment Results ({len(comparison)} runs)")
            print("-" * 50)

            for idx, run in comparison.head(5).iterrows():
                model_type = run["model_type"]
                val_rmse = run["val_rmse"]
                val_accuracy = run["val_accuracy"]
                status = run["status"]

                print(f"  {idx+1}. {model_type:12s} | RMSE: {val_rmse:.4f} | Acc: {val_accuracy:.3f} | {status}")

            best_model = comparison.iloc[0]
            print(f"\n🏆 Best Model: {best_model['model_type']} (RMSE: {best_model['val_rmse']:.4f})")
        else:
            print("❌ No experiments found")

    except Exception as e:
        print(f"❌ Error accessing experiments: {e}")


def main():
    """Main monitoring function."""

    print("🔬 MLflow Status Monitor")
    print("=" * 40)

    # Check server status
    if check_server_status():
        print("✅ MLflow server is running: http://127.0.0.1:5000")

        # Show latest experiments
        show_latest_experiments()

        print(f"\n💡 Server is live and tracking experiments!")
        print(f"📈 Open browser: http://127.0.0.1:5000")
        print(f"🔄 Run experiments: python train_hydra.py -m model=hybrid,collaborative")

    else:
        print("❌ MLflow server is not running")
        print("🚀 Start server: python start_mlflow.py")
        print("🔧 Or manually: python -m mlflow ui --port 5000")


if __name__ == "__main__":
    main()
