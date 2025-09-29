import sys
import os

# Add parent directory to path to access utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mlflow_utils import MLflowModelSelector

# Initialize model selector
selector = MLflowModelSelector(experiment_name="movie_recommendation")

# Get comparison of all models
print("MLflow Experiment Results")
print("=" * 50)

try:
    comparison = selector.compare_models()

    if not comparison.empty:
        print(f"Found {len(comparison)} model runs!")

        # Show best model prominently
        best_model = comparison.iloc[0]
        print(f"\n*** BEST MODEL ***")
        print(f"   Model: {best_model['model_type']}")
        print(f"   RMSE: {best_model['val_rmse']:.4f}")
        print(f"   Accuracy: {best_model['val_accuracy']:.4f}")
        print(f"   Status: {best_model['status']}")

        print("\nModel Performance Summary:")
        print("-" * 40)

        for idx, row in comparison.head(10).iterrows():
            prefix = ">>>" if idx == 0 else "   "
            print(f"{prefix} {idx+1}. Model: {row['model_type']}")
            print(f"      RMSE: {row['val_rmse']:.4f}")
            print(f"      Accuracy: {row['val_accuracy']:.4f}")
            print(f"      Status: {row['status']}")
            print()
    else:
        print("No completed runs found yet.")
        print("Models may still be training...")

except Exception as e:
    print(f"Error accessing MLflow: {e}")
    print("Make sure training has completed at least one epoch.")

print("\nTo view detailed results:")
print("1. Open MLflow UI: http://127.0.0.1:5000")
print("2. Or check files in: ./mlruns/")
