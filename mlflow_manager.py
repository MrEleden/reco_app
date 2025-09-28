#!/usr/bin/env python3
"""
MLflow Model Selection and Comparison Script

This script provides easy access to MLflow experiment results and model selection.

Usage examples:
    # Start MLflow UI
    python mlflow_manager.py --ui

    # Compare all models
    python mlflow_manager.py --compare

    # Get best model by RMSE
    python mlflow_manager.py --best --metric val_rmse

    # Generate comparison report
    python mlflow_manager.py --report

    # Load and test best model
    python mlflow_manager.py --test --metric val_rmse
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.mlflow_utils import MLflowModelSelector, start_mlflow_ui, create_model_comparison_report


def main():
    parser = argparse.ArgumentParser(description="MLflow Model Management")

    # Action arguments
    parser.add_argument("--ui", action="store_true", help="Start MLflow UI")
    parser.add_argument("--compare", action="store_true", help="Compare all models")
    parser.add_argument("--best", action="store_true", help="Get best model")
    parser.add_argument("--report", action="store_true", help="Generate comparison report")
    parser.add_argument("--test", action="store_true", help="Load and test best model")

    # Configuration arguments
    parser.add_argument("--metric", default="val_rmse", help="Metric for model selection")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top models to show")
    parser.add_argument("--experiment", default="movie_recommendation", help="MLflow experiment name")
    parser.add_argument("--port", type=int, default=5000, help="Port for MLflow UI")

    args = parser.parse_args()

    # Initialize model selector
    selector = MLflowModelSelector(experiment_name=args.experiment)

    if args.ui:
        print(f"🚀 Starting MLflow UI for experiment: {args.experiment}")
        process = start_mlflow_ui(port=args.port)
        print(f"MLflow UI started on port {args.port}")
        print("Press Ctrl+C to stop the server")
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nStopping MLflow UI...")
            process.terminate()

    elif args.compare:
        print(f"📊 Comparing models in experiment: {args.experiment}")
        comparison = selector.compare_models()

        if not comparison.empty:
            print("\n🏆 Model Performance Comparison:")
            print("=" * 80)

            # Format the output nicely
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            pd.set_option("display.float_format", "{:.4f}".format)

            print(comparison.to_string(index=False))

            print(f"\n📈 Summary:")
            print(f"   • Total models: {len(comparison)}")
            print(f"   • Best model: {comparison.iloc[0]['model_type']} (RMSE: {comparison.iloc[0]['val_rmse']:.4f})")
            print(f"   • Average RMSE: {comparison['val_rmse'].mean():.4f}")
            print(f"   • Average Accuracy: {comparison['val_accuracy'].mean():.4f}")
        else:
            print("❌ No models found in the experiment.")

    elif args.best:
        print(f"🎯 Finding best model by {args.metric}")
        best_models = selector.get_best_models(metric_name=args.metric, top_k=args.top_k)

        if not best_models.empty:
            print(f"\n🏆 Top {args.top_k} models by {args.metric}:")
            print("=" * 80)

            for idx, row in best_models.iterrows():
                print(
                    f"{idx+1}. {row.get('tags.model_type', 'Unknown')} - {args.metric}: {row.get(f'metrics.{args.metric}', 'N/A'):.4f}"
                )
                print(f"   Run ID: {row['run_id'][:8]}...")
                print()
        else:
            print("❌ No models found in the experiment.")

    elif args.report:
        print(f"📋 Generating model comparison report for experiment: {args.experiment}")
        report = create_model_comparison_report(experiment_name=args.experiment)

        # Save report to file
        report_file = f"model_comparison_report.md"
        with open(report_file, "w") as f:
            f.write(report)

        print(f"✅ Report saved to: {report_file}")
        print(f"\n{report}")

    elif args.test:
        print(f"🔬 Loading best model by {args.metric} for testing")
        model, run_id = selector.load_best_model(metric_name=args.metric)

        if model is not None:
            print(f"✅ Successfully loaded model from run: {run_id[:8]}...")
            print(f"   Model type: {type(model).__name__}")

            # You can add more testing logic here
            print("   Model is ready for inference!")
        else:
            print("❌ Could not load model.")

    else:
        print("ℹ️  Please specify an action:")
        print("   --ui      : Start MLflow UI")
        print("   --compare : Compare all models")
        print("   --best    : Get best models")
        print("   --report  : Generate report")
        print("   --test    : Load best model")
        print("\nUse --help for more options.")


if __name__ == "__main__":
    main()
