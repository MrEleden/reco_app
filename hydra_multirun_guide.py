"""
Hydra Multirun Guide - Multiple Models & Experiments
=====================================================

Complete guide for running systematic experiments with Hydra multirun.
All experiments are automatically tracked with MLflow!
"""


def show_multirun_examples():
    """Show comprehensive multirun examples."""

    print("HYDRA MULTIRUN GUIDE")
    print("=" * 60)

    print("\n1. MODEL COMPARISON EXPERIMENTS")
    print("-" * 40)

    print("A. Compare All Model Types:")
    print("   python train_hydra.py -m model=collaborative,content_based,hybrid,deep_cf")

    print("\nB. Compare Best Models Only:")
    print("   python train_hydra.py -m model=hybrid,deep_cf")

    print("\nC. Focus on Collaborative Models:")
    print("   python train_hydra.py -m model=collaborative,hybrid")

    print("\n2. HYPERPARAMETER SWEEPS")
    print("-" * 30)

    print("A. Learning Rate Optimization:")
    print("   python train_hydra.py -m model=hybrid train.learning_rate=0.0001,0.001,0.01,0.1")

    print("\nB. Batch Size Comparison:")
    print("   python train_hydra.py -m model=hybrid train.batch_size=64,128,256,512")

    print("\nC. Embedding Dimension Sweep:")
    print("   python train_hydra.py -m model=collaborative model.embedding_dim=32,64,128,256")

    print("\nD. Optimizer Comparison:")
    print("   python train_hydra.py -m model=hybrid optimizer=adam,sgd")

    print("\nE. Dropout Regularization:")
    print("   python train_hydra.py -m model=deep_cf model.dropout=0.1,0.2,0.3,0.5")

    print("\n3. ADVANCED GRID SEARCHES")
    print("-" * 32)

    print("A. Learning Rate + Batch Size Grid:")
    print("   python train_hydra.py -m model=hybrid \\")
    print("     train.learning_rate=0.001,0.01 \\")
    print("     train.batch_size=128,256")
    print("   # This runs 4 experiments: 2 lr × 2 batch_size")

    print("\nB. Model + Optimizer Grid:")
    print("   python train_hydra.py -m \\")
    print("     model=collaborative,hybrid \\")
    print("     optimizer=adam,sgd")
    print("   # This runs 4 experiments: 2 models × 2 optimizers")

    print("\nC. Full Hyperparameter Grid:")
    print("   python train_hydra.py -m model=hybrid \\")
    print("     train.learning_rate=0.001,0.01 \\")
    print("     train.batch_size=128,256 \\")
    print("     model.embedding_dim=64,128")
    print("   # This runs 8 experiments: 2×2×2 combinations")

    print("\n4. TRAINING CONFIGURATION SWEEPS")
    print("-" * 38)

    print("A. Training Duration Comparison:")
    print("   python train_hydra.py -m model=hybrid train.epochs=10,20,30")

    print("\nB. Early Stopping Patience:")
    print("   python train_hydra.py -m model=hybrid train.patience=3,5,10")

    print("\nC. Weight Decay Regularization:")
    print("   python train_hydra.py -m model=hybrid train.weight_decay=0.0001,0.001,0.01")

    print("\n5. MODEL ARCHITECTURE VARIATIONS")
    print("-" * 38)

    print("A. Hybrid Model Fusion Weights:")
    print("   python train_hydra.py -m model=hybrid \\")
    print("     model.collaborative_weight=0.5,0.7,0.9")

    print("\nB. Deep CF Hidden Layers (Advanced):")
    print("   # Note: Requires YAML config modification for lists")
    print("   python train_hydra.py -m model=deep_cf")

    print("\n6. SYSTEMATIC EXPERIMENT CAMPAIGNS")
    print("-" * 42)

    print("A. Quick Model Screening (Fast):")
    print("   python train_hydra.py -m \\")
    print("     model=collaborative,hybrid,deep_cf \\")
    print("     train.epochs=5")

    print("\nB. Thorough Hyperparameter Search:")
    print("   python train_hydra.py -m \\")
    print("     model=hybrid \\")
    print("     train.learning_rate=0.0001,0.001,0.01 \\")
    print("     train.batch_size=128,256,512 \\")
    print("     model.embedding_dim=64,128")
    print("   # This runs 18 experiments!")

    print("\nC. Optimizer Performance Study:")
    print("   python train_hydra.py -m \\")
    print("     model=collaborative,hybrid \\")
    print("     optimizer=adam,sgd \\")
    print("     train.learning_rate=0.001,0.01")
    print("   # This runs 8 experiments")

    print("\n7. VIEWING MULTIRUN RESULTS")
    print("-" * 32)

    print("After multirun experiments complete:")
    print("A. Quick Results:")
    print("   python check_mlflow.py")

    print("\nB. Comprehensive Analysis:")
    print("   python mlflow_simple_guide.py")

    print("\nC. MLflow Web UI:")
    print("   python -m mlflow ui --port 5000")
    print("   # Open: http://127.0.0.1:5000")

    print("\n8. MULTIRUN OUTPUT ORGANIZATION")
    print("-" * 38)

    print("Hydra automatically organizes multirun outputs:")
    print("outputs/movie_recommendation/multirun/YYYY-MM-DD_HH-MM-SS/")
    print("├── 0_model=collaborative/     # First job")
    print("├── 1_model=hybrid/            # Second job")
    print("├── 2_model=deep_cf/           # Third job")
    print("└── multirun.yaml              # Multirun config")

    print("\nMLflow tracks all runs in: mlruns/")

    print("\n9. TIPS FOR EFFECTIVE MULTIRUN")
    print("-" * 35)

    print("✅ Best Practices:")
    print("• Start with quick model comparison (epochs=5)")
    print("• Use hybrid model for hyperparameter sweeps (best performer)")
    print("• Check intermediate results with check_mlflow.py")
    print("• Use MLflow UI for visual comparison")
    print("• Save systematic experiments for production runs")

    print("\n❌ Things to Avoid:")
    print("• Don't run huge grids without testing small ones first")
    print("• Don't forget the -m flag (multirun mode)")
    print("• Don't interrupt long-running experiments")
    print("• Don't run multiple multiruns simultaneously")

    print("\n" + "=" * 60)
    print("CURRENTLY RUNNING EXPERIMENT")
    print("=" * 60)
    print("✅ Active: python train_hydra.py -m model=collaborative,hybrid,deep_cf")
    print("📊 Progress: Check with 'python check_mlflow.py'")
    print("🌐 View: http://127.0.0.1:5000 (MLflow UI)")


if __name__ == "__main__":
    show_multirun_examples()
