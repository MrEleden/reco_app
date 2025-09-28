"""
Demo script showing Hydra multirun functionality with GPU acceleration.
"""

import subprocess
import os
from pathlib import Path


def run_demo():
    """Run a demo of multirun functionality."""
    print("🚀 Hydra Multirun Demo - GPU Accelerated Training")
    print("=" * 60)

    # Check GPU availability
    print("\n📊 GPU Status:")
    result = subprocess.run(
        [
            "python",
            "-c",
            "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); "
            "print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    print("🎯 Running Multi-Model Comparison...")
    print("Command: python train_hydra.py -m model=collaborative,content_based train.epochs=2 train.batch_size=256")
    print("\nThis will:")
    print("  • Train 2 different models (Collaborative Filtering & Content-Based)")
    print("  • Use GPU acceleration automatically")
    print("  • Save outputs in organized directory structure")
    print("  • Generate comparison logs and metrics")

    print(f"\n📁 Outputs will be saved in:")
    print(f"  outputs/movie_recommendation/multirun/YYYY-MM-DD_HH-MM-SS/")
    print(f"  ├── 0_collaborative/    # Collaborative filtering results")
    print(f"  ├── 1_content_based/    # Content-based filtering results")
    print(f"  └── multirun.yaml       # Multirun configuration")

    # Check if outputs directory exists
    outputs_dir = Path("outputs/movie_recommendation/multirun")
    if outputs_dir.exists():
        print(f"\n📊 Existing multirun sessions:")
        for session_dir in sorted(outputs_dir.iterdir()):
            if session_dir.is_dir():
                print(f"  • {session_dir.name}/")
                # List job directories
                for job_dir in sorted(session_dir.iterdir()):
                    if job_dir.is_dir() and not job_dir.name.startswith("."):
                        print(f"    └── {job_dir.name}/")

    print(f"\n💡 Additional multirun examples:")
    print(f"  # Hyperparameter sweep:")
    print(f"  python train_hydra.py -m train.learning_rate=0.001,0.01,0.1 train.batch_size=256,512")
    print(f"  ")
    print(f"  # All model comparison:")
    print(f"  python train_hydra.py -m model=collaborative,content_based,hybrid,deep_cf")
    print(f"  ")
    print(f"  # Advanced sweep:")
    print(f"  python train_hydra.py -m model=collaborative,deep_cf train.epochs=10,20 train.learning_rate=0.001,0.01")

    print(f"\n✨ Benefits of Hydra multirun in outputs directory:")
    print(f"  • 🗂️  Organized output structure")
    print(f"  • 🔄  Parallel job execution")
    print(f"  • 📊  Easy experiment comparison")
    print(f"  • 🏷️  Automatic job naming and numbering")
    print(f"  • 💾  Complete configuration tracking")
    print(f"  • 🚀  GPU acceleration for all jobs")


if __name__ == "__main__":
    run_demo()
