"""
Simple demo showing the 2 available optimizers: Adam and SGD.
"""

import subprocess
import os


def demo_simple_optimizers():
    """Demo the 2 simple PyTorch optimizers."""
    print("🔧 Simple Optimizer Demo")
    print("=" * 50)

    print("📋 Available Optimizers:")
    print("  • Adam - Adaptive learning rates, good for most cases")
    print("  • SGD  - Simple gradient descent with momentum")

    print(f"\n💡 Usage Examples:")
    print(f"  # Default Adam optimizer")
    print(f"  python train_hydra.py model=collaborative")
    print(f"")
    print(f"  # Use SGD with momentum")
    print(f"  python train_hydra.py optimizer=sgd model=collaborative")
    print(f"")
    print(f"  # Compare both optimizers")
    print(f"  python train_hydra.py -m optimizer=adam,sgd model=collaborative")
    print(f"")
    print(f"  # SGD with custom learning rate")
    print(f"  python train_hydra.py optimizer=sgd train.learning_rate=0.1")

    print(f"\n⚙️  Optimizer Configuration Files:")
    print(f"  📄 conf/optimizer/adam.yaml - Adam settings (betas, eps, etc.)")
    print(f"  📄 conf/optimizer/sgd.yaml  - SGD settings (momentum, nesterov, etc.)")

    print(f"\n🚀 Benefits of Simple Setup:")
    print(f"  ✅ Easy to understand and configure")
    print(f"  ✅ Battle-tested PyTorch optimizers")
    print(f"  ✅ Good performance for most recommendation tasks")
    print(f"  ✅ Less complexity, fewer hyperparameters to tune")
    print(f"  ✅ Fast experimentation with -m multirun")

    print(f"\n🎯 Recommendations:")
    print(f"  • Start with Adam (default) - works well for most cases")
    print(f"  • Try SGD if you want simplicity and robustness")
    print(f"  • Use multirun to compare both: -m optimizer=adam,sgd")

    # Check if config files exist
    adam_config = "conf/optimizer/adam.yaml"
    sgd_config = "conf/optimizer/sgd.yaml"

    print(f"\n📁 Configuration Status:")
    if os.path.exists(adam_config):
        print(f"  ✅ {adam_config} - Ready")
    else:
        print(f"  ❌ {adam_config} - Missing")

    if os.path.exists(sgd_config):
        print(f"  ✅ {sgd_config} - Ready")
    else:
        print(f"  ❌ {sgd_config} - Missing")


if __name__ == "__main__":
    demo_simple_optimizers()
