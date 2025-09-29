"""
Comprehensive tech stack integration demonstration script.

This script showcases how PyTorch, Hydra, MLflow, and Optuna work together
for modern ML experimentation and hyperparameter optimization.

Usage examples:
    # 1. Basic training with MLflow tracking
    python demo_tech_stack.py model=hybrid train.epochs=10

    # 2. Multi-model comparison
    python demo_tech_stack.py -m model=collaborative,hybrid

    # 3. Optuna hyperparameter optimization
    python demo_tech_stack.py --config-name=optuna_demo -m

    # 4. View all results
    python check_mlflow.py
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def demo_tech_stack(cfg: DictConfig) -> float:
    """
    Demonstration of complete tech stack integration.

    This function shows how all components work together:
    - PyTorch: Deep learning model training
    - Hydra: Configuration management and multirun
    - MLflow: Automatic experiment tracking
    - Optuna: Intelligent hyperparameter optimization
    """

    log.info("🚀 Tech Stack Integration Demo")
    log.info("=" * 50)

    # Show configuration (Hydra)
    log.info("⚙️ Hydra Configuration:")
    log.info(f"   Model: {cfg.model.name}")
    log.info(f"   Learning Rate: {cfg.train.learning_rate}")
    log.info(f"   Batch Size: {cfg.train.batch_size}")
    log.info(f"   Epochs: {cfg.train.epochs}")

    # Import the main training function
    from train_hydra import main as train_main

    # Run the complete training pipeline
    log.info("🔥 Starting PyTorch Training...")
    best_val_loss = train_main(cfg)

    log.info("✅ Demo completed!")
    log.info(f"📊 Best Validation Loss: {best_val_loss:.4f}")
    log.info("🔬 Results automatically logged to MLflow")
    log.info("🎯 Run 'python check_mlflow.py' to view results")

    return best_val_loss


if __name__ == "__main__":
    demo_tech_stack()
