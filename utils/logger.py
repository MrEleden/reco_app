"""
Logging utilities for movie recommendation system.
"""

import logging
import os
from datetime import datetime
from typing import Optional


class Logger:
    """Custom logger for training and evaluation."""

    def __init__(self, log_file: Optional[str] = None, level: int = logging.INFO):
        self.logger = logging.getLogger("MovieRecommender")
        self.logger.setLevel(level)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def log_model_info(self, model, config: dict):
        """Log model information."""
        self.info("=" * 50)
        self.info("MODEL INFORMATION")
        self.info("=" * 50)
        self.info(f"Model: {model.__class__.__name__}")
        self.info(f"Parameters: {model.get_num_parameters():,}")
        self.info("Configuration:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")
        self.info("=" * 50)

    def log_training_start(self, epochs: int, batch_size: int, lr: float):
        """Log training start."""
        self.info("=" * 50)
        self.info("TRAINING START")
        self.info("=" * 50)
        self.info(f"Epochs: {epochs}")
        self.info(f"Batch Size: {batch_size}")
        self.info(f"Learning Rate: {lr}")
        self.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("=" * 50)

    def log_epoch(
        self, epoch: int, total_epochs: int, train_loss: float, val_loss: float, epoch_time: float, metrics: dict = None
    ):
        """Log epoch results."""
        message = (
            f"Epoch {epoch+1}/{total_epochs}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )

        if metrics:
            for metric_name, metric_value in metrics.items():
                message += f", {metric_name}: {metric_value:.4f}"

        self.info(message)

    def log_training_end(self, best_loss: float, total_time: float):
        """Log training end."""
        self.info("=" * 50)
        self.info("TRAINING COMPLETE")
        self.info("=" * 50)
        self.info(f"Best Validation Loss: {best_loss:.4f}")
        self.info(f"Total Training Time: {total_time:.2f}s")
        self.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("=" * 50)
