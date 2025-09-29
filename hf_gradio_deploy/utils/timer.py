"""
Timer utilities for measuring execution time.
"""

import time
from typing import Optional


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer not started. Call start() first.")

        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        return elapsed

    def elapsed(self) -> float:
        """Get elapsed time without stopping the timer."""
        if self.start_time is None:
            raise RuntimeError("Timer not started. Call start() first.")

        return time.time() - self.start_time

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class TrainingTimer:
    """Timer for tracking training progress."""

    def __init__(self):
        self.total_timer = Timer()
        self.epoch_timer = Timer()
        self.batch_timer = Timer()
        self.epoch_times = []

    def start_training(self):
        """Start total training timer."""
        self.total_timer.start()

    def start_epoch(self):
        """Start epoch timer."""
        self.epoch_timer.start()

    def start_batch(self):
        """Start batch timer."""
        self.batch_timer.start()

    def end_batch(self) -> float:
        """End batch and return batch time."""
        return self.batch_timer.stop()

    def end_epoch(self) -> float:
        """End epoch and return epoch time."""
        epoch_time = self.epoch_timer.stop()
        self.epoch_times.append(epoch_time)
        return epoch_time

    def end_training(self) -> float:
        """End training and return total time."""
        return self.total_timer.stop()

    def get_avg_epoch_time(self) -> float:
        """Get average epoch time."""
        return sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0.0

    def get_estimated_remaining_time(self, current_epoch: int, total_epochs: int) -> float:
        """Estimate remaining training time."""
        if not self.epoch_times:
            return 0.0

        avg_epoch_time = self.get_avg_epoch_time()
        remaining_epochs = total_epochs - current_epoch
        return avg_epoch_time * remaining_epochs

    def format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
