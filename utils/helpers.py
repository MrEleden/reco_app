"""
Helper utilities for the movie recommendation system.
"""

import torch
import numpy as np
import random
import json
import os
from typing import Dict, Any
import sys

# Add config to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import RANDOM_SEED


def setup_seed(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(config: Dict[str, Any], filepath: str):
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (dict, list, str, int, float, bool, type(None))):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)

    with open(filepath, "w") as f:
        json.dump(serializable_config, f, indent=2)


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, "r") as f:
        config = json.load(f)

    return config


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def print_model_summary(model: torch.nn.Module, input_size: tuple = None):
    """Print a summary of the model architecture."""
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 70)

    total_params = 0
    trainable_params = 0

    for name, parameter in model.named_parameters():
        param_count = parameter.numel()
        total_params += param_count
        if parameter.requires_grad:
            trainable_params += param_count

        print(f"{name:40} {list(parameter.shape):20} {param_count:10,}")

    print("=" * 70)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 70)


def ensure_dir(directory: str):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb
