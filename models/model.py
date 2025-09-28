"""
Base model class for all recommendation models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all recommendation models.
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        Make predictions with the model.
        Must be implemented by subclasses.
        """
        pass

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_model(self, path: str, **kwargs):
        """Save model state dict and additional info."""
        save_dict = {"model_state_dict": self.state_dict(), "model_config": getattr(self, "config", {}), **kwargs}
        torch.save(save_dict, path)

    def load_model(self, path: str, map_location="cpu"):
        """Load model from saved state dict."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
