"""
Optimizer implementations for movie recommendation system.
"""

import torch
from torch.optim import Optimizer
from typing import Dict, Any, Union


class RecommenderOptimizer:
    """Default optimizer factory for recommendation systems."""

    def __init__(self, optimizer_type: str = "adam"):
        """
        Initialize optimizer factory.

        Args:
            optimizer_type: Type of optimizer ('adam' or 'sgd')
        """
        self.optimizer_type = optimizer_type.lower()
        if self.optimizer_type not in ["adam", "sgd"]:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Use 'adam' or 'sgd'.")

    def create_optimizer(
        self, model: torch.nn.Module, lr: float = 0.001, weight_decay: float = 0.0, **kwargs
    ) -> Optimizer:
        """
        Create optimizer instance.

        Args:
            model: Model to optimize
            lr: Learning rate
            weight_decay: Weight decay coefficient
            **kwargs: Additional optimizer-specific parameters

        Returns:
            Optimizer instance
        """
        if self.optimizer_type == "adam":
            return AdamOptimizer().create_optimizer(model, lr, weight_decay, **kwargs)
        elif self.optimizer_type == "sgd":
            return SGDOptimizer().create_optimizer(model, lr, weight_decay, **kwargs)


class AdamOptimizer:
    """Adam optimizer implementation."""

    def create_optimizer(
        self,
        model: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        amsgrad: bool = False,
        **kwargs,
    ) -> torch.optim.Adam:
        """
        Create Adam optimizer.

        Args:
            model: Model to optimize
            lr: Learning rate
            weight_decay: Weight decay coefficient
            betas: Coefficients for momentum and squared gradient
            eps: Term for numerical stability
            amsgrad: Whether to use AMSGrad variant

        Returns:
            Adam optimizer instance
        """
        return torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )


class SGDOptimizer:
    """SGD optimizer implementation."""

    def create_optimizer(
        self,
        model: torch.nn.Module,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        dampening: float = 0.0,
        nesterov: bool = False,
        **kwargs,
    ) -> torch.optim.SGD:
        """
        Create SGD optimizer.

        Args:
            model: Model to optimize
            lr: Learning rate
            weight_decay: Weight decay coefficient
            momentum: Momentum factor
            dampening: Dampening for momentum
            nesterov: Whether to use Nesterov momentum

        Returns:
            SGD optimizer instance
        """
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )


# Factory function for easy access (similar to loss.py pattern)
def create_optimizer(
    model: torch.nn.Module, optimizer_type: str = "adam", lr: float = 0.001, weight_decay: float = 0.0, **kwargs
) -> Optimizer:
    """
    Factory function to create optimizers.

    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adam' or 'sgd')
        lr: Learning rate
        weight_decay: Weight decay coefficient
        **kwargs: Additional optimizer-specific parameters

    Returns:
        Optimizer instance

    Example:
        optimizer = create_optimizer(model, "adam", lr=0.001, betas=(0.9, 0.999))
        optimizer = create_optimizer(model, "sgd", lr=0.01, momentum=0.9)
    """
    optimizer_factory = RecommenderOptimizer(optimizer_type)
    return optimizer_factory.create_optimizer(model, lr, weight_decay, **kwargs)
