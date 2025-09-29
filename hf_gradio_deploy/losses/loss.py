"""
Loss functions for movie recommendation system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecommenderLoss(nn.Module):
    """Default loss function for recommendation systems."""

    def __init__(self, loss_type: str = "bce"):
        super().__init__()
        if loss_type == "bce":
            self.loss_fn = nn.BCELoss()
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, targets)


class MSELoss(nn.Module):
    """Mean Squared Error Loss for rating prediction."""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, targets)


class BCELoss(nn.Module):
    """Binary Cross Entropy Loss for implicit feedback."""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, targets)


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Loss for implicit feedback."""

    def __init__(self):
        super().__init__()

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute BPR loss.

        Args:
            pos_scores: Scores for positive items
            neg_scores: Scores for negative items
        """
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss


class RankingLoss(nn.Module):
    """Ranking loss for recommendation systems."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute ranking loss.

        Args:
            pos_scores: Scores for positive items
            neg_scores: Scores for negative items
        """
        loss = torch.mean(torch.clamp(self.margin - pos_scores + neg_scores, min=0))
        return loss
