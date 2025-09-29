"""
Evaluation metrics for movie recommendation system.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RecommenderMetrics:
    """Collection of recommendation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new batch."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        if len(self.predictions) == 0:
            return {}

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        metrics = {
            "rmse": np.sqrt(mean_squared_error(targets, predictions)),
            "mae": mean_absolute_error(targets, predictions),
        }

        # For binary classification metrics
        if np.all(np.isin(targets, [0, 1])):
            binary_preds = (predictions > 0.5).astype(int)
            metrics.update(
                {
                    "accuracy": np.mean(binary_preds == targets),
                    "precision": self._precision(targets, binary_preds),
                    "recall": self._recall(targets, binary_preds),
                    "f1": self._f1_score(targets, binary_preds),
                }
            )

        return metrics

    def _precision(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate precision."""
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def _recall(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate recall."""
        tp = np.sum((predictions == 1) & (targets == 1))
        fn = np.sum((predictions == 0) & (targets == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def _f1_score(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate F1 score."""
        precision = self._precision(targets, predictions)
        recall = self._recall(targets, predictions)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


class RMSE:
    """Root Mean Square Error."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())

    def compute(self) -> float:
        if len(self.predictions) == 0:
            return 0.0
        return np.sqrt(mean_squared_error(self.targets, self.predictions))


class MAE:
    """Mean Absolute Error."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())

    def compute(self) -> float:
        if len(self.predictions) == 0:
            return 0.0
        return mean_absolute_error(self.targets, self.predictions)


class Precision:
    """Precision at K for ranking metrics."""

    def __init__(self, k: int = 10):
        self.k = k
        self.reset()

    def reset(self):
        self.scores = []

    def update(self, recommendations: List[int], relevant_items: List[int]):
        """Update with recommendations and relevant items for a user."""
        top_k = recommendations[: self.k]
        relevant_in_top_k = len(set(top_k) & set(relevant_items))
        precision = relevant_in_top_k / self.k
        self.scores.append(precision)

    def compute(self) -> float:
        return np.mean(self.scores) if self.scores else 0.0


class Recall:
    """Recall at K for ranking metrics."""

    def __init__(self, k: int = 10):
        self.k = k
        self.reset()

    def reset(self):
        self.scores = []

    def update(self, recommendations: List[int], relevant_items: List[int]):
        """Update with recommendations and relevant items for a user."""
        if len(relevant_items) == 0:
            return

        top_k = recommendations[: self.k]
        relevant_in_top_k = len(set(top_k) & set(relevant_items))
        recall = relevant_in_top_k / len(relevant_items)
        self.scores.append(recall)

    def compute(self) -> float:
        return np.mean(self.scores) if self.scores else 0.0


class NDCG:
    """Normalized Discounted Cumulative Gain."""

    def __init__(self, k: int = 10):
        self.k = k
        self.reset()

    def reset(self):
        self.scores = []

    def update(self, recommendations: List[int], relevance_scores: Dict[int, float]):
        """Update with recommendations and relevance scores."""
        dcg = self._dcg(recommendations[: self.k], relevance_scores)
        ideal_order = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
        idcg = self._dcg(ideal_order[: self.k], relevance_scores)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        self.scores.append(ndcg)

    def _dcg(self, recommendations: List[int], relevance_scores: Dict[int, float]) -> float:
        """Calculate Discounted Cumulative Gain."""
        dcg = 0.0
        for i, item_id in enumerate(recommendations):
            relevance = relevance_scores.get(item_id, 0.0)
            dcg += relevance / np.log2(i + 2)  # +2 because log2(1) = 0
        return dcg

    def compute(self) -> float:
        return np.mean(self.scores) if self.scores else 0.0
