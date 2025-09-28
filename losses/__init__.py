"""
Losses package for movie recommendation system.
"""

from .loss import RecommenderLoss, MSELoss, BCELoss, BPRLoss

__all__ = ["RecommenderLoss", "MSELoss", "BCELoss", "BPRLoss"]
