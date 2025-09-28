"""
Models package for movie recommendation system.
"""

from .model import BaseModel
from .collaborative.collaborative_filtering import CollaborativeFilteringModel

__all__ = [
    "BaseModel",
    "CollaborativeFilteringModel",
]
