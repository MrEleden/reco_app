"""
Models package for movie recommendation system.
"""

from .model import BaseModel
from .collaborative.collaborative_filtering import CollaborativeFilteringModel
from .content_based.content_based_model import ContentBasedModel
from .hybrid.hybrid_model import HybridModel
from .deep.deep_collaborative_filtering import DeepCollaborativeFiltering

__all__ = [
    "BaseModel",
    "CollaborativeFilteringModel",
    "ContentBasedModel",
    "HybridModel",
    "DeepCollaborativeFiltering",
]
