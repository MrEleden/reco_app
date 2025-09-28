"""
Content-based filtering models for recommendation system.
"""

import torch
import torch.nn as nn
from ..model import BaseModel


class ContentBasedModel(BaseModel):
    """Content-based recommendation model using movie features."""

    def __init__(self, n_genres: int, n_factors: int = 50, dropout: float = 0.2):
        super().__init__()
        self.n_genres = n_genres
        self.n_factors = n_factors

        # Store config for saving/loading
        self.config = {"n_genres": n_genres, "n_factors": n_factors, "dropout": dropout}

        # Genre embedding
        self.genre_embedding = nn.Embedding(n_genres, n_factors)

        # User preference layers
        self.user_preference = nn.Sequential(
            nn.Linear(n_factors, n_factors * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_factors * 2, n_factors),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_factors, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.genre_embedding.weight, std=0.01)

        for layer in self.user_preference:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, genre_features: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Average genre embeddings for multi-hot encoding
        genre_emb = self.genre_embedding(genre_features).mean(dim=1)
        rating = self.user_preference(genre_emb)
        return rating.squeeze()

    def predict(self, genre_features: torch.Tensor) -> torch.Tensor:
        """Make predictions (same as forward for this model)."""
        self.eval()
        with torch.no_grad():
            return self.forward(genre_features)

    def get_genre_embeddings(self, genre_ids: torch.Tensor) -> torch.Tensor:
        """Get genre embeddings."""
        return self.genre_embedding(genre_ids)
