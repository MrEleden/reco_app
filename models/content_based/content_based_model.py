"""
Content-based filtering models for recommendation system.
"""

import torch
import torch.nn as nn
from ..model import BaseModel


class ContentBasedModel(BaseModel):
    """Content-based recommendation model using movie features."""

    def __init__(self, n_genres: int, n_factors: int = 50, dropout: float = 0.2, n_movies: int = None):
        super().__init__()
        self.n_genres = n_genres
        self.n_factors = n_factors
        self.n_movies = n_movies

        # Store config for saving/loading
        self.config = {"n_genres": n_genres, "n_factors": n_factors, "dropout": dropout, "n_movies": n_movies}

        # Genre embedding
        self.genre_embedding = nn.Embedding(n_genres, n_factors)

        # Movie-to-genre mapping (will be set externally)
        self.register_buffer("movie_genres", torch.zeros(n_movies or 1000, n_genres))

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

    def set_movie_genres(self, movie_genres: torch.Tensor):
        """Set the movie-genre mapping."""
        self.movie_genres = movie_genres

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.genre_embedding.weight, std=0.01)

        for layer in self.user_preference:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self, user_ids: torch.Tensor = None, movie_ids: torch.Tensor = None, genre_features: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass."""
        if genre_features is None and movie_ids is not None:
            # Get genre features from movie IDs
            genre_features = self.movie_genres[movie_ids]

        if genre_features is None:
            raise ValueError("Either genre_features or movie_ids must be provided")

        # Average genre embeddings for multi-hot encoding
        if len(genre_features.shape) == 2:
            # Multi-hot encoding case
            genre_emb = (self.genre_embedding.weight.unsqueeze(0) * genre_features.unsqueeze(-1)).sum(dim=1)
            genre_emb = genre_emb / (genre_features.sum(dim=1, keepdim=True) + 1e-8)
        else:
            # Single genre case
            genre_emb = self.genre_embedding(genre_features).mean(dim=1)

        rating = self.user_preference(genre_emb)
        return rating.squeeze()

    def predict(
        self, user_ids: torch.Tensor = None, movie_ids: torch.Tensor = None, genre_features: torch.Tensor = None
    ) -> torch.Tensor:
        """Make predictions (same as forward for this model)."""
        self.eval()
        with torch.no_grad():
            return self.forward(user_ids, movie_ids, genre_features)

    def get_genre_embeddings(self, genre_ids: torch.Tensor) -> torch.Tensor:
        """Get genre embeddings."""
        return self.genre_embedding(genre_ids)
