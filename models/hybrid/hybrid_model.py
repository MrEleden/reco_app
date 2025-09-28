"""
Hybrid models combining collaborative and content-based approaches.
"""

import torch
import torch.nn as nn
from ..model import BaseModel


class HybridModel(BaseModel):
    """Hybrid model combining collaborative and content-based approaches."""

    def __init__(self, n_users: int, n_movies: int, n_genres: int, n_factors: int = 50, dropout: float = 0.2):
        super().__init__()
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_genres = n_genres
        self.n_factors = n_factors

        # Store config for saving/loading
        self.config = {
            "n_users": n_users,
            "n_movies": n_movies,
            "n_genres": n_genres,
            "n_factors": n_factors,
            "dropout": dropout,
        }

        # Collaborative filtering components
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)

        # Content-based components
        self.genre_embedding = nn.Embedding(n_genres, n_factors)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(n_factors * 3, n_factors * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_factors * 2, n_factors),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_factors, 1),
            nn.Sigmoid(),
        )

        self.global_bias = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.normal_(self.genre_embedding.weight, std=0.01)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.movie_bias.weight, std=0.01)

        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor, genre_features: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Collaborative filtering features
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)

        # Content-based features
        genre_emb = self.genre_embedding(genre_features).mean(dim=1)

        # Combine features
        combined_features = torch.cat([user_emb, movie_emb, genre_emb], dim=1)

        # Get biases
        user_bias = self.user_bias(user_ids).squeeze()
        movie_bias = self.movie_bias(movie_ids).squeeze()

        # Fusion and final prediction
        rating = self.fusion(combined_features).squeeze()
        rating = rating + user_bias + movie_bias + self.global_bias

        return rating

    def predict(self, user_ids: torch.Tensor, movie_ids: torch.Tensor, genre_features: torch.Tensor) -> torch.Tensor:
        """Make predictions (same as forward for this model)."""
        self.eval()
        with torch.no_grad():
            return self.forward(user_ids, movie_ids, genre_features)

    def get_user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user embeddings."""
        return self.user_embedding(user_ids)

    def get_movie_embeddings(self, movie_ids: torch.Tensor) -> torch.Tensor:
        """Get movie embeddings."""
        return self.movie_embedding(movie_ids)

    def get_genre_embeddings(self, genre_ids: torch.Tensor) -> torch.Tensor:
        """Get genre embeddings."""
        return self.genre_embedding(genre_ids)
