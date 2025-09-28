"""
Collaborative Filtering models for recommendation system.
"""

import torch
import torch.nn as nn
from ..model import BaseModel


class CollaborativeFilteringModel(BaseModel):
    """Matrix Factorization model for collaborative filtering."""

    def __init__(self, n_users: int, n_movies: int, n_factors: int = 50, dropout: float = 0.2):
        super().__init__()
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factors = n_factors

        # Store config for saving/loading
        self.config = {"n_users": n_users, "n_movies": n_movies, "n_factors": n_factors, "dropout": dropout}

        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)

        # Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings with small random values."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.movie_bias.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)

        # Get biases
        user_bias = self.user_bias(user_ids).squeeze()
        movie_bias = self.movie_bias(movie_ids).squeeze()

        # Compute dot product
        dot_product = torch.sum(user_emb * movie_emb, dim=1)

        # Apply dropout
        dot_product = self.dropout(dot_product)

        # Add biases
        rating = dot_product + user_bias + movie_bias + self.global_bias

        return torch.sigmoid(rating)

    def predict(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        """Make predictions (same as forward for this model)."""
        self.eval()
        with torch.no_grad():
            return self.forward(user_ids, movie_ids)

    def get_user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user embeddings."""
        return self.user_embedding(user_ids)

    def get_movie_embeddings(self, movie_ids: torch.Tensor) -> torch.Tensor:
        """Get movie embeddings."""
        return self.movie_embedding(movie_ids)
