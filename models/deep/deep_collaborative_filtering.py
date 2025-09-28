"""
Deep neural collaborative filtering models.
"""

import torch
import torch.nn as nn
from ..model import BaseModel


class DeepCollaborativeFiltering(BaseModel):
    """Deep neural collaborative filtering model."""

    def __init__(
        self, n_users: int, n_movies: int, n_factors: int = 50, hidden_dims: list = [128, 64], dropout: float = 0.2
    ):
        super().__init__()
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factors = n_factors
        self.hidden_dims = hidden_dims

        # Store config for saving/loading
        self.config = {
            "n_users": n_users,
            "n_movies": n_movies,
            "n_factors": n_factors,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
        }

        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)

        # Deep neural network
        layers = []
        input_dim = n_factors * 2

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        self.deep_layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)

        for layer in self.deep_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)

        # Concatenate embeddings
        combined = torch.cat([user_emb, movie_emb], dim=1)

        # Deep neural network
        rating = self.deep_layers(combined).squeeze()

        return rating

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

    def get_combined_features(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        """Get combined user-movie features before deep layers."""
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        return torch.cat([user_emb, movie_emb], dim=1)
