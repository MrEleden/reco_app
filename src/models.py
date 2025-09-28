"""
PyTorch Models for Movie Recommendation System
"""

import torch
import torch.nn as nn
import numpy as np


class CollaborativeFilteringModel(nn.Module):
    """Matrix Factorization model for collaborative filtering."""

    def __init__(self, n_users: int, n_movies: int, n_factors: int = 50, dropout: float = 0.2):
        super().__init__()
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factors = n_factors

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


class ContentBasedModel(nn.Module):
    """Content-based recommendation model using movie features."""

    def __init__(self, n_genres: int, n_factors: int = 50, dropout: float = 0.2):
        super().__init__()
        self.n_genres = n_genres
        self.n_factors = n_factors

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


class HybridModel(nn.Module):
    """Hybrid model combining collaborative and content-based approaches."""

    def __init__(self, n_users: int, n_movies: int, n_genres: int, n_factors: int = 50, dropout: float = 0.2):
        super().__init__()

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


class DeepCollaborativeFiltering(nn.Module):
    """Deep neural collaborative filtering model."""

    def __init__(
        self, n_users: int, n_movies: int, n_factors: int = 50, hidden_dims: list = [128, 64], dropout: float = 0.2
    ):
        super().__init__()

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
