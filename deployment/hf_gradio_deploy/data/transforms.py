"""
Data transformation utilities for movie recommendation system.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import sys
import os

# Add config to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import RANDOM_SEED


class NormalizeRatings:
    """Normalize ratings to [0, 1] range."""

    def __init__(self, min_rating: float = 0.5, max_rating: float = 5.0):
        self.min_rating = min_rating
        self.max_rating = max_rating

    def __call__(self, ratings: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        return (ratings - self.min_rating) / (self.max_rating - self.min_rating)

    def inverse_transform(self, normalized_ratings: np.ndarray) -> np.ndarray:
        """Inverse transformation."""
        return normalized_ratings * (self.max_rating - self.min_rating) + self.min_rating


class NegativeSampling:
    """Generate negative samples for implicit feedback."""

    def __init__(self, neg_ratio: int = 4, threshold: float = 3.0, seed: Optional[int] = None):
        self.neg_ratio = neg_ratio
        self.threshold = threshold
        self.seed = seed or RANDOM_SEED

    def __call__(self, user_movie_pairs: np.ndarray, n_users: int, n_movies: int) -> np.ndarray:
        """Generate negative samples."""
        np.random.seed(self.seed)

        # Create set of positive interactions
        positive_pairs = set()
        for user_id, movie_id, rating in user_movie_pairs:
            if rating > self.threshold:
                positive_pairs.add((user_id, movie_id))

        negative_samples = []

        for user_id, movie_id, rating in user_movie_pairs:
            # Add positive sample
            label = 1 if rating > self.threshold else 0
            negative_samples.append([user_id, movie_id, label])

            # Add negative samples for positive interactions
            if label == 1:
                for _ in range(self.neg_ratio):
                    neg_movie_id = np.random.randint(0, n_movies)
                    while (user_id, neg_movie_id) in positive_pairs:
                        neg_movie_id = np.random.randint(0, n_movies)
                    negative_samples.append([user_id, neg_movie_id, 0])

        return np.array(negative_samples)


class ToTensor:
    """Convert numpy arrays to PyTorch tensors."""

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        """Convert to tensor."""
        return torch.tensor(data, dtype=self.dtype)


class GenreEncoder:
    """Encode movie genres to fixed-length vectors."""

    def __init__(self, max_genres: int = 5):
        self.max_genres = max_genres

    def __call__(self, genres_str: str, genre_encoder) -> np.ndarray:
        """Encode genres string to fixed-length array."""
        genres = genres_str.split("|")
        encoded_genres = genre_encoder.transform(genres)

        # Pad or truncate to fixed length
        if len(encoded_genres) > self.max_genres:
            encoded_genres = encoded_genres[: self.max_genres]
        else:
            padding = np.zeros(self.max_genres - len(encoded_genres))
            encoded_genres = np.concatenate([encoded_genres, padding])

        return encoded_genres.astype(np.int64)
