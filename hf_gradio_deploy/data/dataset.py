"""
Dataset classes for movie recommendation system.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
import sys
import os

# Add config to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import RANDOM_SEED, RATING_THRESHOLD


class RecommenderDataset(Dataset):
    """PyTorch Dataset for recommendation systems.

    Args:
        user_movie_pairs (np.ndarray): Array of [user_id, movie_id, rating] triplets
        negative_sampling (bool): Whether to perform negative sampling for implicit feedback
        neg_ratio (int): Ratio of negative samples to positive samples
        n_users (int): Total number of users (for negative sampling)
        n_movies (int): Total number of movies (for negative sampling)
        rating_threshold (float): Threshold for converting ratings to binary labels.
                                  Ratings > threshold become 1 (positive), else 0 (negative)
    """

    def __init__(
        self,
        user_movie_pairs: np.ndarray,
        negative_sampling: bool = True,
        neg_ratio: int = 4,
        n_users: int = None,
        n_movies: int = None,
        rating_threshold: float = RATING_THRESHOLD,
    ):
        self.user_movie_pairs = user_movie_pairs
        self.negative_sampling = negative_sampling
        self.neg_ratio = neg_ratio
        self.n_users = n_users
        self.n_movies = n_movies
        self.rating_threshold = rating_threshold

        if negative_sampling:
            self.data = self._create_negative_samples()
        else:
            # Normalize ratings to [0, 1] range for BCE loss compatibility
            normalized_data = user_movie_pairs.copy()
            normalized_data[:, 2] = (normalized_data[:, 2] - 0.5) / 4.5  # Scale 0.5-5.0 to 0-1
            self.data = normalized_data

    def _create_negative_samples(self) -> np.ndarray:
        """Create negative samples for implicit feedback."""
        positive_pairs = set()
        for user_id, movie_id, _ in self.user_movie_pairs:
            positive_pairs.add((user_id, movie_id))

        negative_samples = []
        np.random.seed(RANDOM_SEED)

        for user_id, movie_id, rating in self.user_movie_pairs:
            # Add positive sample (convert rating > threshold to 1, else 0)
            label = 1 if rating > self.rating_threshold else 0
            negative_samples.append([user_id, movie_id, label])

            # Add negative samples
            if label == 1:  # Only add negative samples for positive interactions
                for _ in range(self.neg_ratio):
                    neg_movie_id = np.random.randint(0, self.n_movies)
                    while (user_id, neg_movie_id) in positive_pairs:
                        neg_movie_id = np.random.randint(0, self.n_movies)
                    negative_samples.append([user_id, neg_movie_id, 0])

        return np.array(negative_samples)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_id, movie_id, label = self.data[idx]
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(movie_id, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


class ContentBasedDataset(Dataset):
    """Dataset for content-based recommendations."""

    def __init__(self, user_genre_pairs: np.ndarray, genre_encoder: LabelEncoder = None):
        self.user_genre_pairs = user_genre_pairs
        self.genre_encoder = genre_encoder or LabelEncoder()

        if genre_encoder is None:
            self._encode_genres()

    def _encode_genres(self):
        """Encode genre strings to integers."""
        all_genres = set()
        for _, genres_str, _ in self.user_genre_pairs:
            genres = genres_str.split("|")
            all_genres.update(genres)

        self.genre_encoder.fit(list(all_genres))

    def __len__(self) -> int:
        return len(self.user_genre_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        user_id, genres_str, rating = self.user_genre_pairs[idx]

        # Encode genres
        genres = genres_str.split("|")
        encoded_genres = self.genre_encoder.transform(genres)

        # Pad or truncate to fixed length
        max_genres = 5
        if len(encoded_genres) > max_genres:
            encoded_genres = encoded_genres[:max_genres]
        else:
            padding = np.zeros(max_genres - len(encoded_genres))
            encoded_genres = np.concatenate([encoded_genres, padding])

        return (torch.tensor(encoded_genres, dtype=torch.long), torch.tensor(rating, dtype=torch.float32))
