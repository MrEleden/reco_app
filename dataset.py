"""
Dataset classes and data loading utilities for Movie Recommendation System
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Set
import os


class RecommenderDataset(Dataset):
    """PyTorch Dataset for recommendation systems."""

    def __init__(
        self,
        user_movie_pairs: np.ndarray,
        negative_sampling: bool = True,
        neg_ratio: int = 4,
        n_users: int = None,
        n_movies: int = None,
    ):
        self.user_movie_pairs = user_movie_pairs
        self.negative_sampling = negative_sampling
        self.neg_ratio = neg_ratio
        self.n_users = n_users
        self.n_movies = n_movies

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
        np.random.seed(42)

        for user_id, movie_id, rating in self.user_movie_pairs:
            # Add positive sample (convert rating > 3 to 1, else 0)
            label = 1 if rating > 3.0 else 0
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


class MovieLensDataLoader:
    """Data loader for MovieLens dataset."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.movies_df = None
        self.ratings_df = None
        self.links_df = None
        self.tags_df = None

        # Encoders
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.genre_encoder = LabelEncoder()

        self._load_data()
        self._prepare_encoders()

    def _load_data(self):
        """Load MovieLens data files."""
        try:
            movies_path = os.path.join(self.data_dir, "movies.csv")
            ratings_path = os.path.join(self.data_dir, "ratings.csv")

            if os.path.exists(movies_path) and os.path.exists(ratings_path):
                self.movies_df = pd.read_csv(movies_path)
                self.ratings_df = pd.read_csv(ratings_path)
                print(f"✅ Loaded {len(self.movies_df)} movies and {len(self.ratings_df)} ratings")

                # Load optional files
                links_path = os.path.join(self.data_dir, "links.csv")
                tags_path = os.path.join(self.data_dir, "tags.csv")

                if os.path.exists(links_path):
                    self.links_df = pd.read_csv(links_path)
                    print(f"✅ Loaded {len(self.links_df)} movie links")

                if os.path.exists(tags_path):
                    self.tags_df = pd.read_csv(tags_path)
                    print(f"✅ Loaded {len(self.tags_df)} tags")
            else:
                print("❌ MovieLens data files not found")
                self._create_sample_data()

        except Exception as e:
            print(f"Error loading data: {e}")
            self._create_sample_data()

    def _create_sample_data(self):
        """Create sample data if real data isn't available."""
        # Create sample movies
        sample_movies = [
            (1, "Toy Story (1995)", "Animation|Children|Comedy"),
            (2, "Jumanji (1995)", "Adventure|Children|Fantasy"),
            (3, "The Lion King (1994)", "Animation|Children|Drama|Musical"),
            (4, "Forrest Gump (1994)", "Comedy|Drama|Romance"),
            (5, "Pulp Fiction (1994)", "Crime|Drama"),
            (6, "The Matrix (1999)", "Action|Sci-Fi|Thriller"),
            (7, "Titanic (1997)", "Drama|Romance"),
            (8, "Star Wars (1977)", "Action|Adventure|Sci-Fi"),
            (9, "The Godfather (1972)", "Crime|Drama"),
            (10, "Goodfellas (1990)", "Crime|Drama"),
        ]

        self.movies_df = pd.DataFrame(sample_movies, columns=["movieId", "title", "genres"])

        # Create sample ratings
        np.random.seed(42)
        sample_ratings = []
        for user_id in range(1, 11):
            for movie_id in range(1, 11):
                if np.random.random() > 0.3:  # 70% chance of rating
                    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
                    sample_ratings.append((user_id, movie_id, rating, 964982703))

        self.ratings_df = pd.DataFrame(sample_ratings, columns=["userId", "movieId", "rating", "timestamp"])
        print("✅ Created sample data for demonstration")

    def _prepare_encoders(self):
        """Prepare label encoders for users, movies, and genres."""
        if self.ratings_df is None:
            return

        # Encode users and movies
        self.user_encoder.fit(self.ratings_df["userId"].unique())
        self.movie_encoder.fit(self.movies_df["movieId"].unique())

        # Encode genres
        all_genres = set()
        for genres_str in self.movies_df["genres"].dropna():
            genres = genres_str.split("|")
            all_genres.update(genres)

        self.genre_encoder.fit(list(all_genres))

        print(
            f"✅ Prepared encoders: {len(self.user_encoder.classes_)} users, "
            f"{len(self.movie_encoder.classes_)} movies, {len(self.genre_encoder.classes_)} genres"
        )

    def get_collaborative_data(self) -> Tuple[np.ndarray, Dict]:
        """Get data for collaborative filtering."""
        if self.ratings_df is None:
            return None, {}

        # Encode user and movie IDs
        ratings_encoded = self.ratings_df.copy()
        ratings_encoded["user_id"] = self.user_encoder.transform(self.ratings_df["userId"])
        ratings_encoded["movie_id"] = self.movie_encoder.transform(self.ratings_df["movieId"])

        # Create user-movie pairs
        user_movie_pairs = ratings_encoded[["user_id", "movie_id", "rating"]].values

        return user_movie_pairs, {
            "n_users": len(self.user_encoder.classes_),
            "n_movies": len(self.movie_encoder.classes_),
            "user_encoder": self.user_encoder,
            "movie_encoder": self.movie_encoder,
        }

    def get_content_based_data(self) -> Tuple[np.ndarray, Dict]:
        """Get data for content-based filtering."""
        if self.ratings_df is None or self.movies_df is None:
            return None, {}

        # Merge ratings with movie genres
        merged_df = self.ratings_df.merge(self.movies_df, on="movieId")

        # Create user-genre pairs
        user_genre_pairs = merged_df[["userId", "genres", "rating"]].values

        return user_genre_pairs, {"n_genres": len(self.genre_encoder.classes_), "genre_encoder": self.genre_encoder}

    def get_hybrid_data(self) -> Tuple[np.ndarray, Dict]:
        """Get data for hybrid models."""
        collaborative_data, collab_info = self.get_collaborative_data()
        content_data, content_info = self.get_content_based_data()

        if collaborative_data is None or content_data is None:
            return None, {}

        # Merge information
        hybrid_info = {**collab_info, **content_info}

        return collaborative_data, hybrid_info

    def create_train_val_split(self, data: np.ndarray, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into train and validation sets."""
        np.random.seed(42)
        n_samples = len(data)
        n_val = int(n_samples * val_ratio)

        # Shuffle data
        shuffled_indices = np.random.permutation(n_samples)
        val_indices = shuffled_indices[:n_val]
        train_indices = shuffled_indices[n_val:]

        return data[train_indices], data[val_indices]

    def get_movie_info(self, movie_id: int) -> Dict:
        """Get movie information by ID."""
        if self.movies_df is None:
            return {}

        movie_info = self.movies_df[self.movies_df["movieId"] == movie_id]
        if movie_info.empty:
            return {}

        return {"title": movie_info.iloc[0]["title"], "genres": movie_info.iloc[0]["genres"]}

    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        """Get all ratings for a specific user."""
        if self.ratings_df is None:
            return pd.DataFrame()

        return self.ratings_df[self.ratings_df["userId"] == user_id]

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        if self.ratings_df is None:
            return {}

        return {
            "n_users": len(self.ratings_df["userId"].unique()),
            "n_movies": len(self.ratings_df["movieId"].unique()),
            "n_ratings": len(self.ratings_df),
            "avg_rating": self.ratings_df["rating"].mean(),
            "rating_density": len(self.ratings_df)
            / (len(self.ratings_df["userId"].unique()) * len(self.ratings_df["movieId"].unique())),
        }
