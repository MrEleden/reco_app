"""
Data loading utilities for movie recommendation system.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict
import sys

# Add config to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR, RANDOM_SEED


class MovieLensDataLoader:
    """Data loader for MovieLens dataset."""

    def __init__(self, data_dir: str = DATA_DIR):
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
            movies_path = os.path.join(self.data_dir, "raw", "movies.csv")
            ratings_path = os.path.join(self.data_dir, "raw", "ratings.csv")

            if os.path.exists(movies_path) and os.path.exists(ratings_path):
                self.movies_df = pd.read_csv(movies_path)
                self.ratings_df = pd.read_csv(ratings_path)
                print(f"Loaded {len(self.movies_df)} movies and {len(self.ratings_df)} ratings")

                # Load optional files
                links_path = os.path.join(self.data_dir, "raw", "links.csv")
                tags_path = os.path.join(self.data_dir, "raw", "tags.csv")

                if os.path.exists(links_path):
                    self.links_df = pd.read_csv(links_path)
                    print(f"Loaded {len(self.links_df)} movie links")

                if os.path.exists(tags_path):
                    self.tags_df = pd.read_csv(tags_path)
                    print(f"Loaded {len(self.tags_df)} tags")
            else:
                print("MovieLens data files not found")
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
        np.random.seed(RANDOM_SEED)
        sample_ratings = []
        for user_id in range(1, 11):
            for movie_id in range(1, 11):
                if np.random.random() > 0.3:  # 70% chance of rating
                    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
                    sample_ratings.append((user_id, movie_id, rating, 964982703))

        self.ratings_df = pd.DataFrame(sample_ratings, columns=["userId", "movieId", "rating", "timestamp"])
        print("Created sample data for demonstration")

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
            f"Prepared encoders: {len(self.user_encoder.classes_)} users, "
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

    def create_train_val_split(self, data: np.ndarray, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into train and validation sets."""
        np.random.seed(RANDOM_SEED)
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
