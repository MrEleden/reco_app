"""
Movie Recommendation System - Main recommendation logic
"""

import pandas as pd
import numpy as np
import os
import torch
from typing import List, Dict, Tuple
from sklearn.preprocessing import LabelEncoder

from dataset import MovieLensDataLoader
from trainer import ModelTrainer
from models import CollaborativeFilteringModel


class MovieRecommendationSystem:
    """Main recommendation system with multiple recommendation strategies."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.data_loader = MovieLensDataLoader(data_dir)

        # DataFrames for easy access
        self.movies_df = self.data_loader.movies_df
        self.ratings_df = self.data_loader.ratings_df

        # Recommendation data structures
        self.movie_stats = None
        self.user_movie_matrix = None

        # Trained models
        self.trained_models = {}

        self._prepare_recommendations()

    def _prepare_recommendations(self):
        """Prepare recommendation data structures."""
        if self.ratings_df is None:
            return

        # Calculate movie popularity and ratings
        self.movie_stats = self.ratings_df.groupby("movieId").agg({"rating": ["mean", "count"]}).round(2)
        self.movie_stats.columns = ["avg_rating", "rating_count"]
        self.movie_stats = self.movie_stats.reset_index()

        # Create user-item matrix for collaborative filtering
        self.user_movie_matrix = self.ratings_df.pivot_table(
            index="userId", columns="movieId", values="rating", fill_value=0
        )

        print("✅ Recommendation system ready!")

    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10, method: str = "collaborative") -> str:
        """Get personalized recommendations for a user."""
        try:
            if self.ratings_df is None:
                return "❌ No rating data available"

            available_users = self.ratings_df["userId"].unique()
            if user_id not in available_users:
                return f"❌ User {user_id} not found. Try user IDs: {sorted(available_users)[:10]}"

            # Get user's ratings
            user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
            rated_movies = set(user_ratings["movieId"].values)

            if len(user_ratings) == 0:
                return f"❌ User {user_id} hasn't rated any movies"

            if method == "ml" and "collaborative" in self.trained_models:
                return self._get_ml_recommendations(user_id, n_recommendations)
            else:
                return self._get_collaborative_recommendations(user_id, n_recommendations, rated_movies)

        except Exception as e:
            return f"❌ Error generating recommendations: {str(e)}"

    def _get_collaborative_recommendations(self, user_id: int, n_recommendations: int, rated_movies: set) -> str:
        """Get collaborative filtering recommendations."""
        recommendations = []

        # Find similar users
        similar_users = self._find_similar_users(user_id)

        if similar_users:
            # Get movies liked by similar users
            for sim_user_id, similarity in similar_users[:5]:
                sim_user_ratings = self.ratings_df[
                    (self.ratings_df["userId"] == sim_user_id) & (self.ratings_df["rating"] >= 4)
                ]

                for _, rating_row in sim_user_ratings.iterrows():
                    movie_id = rating_row["movieId"]
                    if movie_id not in rated_movies:
                        movie_info = self.movies_df[self.movies_df["movieId"] == movie_id]
                        if not movie_info.empty:
                            score = rating_row["rating"] * similarity
                            recommendations.append(
                                {
                                    "movieId": movie_id,
                                    "title": movie_info.iloc[0]["title"],
                                    "genres": movie_info.iloc[0]["genres"],
                                    "score": score,
                                    "predicted_rating": min(5.0, score),
                                }
                            )

        # Genre-based recommendations if collaborative filtering yields few results
        if len(recommendations) < n_recommendations:
            genre_recs = self._get_genre_recommendations(user_id, rated_movies, n_recommendations)
            recommendations.extend(genre_recs)

        # Sort and remove duplicates
        seen_movies = set()
        unique_recs = []
        for rec in sorted(recommendations, key=lambda x: x["score"], reverse=True):
            if rec["movieId"] not in seen_movies:
                unique_recs.append(rec)
                seen_movies.add(rec["movieId"])

        # Format output
        if not unique_recs:
            return self._get_popular_recommendations(n_recommendations, rated_movies)

        result = f"🎬 **Personalized Recommendations for User {user_id}:**\n\n"

        # Show user's preference summary
        avg_rating = self.ratings_df[self.ratings_df["userId"] == user_id]["rating"].mean()
        fav_genres = self._get_user_favorite_genres(user_id)
        result += f"*Your average rating: {avg_rating:.1f}/5.0*\n"
        result += f"*Your favorite genres: {', '.join(fav_genres[:3])}*\n\n"

        for i, rec in enumerate(unique_recs[:n_recommendations], 1):
            result += f"**{i}. {rec['title']}** (⭐ {rec['predicted_rating']:.1f}/5.0)\n"
            result += f"   *{rec['genres']}*\n\n"

        return result

    def _get_ml_recommendations(self, user_id: int, n_recommendations: int = 10) -> str:
        """Get recommendations using trained ML model."""
        try:
            model_info = self.trained_models.get("collaborative")
            if not model_info:
                return "❌ No trained model available. Train a model first!"

            model = model_info["model"]
            user_encoder = model_info["user_encoder"]
            movie_encoder = model_info["movie_encoder"]

            # Check if user exists in training data
            if user_id not in self.ratings_df["userId"].values:
                return f"❌ User {user_id} not found in training data"

            # Get user's rated movies to exclude
            user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
            rated_movie_ids = set(user_ratings["movieId"].values)

            # Get all movies for prediction
            all_movie_ids = self.movies_df["movieId"].values
            unrated_movies = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

            if not unrated_movies:
                return "❌ User has rated all movies in the database"

            # Make predictions
            try:
                user_encoded = user_encoder.transform([user_id])[0]
            except ValueError:
                return f"❌ User {user_id} not in training data"

            predictions = []
            model.eval()

            with torch.no_grad():
                for movie_id in unrated_movies[:1000]:  # Limit for performance
                    try:
                        movie_encoded = movie_encoder.transform([movie_id])[0]

                        user_tensor = torch.tensor([user_encoded], dtype=torch.long)
                        movie_tensor = torch.tensor([movie_encoded], dtype=torch.long)

                        pred = model(user_tensor, movie_tensor)
                        # Convert to 1-5 rating scale
                        rating_pred = pred.item() * 4 + 1

                        predictions.append((movie_id, rating_pred))

                    except ValueError:
                        continue  # Movie not in training data

            if not predictions:
                return "❌ Could not generate ML predictions"

            # Sort by predicted rating
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = predictions[:n_recommendations]

            result = f"🤖 **AI Model Recommendations for User {user_id}:**\n\n"

            for i, (movie_id, pred_rating) in enumerate(top_predictions, 1):
                movie_info = self.movies_df[self.movies_df["movieId"] == movie_id]
                if not movie_info.empty:
                    title = movie_info.iloc[0]["title"]
                    genres = movie_info.iloc[0]["genres"]
                    result += f"**{i}. {title}** (🤖 Predicted: {pred_rating:.1f}/5.0)\n"
                    result += f"   *{genres}*\n\n"

            return result

        except Exception as e:
            return f"❌ Error generating ML recommendations: {str(e)}"

    def _find_similar_users(self, user_id: int, n_similar: int = 10) -> List:
        """Find users with similar rating patterns."""
        try:
            user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
            user_movies = set(user_ratings["movieId"].values)

            similarities = []

            for other_user in self.ratings_df["userId"].unique():
                if other_user == user_id:
                    continue

                other_ratings = self.ratings_df[self.ratings_df["userId"] == other_user]
                other_movies = set(other_ratings["movieId"].values)

                # Find common movies
                common_movies = user_movies.intersection(other_movies)

                if len(common_movies) >= 2:  # Need at least 2 movies in common
                    # Calculate similarity using common ratings
                    user_common_ratings = user_ratings[user_ratings["movieId"].isin(common_movies)]["rating"].values
                    other_common_ratings = other_ratings[other_ratings["movieId"].isin(common_movies)]["rating"].values

                    if len(user_common_ratings) == len(other_common_ratings):
                        # Cosine similarity
                        similarity = np.dot(user_common_ratings, other_common_ratings) / (
                            np.linalg.norm(user_common_ratings) * np.linalg.norm(other_common_ratings)
                        )
                        similarities.append((other_user, similarity))

            return sorted(similarities, key=lambda x: x[1], reverse=True)[:n_similar]

        except Exception as e:
            print(f"Error finding similar users: {e}")
            return []

    def _get_genre_recommendations(self, user_id: int, rated_movies: set, n_recs: int) -> List:
        """Get recommendations based on user's favorite genres."""
        try:
            favorite_genres = self._get_user_favorite_genres(user_id)

            recommendations = []
            for _, movie in self.movies_df.iterrows():
                if movie["movieId"] in rated_movies:
                    continue

                movie_genres = set(movie["genres"].split("|"))
                genre_match_score = len(set(favorite_genres).intersection(movie_genres))

                if genre_match_score > 0:
                    # Get movie's average rating
                    movie_stats = self.movie_stats[self.movie_stats["movieId"] == movie["movieId"]]
                    avg_rating = movie_stats["avg_rating"].iloc[0] if not movie_stats.empty else 3.0

                    recommendations.append(
                        {
                            "movieId": movie["movieId"],
                            "title": movie["title"],
                            "genres": movie["genres"],
                            "score": genre_match_score * avg_rating,
                            "predicted_rating": avg_rating,
                        }
                    )

            return recommendations

        except Exception as e:
            print(f"Error in genre recommendations: {e}")
            return []

    def _get_user_favorite_genres(self, user_id: int) -> List[str]:
        """Get user's favorite genres based on their ratings."""
        try:
            user_ratings = self.ratings_df[(self.ratings_df["userId"] == user_id) & (self.ratings_df["rating"] >= 4)]

            genre_scores = {}
            for _, rating in user_ratings.iterrows():
                movie_info = self.movies_df[self.movies_df["movieId"] == rating["movieId"]]
                if not movie_info.empty:
                    genres = movie_info.iloc[0]["genres"].split("|")
                    for genre in genres:
                        genre_scores[genre] = genre_scores.get(genre, 0) + rating["rating"]

            return sorted(genre_scores.keys(), key=genre_scores.get, reverse=True)

        except Exception as e:
            return []

    def _get_popular_recommendations(self, n_recs: int, exclude_movies: set = None) -> str:
        """Get popular movie recommendations."""
        exclude_movies = exclude_movies or set()

        try:
            # Get movies with good ratings and enough votes
            popular = self.movie_stats[
                (self.movie_stats["rating_count"] >= 3) & (~self.movie_stats["movieId"].isin(exclude_movies))
            ].nlargest(n_recs, "avg_rating")

            result = "🔥 **Popular Movies You Might Like:**\n\n"

            for i, (_, row) in enumerate(popular.iterrows(), 1):
                movie_info = self.movies_df[self.movies_df["movieId"] == row["movieId"]]
                if not movie_info.empty:
                    title = movie_info.iloc[0]["title"]
                    genres = movie_info.iloc[0]["genres"]
                    result += f"**{i}. {title}** (⭐ {row['avg_rating']:.1f}/5.0)\n"
                    result += f"   *{genres}* - {row['rating_count']} ratings\n\n"

            return result

        except Exception as e:
            return f"❌ Error getting popular recommendations: {str(e)}"

    def search_movies(self, query: str) -> str:
        """Search for movies by title or genre."""
        if self.movies_df is None:
            return "❌ Movie data not available"

        try:
            # Search in titles and genres
            title_matches = self.movies_df[self.movies_df["title"].str.contains(query, case=False, na=False)]
            genre_matches = self.movies_df[self.movies_df["genres"].str.contains(query, case=False, na=False)]

            # Combine results
            all_matches = pd.concat([title_matches, genre_matches]).drop_duplicates()

            if all_matches.empty:
                return f"❌ No movies found for '{query}'"

            result = f"🔍 **Search Results for '{query}':**\n\n"

            for i, (_, movie) in enumerate(all_matches.head(10).iterrows(), 1):
                movie_stats = self.movie_stats[self.movie_stats["movieId"] == movie["movieId"]]

                result += f"**{i}. {movie['title']}**\n"
                result += f"   *{movie['genres']}*\n"

                if not movie_stats.empty:
                    avg_rating = movie_stats["avg_rating"].iloc[0]
                    rating_count = movie_stats["rating_count"].iloc[0]
                    result += f"   ⭐ {avg_rating:.1f}/5.0 ({rating_count} ratings)\n\n"
                else:
                    result += f"   No ratings yet\n\n"

            return result

        except Exception as e:
            return f"❌ Error searching: {str(e)}"

    def get_popular_movies(self, n_movies: int = 15) -> str:
        """Get the most popular movies."""
        return self._get_popular_recommendations(n_movies)

    def get_user_profile(self, user_id: int) -> str:
        """Get user profile and rating history."""
        try:
            if self.ratings_df is None:
                return "❌ No rating data available"

            available_users = self.ratings_df["userId"].unique()
            if user_id not in available_users:
                return f"❌ User {user_id} not found. Try: {sorted(available_users)[:10]}"

            user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]

            result = f"👤 **User {user_id} Profile:**\n\n"
            result += f"📊 **Statistics:**\n"
            result += f"- Movies rated: {len(user_ratings)}\n"
            result += f"- Average rating: {user_ratings['rating'].mean():.1f}/5.0\n"
            result += f"- Highest rating: {user_ratings['rating'].max()}/5.0\n"
            result += f"- Lowest rating: {user_ratings['rating'].min()}/5.0\n\n"

            # Favorite genres
            fav_genres = self._get_user_favorite_genres(user_id)
            result += f"🎭 **Favorite Genres:** {', '.join(fav_genres[:5])}\n\n"

            # Top rated movies
            top_rated = user_ratings.nlargest(5, "rating")
            result += f"⭐ **Top Rated Movies:**\n"

            for i, (_, rating_row) in enumerate(top_rated.iterrows(), 1):
                movie_info = self.movies_df[self.movies_df["movieId"] == rating_row["movieId"]]
                if not movie_info.empty:
                    title = movie_info.iloc[0]["title"]
                    result += f"{i}. **{title}** - {rating_row['rating']}/5.0\n"

            return result

        except Exception as e:
            return f"❌ Error getting user profile: {str(e)}"

    def train_model(
        self, model_type: str = "collaborative", epochs: int = 20, batch_size: int = 256, learning_rate: float = 0.01
    ) -> str:
        """Train a recommendation model."""
        try:
            from trainer import train_collaborative_model

            if model_type == "collaborative":
                result = train_collaborative_model(epochs, batch_size, learning_rate)

                # Load the trained model
                if "✅" in result:  # Training successful
                    self.load_trained_model("collaborative")

                return result
            else:
                return f"❌ Model type '{model_type}' not implemented yet"

        except Exception as e:
            return f"❌ Training error: {str(e)}"

    def load_trained_model(self, model_type: str = "collaborative") -> str:
        """Load a trained model."""
        try:
            if model_type == "collaborative":
                model_path = "models/best_collaborative_model.pt"
                metadata_path = "models/collaborative_metadata.pt"

                if not os.path.exists(model_path):
                    return "❌ No trained collaborative model found. Train a model first!"

                # Load metadata
                metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)

                # Create and load model
                model = CollaborativeFilteringModel(
                    n_users=metadata["n_users"], n_movies=metadata["n_movies"], n_factors=50
                )

                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()

                # Store in trained models
                self.trained_models[model_type] = {
                    "model": model,
                    "user_encoder": metadata["user_encoder"],
                    "movie_encoder": metadata["movie_encoder"],
                    "metadata": metadata,
                }

                return (
                    f"✅ **{model_type.capitalize()} Model Loaded!**\n"
                    f"- Users: {metadata['n_users']}\n"
                    f"- Movies: {metadata['n_movies']}\n"
                    f"- Validation Loss: {checkpoint.get('val_loss', 'unknown'):.4f}"
                )
            else:
                return f"❌ Model type '{model_type}' not supported"

        except Exception as e:
            return f"❌ Error loading model: {str(e)}"

    def get_data_stats(self) -> str:
        """Get dataset statistics."""
        stats = self.data_loader.get_stats()
        if not stats:
            return "❌ No data available"

        result = "📊 **Dataset Statistics:**\n\n"
        result += f"- **Users:** {stats['n_users']:,}\n"
        result += f"- **Movies:** {stats['n_movies']:,}\n"
        result += f"- **Ratings:** {stats['n_ratings']:,}\n"
        result += f"- **Average Rating:** {stats['avg_rating']:.2f}/5.0\n"
        result += f"- **Data Density:** {stats['rating_density']:.4f} ({stats['rating_density']*100:.2f}%)\n"

        return result
