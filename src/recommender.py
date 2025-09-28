"""
Movie Recommendation System - Core Logic
A clean, PyTorch-based recommendation system for local testing.
"""

import pandas as pd
import numpy as np
import torch
import os
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder

# Import our existing modules
from dataset import MovieLensDataLoader, RecommenderDataset
from trainer import ModelTrainer


class MovieRecommendationSystem:
    """Main recommendation system for local PyTorch training and testing."""

    def __init__(self, data_dir: str = "../data"):
        """Initialize the recommendation system.

        Args:
            data_dir: Directory containing MovieLens data files
        """
        self.data_dir = data_dir
        self.data_loader = MovieLensDataLoader(data_dir)

        # DataFrames for easy access
        self.movies_df = self.data_loader.movies_df
        self.ratings_df = self.data_loader.ratings_df

        # Recommendation data structures
        self.movie_stats = None
        self.user_movie_matrix = None

        # Trained models cache
        self.trained_models = {}

        self._prepare_recommendations()
        print("✅ Movie Recommendation System initialized for local PyTorch training")

    def _prepare_recommendations(self):
        """Prepare recommendation data structures."""
        if self.ratings_df is None or self.ratings_df.empty:
            print("⚠️ No ratings data available")
            return

        # Calculate movie popularity and ratings
        self.movie_stats = self.ratings_df.groupby("movieId").agg({"rating": ["mean", "count"]}).round(2)
        self.movie_stats.columns = ["avg_rating", "rating_count"]
        self.movie_stats = self.movie_stats.reset_index()

        # Create user-item matrix for collaborative filtering
        self.user_movie_matrix = self.ratings_df.pivot_table(
            index="userId", columns="movieId", values="rating", fill_value=0
        )

        print(f"📊 Prepared stats for {len(self.movie_stats)} movies")

    def get_movie_info(self, movie_id: int) -> Dict:
        """Get detailed movie information."""
        movie_info = self.movies_df[self.movies_df["movieId"] == movie_id]
        if movie_info.empty:
            return {"error": f"Movie {movie_id} not found"}

        movie_row = movie_info.iloc[0]
        stats = self.movie_stats[self.movie_stats["movieId"] == movie_id]

        result = {
            "movieId": int(movie_id),
            "title": movie_row["title"],
            "genres": movie_row["genres"],
            "year": movie_row.get("year", "Unknown"),
        }

        if not stats.empty:
            stat_row = stats.iloc[0]
            result.update({"avg_rating": float(stat_row["avg_rating"]), "rating_count": int(stat_row["rating_count"])})

        return result

    def search_movies(self, query: str, limit: int = 10) -> List[Dict]:
        """Search movies by title."""
        if not query:
            return []

        # Case-insensitive search
        mask = self.movies_df["title"].str.contains(query, case=False, na=False)
        matching_movies = self.movies_df[mask].head(limit)

        results = []
        for _, movie in matching_movies.iterrows():
            movie_info = self.get_movie_info(movie["movieId"])
            if "error" not in movie_info:
                results.append(movie_info)

        return results

    def get_popular_movies(self, n_movies: int = 10, min_ratings: int = 100) -> List[Dict]:
        """Get most popular movies by rating count and average rating."""
        if self.movie_stats is None:
            return []

        # Filter movies with minimum ratings and sort by avg rating
        popular = self.movie_stats[self.movie_stats["rating_count"] >= min_ratings].nlargest(n_movies, "avg_rating")

        results = []
        for _, stats in popular.iterrows():
            movie_info = self.get_movie_info(stats["movieId"])
            if "error" not in movie_info:
                results.append(movie_info)

        return results

    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10, method: str = "collaborative") -> str:
        """Get personalized recommendations for a user.

        Args:
            user_id: User ID for recommendations
            n_recommendations: Number of recommendations to return
            method: Recommendation method ('collaborative' or 'ml')

        Returns:
            Formatted string with recommendations
        """
        try:
            if self.ratings_df is None:
                return "❌ No rating data available"

            # Check if user exists
            available_users = self.ratings_df["userId"].unique()
            if user_id not in available_users:
                sample_users = sorted(available_users)[:10]
                return f"❌ User {user_id} not found. Try user IDs: {sample_users}"

            # Get user's ratings
            user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
            if len(user_ratings) == 0:
                return f"❌ User {user_id} hasn't rated any movies"

            rated_movies = set(user_ratings["movieId"].values)

            if method == "ml" and "collaborative" in self.trained_models:
                return self._get_ml_recommendations(user_id, n_recommendations)
            else:
                return self._get_collaborative_recommendations(user_id, n_recommendations, rated_movies)

        except Exception as e:
            return f"❌ Error generating recommendations: {str(e)}"

    def _get_collaborative_recommendations(self, user_id: int, n_recommendations: int, rated_movies: set) -> str:
        """Get collaborative filtering recommendations without ML."""
        recommendations = []

        # Find similar users based on ratings
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
                        movie_info = self.get_movie_info(movie_id)
                        if "error" not in movie_info:
                            score = rating_row["rating"] * similarity
                            recommendations.append(
                                {
                                    **movie_info,
                                    "score": score,
                                    "reason": f"Users like you rated it {rating_row['rating']:.1f}/5",
                                }
                            )

        # Remove duplicates and sort by score
        seen_movies = set()
        unique_recs = []
        for rec in recommendations:
            if rec["movieId"] not in seen_movies:
                seen_movies.add(rec["movieId"])
                unique_recs.append(rec)

        unique_recs.sort(key=lambda x: x["score"], reverse=True)
        top_recs = unique_recs[:n_recommendations]

        # Format output
        if not top_recs:
            return f"❌ No recommendations found for user {user_id}"

        result = f"🎬 **Recommendations for User {user_id}** (Collaborative Filtering)\n\n"
        for i, rec in enumerate(top_recs, 1):
            result += f"{i}. **{rec['title']}**\n"
            result += f"   📊 Rating: {rec.get('avg_rating', 'N/A')}/5 "
            result += f"({rec.get('rating_count', 0)} reviews)\n"
            result += f"   🎭 Genres: {rec['genres']}\n"
            result += f"   💡 {rec['reason']}\n\n"

        return result

    def _get_ml_recommendations(self, user_id: int, n_recommendations: int) -> str:
        """Get ML-based recommendations using trained model."""
        try:
            model_info = self.trained_models.get("collaborative")
            if not model_info:
                return "❌ No trained model available. Train a model first."

            model = model_info["model"]
            config = model_info["config"]

            # Get user and movie encoders
            user_encoder = config["user_encoder"]
            movie_encoder = config["movie_encoder"]

            # Encode user ID
            if user_id not in user_encoder.classes_:
                return f"❌ User {user_id} not in training data"

            user_encoded = user_encoder.transform([user_id])[0]

            # Get all movies user hasn't rated
            user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
            rated_movies = set(user_ratings["movieId"].values)
            all_movies = set(self.movies_df["movieId"].values)
            unrated_movies = list(all_movies - rated_movies)

            # Predict ratings for unrated movies
            predictions = []
            model.eval()
            with torch.no_grad():
                for movie_id in unrated_movies:
                    if movie_id in movie_encoder.classes_:
                        movie_encoded = movie_encoder.transform([movie_id])[0]
                        user_tensor = torch.LongTensor([user_encoded])
                        movie_tensor = torch.LongTensor([movie_encoded])

                        pred = model(user_tensor, movie_tensor).item()
                        predictions.append((movie_id, pred))

            # Sort by predicted rating
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = predictions[:n_recommendations]

            # Format results
            result = f"🤖 **AI Recommendations for User {user_id}** (Neural Network)\n\n"
            for i, (movie_id, pred_rating) in enumerate(top_predictions, 1):
                movie_info = self.get_movie_info(movie_id)
                if "error" not in movie_info:
                    result += f"{i}. **{movie_info['title']}**\n"
                    result += f"   🧠 AI Predicted Rating: {pred_rating:.2f}/5\n"
                    result += f"   📊 Avg Rating: {movie_info.get('avg_rating', 'N/A')}/5\n"
                    result += f"   🎭 Genres: {movie_info['genres']}\n\n"

            return result

        except Exception as e:
            return f"❌ ML recommendation error: {str(e)}"

    def _find_similar_users(self, user_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Find users similar to the given user based on rating patterns."""
        if self.user_movie_matrix is None:
            return []

        try:
            # Get user's ratings
            if user_id not in self.user_movie_matrix.index:
                return []

            user_ratings = self.user_movie_matrix.loc[user_id]

            # Calculate similarity with other users
            similarities = []
            for other_user in self.user_movie_matrix.index:
                if other_user != user_id:
                    other_ratings = self.user_movie_matrix.loc[other_user]

                    # Find commonly rated movies
                    common_movies = (user_ratings > 0) & (other_ratings > 0)
                    if common_movies.sum() >= 3:  # Need at least 3 common movies
                        # Calculate cosine similarity
                        user_common = user_ratings[common_movies]
                        other_common = other_ratings[common_movies]

                        similarity = np.corrcoef(user_common, other_common)[0, 1]
                        if not np.isnan(similarity):
                            similarities.append((other_user, similarity))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:n_similar]

        except Exception as e:
            print(f"Error finding similar users: {e}")
            return []

    def train_model(
        self,
        model_type: str = "collaborative",
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 0.01,
        embedding_dim: int = 50,
        dropout_rate: float = 0.2,
    ) -> str:
        """Train a recommendation model using PyTorch.

        Args:
            model_type: Type of model to train
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            embedding_dim: Embedding dimension for neural network
            dropout_rate: Dropout rate for regularization

        Returns:
            Training result message
        """
        try:
            print(f"🚀 Starting PyTorch training...")

            # Get training data
            user_movie_pairs, config = self.data_loader.get_collaborative_data()
            if user_movie_pairs is None:
                return "❌ No training data available"

            # Update config with parameters
            config.update({"n_factors": embedding_dim, "dropout": dropout_rate})

            # Create datasets
            train_pairs, val_pairs = self.data_loader.create_train_val_split(user_movie_pairs)

            train_dataset = RecommenderDataset(
                train_pairs, negative_sampling=True, n_users=config["n_users"], n_movies=config["n_movies"]
            )

            val_dataset = RecommenderDataset(
                val_pairs, negative_sampling=True, n_users=config["n_users"], n_movies=config["n_movies"]
            )

            from torch.utils.data import DataLoader

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Train using standard trainer
            results = self.trainer.train_model(
                model_type=model_type,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate,
            )

            if results["success"]:
                # Store trained model for inference
                if os.path.exists(results["model_path"]):
                    model_state = torch.load(results["model_path"], map_location="cpu", weights_only=False)

                    # Initialize model for inference
                    from models import CollaborativeFilteringModel

                    model = CollaborativeFilteringModel(
                        n_users=config["n_users"],
                        n_movies=config["n_movies"],
                        n_factors=embedding_dim,
                        dropout=dropout_rate,
                    )
                    model.load_state_dict(model_state["model_state_dict"])
                    model.eval()

                    self.trained_models[model_type] = {"model": model, "config": config, "results": results}

                # Format success message
                improvement = (
                    (results["metrics"][0]["val_loss"] - results["best_val_loss"])
                    / results["metrics"][0]["val_loss"]
                    * 100
                )

                result_msg = f"✅ **Training Completed Successfully!**\n\n"
                result_msg += f"📊 **Results:**\n"
                result_msg += f"• Best Validation Loss: {results['best_val_loss']:.4f}\n"
                result_msg += f"• Final Train Loss: {results['final_train_loss']:.4f}\n"
                result_msg += f"• Total Epochs: {results['total_epochs']}\n"
                result_msg += f"• Improvement: {improvement:.1f}%\n\n"
                result_msg += f" **Model saved and ready for recommendations!**\n"
                result_msg += f"🎯 **Next:** Try AI recommendations with method='ml'"

                return result_msg
            else:
                return f"❌ Training failed: {results.get('error', 'Unknown error')}"

        except Exception as e:
            return f"❌ Training error: {str(e)}"

    def load_trained_model(self, model_type: str = "collaborative") -> str:
        """Load a previously trained model."""
        try:
            model_path = f"models/best_{model_type}_model.pt"
            if not os.path.exists(model_path):
                return f"❌ No trained {model_type} model found. Train one first."

            # Load model state with weights_only=False to allow sklearn objects
            model_state = torch.load(model_path, map_location="cpu", weights_only=False)

            # Reconstruct model
            from models import CollaborativeFilteringModel

            model = CollaborativeFilteringModel(
                n_users=model_state["n_users"],
                n_movies=model_state["n_movies"],
                n_factors=model_state.get("n_factors", 50),
                dropout=model_state.get("dropout", 0.2),
            )
            model.load_state_dict(model_state["model_state_dict"])
            model.eval()

            # Store for inference
            self.trained_models[model_type] = {
                "model": model,
                "config": {
                    "n_users": model_state["n_users"],
                    "n_movies": model_state["n_movies"],
                    "user_encoder": model_state["user_encoder"],
                    "movie_encoder": model_state["movie_encoder"],
                },
            }

            return f"✅ {model_type.capitalize()} model loaded successfully! Ready for AI recommendations."

        except Exception as e:
            return f"❌ Error loading model: {str(e)}"

    def get_training_stats(self) -> Dict:
        """Get statistics about available data and training."""
        stats = {
            "data": {
                "n_users": len(self.ratings_df["userId"].unique()) if self.ratings_df is not None else 0,
                "n_movies": len(self.movies_df) if self.movies_df is not None else 0,
                "n_ratings": len(self.ratings_df) if self.ratings_df is not None else 0,
            },
            "models": {"trained_models": list(self.trained_models.keys()), "available_types": ["collaborative"]},
        }

        return stats
