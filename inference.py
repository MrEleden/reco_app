#!/usr/bin/env python3
"""
🎯 Global Movie Recommendation Inference API
===========================================

Simple, clean interface for movie recommendations.
Handles all the complexity internally:
- MLflow model loading
- User/Movie ID encoding  
- Data preprocessing
- Recommendation generation
"""

import os
import logging
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.pytorch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Global inference engine for movie recommendations."""
    
    def __init__(self, mlruns_path: str = None):
        """Initialize the recommendation engine."""
        if mlruns_path is None:
            # Auto-detect mlruns path relative to this file
            current_dir = Path(__file__).parent
            mlruns_path = current_dir / "mlruns"
            
        self.mlruns_path = mlruns_path
        self.model = None
        self.user_encoder = None
        self.movie_encoder = None
        self.movies_df = None
        self.ratings_df = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize everything
        self._setup_mlflow()
        self._load_data()
        self._load_encoders()
        self._load_best_model()
        
        logger.info("🚀 RecommendationEngine initialized successfully!")
    
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        # Use relative path for MLflow tracking (works better on Windows)
        tracking_uri = f"file:./{self.mlruns_path}"
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    def _load_data(self):
        """Load movies and ratings data."""
        try:
            # Get data directory relative to this file
            current_dir = Path(__file__).parent
            data_dir = current_dir / "data"
            
            # Load movies - PRIORITIZE REAL MOVIELENS DATA over sample data
            movies_path_raw = data_dir / "raw" / "movies.csv"
            if movies_path_raw.exists():
                self.movies_df = pd.read_csv(movies_path_raw)
                logger.info(f"✅ Loaded {len(self.movies_df)} movies from {movies_path_raw} (Real MovieLens data)")
            else:
                # Fallback to sample data
                movies_path = data_dir / "movies.csv"
                if movies_path.exists():
                    self.movies_df = pd.read_csv(movies_path)
                    logger.info(f"⚠️ Loaded {len(self.movies_df)} movies from {movies_path} (Sample data)")
                else:
                    logger.warning(f"❌ Movies data not found at {movies_path_raw} or {movies_path}")
                    # Create sample data with real movie titles
                    self.movies_df = pd.DataFrame({
                        "movieId": list(range(1, 21)),
                        "title": [
                            "Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)",
                            "Waiting to Exhale (1995)", "Father of the Bride Part II (1995)",
                            "Heat (1995)", "Sabrina (1995)", "Tom and Huck (1995)",
                            "Sudden Death (1995)", "GoldenEye (1995)", "The American President (1995)",
                            "Dracula: Dead and Loving It (1995)", "Balto (1995)", "Nixon (1995)",
                            "Cutthroat Island (1995)", "Casino (1995)", "Sense and Sensibility (1995)",
                            "Four Weddings and a Funeral (1994)", "The Shawshank Redemption (1994)",
                            "The Lion King (1994)"
                        ],
                        "genres": [
                            "Animation|Children's|Comedy", "Adventure|Children's|Fantasy",
                            "Comedy|Romance", "Comedy|Drama", "Comedy", "Action|Crime|Thriller",
                            "Comedy|Romance", "Adventure|Children's", "Action", "Action|Adventure|Thriller",
                            "Comedy|Drama|Romance", "Comedy|Horror", "Animation|Children's",
                            "Drama", "Action|Adventure", "Drama|Thriller", "Drama|Romance",
                            "Comedy|Romance", "Drama", "Animation|Children's|Drama|Musical"
                        ]
                    })
                    logger.info("Created sample movie data")
            
            # Load ratings for encoder training - PRIORITIZE REAL MOVIELENS DATA
            ratings_path = data_dir / "raw" / "ratings.csv"
            if not ratings_path.exists():
                ratings_path = data_dir / "ratings.csv"
            
            if ratings_path.exists():
                self.ratings_df = pd.read_csv(ratings_path)
                logger.info(f"✅ Loaded {len(self.ratings_df)} ratings from {ratings_path}")
            else:
                logger.warning("❌ Ratings data not found")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _load_encoders(self):
        """Create user and movie ID encoders."""
        try:
            if self.ratings_df is not None:
                # Create encoders from the same data used in training
                self.user_encoder = LabelEncoder()
                self.movie_encoder = LabelEncoder()
                
                # Fit encoders on unique IDs
                unique_users = sorted(self.ratings_df['userId'].unique())
                unique_movies = sorted(self.ratings_df['movieId'].unique())
                
                self.user_encoder.fit(unique_users)
                self.movie_encoder.fit(unique_movies)
                
                logger.info(f"✅ Encoders ready: {len(unique_users)} users, {len(unique_movies)} movies")
            else:
                logger.warning("❌ No ratings data available for encoder training")
                
        except Exception as e:
            logger.error(f"Error creating encoders: {e}")
            raise
    
    def _load_best_model(self):
        """Load the best performing model from MLflow."""
        try:
            # Find the best experiment
            experiment = mlflow.get_experiment_by_name("movie_recommendation")
            if not experiment:
                logger.error("❌ No 'movie_recommendation' experiment found")
                return
            
            # Get all finished runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["metrics.val_rmse ASC"]
            )
            
            if runs.empty:
                logger.error("❌ No finished runs found")
                return
            
            # Store available models for potential selection
            self.available_models = []
            for _, run in runs.head(10).iterrows():
                model_info = {
                    'run_id': run['run_id'],
                    'name': run.get('params.model.name', run.get('params.model', f"Model {run['run_id'][:8]}")),
                    'rmse': run.get('metrics.val_rmse', float('inf')),
                    'accuracy': run.get('metrics.val_accuracy', 0)
                }
                self.available_models.append(model_info)
            
            # Load the best model
            best_run = runs.iloc[0]
            run_id = best_run['run_id']
            
            logger.info(f"✅ Loading best model from run: {run_id}")
            logger.info(f"📊 Model RMSE: {best_run.get('metrics.val_rmse', 'N/A')}")
            logger.info(f"📊 Model Accuracy: {best_run.get('metrics.val_accuracy', 'N/A')}")
            
            # Load the model
            model_uri = f"runs:/{run_id}/model"
            self.model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            self.model.eval()
            
            # Store current model info
            self.current_model_info = {
                'run_id': run_id,
                'name': best_run.get('params.model.name', 'Unknown'),
                'rmse': best_run.get('metrics.val_rmse', 0),
                'accuracy': best_run.get('metrics.val_accuracy', 0)
            }
            
            logger.info("✅ Best model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
    
    def get_recommendations(self, user_id: int, num_recommendations: int = 10) -> List[Dict]:
        """Get movie recommendations for a user."""
        try:
            if self.model is None:
                logger.error("❌ No model available for inference")
                return self._get_fallback_recommendations(num_recommendations)
            
            if self.user_encoder is None or self.movie_encoder is None:
                logger.error("❌ Encoders not available")
                return self._get_fallback_recommendations(num_recommendations)
            
            # Validate user ID
            if user_id not in self.user_encoder.classes_:
                available_users = sorted(self.user_encoder.classes_)
                logger.warning(f"⚠️ User {user_id} not in training data. Available: {available_users[0]}-{available_users[-1]}")
                return self._get_fallback_recommendations(num_recommendations)
            
            # Encode user ID
            encoded_user_id = self.user_encoder.transform([user_id])[0]
            
            # Get available movie IDs (encoded)
            available_movie_ids = list(range(len(self.movie_encoder.classes_)))[:1000]  # Limit for performance
            
            # Create tensors
            user_tensor = torch.tensor([encoded_user_id] * len(available_movie_ids), dtype=torch.long, device=self.device)
            movie_tensor = torch.tensor(available_movie_ids, dtype=torch.long, device=self.device)
            
            # Generate predictions
            with torch.no_grad():
                scores = self.model(user_tensor, movie_tensor)
            
            # Get top recommendations
            scores_np = scores.squeeze().cpu().numpy()
            top_indices = np.argsort(scores_np)[-num_recommendations:][::-1]
            
            recommendations = []
            for idx in top_indices:
                encoded_movie_id = available_movie_ids[idx]
                score = float(scores_np[idx])
                
                # Decode movie ID back to original
                original_movie_id = self.movie_encoder.classes_[encoded_movie_id]
                
                # Get movie info from our DataFrame
                movie_info = self.movies_df[self.movies_df["movieId"] == original_movie_id]
                if not movie_info.empty:
                    title = movie_info["title"].iloc[0]
                    genres = movie_info["genres"].iloc[0] if "genres" in movie_info.columns else "Unknown"
                    logger.debug(f"Found movie: {original_movie_id} -> {title}")
                else:
                    # If not found in main DataFrame, create a descriptive name
                    title = f"Movie {original_movie_id}"
                    genres = "Unknown"
                    logger.debug(f"Movie {original_movie_id} not found in DataFrame")
                
                recommendations.append({
                    "movieId": int(original_movie_id),
                    "title": title,
                    "score": score,
                    "genres": genres
                })
            
            logger.info(f"✅ Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            return self._get_fallback_recommendations(num_recommendations)
    
    def _get_fallback_recommendations(self, num_recommendations: int) -> List[Dict]:
        """Return popular movies as fallback recommendations."""
        try:
            if self.movies_df is not None and not self.movies_df.empty:
                # Return first N movies as fallback
                fallback_movies = self.movies_df.head(num_recommendations)
                return [
                    {
                        "movieId": int(row["movieId"]),
                        "title": row["title"],
                        "score": 0.5,  # Neutral score
                        "genres": row.get("genres", "Unknown")
                    }
                    for _, row in fallback_movies.iterrows()
                ]
            else:
                # Generate sample recommendations
                return [
                    {
                        "movieId": i,
                        "title": f"Popular Movie {i}",
                        "score": 0.5,
                        "genres": "Drama"
                    }
                    for i in range(1, num_recommendations + 1)
                ]
        except Exception as e:
            logger.error(f"❌ Error generating fallback recommendations: {e}")
            return []
    
    def get_available_users(self) -> List[int]:
        """Get list of available user IDs."""
        if self.user_encoder is not None:
            return sorted(self.user_encoder.classes_.tolist())
        return list(range(1, 611))  # Default range
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        info = {
            "model_loaded": self.model is not None,
            "encoders_loaded": self.user_encoder is not None and self.movie_encoder is not None,
            "num_users": len(self.user_encoder.classes_) if self.user_encoder else 0,
            "num_movies": len(self.movie_encoder.classes_) if self.movie_encoder else 0,
            "device": str(self.device)
        }
        
        # Add current model info if available
        if hasattr(self, 'current_model_info'):
            info.update(self.current_model_info)
            
        return info
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available models."""
        if hasattr(self, 'available_models'):
            return self.available_models
        return []
    
    def load_model_by_run_id(self, run_id: str) -> bool:
        """Load a specific model by run ID."""
        try:
            model_uri = f"runs:/{run_id}/model"
            self.model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            self.model.eval()
            
            # Update current model info
            if hasattr(self, 'available_models'):
                for model in self.available_models:
                    if model['run_id'] == run_id:
                        self.current_model_info = model
                        break
            
            logger.info(f"✅ Successfully loaded model {run_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model {run_id}: {e}")
            return False


# Global instance - singleton pattern
_recommendation_engine = None

def get_recommendation_engine() -> RecommendationEngine:
    """Get the global recommendation engine instance."""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
    return _recommendation_engine


# Convenience functions for simple usage
def recommend_movies(user_id: int, num_recommendations: int = 10) -> List[Dict]:
    """Simple function to get movie recommendations."""
    engine = get_recommendation_engine()
    return engine.get_recommendations(user_id, num_recommendations)


def get_user_list() -> List[int]:
    """Get list of available user IDs."""
    engine = get_recommendation_engine()
    return engine.get_available_users()


if __name__ == "__main__":
    # Test the inference engine
    print("🎬 Testing Movie Recommendation Engine")
    print("=" * 50)
    
    engine = RecommendationEngine()
    
    # Get model info
    info = engine.get_model_info()
    print(f"Model Info: {info}")
    
    # Test recommendations
    test_user_id = 123
    print(f"\nTesting recommendations for user {test_user_id}:")
    recommendations = engine.get_recommendations(test_user_id, 5)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} (Score: {rec['score']:.3f})")