#!/usr/bin/env python3
"""
🎬 Movie Recommendation System - Gradio Demo
=============================================

A Gradio interface for the movie recommendation system showcasing:
- Multiple ML models (Deep CF, Hybrid, Collaborative, Content-based)
- Real-time movie recommendations
- Model performance comparison
- Complete tech stack demonstration

Usage:
    python app_gradio.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to Python path for model imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import gradio as gr
import mlflow
import mlflow.pytorch
import torch
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowModelManager:
    """Manages MLflow models and experiments."""

    def __init__(self, tracking_uri: str = None):
        if tracking_uri is None:
            # Auto-detect the best MLflow location by checking for movie_recommendation experiment
            local_mlruns = Path("./mlruns")
            parent_mlruns = Path("../mlruns")

            # Check which location has the movie_recommendation experiment
            self.tracking_uri = None

            # Test parent mlruns first (more likely to have models)
            if parent_mlruns.exists():
                test_uri = f"file:../mlruns"
                mlflow.set_tracking_uri(test_uri)
                try:
                    experiment = mlflow.get_experiment_by_name("movie_recommendation")
                    if experiment:
                        self.tracking_uri = test_uri
                        logger.info("✅ Found movie_recommendation experiment in parent MLflow: ../mlruns")
                except:
                    pass

            # If not found in parent, try local
            if not self.tracking_uri and local_mlruns.exists():
                test_uri = f"file:./mlruns"
                mlflow.set_tracking_uri(test_uri)
                try:
                    experiment = mlflow.get_experiment_by_name("movie_recommendation")
                    if experiment:
                        self.tracking_uri = test_uri
                        logger.info("✅ Found movie_recommendation experiment in local MLflow: ./mlruns")
                except:
                    pass

            # Default fallback to parent
            if not self.tracking_uri:
                self.tracking_uri = f"file:../mlruns"
                logger.warning("❌ movie_recommendation experiment not found, using default: ../mlruns")
        else:
            self.tracking_uri = tracking_uri

        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info(f"🔗 MLflow tracking URI set to: {self.tracking_uri}")

    def get_all_models(self) -> pd.DataFrame:
        """Get all models from MLflow with their metrics - SAME ORDER AS INFERENCE.PY."""
        try:
            experiment = mlflow.get_experiment_by_name("movie_recommendation")
            if not experiment:
                return pd.DataFrame()

            # Use EXACT same logic as inference.py for consistency
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["metrics.val_rmse ASC"],  # Same as inference.py
            )

            return runs  # Return ALL runs, let caller handle filtering
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return pd.DataFrame()

    def get_best_model(self) -> Tuple[str, Dict]:
        """Get the best performing model."""
        runs = self.get_all_models()
        if runs.empty:
            return None, {}

        best_run = runs.iloc[0]
        return best_run.run_id, best_run.to_dict()

    def load_model(self, run_id: str):
        """Load a model from MLflow."""
        try:
            model_uri = f"runs:/{run_id}/model"
            # Force CPU for consistency and to avoid device issues
            model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model {run_id}: {e}")
            return None


class MovieRecommendationDemo:
    """Main demo application class."""

    def __init__(self):
        self.mlflow_manager = MLflowModelManager()
        self.movies_df = self.load_movie_data()
        self.ratings_df = self.load_ratings_data()
        self.model = None
        self.current_run_id = None
        self.models_info = self.get_models_info()

        # Initialize encoders for proper user/movie ID mapping
        self.user_encoder = None
        self.movie_encoder = None
        self._setup_encoders()

        # Auto-load the best model for immediate use
        self._auto_load_best_model()

    def load_movie_data(self) -> pd.DataFrame:
        """Load movie metadata for recommendations - prioritize real MovieLens data."""
        try:
            # PRIORITIZE REAL MOVIELENS DATA over sample data
            # Fix path resolution from apps/ directory
            movies_path_raw = Path("../data/raw/movies.csv")
            if movies_path_raw.exists():
                movies_df = pd.read_csv(movies_path_raw)
                logger.info(f"✅ Loaded {len(movies_df)} movies from {movies_path_raw} (Real MovieLens data)")
                return movies_df
            else:
                # Try from current directory
                movies_path_raw = Path("data/raw/movies.csv")
                if movies_path_raw.exists():
                    movies_df = pd.read_csv(movies_path_raw)
                    logger.info(f"✅ Loaded {len(movies_df)} movies from {movies_path_raw} (Real MovieLens data)")
                    return movies_df
                else:
                    # Fallback options
                    movies_path = (
                        Path("../data/movies.csv") if not Path("data/movies.csv").exists() else Path("data/movies.csv")
                    )
                    if movies_path.exists():
                        movies_df = pd.read_csv(movies_path)
                        logger.info(f"⚠️ Loaded {len(movies_df)} movies from {movies_path} (Sample data)")
                        return movies_df
                    else:
                        # Create sample movie data for demo
                        logger.warning("❌ No movie data found, creating sample data")
                        sample_movies = {
                            "movieId": list(range(1, 101)),
                            "title": [f"Movie {i}" for i in range(1, 101)],
                            "genres": ["Action|Adventure"] * 100,
                        }
                        return pd.DataFrame(sample_movies)
        except Exception as e:
            logger.error(f"Error loading movie data: {e}")
            return pd.DataFrame()

    def load_ratings_data(self) -> pd.DataFrame:
        """Load ratings data for encoder training."""
        try:
            # PRIORITIZE REAL MOVIELENS DATA over sample data
            # Fix path resolution from apps/ directory
            ratings_path_raw = Path("../data/raw/ratings.csv")
            if ratings_path_raw.exists():
                ratings_df = pd.read_csv(ratings_path_raw)
                logger.info(f"✅ Loaded {len(ratings_df)} ratings from {ratings_path_raw} (Real MovieLens data)")
                return ratings_df
            else:
                # Try from current directory
                ratings_path_raw = Path("data/raw/ratings.csv")
                if ratings_path_raw.exists():
                    ratings_df = pd.read_csv(ratings_path_raw)
                    logger.info(f"✅ Loaded {len(ratings_df)} ratings from {ratings_path_raw} (Real MovieLens data)")
                    return ratings_df
                else:
                    # Fallback to sample data
                    ratings_path = (
                        Path("../data/ratings.csv")
                        if not Path("data/ratings.csv").exists()
                        else Path("data/ratings.csv")
                    )
                    if ratings_path.exists():
                        ratings_df = pd.read_csv(ratings_path)
                        logger.info(f"⚠️ Loaded {len(ratings_df)} ratings from {ratings_path} (Sample data)")
                        return ratings_df
                    else:
                        logger.warning("❌ No ratings data found")
                        return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading ratings data: {e}")
            return pd.DataFrame()

    def _setup_encoders(self):
        """Set up user and movie ID encoders."""
        try:
            if not self.ratings_df.empty:
                # Create encoders from the same data used in training
                self.user_encoder = LabelEncoder()
                self.movie_encoder = LabelEncoder()

                # Fit encoders on unique IDs
                unique_users = sorted(self.ratings_df["userId"].unique())
                unique_movies = sorted(self.ratings_df["movieId"].unique())

                self.user_encoder.fit(unique_users)
                self.movie_encoder.fit(unique_movies)

                logger.info(f"✅ Encoders ready: {len(unique_users)} users, {len(unique_movies)} movies")
            else:
                logger.warning("❌ No ratings data available for encoder training")
        except Exception as e:
            logger.error(f"Error setting up encoders: {e}")

    def _auto_load_best_model(self):
        """Auto-load the best model on startup (same as inference.py uses)."""
        try:
            if self.models_info and len(self.models_info) > 0:
                # Load the first model (best performing)
                best_model_info = self.models_info[0]
                run_id = best_model_info[1]

                if run_id and run_id != "N/A":
                    self.model = self.mlflow_manager.load_model(run_id)
                    self.current_run_id = run_id
                    if self.model:
                        logger.info(f"✅ Auto-loaded best model: {best_model_info[0]}")
                    else:
                        logger.warning("❌ Failed to auto-load best model")
                else:
                    logger.warning("❌ No valid run_id for best model")
            else:
                logger.warning("❌ No models available for auto-loading")
        except Exception as e:
            logger.error(f"Error auto-loading best model: {e}")

    def get_models_info(self) -> List[Tuple[str, str, float, float]]:
        """Get information about available models - SAME SELECTION AS INFERENCE.PY."""
        models_df = self.mlflow_manager.get_all_models()

        if models_df.empty:
            return [("No models available", "N/A", 0.0, 0.0)]

        # Use SAME logic as inference.py: take top 10 runs ordered by RMSE
        models_info = []
        for idx, row in models_df.head(10).iterrows():
            # Get model name with fallback logic
            model_name = row.get("params.model.name", row.get("params.model", f"Model {row['run_id'][:8]}"))
            if pd.isna(model_name) or model_name == "None":
                model_name = f"Model {row['run_id'][:8]}"

            rmse = row.get("metrics.val_rmse", 0.0)
            accuracy = row.get("metrics.val_accuracy", 0.0)
            run_id = row.get("run_id", "")

            # Mark the BEST model (same as inference.py uses) for easy identification
            best_marker = " ⭐ BEST" if idx == 0 else ""
            display_name = f"{model_name} (RMSE: {rmse:.4f}, Acc: {accuracy:.1%}){best_marker}"
            models_info.append((display_name, run_id, rmse, accuracy))

        return models_info

    def load_selected_model(self, model_selection: str) -> str:
        """Load the selected model."""
        try:
            # Extract run_id from selection
            if "No models available" in model_selection:
                return "❌ No models available for loading."

            # Find the run_id from the selection
            for display_name, run_id, rmse, accuracy in self.models_info:
                if display_name == model_selection:
                    if run_id and run_id != "N/A":
                        self.model = self.mlflow_manager.load_model(run_id)
                        self.current_run_id = run_id
                        if self.model:
                            model_name = display_name.split(" (")[0]
                            return f"✅ Successfully loaded {model_name} model!"
                        else:
                            return "❌ Failed to load model."

            return "❌ Model not found."
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"

    def get_recommendations(self, user_id: int, num_recommendations: int) -> Tuple[str, str]:
        """Generate movie recommendations with proper user/movie ID encoding."""
        try:
            if self.model is None:
                return "❌ No model loaded. Please select and load a model first.", ""

            if user_id < 1 or user_id > 610:
                return "❌ User ID must be between 1 and 610.", ""

            # Check if encoders are available
            if self.user_encoder is None or self.movie_encoder is None:
                return "❌ User/Movie encoders not available. Please check data loading.", ""

            # CRITICAL: Encode user ID (1-610 → 0-609 for model input)
            try:
                encoded_user_id = self.user_encoder.transform([user_id])[0]
            except ValueError:
                return f"❌ User ID {user_id} not found in training data. Please use a valid user ID.", ""

            # CRITICAL: Use encoded movie indices (not original IDs) for model input
            num_movies = len(self.movie_encoder.classes_)
            available_movie_indices = list(range(min(1000, num_movies)))  # Limit for performance

            # Create tensors with ENCODED indices (force CPU for consistency)
            user_tensor = torch.tensor([encoded_user_id] * len(available_movie_indices), dtype=torch.long, device="cpu")
            movie_tensor = torch.tensor(available_movie_indices, dtype=torch.long, device="cpu")

            # Ensure model is on CPU
            self.model = self.model.cpu()

            # Generate predictions
            self.model.eval()
            with torch.no_grad():
                scores = self.model(user_tensor, movie_tensor)

            # Get top-K recommendations
            scores_np = scores.squeeze().cpu().numpy()
            top_indices = np.argsort(scores_np)[-num_recommendations:][::-1]

            recommendations = []
            for idx in top_indices:
                encoded_movie_id = available_movie_indices[idx]  # This is the encoded index
                score = scores_np[idx]

                # CRITICAL: Decode back to original movie ID
                original_movie_id = self.movie_encoder.classes_[encoded_movie_id]

                # Get movie title
                if not self.movies_df.empty:
                    movie_row = self.movies_df[self.movies_df["movieId"] == original_movie_id]
                    title = movie_row["title"].iloc[0] if not movie_row.empty else f"Movie {original_movie_id}"
                    genre = movie_row["genres"].iloc[0] if not movie_row.empty else "Unknown"
                else:
                    title = f"Movie {original_movie_id}"
                    genre = "Action|Adventure"

                recommendations.append(
                    {"rank": len(recommendations) + 1, "title": title, "genres": genre, "score": f"{score:.4f}"}
                )

            # Format recommendations as a nice table
            rec_text = f"🎬 **Top {num_recommendations} Movie Recommendations for User {user_id}**\n"
            rec_text += f"📊 *User {user_id} encoded as {encoded_user_id} for model input*\n\n"

            for rec in recommendations:
                rec_text += f"**{rec['rank']}.** {rec['title']}\n"
                rec_text += f"   📝 *{rec['genres']}*\n"
                rec_text += f"   ⭐ Score: {rec['score']}\n\n"

            # Create a simple plot
            titles = [rec["title"][:30] + "..." if len(rec["title"]) > 30 else rec["title"] for rec in recommendations]
            scores = [float(rec["score"]) for rec in recommendations]

            try:
                import matplotlib.pyplot as plt
                import matplotlib

                matplotlib.use("Agg")  # Use non-interactive backend

                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(range(len(titles)), scores)
                ax.set_yticks(range(len(titles)))
                ax.set_yticklabels(titles)
                ax.set_xlabel("Recommendation Score")
                ax.set_title(f"Top {num_recommendations} Movies for User {user_id}")
                ax.invert_yaxis()

                # Color bars
                for i, bar in enumerate(bars):
                    bar.set_color(plt.cm.viridis(i / len(bars)))

                plt.tight_layout()

                # Close the figure to free memory
                return rec_text, fig

            except Exception as plot_error:
                logger.warning(f"Could not create plot: {plot_error}")
                return rec_text, None

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return f"❌ Error generating recommendations: {str(e)}", None

    def get_model_comparison(self) -> Tuple[str, any]:
        """Get model performance comparison."""
        try:
            models_df = self.mlflow_manager.get_all_models()

            if models_df.empty:
                return "No models available for comparison.", None

            # Prepare data for comparison
            model_names = []
            rmse_values = []
            accuracy_values = []

            for _, row in models_df.head(8).iterrows():
                model_name = row.get("params.model.name", "Unknown")
                rmse = row.get("metrics.val_rmse")
                accuracy = row.get("metrics.val_accuracy")

                if rmse is not None and accuracy is not None:
                    model_names.append(model_name)
                    rmse_values.append(rmse)
                    accuracy_values.append(accuracy * 100)  # Convert to percentage

            if not model_names:
                return "No valid model metrics found.", None

            # Create comparison plot
            try:
                import matplotlib.pyplot as plt
                import matplotlib

                matplotlib.use("Agg")  # Use non-interactive backend

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # RMSE comparison
                bars1 = ax1.bar(model_names, rmse_values, color="lightcoral")
                ax1.set_title("Model RMSE Comparison (Lower is Better)")
                ax1.set_ylabel("RMSE")
                ax1.tick_params(axis="x", rotation=45)

                # Accuracy comparison
                bars2 = ax2.bar(model_names, accuracy_values, color="lightblue")
                ax2.set_title("Model Accuracy Comparison (Higher is Better)")
                ax2.set_ylabel("Accuracy (%)")
                ax2.tick_params(axis="x", rotation=45)

                plt.tight_layout()

            except Exception as plot_error:
                logger.warning(f"Could not create comparison plot: {plot_error}")
                fig = None

            # Create summary text
            best_rmse_idx = np.argmin(rmse_values)
            best_acc_idx = np.argmax(accuracy_values)

            summary = f"""## 📊 Model Performance Summary

**🏆 Best RMSE:** {model_names[best_rmse_idx]} ({rmse_values[best_rmse_idx]:.4f})
**🎯 Best Accuracy:** {model_names[best_acc_idx]} ({accuracy_values[best_acc_idx]:.1f}%)

**📈 Performance Rankings:**
"""

            # Sort by RMSE for ranking
            sorted_indices = np.argsort(rmse_values)
            for i, idx in enumerate(sorted_indices):
                summary += f"{i+1}. **{model_names[idx]}** - RMSE: {rmse_values[idx]:.4f}, Accuracy: {accuracy_values[idx]:.1f}%\n"

            return summary, fig

        except Exception as e:
            logger.error(f"Error in model comparison: {e}")
            return f"Error creating model comparison: {str(e)}", None


def create_gradio_interface():
    """Create and return the Gradio interface."""

    # Initialize the demo
    demo_app = MovieRecommendationDemo()

    # Get model choices
    model_choices = [info[0] for info in demo_app.models_info]

    with gr.Blocks(title="🎬 Movie Recommendation System", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
        # 🎬 Movie Recommendation System Demo
        
        **Powered by PyTorch + Hydra + MLflow + Optuna**
        
        This demo showcases a complete machine learning pipeline from scratch PyTorch implementation 
        to production-ready recommendation system with experiment tracking and hyperparameter optimization.
        """
        )

        with gr.Tab("🎯 Get Recommendations"):
            with gr.Row():
                with gr.Column():
                    model_dropdown = gr.Dropdown(
                        choices=model_choices,
                        value=model_choices[0] if model_choices else None,  # Auto-select best model
                        label="🤖 Select Model",
                        info="Choose from trained recommendation models (⭐ marks the best performing model)",
                    )
                    load_btn = gr.Button("📥 Load Model", variant="primary")
                    load_status = gr.Textbox(label="Load Status", interactive=False)

                    gr.Markdown("---")

                    user_id_input = gr.Number(
                        value=1, label="👤 User ID", info="Enter user ID (1-610)", minimum=1, maximum=610
                    )
                    num_recs_input = gr.Slider(
                        minimum=5, maximum=20, value=10, step=1, label="📊 Number of Recommendations"
                    )
                    recommend_btn = gr.Button("🎬 Get Recommendations", variant="primary")

                with gr.Column():
                    recommendations_output = gr.Markdown(label="🎬 Recommendations")
                    recommendation_plot = gr.Plot(label="📊 Recommendation Scores")

        with gr.Tab("📊 Model Comparison"):
            compare_btn = gr.Button("📈 Compare All Models", variant="primary")
            with gr.Row():
                comparison_text = gr.Markdown()
                comparison_plot = gr.Plot()

        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
            ## 🚀 Technology Stack
            
            This movie recommendation system demonstrates the complete evolution from basic PyTorch to production ML:
            
            ### 🔥 Core Technologies
            - **PyTorch**: Deep learning framework for model training
            - **Hydra**: Configuration management for reproducible experiments  
            - **MLflow**: Experiment tracking and model registry
            - **Optuna**: Automated hyperparameter optimization
            - **Gradio**: Interactive machine learning demos
            
            ### 🧠 Recommendation Models
            - **Deep CF**: Neural collaborative filtering with embeddings
            - **Hybrid**: Combined collaborative + content-based approach
            - **Collaborative**: Matrix factorization technique
            - **Content-Based**: Genre and metadata driven recommendations
            
            ### 📈 Key Features
            - **Real-time Predictions**: GPU-accelerated inference with CPU fallback
            - **Model Comparison**: Performance metrics across different architectures
            - **Experiment Tracking**: Complete MLflow integration for reproducibility
            - **Smart Optimization**: Optuna-powered hyperparameter tuning
            
            ### 🎯 Project Evolution
            1. **From Scratch** (`train.py`): Pure PyTorch implementation
            2. **Production Scale** (`train_hydra.py`): Hydra + MLflow integration
            3. **Smart Tuning**: Optuna hyperparameter optimization  
            4. **Interactive Demo**: This Gradio application
            
            ---
            *Built with ❤️ using modern ML engineering practices*
            """
            )

        # Event handlers
        load_btn.click(demo_app.load_selected_model, inputs=[model_dropdown], outputs=[load_status])

        recommend_btn.click(
            demo_app.get_recommendations,
            inputs=[user_id_input, num_recs_input],
            outputs=[recommendations_output, recommendation_plot],
        )

        compare_btn.click(demo_app.get_model_comparison, outputs=[comparison_text, comparison_plot])

    return demo


if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    # Find available port starting from 7860
    import socket

    def find_free_port(start_port=7860, max_attempts=50):
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                continue
        return start_port  # Fallback

    port = find_free_port()
    print(f"🚀 Starting Gradio app on port {port}")
    demo.launch(share=False, server_name="0.0.0.0", server_port=port)  # Set share=True for public sharing
