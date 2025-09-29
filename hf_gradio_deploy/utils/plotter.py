"""
Plotting utilities for training and evaluation visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
import os


class TrainingPlotter:
    """Plotter for training metrics and losses."""

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")
        self.figsize = figsize

    def plot_losses(
        self,
        train_losses: List[float],
        val_losses: List[float],
        save_path: Optional[str] = None,
        title: str = "Training and Validation Loss",
    ):
        """Plot training and validation losses."""
        plt.figure(figsize=self.figsize)

        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
        plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add minimum validation loss point
        min_val_idx = np.argmin(val_losses)
        plt.plot(
            min_val_idx + 1,
            val_losses[min_val_idx],
            "ro",
            markersize=8,
            label=f"Best Val Loss: {val_losses[min_val_idx]:.4f}",
        )
        plt.legend(fontsize=12)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_metrics(
        self, metrics_dict: Dict[str, List[float]], save_path: Optional[str] = None, title: str = "Training Metrics"
    ):
        """Plot multiple metrics over epochs."""
        n_metrics = len(metrics_dict)
        if n_metrics == 0:
            return

        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(6 * ((n_metrics + 1) // 2), 8))
        if n_metrics == 1:
            axes = [axes]
        elif n_metrics <= 2:
            axes = axes
        else:
            axes = axes.flatten()

        for i, (metric_name, values) in enumerate(metrics_dict.items()):
            ax = axes[i] if n_metrics > 1 else axes[0]
            epochs = range(1, len(values) + 1)

            ax.plot(epochs, values, "g-", linewidth=2)
            ax.set_title(f"{metric_name.title()}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(metric_name.title(), fontsize=12)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(metrics_dict), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_learning_rate(self, learning_rates: List[float], losses: List[float], save_path: Optional[str] = None):
        """Plot learning rate vs loss for LR scheduling analysis."""
        plt.figure(figsize=self.figsize)

        plt.semilogx(learning_rates, losses, "b-", linewidth=2)
        plt.title("Learning Rate vs Loss", fontsize=16, fontweight="bold")
        plt.xlabel("Learning Rate", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


class RecommendationPlotter:
    """Plotter for recommendation system analysis."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize

    def plot_rating_distribution(self, ratings: np.ndarray, save_path: Optional[str] = None):
        """Plot distribution of ratings."""
        plt.figure(figsize=self.figsize)

        plt.hist(ratings, bins=np.arange(0.5, 6, 1), alpha=0.7, color="skyblue", edgecolor="black")
        plt.title("Rating Distribution", fontsize=16, fontweight="bold")
        plt.xlabel("Rating", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks([1, 2, 3, 4, 5])
        plt.grid(True, alpha=0.3, axis="y")

        # Add statistics
        mean_rating = np.mean(ratings)
        std_rating = np.std(ratings)
        plt.axvline(mean_rating, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_rating:.2f}")
        plt.legend()

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_user_activity(self, user_activity: Dict[int, int], top_k: int = 20, save_path: Optional[str] = None):
        """Plot most active users."""
        plt.figure(figsize=self.figsize)

        # Sort users by activity
        sorted_users = sorted(user_activity.items(), key=lambda x: x[1], reverse=True)
        top_users = sorted_users[:top_k]

        users, activities = zip(*top_users)

        plt.bar(range(len(users)), activities, color="lightcoral", alpha=0.7)
        plt.title(f"Top {top_k} Most Active Users", fontsize=16, fontweight="bold")
        plt.xlabel("User Rank", fontsize=12)
        plt.ylabel("Number of Ratings", fontsize=12)
        plt.xticks(range(0, len(users), max(1, len(users) // 10)))
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_movie_popularity(self, movie_popularity: Dict[int, int], top_k: int = 20, save_path: Optional[str] = None):
        """Plot most popular movies."""
        plt.figure(figsize=self.figsize)

        # Sort movies by popularity
        sorted_movies = sorted(movie_popularity.items(), key=lambda x: x[1], reverse=True)
        top_movies = sorted_movies[:top_k]

        movies, popularities = zip(*top_movies)

        plt.bar(range(len(movies)), popularities, color="lightgreen", alpha=0.7)
        plt.title(f"Top {top_k} Most Popular Movies", fontsize=16, fontweight="bold")
        plt.xlabel("Movie Rank", fontsize=12)
        plt.ylabel("Number of Ratings", fontsize=12)
        plt.xticks(range(0, len(movies), max(1, len(movies) // 10)))
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_embedding_similarity(
        self, embeddings: np.ndarray, labels: List[str] = None, save_path: Optional[str] = None, method: str = "tsne"
    ):
        """Plot 2D visualization of embeddings."""
        try:
            if method == "tsne":
                from sklearn.manifold import TSNE

                reducer = TSNE(n_components=2, random_state=42)
            elif method == "pca":
                from sklearn.decomposition import PCA

                reducer = PCA(n_components=2)
            else:
                raise ValueError("Method must be 'tsne' or 'pca'")

            embeddings_2d = reducer.fit_transform(embeddings)

            plt.figure(figsize=self.figsize)
            scatter = plt.scatter(
                embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, c=range(len(embeddings_2d)), cmap="viridis"
            )

            plt.title(f"Embedding Visualization ({method.upper()})", fontsize=16, fontweight="bold")
            plt.xlabel("Component 1", fontsize=12)
            plt.ylabel("Component 2", fontsize=12)
            plt.colorbar(scatter)

            plt.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.show()

        except ImportError as e:
            print(f"Could not create embedding visualization: {e}")
            print("Install sklearn for dimensionality reduction support")
