"""
Training utilities for Movie Recommendation System
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

from models import CollaborativeFilteringModel, ContentBasedModel, HybridModel, DeepCollaborativeFiltering
from dataset import RecommenderDataset, MovieLensDataLoader


class ModelTrainer:
    """Training manager for recommendation models."""

    def __init__(self, model_type: str = "collaborative", device: str = None):
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_model_state = None

        print(f"🚀 Trainer initialized for {model_type} model on {self.device}")

    def prepare_model(self, model_config: Dict) -> bool:
        """Initialize model based on configuration."""
        try:
            if self.model_type == "collaborative":
                self.model = CollaborativeFilteringModel(
                    n_users=model_config["n_users"],
                    n_movies=model_config["n_movies"],
                    n_factors=model_config.get("n_factors", 50),
                    dropout=model_config.get("dropout", 0.2),
                )
            elif self.model_type == "content":
                self.model = ContentBasedModel(
                    n_genres=model_config["n_genres"],
                    n_factors=model_config.get("n_factors", 50),
                    dropout=model_config.get("dropout", 0.2),
                )
            elif self.model_type == "hybrid":
                self.model = HybridModel(
                    n_users=model_config["n_users"],
                    n_movies=model_config["n_movies"],
                    n_genres=model_config["n_genres"],
                    n_factors=model_config.get("n_factors", 50),
                    dropout=model_config.get("dropout", 0.2),
                )
            elif self.model_type == "deep":
                self.model = DeepCollaborativeFiltering(
                    n_users=model_config["n_users"],
                    n_movies=model_config["n_movies"],
                    n_factors=model_config.get("n_factors", 50),
                    hidden_dims=model_config.get("hidden_dims", [128, 64]),
                    dropout=model_config.get("dropout", 0.2),
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            self.model = self.model.to(self.device)
            print(f"✅ {self.model_type.capitalize()} model initialized")
            return True

        except Exception as e:
            print(f"❌ Error preparing model: {e}")
            return False

    def prepare_training(self, learning_rate: float = 0.01, weight_decay: float = 1e-5):
        """Prepare optimizer, loss function, and scheduler."""
        if self.model is None:
            raise ValueError("Model must be prepared before training setup")

        # Loss function
        self.criterion = nn.BCELoss()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        print(f"✅ Training setup complete - LR: {learning_rate}, Weight Decay: {weight_decay}")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")

        for batch_idx, batch in enumerate(pbar):
            if self.model_type in ["collaborative", "deep"]:
                user_ids, movie_ids, labels = batch
                user_ids = user_ids.to(self.device)
                movie_ids = movie_ids.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                predictions = self.model(user_ids, movie_ids)

            elif self.model_type == "content":
                genre_features, labels = batch
                genre_features = genre_features.to(self.device)
                labels = labels.to(self.device)

                # Normalize labels to [0,1] for content model
                labels = (labels - 1.0) / 4.0

                # Forward pass
                predictions = self.model(genre_features)

            else:  # hybrid
                user_ids, movie_ids, genre_features, labels = batch
                user_ids = user_ids.to(self.device)
                movie_ids = movie_ids.to(self.device)
                genre_features = genre_features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                predictions = self.model(user_ids, movie_ids, genre_features)

            # Compute loss
            loss = self.criterion(predictions, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        return avg_loss

    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if self.model_type in ["collaborative", "deep"]:
                    user_ids, movie_ids, labels = batch
                    user_ids = user_ids.to(self.device)
                    movie_ids = movie_ids.to(self.device)
                    labels = labels.to(self.device)

                    predictions = self.model(user_ids, movie_ids)

                elif self.model_type == "content":
                    genre_features, labels = batch
                    genre_features = genre_features.to(self.device)
                    labels = labels.to(self.device)

                    # Normalize labels
                    labels = (labels - 1.0) / 4.0

                    predictions = self.model(genre_features)

                else:  # hybrid
                    user_ids, movie_ids, genre_features, labels = batch
                    user_ids = user_ids.to(self.device)
                    movie_ids = movie_ids.to(self.device)
                    genre_features = genre_features.to(self.device)
                    labels = labels.to(self.device)

                    predictions = self.model(user_ids, movie_ids, genre_features)

                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        early_stopping_patience: int = 10,
        save_path: str = "models/best_model.pt",
    ) -> str:
        """Full training loop."""
        if self.model is None or self.optimizer is None:
            return "❌ Model or optimizer not prepared"

        # Create models directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        patience_counter = 0

        # Training log
        result_text = f"🚀 **Training {self.model_type.capitalize()} Model**\n\n"
        result_text += f"- Epochs: {epochs}\n"
        result_text += f"- Device: {self.device}\n"
        result_text += f"- Early Stopping: {early_stopping_patience} epochs\n\n"
        result_text += "**Training Progress:**\n\n"

        print(f"🚀 Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            start_time = time.time()

            # Training phase
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            # Validation phase
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "model_type": self.model_type,
                    },
                    save_path,
                )

                best_marker = " ⭐ (Best!)"
            else:
                patience_counter += 1
                best_marker = ""

            # Epoch time
            epoch_time = time.time() - start_time

            # Log progress
            progress_line = (
                f"Epoch {epoch+1:2d}/{epochs} - "
                f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                f"LR: {current_lr:.2e}, Time: {epoch_time:.1f}s{best_marker}"
            )

            result_text += progress_line + "\n"
            print(progress_line)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                result_text += f"\n🛑 **Early stopping triggered after {epoch+1} epochs**\n"
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break

        # Training summary
        result_text += f"\n✅ **Training Complete!**\n"
        result_text += f"- Best Validation Loss: {self.best_val_loss:.4f}\n"
        result_text += f"- Model saved to: `{save_path}`\n"
        result_text += f"- Total epochs: {len(self.train_losses)}\n"

        return result_text

    def plot_training_history(self, save_path: str = None) -> str:
        """Plot training and validation loss curves."""
        if not self.train_losses or not self.val_losses:
            return "❌ No training history available"

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss", marker="o")
        plt.plot(self.val_losses, label="Validation Loss", marker="s")
        plt.title(f"{self.model_type.capitalize()} Model Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return f"✅ Training plot saved to {save_path}"
        else:
            plt.show()
            return "✅ Training plot displayed"

    def load_model(self, model_path: str, model_config: Dict) -> str:
        """Load a trained model."""
        try:
            if not os.path.exists(model_path):
                return f"❌ Model file not found: {model_path}"

            # Prepare model architecture
            if not self.prepare_model(model_config):
                return "❌ Failed to prepare model architecture"

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            # Load training history if available
            if "train_losses" in checkpoint:
                self.train_losses = checkpoint["train_losses"]
            if "val_losses" in checkpoint:
                self.val_losses = checkpoint["val_losses"]

            return (
                f"✅ **Model Loaded Successfully!**\n"
                f"- Model Type: {checkpoint.get('model_type', 'unknown')}\n"
                f"- Epoch: {checkpoint.get('epoch', 'unknown')}\n"
                f"- Validation Loss: {checkpoint.get('val_loss', 'unknown'):.4f}"
            )

        except Exception as e:
            return f"❌ Error loading model: {str(e)}"

    def get_model_predictions(self, data_loader: DataLoader) -> np.ndarray:
        """Get model predictions for a dataset."""
        if self.model is None:
            return np.array([])

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                if self.model_type in ["collaborative", "deep"]:
                    user_ids, movie_ids, _ = batch
                    user_ids = user_ids.to(self.device)
                    movie_ids = movie_ids.to(self.device)

                    preds = self.model(user_ids, movie_ids)

                elif self.model_type == "content":
                    genre_features, _ = batch
                    genre_features = genre_features.to(self.device)

                    preds = self.model(genre_features)

                else:  # hybrid
                    user_ids, movie_ids, genre_features, _ = batch
                    user_ids = user_ids.to(self.device)
                    movie_ids = movie_ids.to(self.device)
                    genre_features = genre_features.to(self.device)

                    preds = self.model(user_ids, movie_ids, genre_features)

                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)


def train_collaborative_model(epochs: int = 20, batch_size: int = 256, learning_rate: float = 0.01) -> str:
    """Train a collaborative filtering model."""
    try:
        # Load data
        data_loader = MovieLensDataLoader()
        user_movie_pairs, model_config = data_loader.get_collaborative_data()

        if user_movie_pairs is None:
            return "❌ No data available for training"

        # Create datasets
        train_pairs, val_pairs = data_loader.create_train_val_split(user_movie_pairs)

        train_dataset = RecommenderDataset(
            train_pairs, negative_sampling=True, n_users=model_config["n_users"], n_movies=model_config["n_movies"]
        )
        val_dataset = RecommenderDataset(
            val_pairs, negative_sampling=True, n_users=model_config["n_users"], n_movies=model_config["n_movies"]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize trainer
        trainer = ModelTrainer(model_type="collaborative")

        # Prepare model and training
        if not trainer.prepare_model(model_config):
            return "❌ Failed to prepare model"

        trainer.prepare_training(learning_rate=learning_rate)

        # Train model
        result = trainer.train(train_loader, val_loader, epochs=epochs, save_path="models/best_collaborative_model.pt")

        # Save additional metadata
        metadata = {**model_config, "data_loader": data_loader}
        torch.save(metadata, "models/collaborative_metadata.pt")

        return result

    except Exception as e:
        return f"❌ **Training Error:** {str(e)}"
