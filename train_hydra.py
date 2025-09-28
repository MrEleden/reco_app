"""
Hydra-based training script for movie recommendation system.

Usage examples:
    # Train with default config
    python train_hydra.py

    # Train with specific model
    python train_hydra.py model=deep_cf

    # Train with specific training config
    python train_hydra.py train=production

    # Override specific parameters
    python train_hydra.py batch_size=512 learning_rate=0.001

    # Run multiple experiments (multirun)
    python train_hydra.py -m model=collaborative,deep_cf,hybrid train=default,fast

    # Run hyperparameter sweep
    python train_hydra.py -m learning_rate=0.001,0.01,0.1 batch_size=256,512
"""

import os
import sys
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models import CollaborativeFilteringModel, ContentBasedModel, HybridModel, DeepCollaborativeFiltering
from data import RecommenderDataset, MovieLensDataLoader
from losses import RecommenderLoss
from metrics import RecommenderMetrics
from optimizers import RecommenderOptimizer
from utils import Logger, Timer, setup_seed

# Set up logging
log = logging.getLogger(__name__)


def create_model(cfg: DictConfig, n_users: int, n_movies: int, n_genres: int = 20) -> nn.Module:
    """Create model based on configuration."""
    model_type = cfg.type

    if model_type == "CollaborativeFilteringModel":
        model = CollaborativeFilteringModel(
            n_users=n_users,
            n_movies=n_movies,
            n_factors=cfg.embedding_dim,
            dropout=cfg.dropout,
        )
    elif model_type == "ContentBasedModel":
        # For now, create a simple version that works with user/movie IDs
        # TODO: Set up proper movie-genre mapping
        model = ContentBasedModel(
            n_genres=n_genres,
            n_factors=cfg.get("n_factors", cfg.get("genre_embedding_dim", 50)),
            dropout=cfg.dropout,
            n_movies=n_movies,
        )
        # Create dummy movie-genre mapping for now
        import torch

        dummy_genres = torch.zeros(n_movies, n_genres)
        # Set random genres for testing
        for i in range(n_movies):
            n_movie_genres = torch.randint(1, 4, (1,)).item()  # 1-3 genres per movie
            genre_indices = torch.randperm(n_genres)[:n_movie_genres]
            dummy_genres[i, genre_indices] = 1.0
        model.set_movie_genres(dummy_genres)
    elif model_type == "HybridModel":
        model = HybridModel(
            n_users=n_users,
            n_movies=n_movies,
            n_genres=n_genres,
            n_factors=cfg.get("embedding_dim", 50),
            dropout=cfg.dropout,
        )
    elif model_type == "DeepCollaborativeFiltering":
        model = DeepCollaborativeFiltering(
            n_users=n_users,
            n_movies=n_movies,
            n_factors=cfg.embedding_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def create_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """Create optimizer using the RecommenderOptimizer."""
    optimizer_type = cfg.optimizer.type
    lr = cfg.train.learning_rate
    weight_decay = cfg.train.weight_decay

    # Get optimizer-specific parameters
    optimizer_params = {}
    for key, value in cfg.optimizer.items():
        if key != "type":
            optimizer_params[key] = value

    # Create optimizer using the new pattern
    optimizer_factory = RecommenderOptimizer(optimizer_type)
    return optimizer_factory.create_optimizer(model=model, lr=lr, weight_decay=weight_decay, **optimizer_params)


def create_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    """Create learning rate scheduler based on configuration."""
    scheduler_type = cfg.scheduler.type.lower()

    if scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.scheduler.factor,
            patience=cfg.scheduler.patience,
            min_lr=cfg.scheduler.min_lr,
        )
    elif scheduler_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.scheduler.gamma,
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.scheduler.T_max,
            eta_min=cfg.scheduler.eta_min,
        )
    else:
        scheduler = None

    return scheduler


def train_epoch(model, data_loader, criterion, optimizer, device, metrics):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    metrics.reset()

    for batch_idx, (user_ids, movie_ids, ratings) in enumerate(data_loader):
        user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)

        optimizer.zero_grad()
        predictions = model(user_ids, movie_ids)
        loss = criterion(predictions.squeeze(), ratings)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        metrics.update(predictions.squeeze(), ratings)

    return total_loss / len(data_loader), metrics.compute()


def validate_epoch(model, data_loader, criterion, device, metrics):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    metrics.reset()

    with torch.no_grad():
        for user_ids, movie_ids, ratings in data_loader:
            user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)

            predictions = model(user_ids, movie_ids)
            loss = criterion(predictions.squeeze(), ratings)

            total_loss += loss.item()
            metrics.update(predictions.squeeze(), ratings)

    return total_loss / len(data_loader), metrics.compute()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""

    # Print configuration
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Set up reproducibility
    setup_seed(cfg.seed)

    # Set up device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    # Load data
    log.info("Loading data...")
    data_loader = MovieLensDataLoader()
    user_movie_pairs, config = data_loader.get_collaborative_data()

    if user_movie_pairs is None:
        log.error("No data available")
        return

    log.info(f"Loaded {len(user_movie_pairs)} user-movie interactions")
    log.info(f"Users: {config['n_users']}, Movies: {config['n_movies']}")

    # Create train/val split
    train_pairs, val_pairs = data_loader.create_train_val_split(user_movie_pairs, cfg.train.validation.val_ratio)

    # Create datasets
    train_dataset = RecommenderDataset(
        train_pairs,
        negative_sampling=cfg.data.negative_sampling,
        neg_ratio=cfg.data.negative_sampling_ratio,
        n_users=config["n_users"],
        n_movies=config["n_movies"],
        rating_threshold=cfg.data.rating_threshold,
    )

    val_dataset = RecommenderDataset(
        val_pairs,
        negative_sampling=cfg.data.negative_sampling,
        neg_ratio=cfg.data.negative_sampling_ratio,
        n_users=config["n_users"],
        n_movies=config["n_movies"],
        rating_threshold=cfg.data.rating_threshold,
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.validation.val_batch_size, shuffle=False)

    # Create model
    log.info(f"Creating model: {cfg.model.type}")
    model = create_model(cfg.model, config["n_users"], config["n_movies"])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model created with {total_params:,} parameters")

    # Create loss, optimizer, scheduler
    criterion = RecommenderLoss(loss_type=cfg.train.loss.type)
    optimizer = create_optimizer(model, cfg)  # Pass the full config, not just cfg.train
    scheduler = create_scheduler(optimizer, cfg.train)

    # Create metrics
    train_metrics = RecommenderMetrics()
    val_metrics = RecommenderMetrics()

    # Training setup
    best_val_loss = float("inf")
    patience_counter = 0

    # Training loop
    log.info("Starting training...")
    for epoch in range(cfg.train.epochs):
        # Train
        train_loss, train_results = train_epoch(model, train_loader, criterion, optimizer, device, train_metrics)

        # Validate
        val_loss, val_results = validate_epoch(model, val_loader, criterion, device, val_metrics)

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Log results
        log.info(f"Epoch {epoch+1}/{cfg.train.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        log.info(f"Train Metrics: {train_results}")
        log.info(f"Val Metrics: {val_results}")

        # Save best model
        if val_loss < best_val_loss - cfg.train.min_delta:
            best_val_loss = val_loss
            patience_counter = 0

            # Save model
            model_path = f"best_model_{cfg.model.name}.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "epoch": epoch,
                    "config": OmegaConf.to_container(cfg),
                },
                model_path,
            )
            log.info(f"New best model saved: {model_path}")

        else:
            patience_counter += 1

        # Early stopping
        if cfg.train.validation.early_stopping and patience_counter >= cfg.train.patience:
            log.info(f"Early stopping after {epoch+1} epochs")
            break

    log.info("Training completed!")
    log.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
