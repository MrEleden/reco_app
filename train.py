"""
Main training script for movie recommendation system.
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from config import *
from models import CollaborativeFilteringModel
from data import RecommenderDataset, MovieLensDataLoader
from losses import RecommenderLoss
from metrics import RecommenderMetrics
from utils import Logger, Timer, setup_seed


def train_epoch(model, train_loader, criterion, optimizer, device, metrics):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    metrics.reset()

    for batch_idx, (user_ids, movie_ids, ratings) in enumerate(train_loader):
        user_ids = user_ids.to(device)
        movie_ids = movie_ids.to(device)
        ratings = ratings.to(device)

        optimizer.zero_grad()
        predictions = model(user_ids, movie_ids)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update metrics
        metrics.update(predictions, ratings)

    return total_loss / num_batches, metrics.compute()


def validate_epoch(model, val_loader, criterion, device, metrics):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    num_batches = 0
    metrics.reset()

    with torch.no_grad():
        for user_ids, movie_ids, ratings in val_loader:
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            ratings = ratings.to(device)

            predictions = model(user_ids, movie_ids)
            loss = criterion(predictions, ratings)

            total_loss += loss.item()
            num_batches += 1

            # Update metrics
            metrics.update(predictions, ratings)

    return total_loss / num_batches, metrics.compute()


def main():
    parser = argparse.ArgumentParser(description="Train Movie Recommendation Model")
    parser.add_argument("--model", type=str, default="collaborative", choices=["collaborative"], help="Model type")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"], help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"], help="Batch size")
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"], help="Learning rate")
    parser.add_argument(
        "--embedding-dim", type=int, default=MODEL_CONFIG["collaborative"]["embedding_dim"], help="Embedding dimension"
    )
    parser.add_argument("--dropout", type=float, default=MODEL_CONFIG["collaborative"]["dropout"], help="Dropout rate")
    parser.add_argument(
        "--save-path", type=str, default=os.path.join(RESULTS_DIR, "best_model.pth"), help="Model save path"
    )
    parser.add_argument("--log-file", type=str, default=os.path.join(LOGS_DIR, "train.log"), help="Log file path")

    args = parser.parse_args()

    # Setup
    setup_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Logger
    logger = Logger(args.log_file)
    logger.log_training_start(args.epochs, args.batch_size, args.lr)
    logger.info(f"Using device: {device}")

    # Load data
    data_loader = MovieLensDataLoader()
    user_movie_pairs, config = data_loader.get_collaborative_data()

    if user_movie_pairs is None:
        logger.error("No data available")
        return

    logger.info(f"Loaded {len(user_movie_pairs)} user-movie interactions")
    logger.info(f"Users: {config['n_users']}, Movies: {config['n_movies']}")

    # Create train/val split
    train_pairs, val_pairs = data_loader.create_train_val_split(user_movie_pairs, TRAIN_CONFIG["val_ratio"])

    # Create datasets
    train_dataset = RecommenderDataset(
        train_pairs, negative_sampling=True, n_users=config["n_users"], n_movies=config["n_movies"]
    )

    val_dataset = RecommenderDataset(
        val_pairs, negative_sampling=True, n_users=config["n_users"], n_movies=config["n_movies"]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = CollaborativeFilteringModel(
        n_users=config["n_users"], n_movies=config["n_movies"], n_factors=args.embedding_dim, dropout=args.dropout
    ).to(device)

    logger.log_model_info(
        model,
        {
            "n_users": config["n_users"],
            "n_movies": config["n_movies"],
            "embedding_dim": args.embedding_dim,
            "dropout": args.dropout,
        },
    )

    # Loss, optimizer, and metrics
    criterion = RecommenderLoss(loss_type="bce")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=TRAIN_CONFIG["weight_decay"])

    train_metrics = RecommenderMetrics()
    val_metrics = RecommenderMetrics()

    # Training loop
    timer = Timer()
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        timer.start()

        # Train
        train_loss, train_metric_results = train_epoch(model, train_loader, criterion, optimizer, device, train_metrics)

        # Validate
        val_loss, val_metric_results = validate_epoch(model, val_loader, criterion, device, val_metrics)

        epoch_time = timer.stop()

        # Log epoch results
        logger.log_epoch(epoch, args.epochs, train_loss, val_loss, epoch_time, val_metric_results)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            model.save_model(
                args.save_path,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metric_results,
                val_metrics=val_metric_results,
            )
            logger.info(f"New best model saved with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= TRAIN_CONFIG["patience"]:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    total_training_time = sum([timer.epoch_times[-i] for i in range(1, len(timer.epoch_times) + 1)])
    logger.log_training_end(best_val_loss, total_training_time)


if __name__ == "__main__":
    main()
