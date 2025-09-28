"""
Simple training script that demonstrates the new structure.
This will be the foundation - you can expand it later.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Import the new data and model structures
from data import MovieLensDataLoader, RecommenderDataset
from models import CollaborativeFilteringModel
from config import *


def main():
    parser = argparse.ArgumentParser(description="Train Movie Recommendation Model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--embedding-dim", type=int, default=50, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    args = parser.parse_args()

    print("Movie Recommendation System - New Structure")
    print(f"Training with: epochs={args.epochs}, batch_size={args.batch_size}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data using existing loader
    data_loader = MovieLensDataLoader(data_dir="data")
    user_movie_pairs, config = data_loader.get_collaborative_data()

    if user_movie_pairs is None:
        print("ERROR: No data available")
        return

    print(f"Loaded data: {len(user_movie_pairs)} interactions")
    print(f"Users: {config['n_users']}, Movies: {config['n_movies']}")

    # Create train/val split
    train_pairs, val_pairs = data_loader.create_train_val_split(user_movie_pairs, 0.2)

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

    # Create model using NEW structure
    model = CollaborativeFilteringModel(
        n_users=config["n_users"], n_movies=config["n_movies"], n_factors=args.embedding_dim, dropout=args.dropout
    ).to(device)

    print(f"Model created with {model.get_num_parameters()} parameters")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Simple training loop
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        for user_ids, movie_ids, ratings in train_loader:
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()
            predictions = model(user_ids, movie_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for user_ids, movie_ids, ratings in val_loader:
                user_ids = user_ids.to(device)
                movie_ids = movie_ids.to(device)
                ratings = ratings.to(device)

                predictions = model(user_ids, movie_ids)
                loss = criterion(predictions, ratings)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("results", exist_ok=True)
            model.save_model("results/best_model.pth", epoch=epoch, train_loss=train_loss, val_loss=val_loss)
            print(f"New best model saved! Val loss: {val_loss:.4f}")

    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
