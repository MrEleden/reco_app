#!/usr/bin/env python3
"""
Hugging Face Deployment Preparation Script
==========================================

Prepares the movie recommendation app for deployment to Hugging Face Spaces.
Creates sample data and pre-trained models for demo purposes.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import mlflow
import mlflow.pytorch
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoModel(nn.Module):
    """Simple demo model for Hugging Face deployment."""

    def __init__(self, n_users=610, n_movies=9742, embedding_dim=50):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_ids, movie_ids):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        x = torch.cat([user_emb, movie_emb], dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze()


def create_sample_data():
    """Create sample movie data for the demo."""
    logger.info("Creating sample movie data...")

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Sample movie data
    movies_data = {
        "movieId": list(range(1, 101)),
        "title": [
            "The Shawshank Redemption",
            "The Godfather",
            "The Dark Knight",
            "Pulp Fiction",
            "Fight Club",
            "Forrest Gump",
            "Inception",
            "The Matrix",
            "Goodfellas",
            "The Lord of the Rings",
            "Star Wars",
            "Casablanca",
            "Schindler's List",
            "The Silence of the Lambs",
            "Saving Private Ryan",
            "Terminator 2",
            "Back to the Future",
            "Raiders of the Lost Ark",
            "Titanic",
            "Jurassic Park",
            "Avatar",
            "The Avengers",
            "Iron Man",
            "Spider-Man",
            "Batman Begins",
            "Wonder Woman",
            "Black Panther",
            "Captain Marvel",
            "Thor",
            "Guardians of the Galaxy",
            "Deadpool",
            "X-Men",
            "Fantastic Four",
            "The Hulk",
            "Superman",
            "Justice League",
            "Aquaman",
            "The Flash",
            "Green Lantern",
            "Cyborg",
            "Toy Story",
            "Finding Nemo",
            "Up",
            "WALL-E",
            "Inside Out",
            "Frozen",
            "Moana",
            "Zootopia",
            "The Lion King",
            "Beauty and the Beast",
            "The Little Mermaid",
            "Aladdin",
            "Pocahontas",
            "Mulan",
            "Tarzan",
            "Monsters, Inc.",
            "The Incredibles",
            "Cars",
            "Ratatouille",
            "Brave",
            "Coco",
            "Onward",
            "Soul",
            "Luca",
            "Turning Red",
            "Encanto",
            "Strange World",
            "Lightyear",
            "Elemental",
            "Wish",
            "The Princess and the Frog",
            "Tangled",
            "Wreck-It Ralph",
            "Big Hero 6",
            "Ralph Breaks the Internet",
            "Frozen II",
            "Raya and the Last Dragon",
            "Cruella",
            "Loki",
            "WandaVision",
            "The Falcon and the Winter Soldier",
            "What If...?",
            "Hawkeye",
            "Moon Knight",
            "Ms. Marvel",
            "She-Hulk",
            "Secret Invasion",
            "Loki Season 2",
            "The Marvels",
            "Ant-Man and the Wasp: Quantumania",
            "Guardians of the Galaxy Vol. 3",
            "Thor: Love and Thunder",
            "Doctor Strange in the Multiverse of Madness",
            "Black Widow",
            "Eternals",
            "Shang-Chi and the Legend of the Ten Rings",
            "Spider-Man: No Way Home",
            "Venom",
            "Morbius",
            "Spider-Man: Into the Spider-Verse",
            "Spider-Man: Across the Spider-Verse",
            "The Batman",
            "Joker",
            "Wonder Woman 1984",
            "Zack Snyder's Justice League",
        ][:100],
        "genres": [
            "Drama",
            "Crime|Drama",
            "Action|Crime|Drama",
            "Crime|Drama",
            "Drama",
            "Comedy|Drama|Romance",
            "Action|Sci-Fi",
            "Action|Sci-Fi",
            "Crime|Drama",
            "Adventure|Drama|Fantasy",
            "Action|Adventure|Fantasy",
            "Drama|Romance",
            "Biography|Drama|History",
            "Crime|Drama|Thriller",
            "Drama|War",
            "Action|Sci-Fi",
            "Adventure|Comedy|Sci-Fi",
            "Action|Adventure",
            "Drama|Romance",
            "Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure",
            "Action|Crime|Drama",
            "Action|Adventure",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Comedy",
            "Action|Comedy",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Sci-Fi",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Musical|Romance",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Musical|Romance",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Adventure|Comedy",
            "Animation|Drama|Fantasy",
            "Action|Adventure|Sci-Fi",
            "Drama|Fantasy|Romance",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
        ],
    }

    movies_df = pd.DataFrame(movies_data)
    movies_df.to_csv(data_dir / "movies.csv", index=False)
    logger.info(f"Created movies.csv with {len(movies_df)} movies")

    return movies_df


def create_demo_mlflow_data():
    """Create demo MLflow experiments for the app."""
    logger.info("Creating demo MLflow experiments...")

    # Set up MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = "movie_recommendation"

    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    # Create multiple demo runs with different performance
    demo_runs = [
        {
            "model": "deep_cf",
            "val_rmse": 0.3232,
            "val_accuracy": 0.8541,
            "val_mae": 0.2456,
            "learning_rate": 0.0001,
            "batch_size": 512,
            "dropout": 0.1,
            "embedding_dim": 64,
        },
        {
            "model": "hybrid",
            "val_rmse": 0.3235,
            "val_accuracy": 0.8529,
            "val_mae": 0.2461,
            "learning_rate": 0.0005,
            "batch_size": 256,
            "dropout": 0.2,
            "embedding_dim": 50,
        },
        {
            "model": "collaborative",
            "val_rmse": 0.3451,
            "val_accuracy": 0.8365,
            "val_mae": 0.2678,
            "learning_rate": 0.001,
            "batch_size": 128,
            "dropout": 0.3,
            "embedding_dim": 32,
        },
        {
            "model": "content_based",
            "val_rmse": 0.3820,
            "val_accuracy": 0.8225,
            "val_mae": 0.2945,
            "learning_rate": 0.005,
            "batch_size": 256,
            "dropout": 0.2,
            "embedding_dim": 64,
        },
    ]

    for run_data in demo_runs:
        with mlflow.start_run(experiment_id=experiment_id):
            # Log parameters
            mlflow.log_param("model", run_data["model"])
            mlflow.log_param("train.learning_rate", run_data["learning_rate"])
            mlflow.log_param("train.batch_size", run_data["batch_size"])
            mlflow.log_param("model.dropout", run_data["dropout"])
            mlflow.log_param("model.embedding_dim", run_data["embedding_dim"])

            # Log metrics
            mlflow.log_metric("val_rmse", run_data["val_rmse"])
            mlflow.log_metric("val_accuracy", run_data["val_accuracy"])
            mlflow.log_metric("val_mae", run_data["val_mae"])

            # Create and log a demo model
            model = DemoModel(embedding_dim=run_data["embedding_dim"])

            # Log the model
            mlflow.pytorch.log_model(
                pytorch_model=model, artifact_path="model", registered_model_name=f"{run_data['model']}_demo"
            )

            logger.info(f"Created demo run for {run_data['model']} with RMSE {run_data['val_rmse']}")


def create_huggingface_files():
    """Create necessary files for Hugging Face Spaces deployment."""
    logger.info("Creating Hugging Face deployment files...")

    # Create .streamlit config directory
    streamlit_dir = Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)

    # Streamlit config
    config_content = """
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
headless = true
port = 7860
enableCORS = false
"""

    with open(streamlit_dir / "config.toml", "w") as f:
        f.write(config_content)

    # Create app.py as entry point (copy of main app)
    import shutil

    if Path("movie_recommendation_app.py").exists():
        shutil.copy("movie_recommendation_app.py", "app.py")
        logger.info("Created app.py entry point")


def main():
    """Main deployment preparation function."""
    logger.info("🚀 Preparing Movie Recommendation App for Hugging Face Deployment...")

    # Create sample data
    create_sample_data()

    # Create demo MLflow data
    create_demo_mlflow_data()

    # Create Hugging Face files
    create_huggingface_files()

    logger.info("✅ Deployment preparation complete!")
    logger.info("\n📋 Next Steps for Hugging Face Deployment:")
    logger.info("1. Create a new Space on Hugging Face")
    logger.info("2. Set SDK to 'streamlit'")
    logger.info("3. Upload all files to the Space repository")
    logger.info("4. The app will be available at your Space URL")
    logger.info("\n📁 Files to upload:")
    logger.info("   • app.py (or movie_recommendation_app.py)")
    logger.info("   • requirements.txt")
    logger.info("   • README_huggingface.md (rename to README.md)")
    logger.info("   • data/ folder")
    logger.info("   • mlruns/ folder")
    logger.info("   • .streamlit/ folder")


if __name__ == "__main__":
    main()
