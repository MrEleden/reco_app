"""
Configuration file for the movie recommendation system.
"""

import os

# Data configuration
DATA_DIR = "data"
MOVIES_FILE = os.path.join(DATA_DIR, "movies.csv")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")
LINKS_FILE = os.path.join(DATA_DIR, "links.csv")
TAGS_FILE = os.path.join(DATA_DIR, "tags.csv")

# Model configuration
MODEL_CONFIG = {
    "collaborative": {
        "embedding_dim": 50,
        "dropout": 0.2,
        "hidden_dims": [128, 64],
    }
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 256,
    "learning_rate": 0.01,
    "weight_decay": 1e-4,
    "epochs": 20,
    "val_ratio": 0.2,
    "patience": 5,
}

# Paths
RESULTS_DIR = "results"
LOGS_DIR = "logs"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# Device
DEVICE = "cuda" if os.path.exists("/usr/local/cuda") else "cpu"

# Random seed for reproducibility
RANDOM_SEED = 42
