"""
Configuration file for the movie recommendation system.
"""

import os

# Import all configurations
from .data_config import *
from .model_config import *
from .train_config import *

# Global settings
RANDOM_SEED = 42
DEVICE = "auto"  # "auto", "cpu", "cuda"

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

# Legacy compatibility (for existing imports)
# Data configuration
DATA_DIR = "data"
MOVIES_FILE = os.path.join(DATA_DIR, "raw", "movies.csv")
RATINGS_FILE = os.path.join(DATA_DIR, "raw", "ratings.csv")
LINKS_FILE = os.path.join(DATA_DIR, "raw", "links.csv")
TAGS_FILE = os.path.join(DATA_DIR, "raw", "tags.csv")
