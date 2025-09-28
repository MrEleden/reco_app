"""
Data configuration for the movie recommendation system.
"""

import os

# Data directories
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "external")

# Raw data files
MOVIES_FILE = os.path.join(RAW_DATA_DIR, "movies.csv")
RATINGS_FILE = os.path.join(RAW_DATA_DIR, "ratings.csv")
LINKS_FILE = os.path.join(RAW_DATA_DIR, "links.csv")
TAGS_FILE = os.path.join(RAW_DATA_DIR, "tags.csv")

# Data processing settings
MIN_RATINGS_PER_USER = 5
MIN_RATINGS_PER_MOVIE = 5
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.1

# Rating conversion settings
RATING_THRESHOLD = 3.0  # Ratings > threshold are considered positive (1), else negative (0)

# Negative sampling settings
NEGATIVE_SAMPLING_RATIO = 4
