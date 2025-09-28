"""
Model configuration for the movie recommendation system.
"""

# Base model configuration
BASE_MODEL_CONFIG = {
    "device": "auto",  # "auto", "cpu", "cuda"
    "seed": 42,
}

# Collaborative Filtering Model
COLLABORATIVE_CONFIG = {
    "embedding_dim": 50,
    "dropout": 0.2,
    "bias": True,
    "init_mean": 0.0,
    "init_std": 0.1,
}

# Content-Based Model
CONTENT_BASED_CONFIG = {
    "genre_embedding_dim": 32,
    "user_embedding_dim": 64,
    "hidden_dims": [128, 64],
    "dropout": 0.3,
    "activation": "relu",
}

# Hybrid Model
HYBRID_CONFIG = {
    "collaborative_weight": 0.7,
    "content_weight": 0.3,
    "fusion_dims": [256, 128],
    "dropout": 0.2,
}

# Deep Collaborative Filtering
DEEP_CF_CONFIG = {
    "embedding_dim": 64,
    "hidden_dims": [256, 128, 64],
    "dropout": 0.4,
    "batch_norm": True,
    "activation": "relu",
}

# Model registry
MODEL_CONFIG = {
    "collaborative": COLLABORATIVE_CONFIG,
    "content_based": CONTENT_BASED_CONFIG,
    "hybrid": HYBRID_CONFIG,
    "deep_cf": DEEP_CF_CONFIG,
}
