"""
Training configuration for the movie recommendation system.
"""

# Training hyperparameters
TRAIN_CONFIG = {
    "batch_size": 256,
    "learning_rate": 0.01,
    "weight_decay": 1e-4,
    "epochs": 20,
    "patience": 5,
    "min_delta": 1e-4,
}

# Optimizer settings
OPTIMIZER_CONFIG = {
    "type": "adam",  # "adam", "sgd", "rmsprop"
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "amsgrad": False,
}

# Scheduler settings
SCHEDULER_CONFIG = {
    "type": "reduce_on_plateau",  # "reduce_on_plateau", "cosine", "exponential", None
    "factor": 0.5,
    "patience": 3,
    "min_lr": 1e-6,
    "verbose": True,
}

# Loss function settings
LOSS_CONFIG = {
    "type": "bce",  # "bce", "mse", "bpr", "ranking"
    "margin": 1.0,  # For ranking loss
    "reduction": "mean",
}

# Validation settings
VALIDATION_CONFIG = {
    "val_ratio": 0.2,
    "val_batch_size": 512,
    "val_frequency": 1,  # Validate every N epochs
    "early_stopping": True,
}

# Logging and checkpointing
LOGGING_CONFIG = {
    "log_dir": "logs",
    "log_frequency": 100,  # Log every N batches
    "save_best_model": True,
    "save_frequency": 5,  # Save checkpoint every N epochs
    "model_dir": "results/models",
    "plot_dir": "results/plots",
}

# Evaluation settings
EVAL_CONFIG = {
    "metrics": ["rmse", "mae", "precision@10", "recall@10", "ndcg@10"],
    "top_k": [5, 10, 20],
    "eval_batch_size": 1024,
}
