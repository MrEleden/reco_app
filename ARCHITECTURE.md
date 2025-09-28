"""
File Structure and Component Overview
====================================

This document explains the purpose and functionality of each file in the modular 
movie recommendation system.

CORE COMPONENTS
===============

models.py
---------
Purpose: PyTorch neural network architectures
Contains:
- CollaborativeFilteringModel: Matrix factorization with user/movie embeddings
- ContentBasedModel: Genre-based neural network  
- HybridModel: Combines collaborative + content-based approaches
- DeepCollaborativeFiltering: Deep neural network for complex patterns

Key Features:
✅ Embedding layers for users and movies
✅ Bias terms for better predictions
✅ Dropout for regularization
✅ Xavier weight initialization
✅ Sigmoid activation for rating prediction

dataset.py
----------
Purpose: Data loading, preprocessing, and PyTorch dataset creation
Contains:
- MovieLensDataLoader: Comprehensive data management
- RecommenderDataset: PyTorch dataset with negative sampling
- ContentBasedDataset: Dataset for genre-based models

Key Features:
✅ MovieLens CSV file loading
✅ Label encoding for users/movies/genres
✅ Train/validation splitting
✅ Negative sampling for implicit feedback
✅ Sample data creation if real data unavailable

trainer.py
----------
Purpose: Model training pipeline and evaluation
Contains:
- ModelTrainer: Unified training interface for all models
- Training loop with progress tracking
- Model checkpointing and early stopping
- Performance plotting and visualization

Key Features:
✅ Automated training loops
✅ Early stopping with patience
✅ Learning rate scheduling
✅ Gradient clipping
✅ Model checkpointing
✅ Training history visualization
✅ Multi-model support

recommender.py
--------------
Purpose: Main recommendation system logic and algorithms
Contains:
- MovieRecommendationSystem: Core recommendation engine
- User similarity computation
- Multiple recommendation strategies
- Model loading and inference

Key Features:
✅ Collaborative filtering recommendations
✅ Genre-based recommendations
✅ Popular movie suggestions
✅ User profile analysis
✅ Movie search functionality
✅ Trained model integration
✅ Statistical analysis

app_modular.py
--------------
Purpose: Gradio web interface
Contains:
- Web UI with multiple tabs
- Interactive training interface
- Real-time model training progress
- User-friendly recommendation interface

Key Features:
✅ Modern, responsive design
✅ Real-time training feedback
✅ Multiple recommendation methods
✅ Interactive parameter tuning
✅ Dataset statistics display
✅ Error handling and user feedback

UTILITY FILES
=============

requirements.txt
----------------
Purpose: Python package dependencies
Contains: torch, gradio, pandas, numpy, scikit-learn, matplotlib, tqdm

test_modular.py
---------------
Purpose: Test script to verify all components work correctly
Tests: imports, data loading, recommendations, search, profiles, stats

README_modular.md
-----------------
Purpose: Comprehensive documentation
Contains: architecture overview, usage instructions, API reference, deployment guide

FILE RELATIONSHIPS
==================

Data Flow:
1. dataset.py loads and preprocesses MovieLens data
2. models.py defines neural network architectures
3. trainer.py trains models using data and model definitions
4. recommender.py uses trained models and data for recommendations
5. app_modular.py provides web interface for all functionality

Import Dependencies:
- app_modular.py → recommender.py
- recommender.py → dataset.py, trainer.py, models.py
- trainer.py → models.py, dataset.py
- dataset.py (standalone)
- models.py (standalone)

DEPLOYMENT OPTIONS
==================

Option 1: Modular Development
- Use all separate files for development and debugging
- Better code organization and maintainability
- Easier to add new models or features

Option 2: Single File Deployment (Original app.py)
- All functionality in one file
- Simpler deployment to platforms like Hugging Face
- No import dependencies

Option 3: Hybrid Approach
- Keep modular structure for development
- Create deployment script that bundles everything
- Best of both worlds

BENEFITS OF MODULAR STRUCTURE
==============================

Development Benefits:
✅ Clear separation of concerns
✅ Easier debugging and testing
✅ Better code reusability
✅ Simplified maintenance
✅ Team collaboration friendly
✅ Extensible architecture

Performance Benefits:
✅ Selective imports (faster startup)
✅ Memory efficiency
✅ Better error isolation
✅ Optimized for specific tasks

Deployment Benefits:
✅ Flexible deployment options
✅ Easy to containerize
✅ Version control friendly
✅ Environment-specific configurations

USAGE RECOMMENDATIONS
=====================

For Development:
- Use the modular structure (app_modular.py + components)
- Run test_modular.py to verify everything works
- Develop new features in separate modules

For Deployment:
- HuggingFace Spaces: Upload all modular files
- Docker: Use modular structure with proper requirements
- Local Server: Run app_modular.py directly

For Learning:
- Start with individual components (models.py, dataset.py)
- Understand data flow through trainer.py
- Explore recommendation algorithms in recommender.py
- Experiment with UI in app_modular.py

NEXT STEPS
==========

Potential Enhancements:
🔮 Add more model architectures (autoencoders, variational autoencoders)
🔮 Implement A/B testing framework
🔮 Add more evaluation metrics (NDCG, MAP, MRR)
🔮 Create REST API endpoints
🔮 Add user authentication and personalization
🔮 Implement real-time model updates
🔮 Add explainable AI features
🔮 Support for additional datasets beyond MovieLens
"""