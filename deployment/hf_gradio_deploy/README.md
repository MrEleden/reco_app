---
title: Movie Recommendation System
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app_gradio.py
pinned: false
license: mit
---

# 🎬 Movie Recommendation System

A complete PyTorch-based movie recommendation system showcasing modern ML engineering practices from scratch implementation to production deployment.

## 🚀 Features

### 🤖 Multiple ML Models
- **Deep CF**: Neural collaborative filtering with embeddings
- **Hybrid**: Combined collaborative + content-based approach  
- **Collaborative**: Matrix factorization technique
- **Content-Based**: Genre and metadata driven recommendations

### 🔧 Production Pipeline
- **PyTorch**: Deep learning framework
- **Hydra**: Configuration management for experiments
- **MLflow**: Experiment tracking and model registry
- **Optuna**: Automated hyperparameter optimization

### 🎯 Interactive Demo
- **Model Selection**: Choose from trained recommendation models
- **Real-time Predictions**: Get personalized movie recommendations
- **Performance Comparison**: Compare different model architectures
- **Visualization**: See recommendation scores and model metrics

## 🎮 How to Use

### 1. Get Recommendations
1. **Select a Model**: Choose from available trained models
2. **Load Model**: Click "Load Model" to initialize
3. **Enter User ID**: Input a user ID (1-610)
4. **Get Recommendations**: Receive personalized movie suggestions

### 2. Compare Models
- View performance metrics across all trained models
- See RMSE and accuracy comparisons
- Understand which models perform best

## 📊 Model Performance

The system includes multiple recommendation approaches:

- **🏆 Deep CF**: Best overall performance (RMSE ~0.323)
- **🔥 Hybrid**: Balanced collaborative + content approach
- **⚡ Collaborative**: Fast matrix factorization
- **🎯 Content-Based**: Genre and metadata driven

## 🏗️ Architecture Evolution

This demo demonstrates the complete ML project lifecycle:

1. **From Scratch** (`train.py`): Pure PyTorch implementation
2. **Production Scale** (`train_hydra.py`): Hydra + MLflow integration  
3. **Smart Optimization**: Optuna hyperparameter tuning
4. **Interactive Demo**: This Gradio application

## 🔬 Technology Showcase

**Modern ML Stack Integration:**
- Configuration-driven experiments with Hydra
- Automatic experiment tracking via MLflow
- Intelligent hyperparameter search with Optuna
- Production-ready model serving
- Interactive visualization and comparison

## 📈 Key Metrics

- **610 Users** in the dataset
- **9,742 Movies** with rich metadata
- **100K+ Ratings** for training
- **4 Different Architectures** compared
- **GPU/CPU Compatible** deployment

---

*Built with modern ML engineering practices showcasing the evolution from basic PyTorch to production-ready recommendation systems.*