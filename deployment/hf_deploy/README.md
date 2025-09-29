---
title: Movie Recommendation System Demo
emoji: 🎬
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🎬 Movie Recommendation System Demo

A production-ready PyTorch movie recommendation system showcasing the complete ML pipeline from scratch implementation to MLflow tracking and Optuna optimization.

## 🚀 Features

- **Multiple Models**: Matrix Factorization, Neural CF, DeepFM, Hybrid
- **Smart Optimization**: Automated hyperparameter tuning with Optuna  
- **Experiment Tracking**: MLflow integration for model management
- **Interactive Demo**: Real-time movie recommendations
- **Tech Stack**: PyTorch + Hydra + MLflow + Optuna + Streamlit

## 🎯 How to Use

1. **Select a Model**: Choose from trained models in the sidebar
2. **Enter User ID**: Input a user ID (1-610) 
3. **Get Recommendations**: Click to receive personalized movie suggestions
4. **Explore Models**: Compare different model performances

## 🏗️ Architecture

This demo demonstrates the evolution from basic PyTorch to production ML:

- **From Scratch**: Pure PyTorch implementation (`train.py`)
- **Production Scale**: Hydra + MLflow integration (`train_hydra.py`)  
- **Smart Tuning**: Optuna hyperparameter optimization
- **Interactive Demo**: Streamlit web application

## 📊 Model Performance

The system includes multiple recommendation approaches:

- **Deep CF**: Neural collaborative filtering (Best: RMSE 0.3232)
- **Hybrid**: Combined collaborative + content-based  
- **Collaborative**: Matrix factorization approach
- **Content-Based**: Genre and metadata driven

## 🔬 Technology Stack

Built with modern ML/AI technologies:

- **PyTorch**: Deep learning framework
- **Hydra**: Configuration management  
- **MLflow**: Experiment tracking
- **Optuna**: Hyperparameter optimization
- **Streamlit**: Interactive web interface

---

*Powered by PyTorch + Hydra + MLflow + Optuna*