---
title: Movie Recommendation System Demo
emoji: 🎬
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.28.0
app_file: movie_recommendation_app.py
pinned: false
license: mit
---

# 🎬 Movie Recommendation System Demo

**Live Demo showcasing PyTorch + Hydra + MLflow + Optuna Integration**

## 🚀 What This Demo Shows

This interactive Streamlit app demonstrates a **complete modern ML pipeline** with:

### 🔥 **PyTorch Models**
- Deep Collaborative Filtering
- Hybrid Recommendation Models  
- Content-Based Filtering
- Real-time inference and predictions

### 🔬 **MLflow Integration**
- **Model Selection**: Choose from trained models in real-time
- **Performance Comparison**: Interactive charts comparing model metrics
- **Model Registry**: Load any experiment directly from MLflow
- **Experiment Tracking**: View hyperparameters and results

### 🎯 **Intelligent Model Selection**
- Automatic best model identification
- Performance-based model ranking
- One-click model switching
- Real-time model loading from MLflow registry

### ⚙️ **Configuration Management**
- Models trained with Hydra configuration management
- Reproducible experiments and hyperparameters
- Professional ML experiment organization

## 🎮 **How to Use**

1. **Select a Model**: Use the sidebar to choose from available trained models
2. **Enter User ID**: Pick a user (1-610) to get recommendations for
3. **Get Recommendations**: Click the button to generate personalized movie suggestions
4. **Compare Models**: View performance metrics and model comparisons
5. **Explore Code**: See the exact code to reproduce and use each model

## 📊 **Features**

### 🏆 **Model Management**
- Real-time model loading from MLflow
- Performance metrics comparison (RMSE, Accuracy, MAE)
- Hyperparameter visualization
- Model selection based on performance

### 🎬 **Recommendation Engine**
- Generate 5-20 personalized movie recommendations
- Real-time inference using PyTorch models
- Star-based rating predictions
- Interactive user interface

### 📈 **Visualization**
- Model performance comparison charts
- Interactive Plotly visualizations
- Real-time metrics updates
- Professional ML dashboard

## 🛠️ **Tech Stack**

| Technology | Purpose | Integration |
|------------|---------|-------------|
| **🔥 PyTorch** | Deep Learning Models | Model training and inference |
| **⚙️ Hydra** | Configuration Management | Reproducible experiments |
| **🔬 MLflow** | Model Registry & Tracking | Model selection and management |
| **🎯 Optuna** | Hyperparameter Optimization | Automated model tuning |
| **🎨 Streamlit** | Interactive Web App | Demo and visualization |

## 🏗️ **Architecture**

```
User Input → Model Selection (MLflow) → PyTorch Inference → Recommendations
     ↓
  MLflow Model Registry ← Hydra Configuration ← Optuna Optimization
```

## 📚 **Learn More**

This demo is part of a comprehensive movie recommendation system that showcases:

- **Professional ML Engineering**: Clean code architecture with proper abstractions
- **Experiment Management**: MLflow for tracking and model versioning  
- **Configuration Management**: Hydra for reproducible experiments
- **Hyperparameter Optimization**: Optuna for intelligent model tuning
- **Interactive Deployment**: Streamlit for user-friendly demonstration

### 🔗 **Related Resources**
- Full codebase and training scripts
- Model training with `python train_hydra.py`
- Hyperparameter optimization examples
- Complete documentation and usage guides

---

**🎯 Ready to explore AI-powered movie recommendations? Select a model and start discovering!**