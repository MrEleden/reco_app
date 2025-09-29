# 🎬 Movie Recommendation System - Gradio Demo

A production-ready movie recommendation system built with PyTorch, MLflow, and Gradio. This app demonstrates multiple ML models for personalized movie recommendations.

## 🌟 Features

- **🤖 Multiple ML Models**: Deep Collaborative Filtering, Hybrid, Content-based, Collaborative Filtering
- **🎬 Real-time Recommendations**: Get personalized movie suggestions for any user
- **📊 Model Comparison**: Compare performance metrics across different models
- **🎯 Interactive Interface**: Clean, professional ML demo interface
- **📈 MLflow Integration**: Complete experiment tracking and model management

## 🚀 Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will start on `http://0.0.0.0:7860`

## 🎯 How to Use

1. **Select Model**: Choose from available trained models
2. **Load Model**: Click "Load Model" to initialize
3. **Get Recommendations**: Enter a user ID and number of recommendations
4. **Compare Models**: View performance metrics across all models

## 🏗️ Tech Stack

- **PyTorch**: Deep learning framework
- **Gradio**: Web interface framework  
- **MLflow**: Experiment tracking and model registry
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization

## 📊 Model Performance

The app includes several pre-trained models:
- **Deep Collaborative Filtering**: Neural network-based recommendations
- **Hybrid Model**: Combines collaborative and content-based filtering
- **Collaborative Filtering**: User-item matrix factorization
- **Content-based**: Feature-based recommendations

## 🗃️ Data

Uses the MovieLens dataset with:
- **Movies**: 9,742 movies with genres and metadata
- **Ratings**: 100,836 user ratings (1-5 scale)
- **Users**: 610 unique users
- **Features**: Movie genres, user preferences, ratings history

## 🔧 Configuration

The app automatically configures:
- **Device Detection**: CUDA GPU or CPU fallback
- **Model Loading**: Automatic MLflow model registry integration
- **Data Paths**: Relative paths for portability
- **Port Selection**: Automatic available port detection

## 📈 MLflow Integration

Complete experiment tracking with:
- **Model Registry**: All trained models stored and versioned
- **Metrics Tracking**: RMSE, accuracy, training time
- **Parameter Logging**: All hyperparameters and configurations
- **Artifact Storage**: Model weights and training artifacts

## 🎪 Demo Usage

Perfect for:
- **ML Demonstrations**: Show recommendation system capabilities
- **Educational Purposes**: Teach ML concepts and model comparison  
- **Prototyping**: Quick testing of recommendation algorithms
- **Client Presentations**: Professional interface for stakeholders

---

Built with ❤️ using modern ML engineering practices

**Live Demo**: Upload to Hugging Face Spaces for instant deployment!