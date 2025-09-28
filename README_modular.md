# 🎬 Movie Recommendation System

A comprehensive, modular movie recommendation system built with PyTorch and Gradio. This system provides personalized movie recommendations using collaborative filtering, content-based filtering, and deep learning approaches.

## 🌟 Features

- **🤖 Multiple AI Models**: Collaborative filtering, content-based, hybrid, and deep neural networks
- **🎯 Personalized Recommendations**: Tailored suggestions based on user preferences
- **🚀 Interactive Training**: Train your own AI models directly in the web interface
- **🔍 Smart Search**: Find movies by title, genre, or keywords
- **👤 User Profiles**: Explore rating patterns and preferences
- **📊 Real-time Analytics**: Dataset statistics and model performance metrics
- **🎨 Beautiful Interface**: Modern, responsive Gradio web interface

## 🏗️ Architecture

The system is built with a clean, modular architecture:

```
reco_app/
├── models.py           # PyTorch neural network models
├── dataset.py          # Data loading and preprocessing
├── trainer.py          # Model training and evaluation
├── recommender.py      # Main recommendation logic
├── app_modular.py      # Gradio web interface
├── data/              # MovieLens dataset
├── models/            # Trained model checkpoints
└── requirements.txt   # Dependencies
```

### 🧠 Core Components

#### `models.py` - Neural Network Models
- **CollaborativeFilteringModel**: Matrix factorization with embeddings
- **ContentBasedModel**: Genre-based neural network
- **HybridModel**: Combines collaborative and content-based approaches
- **DeepCollaborativeFiltering**: Deep neural collaborative filtering

#### `dataset.py` - Data Management
- **MovieLensDataLoader**: Comprehensive data loading and preprocessing
- **RecommenderDataset**: PyTorch dataset with negative sampling
- **ContentBasedDataset**: Dataset for content-based recommendations

#### `trainer.py` - Training Pipeline
- **ModelTrainer**: Unified training interface for all model types
- Training loop with early stopping and learning rate scheduling
- Model checkpointing and performance visualization

#### `recommender.py` - Recommendation Engine
- **MovieRecommendationSystem**: Main recommendation interface
- Multiple recommendation strategies (collaborative, ML, genre-based)
- User similarity computation and preference analysis

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/your-username/reco_app.git
cd reco_app
pip install -r requirements.txt
```

### 2. Launch the Application

```bash
python app_modular.py
```

The app will be available at `http://localhost:7860`

### 3. Usage Options

#### Option A: Use Pre-built Recommendations
- Navigate to the "My Recommendations" tab
- Enter a user ID (1-610)
- Select "collaborative" method
- Get instant recommendations!

#### Option B: Train Your Own AI Model
- Go to the "AI Training" tab
- Adjust training parameters (epochs, batch size, learning rate)
- Click "Train Model" and wait for completion
- Use "ml" method in recommendations for AI-powered suggestions

## 📊 Dataset

Uses the **MovieLens 100K Dataset**:
- **610 users** with diverse movie preferences
- **9,742 movies** across all genres and decades
- **100,836 ratings** from 0.5 to 5.0 stars
- **20+ genres** for content-based filtering

## 🤖 Model Types

### 1. Collaborative Filtering
Traditional user-item matrix factorization using neural embeddings.
- **Best for**: Users with many ratings
- **Pros**: Discovers complex patterns, handles sparse data
- **Cons**: Cold start problem for new users

### 2. Content-Based
Recommends based on movie features (genres, metadata).
- **Best for**: New users, genre-specific preferences
- **Pros**: No cold start problem, explainable recommendations
- **Cons**: Limited to feature space, less diverse

### 3. Hybrid Model
Combines collaborative and content-based approaches.
- **Best for**: Maximum accuracy and coverage
- **Pros**: Addresses limitations of both approaches
- **Cons**: More complex, requires more training time

### 4. Deep Collaborative Filtering
Advanced neural network with multiple hidden layers.
- **Best for**: Complex pattern recognition
- **Pros**: Captures non-linear relationships
- **Cons**: Requires more data and compute resources

## 🎯 Training Your Own Model

The system supports end-to-end model training:

### Quick Training
```python
from recommender import MovieRecommendationSystem

# Initialize system
recommender = MovieRecommendationSystem()

# Train collaborative filtering model
result = recommender.train_model(
    model_type="collaborative",
    epochs=20,
    batch_size=256,
    learning_rate=0.01
)

# Load trained model
recommender.load_trained_model("collaborative")

# Get AI recommendations
recommendations = recommender.get_user_recommendations(
    user_id=123, 
    method="ml"
)
```

### Advanced Training
```python
from trainer import ModelTrainer
from dataset import MovieLensDataLoader

# Load and prepare data
data_loader = MovieLensDataLoader()
user_movie_pairs, config = data_loader.get_collaborative_data()

# Create trainer
trainer = ModelTrainer(model_type="collaborative")
trainer.prepare_model(config)
trainer.prepare_training(learning_rate=0.01)

# Train with custom settings
result = trainer.train(train_loader, val_loader, epochs=50)
```

## 🔧 Configuration

### Model Parameters
```python
# Collaborative Filtering
CollaborativeFilteringModel(
    n_users=610,        # Number of unique users
    n_movies=9742,      # Number of unique movies  
    n_factors=50,       # Embedding dimension
    dropout=0.2         # Dropout rate
)

# Training Parameters
trainer.prepare_training(
    learning_rate=0.01,   # Learning rate
    weight_decay=1e-5     # L2 regularization
)
```

### Performance Tuning
- **Batch Size**: 256-512 for balanced speed/memory
- **Learning Rate**: 0.01-0.001 with scheduler
- **Epochs**: 20-50 with early stopping
- **Embedding Size**: 50-100 factors

## 📈 Performance Metrics

The system tracks multiple metrics during training:
- **Binary Cross-Entropy Loss**: Primary optimization target
- **Validation Loss**: Monitors overfitting
- **Training Time**: Efficiency measurement
- **Memory Usage**: Resource monitoring

## 🎨 Web Interface

### Tabs Overview
- **🎯 My Recommendations**: Get personalized suggestions
- **👤 User Profile**: Explore user preferences and history
- **🔍 Movie Search**: Find movies by title or genre
- **🔥 Trending**: Discover popular movies
- **🚀 AI Training**: Train and manage ML models
- **📊 Dataset Info**: View dataset statistics
- **ℹ️ About**: System documentation and help

### Features
- **Real-time Training**: Watch model training progress live
- **Interactive Controls**: Adjust parameters dynamically
- **Responsive Design**: Works on desktop and mobile
- **Error Handling**: Graceful error messages and recovery

## 🔍 API Reference

### Main Classes

#### MovieRecommendationSystem
```python
# Initialize
recommender = MovieRecommendationSystem(data_dir="data")

# Get recommendations
recommender.get_user_recommendations(user_id, n_recommendations, method)

# Search movies
recommender.search_movies(query)

# Train models
recommender.train_model(model_type, epochs, batch_size, learning_rate)
```

#### ModelTrainer
```python
# Initialize trainer
trainer = ModelTrainer(model_type="collaborative", device="cpu")

# Prepare model
trainer.prepare_model(model_config)

# Train
trainer.train(train_loader, val_loader, epochs=20)
```

## 🚀 Deployment

### Local Development
```bash
python app_modular.py
```

### Hugging Face Spaces
1. Create new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select **Gradio** as SDK
3. Upload these files:
   - `app_modular.py` (rename to `app.py`)
   - `models.py`
   - `dataset.py`
   - `trainer.py`
   - `recommender.py`
   - `requirements.txt`
   - `data/` folder
4. Space will auto-deploy!

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app_modular.py"]
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MovieLens Dataset**: University of Minnesota GroupLens Research
- **PyTorch**: Deep learning framework
- **Gradio**: Web interface framework
- **Hugging Face**: Hosting and deployment platform

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/reco_app/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/reco_app/discussions)
- **Email**: your-email@example.com

---

**🎬 Happy movie watching!** ⭐