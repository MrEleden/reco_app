# 🎬 Professional PyTorch Movie Recommendation System

A production-ready, modular PyTorch-based movie recommendation system following industry best practices. Built with scalability, maintainability, and extensibility in mind.

## 🏆 Key Features

### **🔧 Professional Architecture**
- **Modular Design**: Organized packages for models, data, losses, metrics, and utilities
- **Industry Standards**: Following PyTorch best practices with proper abstractions
- **Scalable Structure**: Easy to extend with new models, losses, and metrics
- **Configuration Management**: Centralized configuration system

### **🤖 Model Implementations**
- **Collaborative Filtering**: Matrix factorization with embeddings and bias terms
- **Content-Based Filtering**: Genre-based recommendations with user preferences
- **Hybrid Models**: Combined collaborative and content-based approaches
- **Deep Collaborative**: Deep neural networks for collaborative filtering
- **Extensible Framework**: BaseModel abstract class for easy model additions

### **📊 Comprehensive Evaluation**
- **Multiple Metrics**: RMSE, MAE, Precision@K, Recall@K, NDCG
- **Training Monitoring**: Professional logging and visualization
- **Model Checkpointing**: Automatic best model saving with early stopping

### **🛠️ Production Ready**
- **Professional Logging**: Comprehensive training and evaluation tracking
- **Visualization Tools**: Training plots, data analysis, embedding visualization
- **Reproducible Training**: Seed management and configuration tracking
- **Error Handling**: Robust error handling throughout the pipeline

## 🚀 Quick Start

### **Installation**
```bash
git clone https://github.com/MrEleden/reco_app.git
cd reco_app
pip install -r requirements.txt
```

### **Basic Training**
```bash
# Train with default parameters
python train.py

# Custom training parameters
python train.py --epochs 50 --batch-size 512 --lr 0.001 --embedding-dim 100

# Simple training (for testing)
python train_simple.py --epochs 10
```

### **Using the Trained Model**
```python
from models import CollaborativeFilteringModel, ContentBasedModel, HybridModel, DeepCollaborativeFiltering
from data import MovieLensDataLoader
import torch

# Load data
data_loader = MovieLensDataLoader()
user_movie_pairs, config = data_loader.get_collaborative_data()

# Load trained model
model = CollaborativeFilteringModel(
    n_users=config['n_users'],
    n_movies=config['n_movies'],
    n_factors=50
)
checkpoint = torch.load('results/best_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Make predictions
model.eval()
user_ids = torch.tensor([0])
movie_ids = torch.tensor([10])
predictions = model.predict(user_ids, movie_ids)
```

## 📁 Professional Project Structure

```
reco_app/
├── README.md                          # This file
├── config.py                         # Configuration management ⚙️
├── train.py                          # Main training script 🚀
├── train_simple.py                   # Simplified training demo 🔧
├── requirements.txt                  # Dependencies 📋
│
├── models/                           # Model architectures 🧠
│   ├── __init__.py                   # Package initialization
│   ├── model.py                      # BaseModel abstract class
│   ├── collaborative/                # Collaborative filtering models
│   │   ├── __init__.py               # Package init
│   │   └── collaborative_filtering.py # Matrix factorization model
│   ├── content_based/                # Content-based models
│   │   ├── __init__.py               # Package init
│   │   └── content_based_model.py    # Genre-based model
│   ├── hybrid/                       # Hybrid recommendation models
│   │   ├── __init__.py               # Package init
│   │   └── hybrid_model.py           # Combined approach model
│   └── deep/                         # Deep learning models
│       ├── __init__.py               # Package init
│       └── deep_collaborative_filtering.py # Deep CF model
│
├── data/                             # Data handling 📊
│   ├── __init__.py                   # Package initialization
│   ├── dataset.py                    # PyTorch datasets
│   ├── dataloader.py                 # Data loading utilities
│   └── transforms.py                 # Data transformations
│
├── losses/                           # Loss functions 🎯
│   ├── __init__.py                   # Package init
│   └── loss.py                       # Loss implementations
│
├── metrics/                          # Evaluation metrics 📈
│   ├── __init__.py                   # Package init
│   └── metric.py                     # Metrics implementations
│
├── utils/                            # Utilities 🛠️
│   ├── __init__.py                   # Package init
│   ├── logger.py                     # Logging utilities
│   ├── timer.py                      # Timing utilities
│   ├── plotter.py                    # Visualization utilities
│   └── helpers.py                    # Helper functions
│
├── results/                          # Training outputs 💾
│   └── plots/                        # Training plots
│
├── logs/                             # Log files 📝
│
└── data/                             # Dataset files 📂
    └── raw/                          # Raw CSV data files
        ├── movies.csv                # Movies data
        ├── ratings.csv               # Ratings data
        ├── links.csv                 # Links data
        ├── tags.csv                  # Tags data
        └── README.txt                # Dataset documentation
```

## 🔧 Configuration System

All settings are managed through `config.py`:

```python
# Model configurations
MODEL_CONFIG = {
    "collaborative": {
        "embedding_dim": 50,
        "dropout": 0.2,
        "hidden_dims": [128, 64],
    }
}

# Training configurations
TRAIN_CONFIG = {
    "batch_size": 256,
    "learning_rate": 0.01,
    "weight_decay": 1e-4,
    "epochs": 20,
    "val_ratio": 0.2,
    "patience": 5,
}
```

## 🧠 Model Architectures

### **Complete Model Suite**
1. **Collaborative Filtering Model**: Matrix factorization with user/movie embeddings and bias terms
2. **Content-Based Model**: Genre-based recommendations using neural networks
3. **Hybrid Model**: Combines collaborative and content-based approaches with fusion layers
4. **Deep Collaborative Filtering**: Multi-layer neural network for complex pattern learning

### **Collaborative Filtering Model**
- **Matrix Factorization**: User and movie embeddings
- **Bias Terms**: User and movie biases for better predictions
- **Regularization**: Dropout for preventing overfitting
- **Flexible Architecture**: Configurable embedding dimensions

### **Content-Based Model**
- **Genre Embeddings**: Learn genre representations
- **User Preferences**: Neural network for user preference modeling
- **Feature Fusion**: Combine genre and user features
- **Configurable Architecture**: Adjustable hidden layer dimensions

### **Hybrid Model**
- **Multi-Component**: Collaborative + Content-based features
- **Fusion Layers**: Neural networks to combine different signals
- **Weighted Combination**: Learnable weights for different components
- **End-to-End Training**: Joint optimization of all components

### **Deep Collaborative Filtering**
- **Multi-Layer Architecture**: Deep neural networks for user-movie interactions
- **Non-Linear Patterns**: Capture complex user-item relationships
- **Configurable Depth**: Adjustable number of hidden layers
- **Dropout Regularization**: Prevent overfitting in deep architectures

### **BaseModel Abstract Class**
```python
class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
    
    def save_model(self, path: str, **kwargs):
        # Professional model saving
    
    def load_model(self, path: str):
        # Professional model loading
```

## 📊 Comprehensive Metrics

### **Implemented Metrics**
- **RMSE**: Root Mean Square Error for rating prediction
- **MAE**: Mean Absolute Error for rating prediction
- **Precision@K**: Precision for top-K recommendations
- **Recall@K**: Recall for top-K recommendations
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **F1-Score**: Harmonic mean of precision and recall

### **Usage Example**
```python
from metrics import RecommenderMetrics

metrics = RecommenderMetrics()
metrics.update(predictions, targets)
results = metrics.compute()
print(f"RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}")
```

## 🎯 Loss Functions

### **Available Losses**
- **RecommenderLoss**: Configurable loss (BCE/MSE)
- **BCELoss**: Binary Cross Entropy for implicit feedback
- **MSELoss**: Mean Squared Error for explicit ratings
- **BPRLoss**: Bayesian Personalized Ranking
- **RankingLoss**: Margin-based ranking loss

### **Usage Example**
```python
from losses import RecommenderLoss, BPRLoss

# For binary classification
criterion = RecommenderLoss(loss_type='bce')

# For ranking
criterion = BPRLoss()
```

## �️ Utilities

### **Logger**
Professional logging with training tracking:
```python
from utils import Logger

logger = Logger('logs/train.log')
logger.log_training_start(epochs=20, batch_size=256, lr=0.01)
logger.log_epoch(epoch=0, total_epochs=20, train_loss=0.5, val_loss=0.4, epoch_time=30.2)
```

### **Timer**
Performance timing:
```python
from utils import Timer

timer = Timer()
timer.start()
# ... training code ...
epoch_time = timer.stop()
```

### **Plotter**
Visualization tools:
```python
from utils import TrainingPlotter

plotter = TrainingPlotter()
plotter.plot_losses(train_losses, val_losses, save_path='results/plots/losses.png')
```

## 🚀 Training Options

### **Command Line Arguments**
```bash
python train.py \
    --epochs 50 \
    --batch-size 512 \
    --lr 0.001 \
    --embedding-dim 100 \
    --dropout 0.3 \
    --save-path results/my_model.pth
```

### **Available Parameters**
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size for training (default: 256)
- `--lr`: Learning rate (default: 0.01)
- `--embedding-dim`: Embedding dimension (default: 50)
- `--dropout`: Dropout rate (default: 0.2)
- `--save-path`: Model save path (default: results/best_model.pth)

## � Data Processing

### **Dataset Classes**
- **RecommenderDataset**: PyTorch dataset with negative sampling
- **ContentBasedDataset**: Dataset for content-based recommendations

### **Data Transformations**
- **NormalizeRatings**: Normalize ratings to [0, 1] range
- **NegativeSampling**: Generate negative samples for implicit feedback
- **ToTensor**: Convert numpy arrays to PyTorch tensors
- **GenreEncoder**: Encode movie genres to fixed-length vectors

### **DataLoader**
- **MovieLensDataLoader**: Complete data loading with label encoders
- **Automatic fallback**: Creates sample data if MovieLens files not found
- **Train/Validation split**: Configurable data splitting

## 🎯 Extending the System

### **Adding New Models**
```python
# Create models/new_model/new_model.py
from ..model import BaseModel

class NewRecommenderModel(BaseModel):
    def __init__(self, ...):
        super().__init__()
        # Your implementation
        
    def forward(self, ...):
        # Forward pass
        
    def predict(self, ...):
        # Prediction logic
```

### **Adding New Losses**
```python
# Add to losses/loss.py
class CustomLoss(nn.Module):
    def forward(self, predictions, targets):
        # Your loss implementation
        return loss
```

### **Adding New Metrics**
```python
# Add to metrics/metric.py
class CustomMetric:
    def update(self, predictions, targets):
        # Update metric state
        
    def compute(self):
        # Compute final metric value
        return metric_value
```

## 📈 Performance & Benchmarking

### **Model Performance**
The collaborative filtering model achieves:
- **Training convergence**: Typically converges within 20 epochs
- **Memory efficiency**: Optimized for large-scale datasets
- **GPU acceleration**: Full CUDA support for faster training

### **Benchmarking**
```python
# Built-in performance timing
python train.py  # Automatically logs training time per epoch
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-model`
3. **Make your changes**: Follow the existing structure
4. **Add tests**: Ensure your code works
5. **Submit a pull request**: Describe your changes

### **Contribution Areas**
- 🧠 **New Models**: CNN, RNN, Transformer-based recommenders
- 🎯 **Loss Functions**: Advanced loss functions for recommendations
- 📊 **Metrics**: Additional evaluation metrics
- 🔧 **Utilities**: Tools for analysis and visualization
- 📚 **Documentation**: Improve docs and examples

## 🐛 Troubleshooting

### **Common Issues**

**Import Errors**:
```bash
# Test all imports
python -c "from models import CollaborativeFilteringModel; from data import MovieLensDataLoader; print('Success!')"
```

**CUDA Issues**:
```python
# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Data Loading Issues**:
- Ensure `data/` folder contains MovieLens CSV files
- System automatically creates sample data if files are missing

## 📄 License

This project is open source and available under the **MIT License**.

## 🙏 Acknowledgments

- **MovieLens Dataset**: GroupLens Research at the University of Minnesota
- **PyTorch Team**: For the excellent deep learning framework
- **Open Source Community**: For inspiration and best practices

## 📞 Support

- **Issues**: Open an issue on GitHub for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check the code comments for detailed implementation notes

---

**Built with ❤️ using PyTorch | Production-Ready | Professionally Structured**

🎬 **Happy Recommending!** ✨