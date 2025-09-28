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
- **Custom Made Logging**: Comprehensive training and evaluation tracking
- **Visualization Tools**: Training plots, data analysis, embedding visualization
- **Reproducible Training**: Seed management and configuration tracking
- **Error Handling**: Robust error handling throughout the pipeline
- **Hydra Integration**: Advanced experiment management and configuration system

### **⚙️ Hydra Experiment Management**
- **Multiple Training Scripts**: Traditional and Hydra-based training options
- **Configuration Management**: YAML-based configuration system with overrides
- **Experiment Tracking**: Organized output directories and automatic logging
- **Multi-Model Support**: Easy switching between different model architectures
- **Hyperparameter Sweeps**: Automated parameter optimization experiments

## 🚀 Quick Start

### **Installation**
```bash
git clone https://github.com/MrEleden/reco_app.git
cd reco_app
pip install -r requirements.txt
```

**For GPU Training (Recommended)**:
```bash
# Install CUDA-enabled PyTorch for faster training
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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

### **Hydra-Based Advanced Training**
```bash
# Train with Hydra configuration system
python train_hydra.py

# Train specific model with custom parameters
python train_hydra.py model=deep_cf train.epochs=50 train.learning_rate=0.001

# Run multiple model comparison
python train_hydra.py -m model=collaborative,content_based,hybrid,deep_cf

# Hyperparameter sweep
python train_hydra.py -m train.learning_rate=0.001,0.01,0.1 train.batch_size=256,512
```

**Output Organization**: All training outputs are saved in the `outputs/` directory:
- **Single runs**: `outputs/movie_recommendation/YYYY-MM-DD_HH-MM-SS/`
- **Multirun**: `outputs/movie_recommendation/multirun/YYYY-MM-DD_HH-MM-SS/N_modelname/`
- **Each run contains**: logs, model checkpoints, configuration files, and metrics

### **Simple Optimizer Configuration**
```bash
# Use Adam optimizer (default)
python train_hydra.py model=collaborative

# Use SGD optimizer with momentum  
python train_hydra.py optimizer=sgd model=collaborative

# Compare Adam vs SGD
python train_hydra.py -m optimizer=adam,sgd model=collaborative
```

**Available Optimizers**:
- **adam**: Adam optimizer with adaptive learning rates (recommended for most cases)
- **sgd**: SGD with momentum support (simple and effective)

### **Direct Usage (Following Loss/Metrics Pattern)**
```python
from optimizers import RecommenderOptimizer, AdamOptimizer, SGDOptimizer, create_optimizer
import torch

# Method 1: Using RecommenderOptimizer factory class (similar to RecommenderLoss)
optimizer_factory = RecommenderOptimizer("adam")
adam_optimizer = optimizer_factory.create_optimizer(model, lr=0.001, betas=(0.9, 0.999))

# Method 2: Using specific optimizer classes directly (similar to BCELoss, MSELoss)
adam_optimizer = AdamOptimizer(lr=0.001, betas=(0.9, 0.999))
sgd_optimizer = SGDOptimizer(lr=0.01, momentum=0.9)

# Method 3: Using factory function
sgd_optimizer = create_optimizer(model, "sgd", lr=0.01, momentum=0.9)
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

## 📁 Custom Made Project Structure

```
reco_app/
├── README.md                          # This file
├── train.py                          # Main training script 🚀
├── train_hydra.py                    # Hydra-based training script ⚙️
├── train_simple.py                   # Simplified training demo 🔧
├── requirements.txt                  # Dependencies 📋
│
├── config/                           # Configuration management ⚙️
│   ├── __init__.py                   # Main config imports
│   ├── data_config.py                # Data processing settings
│   ├── model_config.py               # Model architecture configs
│   └── train_config.py               # Training hyperparameters
│
├── conf/                             # Hydra configuration files 🎯
│   ├── config.yaml                   # Main Hydra config
│   ├── model/                        # Model configurations
│   │   ├── collaborative.yaml        # Matrix factorization settings
│   │   ├── content_based.yaml        # Content-based settings
│   │   ├── hybrid.yaml               # Hybrid model settings
│   │   └── deep_cf.yaml              # Deep CF settings
│   ├── train/                        # Training configurations
│   │   ├── default.yaml              # Standard training
│   │   ├── fast.yaml                 # Quick experiments
│   │   └── production.yaml           # Thorough training
│   ├── data/                         # Data configurations
│   │   └── default.yaml              # Data processing settings
│   └── experiment/                   # Experiment presets
│       ├── quick.yaml                # Fast testing
│       └── model_comparison.yaml     # Model benchmarking
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
├── optimizers/                       # Optimizer implementations (following loss/metric pattern) ⚡
│   ├── __init__.py                   # Package initialization  
│   └── optimizer.py                  # RecommenderOptimizer, AdamOptimizer, SGDOptimizer classes
│
├── utils/                            # Utilities 🛠️
│   ├── __init__.py                   # Package init
│   ├── logger.py                     # Logging utilities
│   ├── timer.py                      # Timing utilities
│   ├── plotter.py                    # Visualization utilities
│   └── helpers.py                    # Helper functions
│
├── results/                          # Traditional training outputs 💾
│   └── plots/                        # Training plots
│
├── outputs/                          # Hydra training outputs 📊
│   └── movie_recommendation/         # Experiment outputs
│       ├── YYYY-MM-DD_HH-MM-SS/      # Single run outputs
│       └── multirun/                 # Multi-run experiment outputs
│           └── YYYY-MM-DD_HH-MM-SS/  # Multi-run session
│               ├── 0_modelname/      # Individual job outputs  
│               ├── 1_modelname/      # Individual job outputs
│               └── multirun.yaml     # Multi-run configuration
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

## ⚡ Optimizer System

### **Available Optimizers (Following Loss/Metrics Pattern)**
- **AdamOptimizer**: Adam optimizer with adaptive learning rates (recommended for most cases)
- **SGDOptimizer**: SGD with momentum support (simple and effective)
- **RecommenderOptimizer**: Factory class for creating optimizers (similar to RecommenderLoss)

### **Usage Pattern (Consistent with Losses/Metrics)**
```python
from optimizers import RecommenderOptimizer, AdamOptimizer, SGDOptimizer, create_optimizer

# Method 1: Factory class approach (similar to RecommenderLoss)
optimizer_factory = RecommenderOptimizer("adam")
optimizer = optimizer_factory.create_optimizer(model, lr=0.001)

# Method 2: Direct class instantiation (similar to BCELoss, MSELoss)
adam_optimizer = AdamOptimizer(lr=0.001, betas=(0.9, 0.999))
sgd_optimizer = SGDOptimizer(lr=0.01, momentum=0.9, weight_decay=1e-4)

# Method 3: Factory function
optimizer = create_optimizer(model, "sgd", lr=0.01, momentum=0.9)
```

### **Integration with Training**
```python
# In train_hydra.py
from optimizers import RecommenderOptimizer

def create_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    optimizer_factory = RecommenderOptimizer(cfg.optimizer.name)
    return optimizer_factory.create_optimizer(
        model=model, 
        lr=cfg.train.learning_rate,
        **cfg.optimizer.params
    )
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

### **Traditional Command Line Arguments**
```bash
python train.py \
    --epochs 50 \
    --batch-size 512 \
    --lr 0.001 \
    --embedding-dim 100 \
    --dropout 0.3 \
    --save-path results/my_model.pth
```

### **Hydra-Based Configuration System**
```bash
# Single model training
python train_hydra.py model=deep_cf train.epochs=50 train.batch_size=512

# Multiple model comparison
python train_hydra.py -m model=collaborative,content_based,hybrid,deep_cf

# Hyperparameter optimization
python train_hydra.py -m train.learning_rate=0.001,0.01,0.1 model.embedding_dim=32,64,128

# Use preset configurations
python train_hydra.py train=production model=hybrid
```

### **Available Parameters (Traditional)**
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

### **Adding New Optimizers**
```python
# Add to optimizers/optimizer.py (following the same pattern as losses/metrics)
class CustomOptimizer:
    def __init__(self, lr=0.001, **kwargs):
        self.lr = lr
        self.kwargs = kwargs
    
    def create_optimizer(self, model, **extra_kwargs):
        # Combine initialization and runtime parameters
        params = {**self.kwargs, **extra_kwargs}
        return torch.optim.CustomOptim(model.parameters(), lr=self.lr, **params)

# Register in RecommenderOptimizer class
OPTIMIZER_MAPPING = {
    "adam": AdamOptimizer,
    "sgd": SGDOptimizer,
    "custom": CustomOptimizer,  # Add your new optimizer here
}
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

**CUDA/GPU Setup**:
```bash
# Check GPU availability
nvidia-smi

# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

# Install CUDA-enabled PyTorch (if needed)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Test GPU training
python test_gpu_simple.py
```

**GPU Training**: The system automatically detects and uses GPU when available. Set `device: "cuda"` in config files to force GPU usage, or `device: "cpu"` to force CPU usage.

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