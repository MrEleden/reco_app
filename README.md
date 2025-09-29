# 🎬 Movie Recommendation System Tech Stack Integration

A **production-ready movie recommendation system** demonstrating how modern ML/AI technologies work together seamlessly. This project showcases the integration of **PyTorch**, **Hydra**, **MLflow**, **Optuna**, and professional software architecture.

## 🎯 **Tech Stack Showcase**

This project demonstrates **complete integration** of modern ML technologies:

| Technology | Purpose | Integration Point |
|------------|---------|-------------------|
| **🔥 PyTorch** | Deep Learning Framework | Model architecture, training loops, GPU acceleration |
| **⚙️ Hydra** | Configuration Management | Experiment configs, multirun sweeps, parameter overrides |
| **🔬 MLflow** | Experiment Tracking | Automatic logging, model registry, performance comparison |
| **🎯 Optuna** | Hyperparameter Optimization | Intelligent parameter search with Hydra integration |
| **🐍 Python OOP** | Code Architecture | Modular design, abstract classes, professional structure |
| **📊 Scientific Stack** | Data Processing | NumPy, Pandas for data manipulation and analysis |

### **🚀 Key Integration Features**
- **One Command, Full Pipeline**: `python train_hydra.py -m model=collaborative,hybrid,deep_cf`
- **Intelligent Optimization**: `python train_hydra.py --config-name=optuna_test -m` 
- **Automatic Tracking**: Every experiment logged to MLflow with zero extra code
- **Configuration Magic**: Change models, optimizers, hyperparameters via YAML configs
- **Smart HPO**: Optuna finds optimal hyperparameters automatically
- **Production Ready**: Professional error handling, logging, and model persistence

## 🏆 **Live Performance Dashboard**

**Current Best Model**: Hybrid Architecture with **RMSE: 0.3239** and **85.21% Accuracy**

| Model | RMSE ⬇️ | Accuracy ⬆️ | Status |
|-------|---------|-------------|---------|
| 🥇 **Hybrid** | **0.3239** | **85.21%** | ✅ Production Ready |
| 🥈 Collaborative | 0.3451 | 83.65% | ✅ Baseline |
| 🥉 Content-Based | 0.3820 | 82.25% | ✅ Specialized |

*Real-time results from MLflow tracking - View full dashboard: http://127.0.0.1:5000*

## 🚀 **Get Started Now**

**Experience the complete tech stack integration in 4 simple commands:**

```bash
# 1️⃣ Clone and setup
git clone https://github.com/MrEleden/reco_app.git && cd reco_app && pip install -r requirements.txt

# 2️⃣ Basic PyTorch + Hydra + MLflow integration
python train_hydra.py model=hybrid train.epochs=5

# 3️⃣ Multi-model comparison with automatic tracking
python train_hydra.py -m model=collaborative,hybrid train.epochs=5

# 4️⃣ Intelligent hyperparameter optimization with Optuna
python train_hydra.py --config-name=optuna_test -m

# 📊 View comprehensive results dashboard  
python check_mlflow.py && python -m mlflow ui --port 5000
```

### **🎯 Complete Tech Stack Examples**

**🔥 PyTorch Models**: Switch architectures via configuration
```bash
python train_hydra.py model=collaborative    # Matrix factorization
python train_hydra.py model=hybrid          # Multi-modal fusion
python train_hydra.py model=deep_cf         # Deep learning
```

**⚙️ Hydra Configuration**: No code changes needed
```bash  
python train_hydra.py train=fast            # Quick 10-epoch training
python train_hydra.py train=production      # Full 30-epoch training
python train_hydra.py optimizer=sgd         # Change optimizer
```

**🎯 Optuna Optimization**: Smart hyperparameter search
```bash
python train_hydra.py --config-name=optuna_test -m     # Quick optimization
python train_hydra.py --config-name=optuna_demo -m     # Full demo
```

**🔬 MLflow Tracking**: Every experiment automatically logged
```bash
python check_mlflow.py                      # Command-line results
python -m mlflow ui --port 5000            # Web dashboard
```

## 🔬 **MLflow Integration - Zero-Configuration Tracking**

**Every experiment is automatically tracked - no extra code required!**

```bash
# 🎯 Run experiment - MLflow handles everything
python train_hydra.py model=hybrid train.epochs=10

# 📊 View results dashboard
python check_mlflow.py

# 🌐 Launch interactive UI 
python -m mlflow ui --port 5000    # → http://127.0.0.1:5000
```

### **🤖 Automatic Tracking Features**
- ✅ **Hyperparameters**: All Hydra configs automatically logged
- ✅ **Metrics**: RMSE, MAE, Precision@K tracked every epoch  
- ✅ **Models**: Best checkpoints saved with metadata
- ✅ **Artifacts**: Training plots, config files, logs
- ✅ **System Info**: Hardware specs, execution time, Git hash

### **🏆 Production Model Loading**
```python
from utils.mlflow_utils import MLflowModelSelector

# One-liner to get best model
selector = MLflowModelSelector()
best_model, run_id = selector.load_best_model('val_rmse')
```

## 🎯 **Optuna Hyperparameter Optimization**

**Intelligent parameter search integrated with Hydra and MLflow:**

```bash
# 🚀 Quick optimization test (6 trials)
python train_hydra.py --config-name=optuna_test -m

# 📊 All trials automatically tracked in MLflow
python check_mlflow.py  # View results
```

### **🧠 Smart Search vs Grid Search**
| Approach | Trials | Time | Result Quality |
|----------|--------|------|----------------|
| **🎯 Optuna** | 20 trials | ⚡ Efficient | 🏆 Optimal |
| 📊 Grid Search | 64 trials | ⏰ Exhaustive | ✅ Complete |
| 🎲 Random | 20 trials | ⚡ Fast | 📊 Variable |

### **🔧 Optuna Configuration**
```yaml
# conf/hydra/sweeper/optuna_quick.yaml
search_space:
  train.learning_rate: 0.001,0.005,0.01,0.05
  train.batch_size: 256,512  
  model.embedding_dim: 32,50,64
```

**🔄 Optuna automatically explores the most promising parameter combinations**

## ⚙️ **Hydra Configuration Magic**

**Change everything without touching code - just modify YAML files or command line**

### **📝 YAML Configuration Files**
```yaml
# conf/model/hybrid.yaml
name: hybrid
embedding_dim: 50
collaborative_weight: 0.7
fusion_dims: [256, 128]

# conf/train/production.yaml
epochs: 30
batch_size: 256
learning_rate: 0.005
patience: 8
```

### **🚀 Command Line Overrides**
```bash
# Override any parameter instantly
python train_hydra.py model=hybrid train.learning_rate=0.001
python train_hydra.py train=production model.embedding_dim=64
```

## 🧠 **PyTorch Model Architectures**

**Four production-ready models demonstrating different ML approaches:**

| Model | Approach | Architecture | Use Case |
|-------|----------|--------------|----------|
| 🤝 **Collaborative** | Matrix Factorization | User/Movie embeddings | High accuracy, cold start issues |
| 🎬 **Content-Based** | Feature Engineering | Genre + Neural Networks | Interpretable, works with new movies |
| 🔀 **Hybrid** | Multi-Modal Fusion | Combined approach | **Best performance** (RMSE: 0.3239) |  
| 🧠 **Deep CF** | Deep Learning | Multi-layer neural nets | Complex patterns, requires more data |

### **🏗️ Professional PyTorch Implementation**
```python
# Extensible base class
class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, user_ids, movie_ids): pass
    @abstractmethod  
    def predict(self, user_ids, movie_ids): pass
    
# Easy model switching via Hydra config
model = create_model(cfg, n_users, n_movies)  # cfg.model.type determines architecture
```

## 📊 **Comprehensive ML Components**

**Professional PyTorch ecosystem with modular components:**

### **🎯 Loss Functions**
```python
from losses import RecommenderLoss, BPRLoss
criterion = RecommenderLoss(loss_type='bce')  # Configurable via Hydra
```

### **📈 Evaluation Metrics** 
```python  
from metrics import RecommenderMetrics
metrics = ['rmse', 'mae', 'precision@10', 'recall@10', 'ndcg@10']  # Auto-tracked
```

### **⚡ Optimizers**
```python
from optimizers import RecommenderOptimizer
optimizer = RecommenderOptimizer("adam").create_optimizer(model, lr=0.001)
```

**🔄 All components follow the same design pattern and integrate seamlessly with Hydra configuration**

## 📁 **Tech Stack Architecture**

**Clean, Professional Structure Showcasing Modern ML Engineering**

```
reco_app/
├── 🚀 train_hydra.py                 # Main entry point - Hydra + PyTorch + MLflow
├── 🔬 check_mlflow.py                # MLflow results dashboard
├── 📋 requirements.txt               # Tech stack dependencies
│
├── ⚙️ conf/                           # 🎯 HYDRA Configuration Hub
│   ├── config.yaml                   # Main orchestration config
│   ├── model/                        # 🧠 Model architecture configs
│   │   ├── collaborative.yaml        # Matrix factorization
│   │   ├── hybrid.yaml               # Multi-modal approach
│   │   └── deep_cf.yaml              # Deep learning model
│   ├── train/                        # 🚀 Training configurations
│   │   ├── fast.yaml                 # Quick experiments (10 epochs)
│   │   └── production.yaml           # Full training (30 epochs)
│   ├── optimizer/                    # ⚡ Optimizer settings
│   │   ├── adam.yaml                 # Adaptive learning
│   │   └── sgd.yaml                  # Stochastic gradient descent
│   └── hydra/sweeper/                # 🎯 Optuna configurations
│       ├── optuna_quick.yaml         # Quick HPO (6 trials)
│       └── optuna_comprehensive.yaml # Full HPO (50 trials)
│
├── 🧠 models/                         # 🔥 PYTORCH Model Architectures
│   ├── collaborative_filtering.py    # Matrix factorization
│   ├── content_based_model.py        # Genre-based recommendations  
│   ├── hybrid_model.py               # Combined approach
│   └── deep_collaborative_filtering.py # Deep neural networks
│
├── 📊 data/                           # Data processing pipeline
│   ├── dataset.py                    # PyTorch datasets
│   └── dataloader.py                 # MovieLens data loader
│
├── 🎯 losses/ & 📈 metrics/           # Training components
│   ├── loss.py                       # BCE, MSE, ranking losses
│   └── metric.py                     # RMSE, MAE, Precision@K
│
├── 🛠️ utils/                          # Professional utilities
│   ├── mlflow_utils.py               # 🔬 MLflow integration
│   ├── logger.py                     # Training logging
│   └── plotter.py                    # Visualization tools
│
└── 📂 Auto-Generated Outputs          # 🤖 Automated organization
    ├── outputs/                      # Hydra experiment outputs
    │   └── movie_recommendation/     # Timestamped runs
    ├── mlruns/                       # 🔬 MLflow tracking database
    │   ├── experiments/              # Organized experiments
    │   └── models/                   # Model registry
    └── data/raw/                     # MovieLens dataset
```

**🎯 Each directory serves a specific purpose in the tech stack integration**

## 📊 **Data Pipeline Integration**

**Professional PyTorch data handling with MovieLens dataset:**

- 🎬 **MovieLens Dataset**: 100K+ ratings, 9K+ movies with genre metadata
- 🔄 **PyTorch DataLoader**: Efficient batching with negative sampling  
- ⚡ **GPU Acceleration**: CUDA-optimized data loading and model training
- 🛡️ **Error Handling**: Robust data validation with clear error messages

## 🎯 **Why This Tech Stack?**

**This project demonstrates professional ML engineering with integrated modern tools:**

| Component | Purpose | Benefit |
|-----------|---------|---------|
| 🔥 **PyTorch** | Model Training | Production-ready deep learning framework |
| ⚙️ **Hydra** | Configuration | Experiment reproducibility without code changes |
| 🔬 **MLflow** | Tracking | Automated experiment logging and model registry |
| 🎯 **Optuna** | Optimization | Intelligent hyperparameter search |

### **🚀 Ready to Extend**
```python
# Adding new models, losses, metrics follows the same pattern
class NewModel(BaseModel):  # Inherit from base
    def forward(self): pass  # Implement required methods
    
# Register in configuration 
# conf/model/new_model.yaml ← Add config
# python train_hydra.py model=new_model ← Use immediately
```

## 🔧 **Quick Troubleshooting**

```bash
# ✅ Test GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# ✅ Test all imports
python -c "from models import *; from data import *; print('✅ All systems ready!')"
```

## 🎯 **Project Goals Achieved**

✅ **Simple**: One command runs complete ML pipeline  
✅ **Consistent**: All components follow same design patterns  
✅ **Professional**: Production-ready code with proper error handling  
✅ **Integrated**: PyTorch + Hydra + MLflow + Optuna work seamlessly together  
✅ **Intelligent**: Automated hyperparameter optimization with smart search
✅ **Extensible**: Easy to add new models, losses, metrics, optimizers  

---

**🎬 Complete Modern ML Stack: PyTorch + Hydra + MLflow + Optuna** ⚡

*The perfect foundation for your next ML project!* 🚀