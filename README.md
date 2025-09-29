# Movie Recommendation System# 🎬 Movie Recommendation System# 🎬 Movie Recommendation System Tech Stack Integration



A PyTorch-based movie recommendation system that demonstrates scaling from scratch implementation to production-ready ML pipeline.



## Project Evolution**From PyTorch Scratch to Production-Ready ML Pipeline**A **production-ready movie recommendation system** demonstrating how modern ML/AI technologies work together seamlessly. This project showcases the integration of **PyTorch**, **Hydra**, **MLflow**, **Optuna**, and professional software architecture.



This project shows the progression from basic PyTorch to a complete ML system:



1. **From Scratch Implementation** (`train.py`)## 🎯 **Why This Project?**## 🎯 **Tech Stack Showcase**

   - Pure PyTorch movie recommendation models

   - Basic training loop and evaluation

   - Simple data handling

Built to demonstrate the **evolution of ML development**:This project demonstrates **complete integration** of modern ML technologies:

2. **Scaled Production System** (`train_hydra.py`) 

   - Hydra configuration management- 🔥 **From Scratch**: Pure PyTorch implementation (`train.py`)

   - MLflow experiment tracking

   - Optuna hyperparameter optimization- 🏗️ **Modular Design**: Reusable components for scripts and notebooks  | Technology | Purpose | Integration Point |

   - Professional logging and model persistence

- 🚀 **Production Stack**: Integrated PyTorch + Hydra + MLflow + Optuna (`train_hydra.py`)|------------|---------|-------------------|

## What I Built

| **🔥 PyTorch** | Deep Learning Framework | Model architecture, training loops, GPU acceleration |

- **Models**: Matrix Factorization, Neural Collaborative Filtering, DeepFM

- **Configuration**: Hydra-based config system for easy experimentation## ⚡ **One Command, Complete Pipeline**| **⚙️ Hydra** | Configuration Management | Experiment configs, multirun sweeps, parameter overrides |

- **Tracking**: MLflow integration for experiment management

- **Optimization**: Optuna for automatic hyperparameter tuning| **🔬 MLflow** | Experiment Tracking | Automatic logging, model registry, performance comparison |

- **Demo**: Streamlit app for interactive recommendations

```bash| **🎯 Optuna** | Hyperparameter Optimization | Intelligent parameter search with Hydra integration |

## Usage

# Basic PyTorch training| **🐍 Python OOP** | Code Architecture | Modular design, abstract classes, professional structure |

Basic PyTorch training:

```bashpython train.py| **📊 Scientific Stack** | Data Processing | NumPy, Pandas for data manipulation and analysis |

python train.py

```



Production pipeline with optimization:# Production-ready with hyperparameter optimization### **🚀 Key Integration Features**

```bash

python train_hydra.py -m model=collaborative,hybrid,deep_cfpython train_hydra.py -m model=collaborative,hybrid,deep_cf train=production hydra/sweeper=optuna_production- **One Command, Full Pipeline**: `python train_hydra.py -m model=collaborative,hybrid,deep_cf`

```

```- **Intelligent Optimization**: `python train_hydra.py --config-name=optuna_test -m` 

Launch demo app:

```bash- **Automatic Tracking**: Every experiment logged to MLflow with zero extra code

streamlit run app.py

```## 🏆 **Tech Stack Evolution**- **Configuration Magic**: Change models, optimizers, hyperparameters via YAML configs



## Key Technologies- **Smart HPO**: Optuna finds optimal hyperparameters automatically



- PyTorch: Deep learning framework| Approach | Command | Features |- **Production Ready**: Professional error handling, logging, and model persistence

- Hydra: Configuration management

- MLflow: Experiment tracking|----------|---------|----------|

- Optuna: Hyperparameter optimization

- Streamlit: Interactive web app| **🔥 Basic** | `python train.py` | Pure PyTorch, manual tuning |## 🏆 **Live Performance Dashboard**

| **⚙️ Structured** | `python train_hydra.py model=hybrid` | Configuration management |

| **🚀 Production** | `python train_hydra.py -m [models] hydra/sweeper=optuna` | Auto-optimization + tracking |**Current Best Model**: Hybrid Architecture with **RMSE: 0.3239** and **85.21% Accuracy**



## 📊 **Current Best Model**| Model | RMSE ⬇️ | Accuracy ⬆️ | Status |

|-------|---------|-------------|---------|

**Hybrid Architecture**: 85.41% accuracy, RMSE 0.3232  | 🥇 **Hybrid** | **0.3239** | **85.21%** | ✅ Production Ready |

*Automatically discovered via Optuna optimization*| 🥈 Collaborative | 0.3451 | 83.65% | ✅ Baseline |

| 🥉 Content-Based | 0.3820 | 82.25% | ✅ Specialized |

## 🎮 **Interactive Demo**

*Real-time results from MLflow tracking - View full dashboard: http://127.0.0.1:5000*

```bash

streamlit run app.py## 🚀 **Get Started Now**

# → Live model selection, real-time recommendations, MLflow integration

```**Experience the complete tech stack integration in 4 simple commands:**



## 🏗️ **Architecture Highlights**```bash

# 1️⃣ Clone and setup

- **📦 Modular Components**: Reusable across scripts and notebooksgit clone https://github.com/MrEleden/reco_app.git && cd reco_app && pip install -r requirements.txt

- **⚙️ Configuration-Driven**: Change models/parameters without code changes  

- **📊 Auto-Tracking**: Every experiment logged to MLflow# 2️⃣ Basic PyTorch + Hydra + MLflow integration

- **🎯 Smart Optimization**: Optuna finds best hyperparameters automaticallypython train_hydra.py model=hybrid train.epochs=5

- **🌐 Deploy-Ready**: Streamlit app ready for Hugging Face Spaces

# 3️⃣ Multi-model comparison with automatic tracking

## 🚀 **Quick Start**python train_hydra.py -m model=collaborative,hybrid train.epochs=5



```bash# 4️⃣ Production-ready hyperparameter optimization with Optuna

# Clone and setuppython train_hydra.py -m model=collaborative,content_based,hybrid,deep_cf train=production hydra/sweeper=optuna_production

git clone [repo-url] && cd reco_app

pip install -r requirements.txt# 📊 View comprehensive results dashboard  

python check_mlflow.py && python -m mlflow ui --port 5000

# Try different approaches```

python train.py                           # Basic PyTorch

python train_hydra.py model=hybrid        # Structured training  ### **🎯 Complete Tech Stack Examples**

python train_hydra.py -m model=all        # Multi-model comparison

streamlit run app.py                      # Interactive demo**🔥 PyTorch Models**: Switch architectures via configuration

``````bash

python train_hydra.py model=collaborative    # Matrix factorization

---python train_hydra.py model=hybrid          # Multi-modal fusion

python train_hydra.py model=deep_cf         # Deep learning

**🎯 From scratch PyTorch to production ML pipeline in simple commands**```

**⚙️ Hydra Configuration**: No code changes needed
```bash  
python train_hydra.py train=fast            # Quick 10-epoch training
python train_hydra.py train=production      # Full 100-epoch training
python train_hydra.py optimizer=sgd         # Change optimizer
```

**🎯 Optuna Optimization**: Smart hyperparameter search
```bash
# Quick test optimization (6 trials)
python train_hydra.py --config-name=optuna_test -m

# Production optimization (100 trials, 4 parallel jobs)
python train_hydra.py -m model=collaborative,content_based,hybrid,deep_cf train=production hydra/sweeper=optuna_production
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

**Production-ready intelligent parameter search integrated with Hydra and MLflow:**

### **🚀 Quick Start Examples**
```bash
# 🧪 Quick optimization test (6 trials)
python train_hydra.py --config-name=optuna_test -m

# 🏭 Production multi-model optimization (100 trials, 4 parallel jobs)
python train_hydra.py -m model=collaborative,content_based,hybrid,deep_cf train=production hydra/sweeper=optuna_production

# 📊 All trials automatically tracked in MLflow
python check_mlflow.py  # View results
```

### **🏆 Production Configuration Features**
- **100 Intelligent Trials**: TPE sampler with smart parameter exploration
- **4 Parallel Jobs**: Faster optimization using multiple CPU cores
- **Multi-Model Search**: Optimizes all 4 models simultaneously
- **Automatic MLflow Tracking**: Every trial logged with hyperparameters and results
- **Production Training**: Uses full 100-epoch training with early stopping

### **🔧 Optimized Parameters**
| Parameter | Search Space | Strategy |
|-----------|--------------|----------|
| `model` | collaborative, content_based, hybrid, deep_cf | Intelligent model selection |
| `learning_rate` | 0.0001 → 0.05 | Log-scale discrete choices |
| `batch_size` | 128, 256, 512, 1024 | Memory-efficient powers of 2 |
| `dropout` | 0.1 → 0.5 | Regularization optimization |
| `weight_decay` | 0.0001 → 0.01 | Log-scale regularization |
| `patience` | 5 → 15 epochs | Early stopping optimization |

### **🧠 Smart Search vs Grid Search**
| Approach | Trials | Time | Result Quality | Parallel |
|----------|--------|------|----------------|----------|
| **🎯 Optuna Production** | 100 trials | ⚡ Efficient | 🏆 Optimal | ✅ 4 jobs |
| **🧪 Optuna Quick** | 6 trials | ⚡ Ultra-fast | 📊 Good | ❌ 1 job |
| 📊 Grid Search | 1000+ trials | ⏰ Exhaustive | ✅ Complete | ❌ Sequential |
| 🎲 Random | 100 trials | ⚡ Fast | 📊 Variable | ❌ Sequential |

**🔄 Optuna's TPE sampler automatically explores the most promising parameter combinations based on previous trial results**

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
│   │   └── production.yaml           # Full training (100 epochs)
│   ├── optimizer/                    # ⚡ Optimizer settings
│   │   ├── adam.yaml                 # Adaptive learning
│   │   └── sgd.yaml                  # Stochastic gradient descent
│   └── hydra/sweeper/                # 🎯 Optuna HPO configurations
│       ├── optuna_quick.yaml         # Quick HPO (6 trials)
│       ├── optuna_comprehensive.yaml # Standard HPO (50 trials)
│       └── optuna_production.yaml    # Production HPO (100 trials, 4 jobs)
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