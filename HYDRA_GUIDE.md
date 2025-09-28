# 🚀 Hydra-Based Training System 

## ✅ **Hydra Implementation Complete!**

Your movie recommendation system now supports custom made experiment management with Hydra! This upgrade adds advanced configuration management alongside the existing traditional training system.

## 🎯 **Key Features Added**
- **Dual Training Systems**: Traditional (`train.py`) + Hydra-based (`train_hydra.py`)
- **Multiple Model Configurations**: Collaborative, Content-based, Hybrid, Deep CF
- **Flexible Training Settings**: Fast, Default, Production configurations
- **Hyperparameter Sweeps**: Easy parameter optimization with multirun
- **Experiment Tracking**: Organized output directories with automatic logging
- **Configuration Overrides**: Command-line parameter modification

## 📁 **Configuration Structure**
```
conf/
├── config.yaml              # Main configuration
├── model/                   # Model configurations
│   ├── collaborative.yaml  # Matrix factorization
│   ├── content_based.yaml  # Genre-based model
│   ├── hybrid.yaml         # Combined approach  
│   └── deep_cf.yaml        # Deep neural network
├── optimizer/               # Optimizer configurations
│   ├── adam.yaml           # Adam optimizer settings
│   └── sgd.yaml            # SGD optimizer settings
├── train/                   # Training configurations
│   ├── default.yaml        # Standard training
│   ├── fast.yaml           # Quick experiments
│   └── production.yaml     # Thorough training
├── data/                    # Data configurations
│   └── default.yaml        # Data processing settings
└── experiment/              # Experiment presets
    ├── quick.yaml          # Fast testing
    └── model_comparison.yaml # Model benchmarking
```

## 🔧 **Usage Examples**

### **Basic Training**
```bash
# Train with default settings (Collaborative Filtering, 20 epochs, Adam optimizer)
python train_hydra.py

# Train specific model with specific optimizer
python train_hydra.py model=deep_cf optimizer=sgd
python train_hydra.py model=hybrid optimizer=adam
python train_hydra.py model=content_based
```

### **Optimizer Configuration**
```bash
# Use Adam optimizer (default)
python train_hydra.py model=collaborative optimizer=adam

# Use SGD optimizer with momentum
python train_hydra.py model=deep_cf optimizer=sgd

# Compare optimizers
python train_hydra.py -m optimizer=adam,sgd model=collaborative
```

### **Override Parameters**
```bash
# Change training settings
python train_hydra.py train.epochs=50 train.learning_rate=0.001

# Change model settings
python train_hydra.py model=collaborative model.embedding_dim=100

# Change data settings  
python train_hydra.py data.rating_threshold=3.5 data.negative_sampling_ratio=6
```

### **Use Different Training Configurations**
```bash
# Fast training (10 epochs, large batch)
python train_hydra.py train=fast

# Production training (100 epochs, careful tuning)
python train_hydra.py train=production

# Custom combination
python train_hydra.py model=deep_cf train=production
```

### **Run Multiple Experiments (Multirun)**
```bash
# Compare all models
python train_hydra.py -m model=collaborative,content_based,hybrid,deep_cf

# Hyperparameter sweep
python train_hydra.py -m train.learning_rate=0.001,0.01,0.1 train.batch_size=256,512

# Model comparison with consistent settings
python train_hydra.py -m model=collaborative,deep_cf train.epochs=30 train.learning_rate=0.005
```

### **Experiment Configurations**
```bash
# Quick test experiment
python train_hydra.py experiment=quick

# Model comparison benchmark
python train_hydra.py -m experiment=model_comparison model=collaborative,hybrid,deep_cf
```

## 📊 **Output Organization**

### **Single Run**
```
outputs/movie_recommendation/2025-09-28_18-30-45/
├── .hydra/                  # Hydra configuration files
├── best_model_deep_cf.pth   # Saved model
├── main.log                 # Training logs  
└── hydra.yaml              # Used configuration
```

### **Multirun**
```
multirun/movie_recommendation/2025-09-28_18-30-45/
├── 0_collaborative/         # First experiment
├── 1_deep_cf/              # Second experiment  
├── 2_hybrid/               # Third experiment
└── ...
```

## 🎯 **Advanced Examples**

### **Comprehensive Model Comparison**
```bash
python train_hydra.py -m \
  model=collaborative,content_based,hybrid,deep_cf \
  optimizer=adam \
  train.epochs=50 \
  train.learning_rate=0.005 \
  train.batch_size=256
```

### **Optimizer Comparison**
```bash
python train_hydra.py -m \
  optimizer=adam,sgd \
  model=deep_cf \
  train.epochs=30 \
  train.batch_size=256
```

### **Hyperparameter Optimization**
```bash
python train_hydra.py -m \
  model=deep_cf \
  optimizer=adam,sgd \
  model.embedding_dim=32,64,128 \
  train.learning_rate=0.001,0.01,0.1 \
  train.dropout=0.2,0.4,0.6
```

### **Data Sensitivity Analysis**  
```bash
python train_hydra.py -m \
  data.rating_threshold=2.5,3.0,3.5,4.0 \
  data.negative_sampling_ratio=2,4,6,8 \
  model=collaborative
```

## 🔍 **Configuration Details**

### **Available Models**
- `collaborative`: Matrix factorization (50D embeddings)
- `content_based`: Genre-based recommendations  
- `hybrid`: Combined collaborative + content
- `deep_cf`: Deep neural collaborative filtering

### **Available Optimizers** 
- `adam`: Adam optimizer with adaptive learning rates (default)
- `sgd`: SGD with momentum support (simple and effective)

### **Training Configurations**
- `default`: 20 epochs, balanced settings
- `fast`: 10 epochs, quick experiments  
- `production`: 100 epochs, thorough training

### **Key Parameters**
- `optimizer`: Optimizer type (adam/sgd)
- `train.epochs`: Number of training epochs
- `train.batch_size`: Training batch size
- `train.learning_rate`: Learning rate
- `model.embedding_dim`: Model embedding dimensions
- `data.rating_threshold`: Rating conversion threshold
- `train.patience`: Early stopping patience

## 🎉 **Benefits**

✅ **Reproducible**: All configurations saved automatically  
✅ **Scalable**: Easy to run hundreds of experiments  
✅ **Organized**: Clean output directory structure  
✅ **Flexible**: Override any parameter from command line  
✅ **Custom Made**: Tailored experiment management system  

Your recommendation system now has enterprise-grade experiment infrastructure! 🚀

## 📈 **System Evolution**

### **Phase 1: Basic Implementation**
- Single model (Collaborative Filtering)
- Basic training script (`train.py`)
- Simple configuration management

### **Phase 2: Custom Made Architecture**
- 4 Model architectures (Collaborative, Content-based, Hybrid, Deep CF)
- Modular package structure (models/, data/, losses/, metrics/, utils/)
- Enhanced configuration system (config/ directory)
- Professional logging and visualization

### **Phase 3: Hydra Integration** ✅ **NEW!**
- Advanced experiment management with Hydra
- YAML-based configuration system (`conf/` directory)
- Multi-run capabilities for model comparison
- Hyperparameter sweep automation
- Organized experiment outputs

### **Current Capabilities**
- **3 Training Scripts**: `train.py`, `train_simple.py`, `train_hydra.py`
- **Dual Config Systems**: Traditional Python configs + Hydra YAML configs
- **4 Model Types**: Ready for immediate experimentation
- **Multiple Training Modes**: Single run, multi-run, hyperparameter sweeps