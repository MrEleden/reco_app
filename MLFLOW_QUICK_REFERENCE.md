# 🎯 MLflow Quick Reference Guide

## 🚀 **Most Common Commands**

### **Training & Experiments**
```bash
# Single model training
python train_hydra.py model=hybrid

# Compare all models  
python train_hydra.py -m model=collaborative,content_based,hybrid,deep_cf

# Hyperparameter sweep (recommended)
python train_hydra.py -m model=hybrid train.learning_rate=0.001,0.01,0.1
```

### **View Results**
```bash
# Quick overview
python check_mlflow.py

# Detailed analysis
python mlflow_simple_guide.py

# Web interface
python -m mlflow ui --port 5000
# Open: http://127.0.0.1:5000
```

## 🏆 **Current Best Performance**
- **Best Model**: `hybrid` (RMSE: **0.3255**)
- **Best Accuracy**: **85.1%**
- **Total Experiments**: 6 completed runs

## 📂 **File Organization**
- `train_hydra.py` - Main training script
- `check_mlflow.py` - Your custom experiment viewer  
- `mlflow_simple_guide.py` - Comprehensive analysis
- `mlflow_workflows.py` - Advanced examples
- `mlruns/` - MLflow tracking data
- `model_comparison_report.md` - Latest results

## 🔥 **Pro Tips**
1. **Always use multirun (`-m`)** for comparing options
2. **Check MLflow UI** for visual experiment comparison
3. **Hybrid models perform best** - focus hyperparameter tuning there
4. **All experiments are automatically tracked** - no manual logging needed
5. **Best models are saved automatically** - load them with `MLflowModelSelector`

## 🎯 **Next Experiments to Try**
```bash
# Fine-tune the best model
python train_hydra.py -m model=hybrid train.batch_size=128,256,512

# Try different optimizers
python train_hydra.py -m model=hybrid optimizer=adam,sgd

# Architecture variations
python train_hydra.py -m model=hybrid model.embedding_dim=64,128,256
```

---
**🎬 Happy experimenting with MLflow! 🚀**