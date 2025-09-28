# 🚀 Hydra Multirun Quick Reference

## **🎯 Most Useful Commands**

### **1. Quick Model Comparison (RECOMMENDED)**
```bash
# Fast comparison with 5 epochs (currently running!)
python train_hydra.py -m model=collaborative,hybrid train.epochs=5

# Full comparison with default epochs (20)
python train_hydra.py -m model=collaborative,hybrid,deep_cf
```

### **2. Hyperparameter Optimization**
```bash
# Learning rate sweep (best model)
python train_hydra.py -m model=hybrid train.learning_rate=0.001,0.01,0.1

# Batch size optimization
python train_hydra.py -m model=hybrid train.batch_size=128,256,512

# Combined hyperparameter grid
python train_hydra.py -m model=hybrid train.learning_rate=0.001,0.01 train.batch_size=128,256
```

### **3. Architecture Experiments**
```bash
# Embedding dimension sweep
python train_hydra.py -m model=collaborative model.embedding_dim=32,64,128,256

# Optimizer comparison
python train_hydra.py -m model=hybrid optimizer=adam,sgd

# Regularization study
python train_hydra.py -m model=hybrid train.weight_decay=0.0001,0.001,0.01
```

### **4. View Results**
```bash
# Quick overview
python check_mlflow.py

# Detailed analysis
python mlflow_simple_guide.py

# MLflow Web UI
python -m mlflow ui --port 5000
# Open: http://127.0.0.1:5000
```

## **💡 Pro Tips**

### **✅ Best Practices**
- **Start small**: Use `train.epochs=5` for quick tests
- **Use `-m` flag**: Required for multirun mode
- **Check progress**: `python check_mlflow.py` while running
- **Focus on hybrid**: It's your best performing model

### **⚡ Multirun Syntax**
- **Single parameter**: `model=hybrid,collaborative`
- **Multiple parameters**: `model=hybrid train.epochs=5,10`
- **Grid search**: Each combination runs separately
- **Output**: Organized in `outputs/movie_recommendation/multirun/`

### **🎯 Current Status**
- **✅ RUNNING**: `collaborative,hybrid` comparison with 5 epochs
- **📊 Progress**: Check MLflow at http://127.0.0.1:5000
- **🔄 Next**: Try hyperparameter sweeps on the best model

---
**🎬 Happy multirun experimenting! 🚀**