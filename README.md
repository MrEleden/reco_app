# 🎬 Movie Recommendation System

Minimal, production-minded PyTorch recommender. Start simple, scale with Hydra, track with MLflow, and tune with Optuna. A Streamlit app provides an interactive demo.

## Stack
PyTorch • Hydra • MLflow • Optuna • Streamlit

## Project Structure
```
reco_app/
├── train.py              # Baseline (pure PyTorch)
├── train_hydra.py        # Hydra + MLflow (+ Optuna)
├── requirements.txt      # Core dependencies
├── conf/                 # Hydra configurations
├── apps/                 # Interactive applications
│   ├── app.py           # Streamlit web app
│   └── app_gradio.py    # Gradio demo app
├── scripts/             # Utility scripts
├── deployment/          # Deployment packages & configs
├── weights/             # Model weights storage
└── data/                # Dataset files
```

## Quick Start
```bash
# 1) Install
git clone <repo-url> reco_app && cd reco_app
pip install -r requirements.txt

# 2) Baseline (from scratch)
python train.py

# 3) Structured training (Hydra + MLflow)
python train_hydra.py model=hybrid train=fast

# 4) Compare multiple models
python train_hydra.py -m model=collaborative,hybrid,deep_cf train.epochs=5

# 5) Hyperparameter optimization (Optuna via Hydra sweeper)
python train_hydra.py -m model=collaborative,content_based,hybrid,deep_cf   train=production hydra/sweeper=optuna_production

# 6) Inspect results (MLflow UI)
python -m mlflow ui --port 5000

# 6b) Quick MLflow summary
python scripts/check_mlflow.py

# 7) Demo apps
python apps/app.py          # Full-featured web app
python apps/app_gradio.py          # Clean ML demo (HF Spaces ready)
```

## Configuration (Hydra)
- All settings live in `conf/` (e.g., `conf/model/*.yaml`, `conf/train/*.yaml`).
- Override any value from the CLI without editing code:
```bash
python train_hydra.py model=hybrid train.learning_rate=0.001 train.epochs=10
```
