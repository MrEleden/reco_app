# MovieLens Recommendation System

A comprehensive PyTorch-based recommendation system implementing collaborative filtering, content-based filtering, and hybrid approaches. Designed for easy deployment on Hugging Face.

## 🚀 Features

- **Multiple Model Types**:
  - Collaborative Filtering (Matrix Factorization)
  - Content-Based Filtering
  - Hybrid Model (Combines both approaches)
  - Deep Collaborative Filtering

- **Production Ready**:
  - Comprehensive evaluation metrics
  - Gradio web interface
  - Hugging Face deployment utilities
  - Configurable training pipeline

- **Dataset**: MovieLens Small (100k ratings, 610 users, 9,742 movies)

## 📊 Model Performance

Our models are evaluated using multiple metrics:
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **MAP@k**: Mean Average Precision
- **Hit Rate@k**: Fraction of users with at least one relevant item in top-k
- **RMSE/MAE**: For rating prediction accuracy

## 🛠️ Installation

```bash
git clone https://github.com/MrEleden/reco_app.git
cd reco_app
pip install -r requirements.txt
```

## 🎯 Quick Start

### 1. Train a Model

```bash
# Train a collaborative filtering model
python src/main.py --model_type collaborative --epochs 50 --batch_size 512

# Train a hybrid model
python src/main.py --model_type hybrid --epochs 100 --use_wandb

# Train with custom configuration
python src/main.py --config config.json --model_type deep_cf
```

### 2. Use the Trained Model

```python
from src import MovieLensDataLoader, create_model, RecommenderEvaluator

# Load data
data_loader = MovieLensDataLoader('data')
data_loader.load_data()
data_dict = data_loader.preprocess_data()

# Load trained model
import torch
model = create_model('collaborative', {
    'n_users': data_dict['n_users'],
    'n_movies': data_dict['n_movies'],
    'n_factors': 50
})
checkpoint = torch.load('models/best_collaborative_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate recommendations
evaluator = RecommenderEvaluator(model)
recommendations = evaluator.generate_recommendations(
    user_id=1, n_recommendations=10,
    user_encoder=data_loader.user_encoder,
    movie_encoder=data_loader.movie_encoder,
    all_movie_ids=data_loader.movies['movieId'].tolist()
)
```

### 3. Launch Web Interface

```python
from src.huggingface_utils import HuggingFaceModelWrapper, create_gradio_interface

# Create model wrapper
wrapper = HuggingFaceModelWrapper('models/best_collaborative_model.pt', 'collaborative')

# Launch Gradio interface
interface = create_gradio_interface(wrapper)
interface.launch()
```

## 📁 Project Structure

```
reco_app/
├── data/                          # MovieLens dataset
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   ├── links.csv
│   └── README.txt
├── src/                           # Source code
│   ├── __init__.py
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── models.py                 # PyTorch model implementations
│   ├── trainer.py                # Training utilities
│   ├── evaluator.py              # Evaluation metrics
│   ├── main.py                   # Training script
│   └── huggingface_utils.py      # HF deployment utilities
├── models/                        # Saved models (created after training)
├── config.json                    # Training configuration
├── requirements.txt               # Dependencies
└── README.md
```

## 🔧 Configuration

Edit `config.json` to customize training parameters:

```json
{
  "model_configs": {
    "collaborative": {
      "n_factors": 50,
      "dropout": 0.2
    },
    "hybrid": {
      "n_factors": 50,
      "hidden_size": 128,
      "dropout": 0.2
    }
  },
  "training": {
    "epochs": 100,
    "batch_size": 512,
    "learning_rate": 0.001,
    "weight_decay": 1e-5
  }
}
```

## 📈 Model Architectures

### Collaborative Filtering
- Matrix factorization with user/movie embeddings
- Bias terms for users and movies
- Dropout for regularization

### Content-Based Filtering
- Incorporates movie genre information
- Neural network for feature combination
- User preferences learning

### Hybrid Model
- Combines collaborative and content-based approaches
- Attention mechanism for feature weighting
- Deep neural networks for complex patterns

### Deep Collaborative Filtering
- Multi-layer neural network
- Batch normalization and dropout
- Non-linear feature interactions

## 🚀 Deployment to Hugging Face

### Upload Model to Hub

```python
from src.huggingface_utils import upload_to_huggingface

upload_to_huggingface(
    model_path='models/best_hybrid_model.pt',
    model_type='hybrid',
    repo_name='your-username/movie-recommender',
    token='your-hf-token'
)
```

### Deploy to Spaces

1. Create a new Space on Hugging Face
2. Use the generated `app.py` file:

```python
from src.huggingface_utils import create_app_py

create_app_py('models/best_hybrid_model.pt', 'hybrid')
```

## 📊 Evaluation Metrics

The system provides comprehensive evaluation:

- **Ranking Metrics**: NDCG@k, MAP@k, Hit Rate@k
- **Rating Prediction**: RMSE, MAE
- **Cold-Start Performance**: Metrics for new users
- **Diversity**: Intra-list diversity and catalog coverage

## 🎯 Use Cases

- **Movie Streaming Platforms**: Personalized movie recommendations
- **E-commerce**: Product recommendations using similar techniques
- **Content Discovery**: Help users find relevant content
- **Research**: Benchmark for recommendation algorithms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- MovieLens dataset provided by GroupLens Research
- PyTorch team for the deep learning framework
- Hugging Face for deployment infrastructure

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Recommending! 🎬✨**