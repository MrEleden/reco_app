#!/usr/bin/env python3
"""
Movie Recommendation System - Command Line Interface
Train and test neural network models with command line arguments
"""

import argparse
import torch
import os
import sys
import time
from torch.utils.data import DataLoader

# Import our core modules
from dataset import MovieLensDataLoader, RecommenderDataset
from trainer import ModelTrainer


def train_model(args):
    """Train a recommendation model."""
    print("🎬 Movie Recommendation Neural Network Training")
    print("=" * 50)
    
    print(f"📊 Training Configuration:")
    print(f"   • Model type: {args.model}")
    print(f"   • Epochs: {args.epochs}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Learning rate: {args.learning_rate}")
    print(f"   • Embedding dimension: {args.embedding_dim}")
    print(f"   • Dropout rate: {args.dropout}")
    print(f"   • Data directory: {args.data_dir}")
    print(f"   • Output directory: {args.output_dir}")
    print()
    
    # Load data
    print("📂 Loading MovieLens dataset...")
    data_loader = MovieLensDataLoader(args.data_dir)
    
    # Get collaborative filtering data
    user_movie_pairs, model_config = data_loader.get_collaborative_data()
    if user_movie_pairs is None:
        print("❌ No training data available")
        return False
    
    # Update config with model parameters
    model_config.update({
        "n_factors": args.embedding_dim,
        "dropout": args.dropout
    })
    
    print(f"✅ Data loaded successfully:")
    print(f"   • Users: {model_config['n_users']:,}")
    print(f"   • Movies: {model_config['n_movies']:,}")
    print(f"   • Ratings: {len(user_movie_pairs):,}")
    print()
    
    # Create train/validation split
    print("🔄 Creating train/validation datasets...")
    train_pairs, val_pairs = data_loader.create_train_val_split(user_movie_pairs)
    
    train_dataset = RecommenderDataset(
        train_pairs, 
        negative_sampling=True, 
        n_users=model_config["n_users"], 
        n_movies=model_config["n_movies"]
    )
    
    val_dataset = RecommenderDataset(
        val_pairs, 
        negative_sampling=True, 
        n_users=model_config["n_users"], 
        n_movies=model_config["n_movies"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"✅ Datasets created:")
    print(f"   • Training samples: {len(train_dataset):,}")
    print(f"   • Validation samples: {len(val_dataset):,}")
    print(f"   • Training batches: {len(train_loader):,}")
    print()
    
    # Initialize trainer
    print("🚀 Initializing neural network trainer...")
    trainer = ModelTrainer(model_type=args.model)
    
    if not trainer.prepare_model(model_config):
        print("❌ Failed to prepare model")
        return False
    
    trainer.prepare_training(learning_rate=args.learning_rate)
    
    print(f"✅ Model initialized:")
    print(f"   • Model type: {args.model}")
    print(f"   • Device: {trainer.device}")
    print(f"   • Parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print()
    
    # Training loop
    print("🎯 Starting training...")
    print("=" * 70)
    
    best_val_loss = float("inf")
    patience_counter = 0
    patience = args.patience
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Training phase
        trainer.model.train()
        total_train_loss = 0.0
        n_train_batches = 0
        
        for batch_idx, (user_ids, movie_ids, labels) in enumerate(train_loader):
            user_ids = user_ids.to(trainer.device)
            movie_ids = movie_ids.to(trainer.device)
            labels = labels.to(trainer.device)
            
            trainer.optimizer.zero_grad()
            predictions = trainer.model(user_ids, movie_ids)
            loss = trainer.criterion(predictions, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
            trainer.optimizer.step()
            
            total_train_loss += loss.item()
            n_train_batches += 1
        
        train_loss = total_train_loss / n_train_batches
        
        # Validation phase
        trainer.model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for user_ids, movie_ids, labels in val_loader:
                user_ids = user_ids.to(trainer.device)
                movie_ids = movie_ids.to(trainer.device)
                labels = labels.to(trainer.device)
                
                predictions = trainer.model(user_ids, movie_ids)
                loss = trainer.criterion(predictions, labels)
                total_val_loss += loss.item()
                n_val_batches += 1
        
        val_loss = total_val_loss / n_val_batches
        
        # Learning rate scheduling
        trainer.scheduler.step(val_loss)
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        
        # Save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(args.output_dir, f"best_{args.model}_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "model_type": args.model,
                "n_users": model_config["n_users"],
                "n_movies": model_config["n_movies"],
                "user_encoder": model_config["user_encoder"],
                "movie_encoder": model_config["movie_encoder"],
                "config": {
                    "embedding_dim": args.embedding_dim,
                    "dropout_rate": args.dropout,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size
                }
            }, model_path, _use_new_zipfile_serialization=False)
            
            best_marker = " 🌟 NEW BEST!"
        else:
            patience_counter += 1
            best_marker = ""
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Print progress
        progress_line = (
            f"Epoch {epoch+1:2d}/{args.epochs} │ "
            f"Train: {train_loss:.4f} │ "
            f"Val: {val_loss:.4f} │ "
            f"LR: {current_lr:.1e} │ "
            f"Time: {epoch_time:.1f}s{best_marker}"
        )
        print(progress_line)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training summary
    total_time = time.time() - start_time
    print("=" * 70)
    print("🎉 Training completed!")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
    print(f"📁 Model saved to: {model_path}")
    
    # Calculate improvement
    if len(trainer.train_losses) > 0:
        initial_loss = trainer.train_losses[0] if trainer.train_losses else train_loss
        improvement = ((initial_loss - train_loss) / initial_loss * 100)
        print(f"📈 Training improvement: {improvement:.1f}%")
    
    print("\n✅ Training successful! Model ready for recommendations.")
    return True


def test_model(args):
    """Test a trained model."""
    print("🧪 Testing trained model...")
    print("=" * 40)
    
    model_path = os.path.join(args.output_dir, f"best_{args.model}_model.pt")
    if not os.path.exists(model_path):
        print(f"❌ No trained model found at: {model_path}")
        print("   Run training first with: python main.py train")
        return False
    
    try:
        # Load model
        model_state = torch.load(model_path, map_location="cpu", weights_only=False)
        
        from models import CollaborativeFilteringModel
        model = CollaborativeFilteringModel(
            n_users=model_state["n_users"],
            n_movies=model_state["n_movies"],
            n_factors=model_state["config"]["embedding_dim"],
            dropout=model_state["config"]["dropout_rate"]
        )
        model.load_state_dict(model_state["model_state_dict"])
        model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"   • Architecture: Collaborative Filtering")
        print(f"   • Users: {model_state['n_users']:,}")
        print(f"   • Movies: {model_state['n_movies']:,}")
        print(f"   • Embedding dim: {model_state['config']['embedding_dim']}")
        print(f"   • Best validation loss: {model_state['best_val_loss']:.4f}")
        print(f"   • Training epochs: {model_state['epoch'] + 1}")
        
        # Test predictions
        print("\n🔮 Sample predictions:")
        with torch.no_grad():
            for i in range(5):
                user_id = torch.LongTensor([i])
                movie_id = torch.LongTensor([i * 10])
                prediction = model(user_id, movie_id).item()
                print(f"   • User {i+1}, Movie {i*10+1}: {prediction:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        return False


def get_recommendations(args):
    """Get recommendations for a user."""
    print(f"🎭 Getting recommendations for User {args.user_id}")
    print("=" * 50)
    
    # Load data
    data_loader = MovieLensDataLoader(args.data_dir)
    movies_df = data_loader.movies_df
    ratings_df = data_loader.ratings_df
    
    # Load model
    model_path = os.path.join(args.output_dir, f"best_{args.model}_model.pt")
    if not os.path.exists(model_path):
        print(f"❌ No trained model found at: {model_path}")
        return False
    
    try:
        model_state = torch.load(model_path, map_location="cpu", weights_only=False)
        
        from models import CollaborativeFilteringModel
        model = CollaborativeFilteringModel(
            n_users=model_state["n_users"],
            n_movies=model_state["n_movies"],
            n_factors=model_state["config"]["embedding_dim"],
            dropout=model_state["config"]["dropout_rate"]
        )
        model.load_state_dict(model_state["model_state_dict"])
        model.eval()
        
        user_encoder = model_state["user_encoder"]
        movie_encoder = model_state["movie_encoder"]
        
        # Check if user exists
        if args.user_id not in user_encoder.classes_:
            print(f"❌ User {args.user_id} not in training data")
            available_users = sorted(user_encoder.classes_)[:10]
            print(f"   Available users (sample): {available_users}")
            return False
        
        # Get user's rated movies
        user_ratings = ratings_df[ratings_df["userId"] == args.user_id]
        rated_movies = set(user_ratings["movieId"].values)
        
        print(f"👤 User {args.user_id} has rated {len(rated_movies)} movies")
        
        # Get all movies
        all_movies = set(movies_df["movieId"].values)
        unrated_movies = list(all_movies - rated_movies)
        
        # Predict ratings
        predictions = []
        user_encoded = user_encoder.transform([args.user_id])[0]
        
        model.eval()
        with torch.no_grad():
            for movie_id in unrated_movies:
                if movie_id in movie_encoder.classes_:
                    movie_encoded = movie_encoder.transform([movie_id])[0]
                    user_tensor = torch.LongTensor([user_encoded])
                    movie_tensor = torch.LongTensor([movie_encoded])
                    
                    pred = model(user_tensor, movie_tensor).item()
                    predictions.append((movie_id, pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:args.num_recommendations]
        
        print(f"\n🤖 Top {len(top_predictions)} AI Recommendations:")
        print("-" * 60)
        
        for i, (movie_id, pred_rating) in enumerate(top_predictions, 1):
            movie_info = movies_df[movies_df["movieId"] == movie_id]
            if not movie_info.empty:
                movie_row = movie_info.iloc[0]
                print(f"{i:2d}. {movie_row['title']}")
                print(f"     🧠 AI Predicted Rating: {pred_rating:.2f}/5")
                print(f"     🎭 Genres: {movie_row['genres']}")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting recommendations: {str(e)}")
        return False


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Movie Recommendation System - Neural Network Training and Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model with default settings
  python main.py train
  
  # Train with custom parameters
  python main.py train --epochs 50 --batch-size 512 --learning-rate 0.005
  
  # Test a trained model
  python main.py test
  
  # Get recommendations for user 42
  python main.py recommend --user-id 42
  
  # Get 20 recommendations for user 15
  python main.py recommend --user-id 15 --num-recommendations 20
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train a recommendation model")
    train_parser.add_argument("--model", default="collaborative", choices=["collaborative"],
                            help="Model type to train (default: collaborative)")
    train_parser.add_argument("--epochs", type=int, default=20,
                            help="Number of training epochs (default: 20)")
    train_parser.add_argument("--batch-size", type=int, default=256,
                            help="Batch size for training (default: 256)")
    train_parser.add_argument("--learning-rate", type=float, default=0.01,
                            help="Learning rate for optimization (default: 0.01)")
    train_parser.add_argument("--embedding-dim", type=int, default=50,
                            help="Embedding dimension (default: 50)")
    train_parser.add_argument("--dropout", type=float, default=0.2,
                            help="Dropout rate (default: 0.2)")
    train_parser.add_argument("--patience", type=int, default=10,
                            help="Early stopping patience (default: 10)")
    train_parser.add_argument("--data-dir", default="data",
                            help="Directory containing MovieLens data (default: data)")
    train_parser.add_argument("--output-dir", default="models",
                            help="Directory to save trained models (default: models)")
    
    # Testing command
    test_parser = subparsers.add_parser("test", help="Test a trained model")
    test_parser.add_argument("--model", default="collaborative", choices=["collaborative"],
                           help="Model type to test (default: collaborative)")
    test_parser.add_argument("--output-dir", default="models",
                           help="Directory containing trained models (default: models)")
    
    # Recommendation command
    rec_parser = subparsers.add_parser("recommend", help="Get recommendations for a user")
    rec_parser.add_argument("--user-id", type=int, required=True,
                          help="User ID to get recommendations for")
    rec_parser.add_argument("--num-recommendations", type=int, default=10,
                          help="Number of recommendations to generate (default: 10)")
    rec_parser.add_argument("--model", default="collaborative", choices=["collaborative"],
                          help="Model type to use (default: collaborative)")
    rec_parser.add_argument("--data-dir", default="data",
                          help="Directory containing MovieLens data (default: data)")
    rec_parser.add_argument("--output-dir", default="models",
                          help="Directory containing trained models (default: models)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        success = train_model(args)
        sys.exit(0 if success else 1)
    
    elif args.command == "test":
        success = test_model(args)
        sys.exit(0 if success else 1)
    
    elif args.command == "recommend":
        success = get_recommendations(args)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("   python main.py train              # Train a model")
        print("   python main.py test               # Test the model") 
        print("   python main.py recommend --user-id 42  # Get recommendations")


if __name__ == "__main__":
    main()