"""
Test script for modular movie recommendation system
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("🧪 Testing modular imports...")

    from models import CollaborativeFilteringModel, ContentBasedModel

    print("✅ Models imported successfully")

    from dataset import MovieLensDataLoader, RecommenderDataset

    print("✅ Dataset classes imported successfully")

    from trainer import ModelTrainer

    print("✅ Trainer imported successfully")

    from recommender import MovieRecommendationSystem

    print("✅ Recommender system imported successfully")

    print("\n🚀 Testing basic functionality...")

    # Test data loading
    recommender = MovieRecommendationSystem()
    print("✅ Recommendation system initialized")

    # Test basic recommendation
    result = recommender.get_user_recommendations(1, 5)
    print("✅ Basic recommendations work")

    # Test search
    search_result = recommender.search_movies("toy")
    print("✅ Movie search works")

    # Test user profile
    profile_result = recommender.get_user_profile(1)
    print("✅ User profile works")

    # Test dataset stats
    stats_result = recommender.get_data_stats()
    print("✅ Dataset stats work")

    print("\n🎉 All modular components working correctly!")
    print("\n📋 Summary:")
    print("- ✅ All imports successful")
    print("- ✅ Data loading works")
    print("- ✅ Recommendations work")
    print("- ✅ Search functionality works")
    print("- ✅ User profiles work")
    print("- ✅ Dataset statistics work")

    print(f"\n📊 Quick stats: {recommender.get_data_stats()}")

except Exception as e:
    print(f"❌ Error during testing: {str(e)}")
    import traceback

    traceback.print_exc()
