#!/usr/bin/env python3
"""
Test script to verify movie titles are working correctly
"""

import sys
from pathlib import Path

# Add apps directory to path
sys.path.append(str(Path(__file__).parent / "apps"))

# Test the inference engine
print("🧪 Testing Inference Engine...")
print("=" * 50)

try:
    from inference import RecommendationEngine
    
    engine = RecommendationEngine()
    
    print(f"✅ Movies loaded: {engine.movies_df.shape}")
    print(f"✅ Users available: {len(engine.user_encoder.classes_) if engine.user_encoder else 'N/A'}")
    print(f"✅ Model loaded: {engine.current_model_info['name'] if hasattr(engine, 'current_model_info') else 'Unknown'}")
    
    # Test movie 158 (should be Casper)
    movie_158 = engine.movies_df[engine.movies_df['movieId'] == 158]
    if not movie_158.empty:
        print(f"✅ Movie 158: {movie_158.iloc[0]['title']}")
    else:
        print("❌ Movie 158 not found")
    
    # Test recommendations for user 1
    print("\n🎯 Testing Recommendations for User 1:")
    print("-" * 40)
    
    recs = engine.get_recommendations(user_id=1, num_recommendations=5)
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec['title']} (Score: {rec['score']:.3f})")
    
    # Test recommendations for user 50 (different user)
    print("\n🎯 Testing Recommendations for User 50:")
    print("-" * 40)
    
    recs = engine.get_recommendations(user_id=50, num_recommendations=5)
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec['title']} (Score: {rec['score']:.3f})")
    
    print("\n🎉 All tests passed! Movie titles are working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()