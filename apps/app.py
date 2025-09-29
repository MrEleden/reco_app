#!/usr/bin/env python3
"""
🎬 Simple Movie Recommendation App
=================================

Clean Streamlit interface using global inference API.
No complex MLflow handling, encoder management, or model loading logic.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict
import sys
from pathlib import Path

# Add parent directory to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our global inference engine
try:
    from inference import get_recommendation_engine, recommend_movies, get_user_list
    INFERENCE_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import inference module: {e}")
    INFERENCE_AVAILABLE = False

# Page configuration
st.set_page_config(page_title="🎬 Movie Recommendations", page_icon="🎬", layout="wide")


def create_star_rating(score: float) -> str:
    """Create a star rating display based on score."""
    # Normalize score to 0-5 star range
    stars = int(score * 5)
    full_stars = "⭐" * stars
    empty_stars = "☆" * (5 - stars)
    return full_stars + empty_stars


def main():
    """Main app interface."""

    # Header
    st.title("🎬 Movie Recommendation System")
    st.markdown("**Powered by PyTorch + Hydra + MLflow + Optuna**")
    st.markdown("*Simple interface with global inference API*")
    st.markdown("---")

    if not INFERENCE_AVAILABLE:
        st.error("❌ Inference engine not available. Please check the setup.")
        return

    # Initialize the global inference engine
    with st.spinner("🚀 Initializing recommendation engine..."):
        try:
            engine = get_recommendation_engine()
            model_info = engine.get_model_info()
        except Exception as e:
            st.error(f"❌ Failed to initialize recommendation engine: {e}")
            return

    # Sidebar with model info
    with st.sidebar:
        st.header("🤖 Model Status")

        if model_info["model_loaded"]:
            st.success("✅ Model loaded")
        else:
            st.error("❌ No model loaded")

        if model_info["encoders_loaded"]:
            st.success("✅ Encoders ready")
        else:
            st.error("❌ Encoders not available")

        st.metric("Users", model_info["num_users"])
        st.metric("Movies", model_info["num_movies"])
        st.info(f"Device: {model_info['device']}")
        
        # Model selection
        st.markdown("---")
        st.header("🎯 Model Selection")
        
        # Current model info
        if 'name' in model_info:
            st.info(f"**Current Model:** {model_info['name']}")
            if 'rmse' in model_info:
                st.metric("RMSE", f"{model_info['rmse']:.4f}")
            if 'accuracy' in model_info:
                st.metric("Accuracy", f"{model_info['accuracy']:.3f}")
        
        # Available models
        available_models = engine.get_available_models()
        if available_models:
            st.markdown("**Available Models:**")
            
            model_options = {}
            for model in available_models:
                display_name = f"{model['name']} (RMSE: {model['rmse']:.4f})"
                model_options[display_name] = model['run_id']
            
            # Current selection
            current_display = None
            if 'run_id' in model_info:
                for display, run_id in model_options.items():
                    if run_id == model_info['run_id']:
                        current_display = display
                        break
            
            # Model selector
            selected_display = st.selectbox(
                "Choose Model:",
                options=list(model_options.keys()),
                index=list(model_options.keys()).index(current_display) if current_display else 0,
                help="Select a different model to use for recommendations"
            )
            
            # Load selected model
            selected_run_id = model_options[selected_display]
            if 'run_id' not in model_info or selected_run_id != model_info['run_id']:
                if st.button("🔄 Load Selected Model"):
                    with st.spinner("Loading model..."):
                        if engine.load_model_by_run_id(selected_run_id):
                            st.success("✅ Model loaded successfully!")
                            st.experimental_rerun()
                        else:
                            st.error("❌ Failed to load model")
        
        # Available users info
        if model_info["encoders_loaded"]:
            available_users = engine.get_available_users()
            st.markdown("---")
            st.markdown(f"**User ID Range:** {min(available_users)} - {max(available_users)}")

    # Main interface
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("🎯 Get Recommendations")

        # User input
        if model_info["encoders_loaded"]:
            available_users = engine.get_available_users()
            default_user = available_users[len(available_users) // 2] if available_users else 1
            user_id = st.number_input(
                "User ID",
                min_value=min(available_users),
                max_value=max(available_users),
                value=default_user,
                help=f"Enter a user ID between {min(available_users)} and {max(available_users)}",
            )
        else:
            user_id = st.number_input("User ID", min_value=1, max_value=610, value=123)

        num_recommendations = st.slider("Number of Recommendations", 1, 20, 10)

        # Generate recommendations button
        if st.button("🎬 Get Recommendations", type="primary"):
            with st.spinner("🔮 Generating personalized recommendations..."):
                try:
                    recommendations = recommend_movies(user_id, num_recommendations)

                    if recommendations:
                        st.session_state.recommendations = recommendations
                        st.session_state.current_user = user_id
                        st.success(f"✅ Generated {len(recommendations)} recommendations!")
                    else:
                        st.error("❌ No recommendations generated")

                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {e}")

    with col2:
        st.header("🍿 Your Personalized Recommendations")

        # Display recommendations if available
        if hasattr(st.session_state, "recommendations") and st.session_state.recommendations:
            st.markdown(f"**For User ID: {st.session_state.current_user}**")

            # Create a nice display
            for i, rec in enumerate(st.session_state.recommendations, 1):
                with st.container():
                    rec_col1, rec_col2, rec_col3 = st.columns([0.5, 3, 1])

                    with rec_col1:
                        st.markdown(f"**#{i}**")

                    with rec_col2:
                        st.markdown(f"**{rec['title']}**")
                        if "genres" in rec:
                            st.caption(rec["genres"])

                    with rec_col3:
                        stars = create_star_rating(rec["score"])
                        st.markdown(stars)
                        st.caption(f"{rec['score']:.3f}")

                    st.markdown("---")
        else:
            st.info("👆 Enter a User ID and click 'Get Recommendations' to see personalized movie suggestions!")

    # Footer
    st.markdown("---")
    st.markdown("**Tech Stack:** PyTorch • Hydra • MLflow • Optuna • Streamlit")


if __name__ == "__main__":
    main()
