"""
🎬 Movie Recommendation System - Gradio App

A comprehensive movie recommendation app with training capabilities using the MovieLens dataset.
Provides personalized recommendations using collaborative filtering techniques.
"""

import gradio as gr
import os
from recommender import MovieRecommendationSystem


# Initialize the recommendation system
print("🚀 Initializing Movie Recommendation System...")
recommender = MovieRecommendationSystem()

# Create Gradio interface
with gr.Blocks(
    title="🎬 Movie Recommendations",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    """,
) as demo:

    with gr.Row():
        gr.HTML(
            """
        <div class="main-header">
            <h1>🎬 AI Movie Recommendation System</h1>
            <p>Discover your next favorite movie with personalized AI recommendations!</p>
        </div>
        """
        )

    with gr.Tab("🎯 My Recommendations", elem_id="rec-tab"):
        gr.Markdown("### Get personalized movie recommendations based on your taste!")

        with gr.Row():
            with gr.Column(scale=1):
                user_input = gr.Number(
                    label="👤 Your User ID",
                    value=1,
                    minimum=1,
                    maximum=610,
                    info="Enter any user ID to see different user profiles",
                )
                num_recs = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="📝 How many recommendations?")
                rec_method = gr.Radio(
                    choices=["collaborative", "ml"],
                    value="collaborative",
                    label="🤖 Recommendation Method",
                    info="Use 'ml' for AI-trained model (requires training first)",
                )
                rec_button = gr.Button("🎬 Get My Recommendations!", variant="primary", size="lg")

                gr.Markdown(
                    """
                **💡 Tips:** 
                - Try different user IDs to see how recommendations change
                - Use 'collaborative' for traditional recommendations
                - Use 'ml' for AI-powered recommendations (train model first)
                """
                )

            with gr.Column(scale=2):
                rec_output = gr.Markdown(
                    value="👆 Enter your user ID and click the button to get started!", elem_id="recommendations"
                )

        def get_recommendations_wrapper(user_id, num_recs, method):
            return recommender.get_user_recommendations(user_id, num_recs, method)

        rec_button.click(fn=get_recommendations_wrapper, inputs=[user_input, num_recs, rec_method], outputs=rec_output)

    with gr.Tab("👤 User Profile", elem_id="profile-tab"):
        gr.Markdown("### Explore different user profiles and their movie preferences")

        with gr.Row():
            with gr.Column(scale=1):
                profile_user_input = gr.Number(label="👤 User ID to explore", value=1, minimum=1, maximum=610)
                profile_button = gr.Button("👤 Show Profile", variant="secondary", size="lg")

            with gr.Column(scale=2):
                profile_output = gr.Markdown(value="Select a user ID to see their movie preferences and rating history")

        profile_button.click(fn=recommender.get_user_profile, inputs=profile_user_input, outputs=profile_output)

    with gr.Tab("🔍 Movie Search", elem_id="search-tab"):
        gr.Markdown("### Find movies by title or genre")

        with gr.Row():
            with gr.Column(scale=1):
                search_input = gr.Textbox(
                    label="🔍 Search movies", placeholder="Try: 'Star Wars', 'Comedy', 'Action'...", lines=1
                )
                search_button = gr.Button("🔍 Search Movies", variant="secondary", size="lg")

                gr.Markdown(
                    """
                **Search tips:**
                - Movie titles: "Toy Story", "Matrix"
                - Genres: "Comedy", "Action", "Romance"
                - Partial matches work too!
                """
                )

            with gr.Column(scale=2):
                search_output = gr.Markdown(value="Enter a movie title or genre to search the database")

        search_button.click(fn=recommender.search_movies, inputs=search_input, outputs=search_output)

    with gr.Tab("🔥 Trending", elem_id="trending-tab"):
        gr.Markdown("### Discover the most popular and highly-rated movies")

        with gr.Row():
            with gr.Column():
                trending_button = gr.Button("🔥 Show Popular Movies", variant="secondary", size="lg")
                trending_output = gr.Markdown(value="Click the button to see the most popular movies!")

        trending_button.click(fn=recommender.get_popular_movies, outputs=trending_output)

    with gr.Tab("🚀 AI Training", elem_id="training-tab"):
        gr.Markdown("### Train an AI model for better recommendations!")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                **🧠 Train a Neural Network:**

                This will train a PyTorch collaborative filtering model that learns user preferences 
                and provides more accurate recommendations.
                """
                )

                model_type_input = gr.Radio(
                    choices=["collaborative"],
                    value="collaborative",
                    label="🤖 Model Type",
                    info="More model types coming soon!",
                )
                epochs_input = gr.Slider(minimum=5, maximum=50, value=20, step=1, label="🔢 Training Epochs")
                batch_size_input = gr.Slider(minimum=128, maximum=1024, value=256, step=128, label="📦 Batch Size")
                lr_input = gr.Slider(minimum=0.001, maximum=0.1, value=0.01, step=0.001, label="📈 Learning Rate")

                train_button = gr.Button("🚀 Train Model", variant="primary", size="lg")

                with gr.Row():
                    load_button = gr.Button("📥 Load Trained Model", variant="secondary")

            with gr.Column(scale=2):
                training_output = gr.Markdown(
                    value="Click 'Train Model' to start training a neural network on your data!"
                )

        # Training button events
        def train_model_wrapper(model_type, epochs, batch_size, learning_rate):
            return recommender.train_model(model_type, epochs, batch_size, learning_rate)

        train_button.click(
            fn=train_model_wrapper,
            inputs=[model_type_input, epochs_input, batch_size_input, lr_input],
            outputs=training_output,
        )

        def load_model_wrapper(model_type):
            return recommender.load_trained_model(model_type)

        load_button.click(fn=load_model_wrapper, inputs=[model_type_input], outputs=training_output)

    with gr.Tab("📊 Dataset Info", elem_id="data-tab"):
        gr.Markdown("### Dataset Statistics and Information")

        with gr.Row():
            with gr.Column():
                stats_button = gr.Button("📊 Show Dataset Stats", variant="secondary", size="lg")
                stats_output = gr.Markdown(value="Click the button to see dataset statistics!")

        stats_button.click(fn=recommender.get_data_stats, outputs=stats_output)

    with gr.Tab("ℹ️ About", elem_id="about-tab"):
        gr.Markdown(
            """
        ## 🤖 How This AI Recommendation System Works

        This movie recommendation system uses **collaborative filtering**, **content-based filtering**, 
        and **deep learning** to suggest movies you'll love!

        ### 🧠 The AI Process:

        1. **📊 Learn Your Preferences**: Analyzes your rating patterns and favorite genres
        2. **👥 Find Similar Users**: Identifies other users with similar taste profiles  
        3. **🎯 Smart Recommendations**: Suggests movies liked by users similar to you
        4. **🎭 Genre Analysis**: Considers your preferred movie genres and themes
        5. **🤖 Neural Networks**: Uses deep learning for advanced pattern recognition

        ### 🎬 Features:

        - **🎯 Personalized Recommendations**: Custom suggestions based on your unique taste
        - **👤 User Profiles**: Explore rating patterns and preferences  
        - **🔍 Smart Search**: Find movies by title, genre, or keywords
        - **🔥 Popular Trends**: Discover highly-rated and trending movies
        - **🚀 AI Training**: Train your own neural network models
        - **📊 Real-time Analysis**: Instant recommendations using machine learning

        ### 📈 The Data:

        Our AI is trained on the **MovieLens dataset** containing:
        - **🎬 Movies**: Thousands of films across all genres and decades
        - **⭐ Ratings**: Real user ratings from movie enthusiasts  
        - **🎭 Genres**: 20+ genre categories for precise matching
        - **📅 Time Period**: Movies from classics to recent releases

        ### 🚀 Getting Started:

        1. **Pick a User ID** to explore different taste profiles
        2. **Get Recommendations** tailored to that user's preferences
        3. **Train AI Models** for even better personalized suggestions
        4. **Explore Profiles** to understand different movie tastes
        5. **Search Movies** to find specific films or genres

        ### 🎯 Pro Tips:

        - Train the AI model first for the best recommendations
        - Try different user IDs to see diverse recommendation styles
        - Users with more ratings get better personalized suggestions  
        - Use ML recommendations for the most accurate results

        ---

        **🎬 Ready to discover your next favorite movie?** Start with the "My Recommendations" tab!

        *Built with PyTorch, Gradio, and machine learning magic* ✨

        ### 🏗️ Architecture:

        This modular system consists of:
        - **models.py**: PyTorch neural network architectures
        - **dataset.py**: Data loading and preprocessing utilities
        - **trainer.py**: Model training and evaluation pipelines
        - **recommender.py**: Main recommendation system logic
        - **app.py**: Gradio web interface
        """
        )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
