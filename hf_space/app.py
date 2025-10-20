"""
🎬 Movie Recommendation System - Hugging Face Space Version
Lightweight version without heavy ML training to fit within storage limits
"""

import gradio as gr
import pandas as pd
import numpy as np
import os

class MovieRecommendationSystem:
    """Lightweight recommendation system for Hugging Face Spaces."""

    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.movie_stats = None
        self._load_data()
        self._prepare_recommendations()

    def _load_data(self):
        """Load the MovieLens data."""
        try:
            if os.path.exists("data/movies.csv") and os.path.exists("data/ratings.csv"):
                self.movies_df = pd.read_csv("data/movies.csv")
                self.ratings_df = pd.read_csv("data/ratings.csv")
                print(f"✅ Loaded {len(self.movies_df)} movies and {len(self.ratings_df)} ratings")
            else:
                print("⚠️ Data files not found, creating sample data...")
                self._create_sample_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            self._create_sample_data()

    def _create_sample_data(self):
        """Create sample data if real data isn't available."""
        sample_movies = [
            (1, "Toy Story (1995)", "Animation|Children|Comedy"),
            (2, "Jumanji (1995)", "Adventure|Children|Fantasy"),
            (3, "The Lion King (1994)", "Animation|Children|Drama|Musical"),
            (4, "Forrest Gump (1994)", "Comedy|Drama|Romance"),
            (5, "Pulp Fiction (1994)", "Crime|Drama"),
            (6, "The Matrix (1999)", "Action|Sci-Fi|Thriller"),
            (7, "Titanic (1997)", "Drama|Romance"),
            (8, "Star Wars (1977)", "Action|Adventure|Sci-Fi"),
            (9, "The Godfather (1972)", "Crime|Drama"),
            (10, "Goodfellas (1990)", "Crime|Drama"),
        ]
        self.movies_df = pd.DataFrame(sample_movies, columns=["movieId", "title", "genres"])

        np.random.seed(42)
        sample_ratings = []
        for user_id in range(1, 21):
            for movie_id in range(1, 11):
                if np.random.random() > 0.3:
                    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
                    sample_ratings.append((user_id, movie_id, rating, 964982703))

        self.ratings_df = pd.DataFrame(sample_ratings, columns=["userId", "movieId", "rating", "timestamp"])
        print("✅ Created sample data for demonstration")

    def _prepare_recommendations(self):
        """Prepare recommendation data structures."""
        if self.ratings_df is None:
            return

        self.movie_stats = self.ratings_df.groupby("movieId").agg({"rating": ["mean", "count"]}).round(2)
        self.movie_stats.columns = ["avg_rating", "rating_count"]
        self.movie_stats = self.movie_stats.reset_index()
        print("✅ Recommendation system ready!")

    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> str:
        """Get personalized recommendations for a user."""
        try:
            if self.ratings_df is None:
                return "❌ No rating data available"

            available_users = self.ratings_df["userId"].unique()
            if user_id not in available_users:
                return f"❌ User {user_id} not found. Try user IDs: {sorted(available_users)[:10]}"

            user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
            rated_movies = set(user_ratings["movieId"].values)

            if len(user_ratings) == 0:
                return f"❌ User {user_id} hasn't rated any movies"

            # Find similar users
            similar_users = self._find_similar_users(user_id)
            recommendations = []

            if similar_users:
                for sim_user_id, similarity in similar_users[:5]:
                    sim_user_ratings = self.ratings_df[
                        (self.ratings_df["userId"] == sim_user_id) & (self.ratings_df["rating"] >= 4)
                    ]

                    for _, rating_row in sim_user_ratings.iterrows():
                        movie_id = rating_row["movieId"]
                        if movie_id not in rated_movies:
                            movie_info = self.movies_df[self.movies_df["movieId"] == movie_id]
                            if not movie_info.empty:
                                score = rating_row["rating"] * similarity
                                recommendations.append({
                                    "movieId": movie_id,
                                    "title": movie_info.iloc[0]["title"],
                                    "genres": movie_info.iloc[0]["genres"],
                                    "score": score,
                                    "predicted_rating": min(5.0, score),
                                })

            # Add genre-based recommendations
            if len(recommendations) < n_recommendations:
                genre_recs = self._get_genre_recommendations(user_id, rated_movies, n_recommendations)
                recommendations.extend(genre_recs)

            # Remove duplicates
            seen_movies = set()
            unique_recs = []
            for rec in sorted(recommendations, key=lambda x: x["score"], reverse=True):
                if rec["movieId"] not in seen_movies:
                    unique_recs.append(rec)
                    seen_movies.add(rec["movieId"])

            if not unique_recs:
                return self._get_popular_recommendations(n_recommendations, rated_movies)

            result = f"🎬 **Personalized Recommendations for User {user_id}:**\n\n"
            avg_rating = user_ratings["rating"].mean()
            fav_genres = self._get_user_favorite_genres(user_id)
            result += f"*Your average rating: {avg_rating:.1f}/5.0*\n"
            result += f"*Your favorite genres: {', '.join(fav_genres[:3])}*\n\n"

            for i, rec in enumerate(unique_recs[:n_recommendations], 1):
                result += f"**{i}. {rec['title']}** (⭐ {rec['predicted_rating']:.1f}/5.0)\n"
                result += f"   *{rec['genres']}*\n\n"

            return result

        except Exception as e:
            return f"❌ Error generating recommendations: {str(e)}"

    def _find_similar_users(self, user_id: int, n_similar: int = 10):
        """Find users with similar rating patterns."""
        try:
            user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
            user_movies = set(user_ratings["movieId"].values)
            similarities = []

            for other_user in self.ratings_df["userId"].unique():
                if other_user == user_id:
                    continue

                other_ratings = self.ratings_df[self.ratings_df["userId"] == other_user]
                other_movies = set(other_ratings["movieId"].values)
                common_movies = user_movies.intersection(other_movies)

                if len(common_movies) >= 2:
                    user_common_ratings = user_ratings[user_ratings["movieId"].isin(common_movies)]["rating"].values
                    other_common_ratings = other_ratings[other_ratings["movieId"].isin(common_movies)]["rating"].values

                    if len(user_common_ratings) == len(other_common_ratings):
                        similarity = np.dot(user_common_ratings, other_common_ratings) / (
                            np.linalg.norm(user_common_ratings) * np.linalg.norm(other_common_ratings)
                        )
                        similarities.append((other_user, similarity))

            return sorted(similarities, key=lambda x: x[1], reverse=True)[:n_similar]
        except:
            return []

    def _get_genre_recommendations(self, user_id: int, rated_movies: set, n_recs: int):
        """Get recommendations based on user's favorite genres."""
        try:
            favorite_genres = self._get_user_favorite_genres(user_id)
            recommendations = []

            for _, movie in self.movies_df.iterrows():
                if movie["movieId"] in rated_movies:
                    continue

                movie_genres = set(movie["genres"].split("|"))
                genre_match_score = len(set(favorite_genres).intersection(movie_genres))

                if genre_match_score > 0:
                    movie_stats = self.movie_stats[self.movie_stats["movieId"] == movie["movieId"]]
                    avg_rating = movie_stats["avg_rating"].iloc[0] if not movie_stats.empty else 3.0

                    recommendations.append({
                        "movieId": movie["movieId"],
                        "title": movie["title"],
                        "genres": movie["genres"],
                        "score": genre_match_score * avg_rating,
                        "predicted_rating": avg_rating,
                    })

            return recommendations
        except:
            return []

    def _get_user_favorite_genres(self, user_id: int):
        """Get user's favorite genres based on their ratings."""
        try:
            user_ratings = self.ratings_df[(self.ratings_df["userId"] == user_id) & (self.ratings_df["rating"] >= 4)]
            genre_scores = {}

            for _, rating in user_ratings.iterrows():
                movie_info = self.movies_df[self.movies_df["movieId"] == rating["movieId"]]
                if not movie_info.empty:
                    genres = movie_info.iloc[0]["genres"].split("|")
                    for genre in genres:
                        genre_scores[genre] = genre_scores.get(genre, 0) + rating["rating"]

            return sorted(genre_scores.keys(), key=genre_scores.get, reverse=True)
        except:
            return []

    def _get_popular_recommendations(self, n_recs: int, exclude_movies: set = None) -> str:
        """Get popular movie recommendations."""
        exclude_movies = exclude_movies or set()

        try:
            popular = self.movie_stats[
                (self.movie_stats["rating_count"] >= 3) & (~self.movie_stats["movieId"].isin(exclude_movies))
            ].nlargest(n_recs, "avg_rating")

            result = "🔥 **Popular Movies You Might Like:**\n\n"

            for i, (_, row) in enumerate(popular.iterrows(), 1):
                movie_info = self.movies_df[self.movies_df["movieId"] == row["movieId"]]
                if not movie_info.empty:
                    title = movie_info.iloc[0]["title"]
                    genres = movie_info.iloc[0]["genres"]
                    result += f"**{i}. {title}** (⭐ {row['avg_rating']:.1f}/5.0)\n"
                    result += f"   *{genres}* - {row['rating_count']} ratings\n\n"

            return result
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_movies(self, query: str) -> str:
        """Search for movies by title or genre."""
        if self.movies_df is None:
            return "❌ Movie data not available"

        try:
            title_matches = self.movies_df[self.movies_df["title"].str.contains(query, case=False, na=False)]
            genre_matches = self.movies_df[self.movies_df["genres"].str.contains(query, case=False, na=False)]
            all_matches = pd.concat([title_matches, genre_matches]).drop_duplicates()

            if all_matches.empty:
                return f"❌ No movies found for '{query}'"

            result = f"🔍 **Search Results for '{query}':**\n\n"

            for i, (_, movie) in enumerate(all_matches.head(10).iterrows(), 1):
                movie_stats = self.movie_stats[self.movie_stats["movieId"] == movie["movieId"]]
                result += f"**{i}. {movie['title']}**\n   *{movie['genres']}*\n"

                if not movie_stats.empty:
                    result += f"   ⭐ {movie_stats['avg_rating'].iloc[0]:.1f}/5.0 ({movie_stats['rating_count'].iloc[0]} ratings)\n\n"
                else:
                    result += f"   No ratings yet\n\n"

            return result
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_user_profile(self, user_id: int) -> str:
        """Get user profile and rating history."""
        try:
            if self.ratings_df is None:
                return "❌ No rating data available"

            available_users = self.ratings_df["userId"].unique()
            if user_id not in available_users:
                return f"❌ User {user_id} not found. Try: {sorted(available_users)[:10]}"

            user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]

            result = f"👤 **User {user_id} Profile:**\n\n"
            result += f"📊 **Statistics:**\n"
            result += f"- Movies rated: {len(user_ratings)}\n"
            result += f"- Average rating: {user_ratings['rating'].mean():.1f}/5.0\n"
            result += f"- Highest rating: {user_ratings['rating'].max()}/5.0\n"
            result += f"- Lowest rating: {user_ratings['rating'].min()}/5.0\n\n"

            fav_genres = self._get_user_favorite_genres(user_id)
            result += f"🎭 **Favorite Genres:** {', '.join(fav_genres[:5])}\n\n"

            top_rated = user_ratings.nlargest(5, "rating")
            result += f"⭐ **Top Rated Movies:**\n"

            for i, (_, rating_row) in enumerate(top_rated.iterrows(), 1):
                movie_info = self.movies_df[self.movies_df["movieId"] == rating_row["movieId"]]
                if not movie_info.empty:
                    title = movie_info.iloc[0]["title"]
                    result += f"{i}. **{title}** - {rating_row['rating']}/5.0\n"

            return result
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_popular_movies(self) -> str:
        """Get the most popular movies."""
        return self._get_popular_recommendations(15)


# Initialize the recommendation system
print("🚀 Initializing Movie Recommendation System (Lightweight HF Version)...")
recommender = MovieRecommendationSystem()

# Create Gradio interface
with gr.Blocks(
    title="🎬 Movie Recommendations",
    theme=gr.themes.Soft(),
) as demo:

    gr.HTML("""
        <div style="text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <h1>🎬 Movie Recommendation System</h1>
            <p>Discover your next favorite movie with AI-powered recommendations!</p>
            <p><em>Lightweight version optimized for Hugging Face Spaces</em></p>
        </div>
    """)

    with gr.Tab("🎯 My Recommendations"):
        gr.Markdown("### Get personalized movie recommendations!")

        with gr.Row():
            with gr.Column(scale=1):
                user_input = gr.Number(label="👤 Your User ID", value=1, minimum=1, maximum=610)
                num_recs = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="📝 How many recommendations?")
                rec_button = gr.Button("🎬 Get Recommendations!", variant="primary", size="lg")

            with gr.Column(scale=2):
                rec_output = gr.Markdown(value="👆 Enter your user ID and click the button to get started!")

        rec_button.click(fn=recommender.get_user_recommendations, inputs=[user_input, num_recs], outputs=rec_output)

    with gr.Tab("👤 User Profile"):
        gr.Markdown("### Explore user profiles and movie preferences")

        with gr.Row():
            with gr.Column(scale=1):
                profile_user_input = gr.Number(label="👤 User ID to explore", value=1, minimum=1, maximum=610)
                profile_button = gr.Button("👤 Show Profile", variant="secondary", size="lg")

            with gr.Column(scale=2):
                profile_output = gr.Markdown(value="Select a user ID to see their preferences")

        profile_button.click(fn=recommender.get_user_profile, inputs=profile_user_input, outputs=profile_output)

    with gr.Tab("🔍 Movie Search"):
        gr.Markdown("### Find movies by title or genre")

        with gr.Row():
            with gr.Column(scale=1):
                search_input = gr.Textbox(label="🔍 Search movies", placeholder="Try: 'Toy Story', 'Action'...")
                search_button = gr.Button("🔍 Search", variant="secondary", size="lg")

            with gr.Column(scale=2):
                search_output = gr.Markdown(value="Enter a movie title or genre to search")

        search_button.click(fn=recommender.search_movies, inputs=search_input, outputs=search_output)

    with gr.Tab("🔥 Trending"):
        gr.Markdown("### Discover popular and highly-rated movies")

        trending_button = gr.Button("🔥 Show Popular Movies", variant="secondary", size="lg")
        trending_output = gr.Markdown(value="Click the button to see popular movies!")

        trending_button.click(fn=recommender.get_popular_movies, outputs=trending_output)

    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## 🤖 Movie Recommendation System

        This is a **lightweight version** optimized for Hugging Face Spaces (within 50GB storage limit).

        ### 🎯 Features:
        - **Personalized Recommendations**: Based on collaborative filtering
        - **User Profiles**: Analyze rating patterns and preferences
        - **Smart Search**: Find movies by title or genre
        - **Popular Trends**: Discover highly-rated movies

        ### 📊 The Data:
        - **MovieLens Dataset** with thousands of movies and ratings
        - Real user preferences and rating patterns

        ### 💡 Note:
        This version uses traditional collaborative filtering instead of heavy PyTorch models to stay within storage limits.
        
        ---
        *Built with Python, Gradio, and Pandas* ✨
        """)

if __name__ == "__main__":
    demo.launch()
