# Raw Data Files

This folder contains the raw CSV data files used by the movie recommendation system.

## Files

- **`movies.csv`**: Movie information with titles and genres
- **`ratings.csv`**: User ratings for movies (userId, movieId, rating, timestamp)
- **`links.csv`**: Links between movieIds and external databases (IMDB, TMDB)
- **`tags.csv`**: User-generated tags for movies
- **`README.txt`**: Original MovieLens dataset documentation

## Data Source

These files are typically from the MovieLens dataset (https://grouplens.org/datasets/movielens/).

If these files are not present, the system will automatically generate sample data for demonstration purposes.

## Usage

The files in this folder are automatically loaded by the `MovieLensDataLoader` class in the parent directory.

```python
from data import MovieLensDataLoader

# DataLoader automatically looks for files in data/raw/
loader = MovieLensDataLoader()
```