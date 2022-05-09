from typing import Dict, Iterable

# In Memory Repo
from models.movie import Movie

_loaded_movies: Dict[str, Movie] = {}  # Movies by name

def contains_movie(movie_name: str) -> bool:
    return movie_name in _loaded_movies

def get_movie(movie_name: str) -> Movie:
    return _loaded_movies[movie_name]

def persist_movie(movie: Movie) -> None:
    _loaded_movies[movie.name] = movie
