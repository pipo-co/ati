from typing import Dict, Iterable

# In Memory Repo
from models.movie import Movie

_loaded_movies: Dict[str, Movie] = {}  # Movies by name

def contains_movie(movie_name: str) -> bool:
    return movie_name in _loaded_movies

def get_movie(movie_name: str) -> Movie:
    if movie_name not in _loaded_movies:
        raise ValueError(f'{movie_name} is not in repository')
    return _loaded_movies[movie_name]

def persist_movie(movie: Movie) -> None:
    if movie.name in _loaded_movies:
        raise ValueError(f'{movie.name} already has mapped {_loaded_movies[movie.name]}')
    _loaded_movies[movie.name] = movie
