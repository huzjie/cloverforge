# -*- coding: utf-8 -*-
"""Small utility helpers: deterministic RNG, minimal YAML, text distance."""
from .rng import stable_float, stable_int, stable_vector, stable_choice, stable_matrix
from .text import edit_distance, tokenize_words, word_error_rate

__all__ = [
    "stable_float", "stable_int", "stable_vector", "stable_choice", "stable_matrix",
    "edit_distance", "tokenize_words", "word_error_rate",
]
