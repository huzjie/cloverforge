# -*- coding: utf-8 -*-
"""Deterministic token embedding table."""
from __future__ import annotations

from ..utils.rng import stable_matrix


class Embedding:
    def __init__(self, vocab_size: int, dim: int, key: str):
        self.vocab_size = vocab_size
        self.dim = dim
        self.table = stable_matrix("%s:emb" % key, vocab_size, dim, -0.5, 0.5)

    def embed(self, token_id: int) -> list:
        return self.table[token_id]
