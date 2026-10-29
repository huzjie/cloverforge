# -*- coding: utf-8 -*-
"""A single expert: a two-layer ReLU feed-forward network."""
from __future__ import annotations

from ..utils.linalg import matvec, relu, add
from ..utils.rng import stable_matrix, stable_vector


class ExpertFFN:
    def __init__(self, dim: int, hidden: int, key: str):
        self.dim = dim
        self.hidden = hidden
        # W1: [hidden, dim]; b1: [hidden]; W2: [dim, hidden]; b2: [dim]
        self.W1 = stable_matrix("%s:W1" % key, hidden, dim)
        self.b1 = stable_vector("%s:b1" % key, hidden)
        self.W2 = stable_matrix("%s:W2" % key, dim, hidden)
        self.b2 = stable_vector("%s:b2" % key, dim)

    def forward(self, x: list) -> list:
        h = relu(add(matvec(self.W1, x), self.b1))
        out = add(matvec(self.W2, h), self.b2)
        return out
