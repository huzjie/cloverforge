# -*- coding: utf-8 -*-
"""Output projection: hidden state -> vocabulary logits."""
from __future__ import annotations

from ..utils.linalg import matvec
from ..utils.rng import stable_matrix


class LMHead:
    def __init__(self, dim: int, vocab_size: int, key: str):
        self.W = stable_matrix("%s:head" % key, vocab_size, dim)

    def forward(self, h: list) -> list:
        return matvec(self.W, h)
