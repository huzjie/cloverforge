# -*- coding: utf-8 -*-
"""Deterministic rank oracle: the ground-truth total order for joint decisions."""
from __future__ import annotations

import hashlib


class RankOracle:
    """Assigns a deterministic true rank to ``n_items`` labeled items."""

    def __init__(self, seed: int = 20261029):
        self.seed = seed

    def ranks(self, n_items: int) -> list:
        """Return a permutation of [0..n_items) giving each item's true rank."""
        keys = [self._key(i) for i in range(n_items)]
        return sorted(range(n_items), key=lambda i: keys[i])

    def compare(self, i: int, j: int) -> bool:
        """True iff item i outranks item j (rank_i < rank_j)."""
        return self._key(i) < self._key(j)

    def _key(self, i: int) -> int:
        h = hashlib.md5(("%d|%d" % (self.seed, i)).encode("utf-8")).hexdigest()[:12]
        return int(h, 16)
