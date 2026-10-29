# -*- coding: utf-8 -*-
"""Reveal schedule: how many tokens to unmask at each reverse step."""
from __future__ import annotations

import math


def alpha_bar(t: int, T: int, kind: str = "cosine") -> float:
    """Fraction of the target region revealed at reverse step t (1..T).

    t=1 reveals the few most-confident tokens; t=T reveals everything.
    """
    if T <= 0:
        return 1.0
    if kind == "linear":
        return t / T
    if kind == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * t / T))
    raise ValueError("unknown schedule: %s" % kind)


def reveal_count(alpha: float, length: int) -> int:
    """Number of tokens to reveal for a given alpha (clamped to [0, length])."""
    n = int(round(alpha * length))
    return max(0, min(length, n))
