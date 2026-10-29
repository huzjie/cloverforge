# -*- coding: utf-8 -*-
"""Remasking policies: which newly-predicted tokens to keep vs. re-mask."""
from __future__ import annotations

from ..utils.linalg import softmax, argmax


def remask_greedy(masked_positions: list, logits_at: dict, n_new: int, temperature: float = 1.0):
    """Reveal the n_new most-confident predictions among masked positions.

    Returns (kept_positions, kept_tokens) ordered by descending confidence.
    """
    scored = []
    for pos in masked_positions:
        lg = logits_at[pos]
        p = softmax([x / temperature for x in lg]) if temperature and temperature != 1.0             else softmax(lg)
        scored.append((pos, argmax(lg), max(p)))
    scored.sort(key=lambda t: t[2], reverse=True)
    kept = scored[:n_new]
    return [(pos, tok) for pos, tok, _conf in kept]
