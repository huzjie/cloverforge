# -*- coding: utf-8 -*-
"""Forward corruption: mask a deterministic subset of tokens."""
from __future__ import annotations

import random

from ..utils.rng import _seed


def forward_mask(tokens: list, mask_ratio: float, mask_id: int, key: str):
    """Mask ``mask_ratio`` of tokens at deterministically chosen positions.

    Returns (masked_tokens, masked_positions).
    """
    L = len(tokens)
    n_mask = int(round(mask_ratio * L))
    n_mask = max(0, min(L, n_mask))
    if n_mask == 0:
        return list(tokens), []
    r = random.Random(_seed(key))
    pos = r.sample(range(L), n_mask)
    out = list(tokens)
    for p in pos:
        out[p] = mask_id
    return out, pos
