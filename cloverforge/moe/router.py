# -*- coding: utf-8 -*-
"""Noisy top-k router for sparse MoE."""
from __future__ import annotations

from ..utils.linalg import matvec, softmax
from ..utils.rng import stable_float


def router_logits(hidden: list, Wg: list) -> list:
    """Gate logits for each expert: logit_j = Wg[j] . hidden."""
    return matvec(Wg, hidden)


def softmax_probs(logits: list) -> list:
    return softmax(logits)


def noisy_topk(logits: list, top_k: int, noise_scale: float, key: str):
    """Pick top-k experts with deterministic gating noise (Shazeer-style)."""
    n = len(logits)
    noisy = [logits[j] + noise_scale * stable_float("%s:%d" % (key, j), -1.0, 1.0)
             for j in range(n)]
    order = sorted(range(n), key=lambda j: noisy[j], reverse=True)[:top_k]
    selected = [noisy[j] for j in order]
    weights = softmax(selected)
    return order, weights
