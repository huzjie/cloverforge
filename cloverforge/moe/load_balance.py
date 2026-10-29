# -*- coding: utf-8 -*-
"""Load-balancing auxiliary loss (Switch-Transformer style)."""
from __future__ import annotations


def load_balance_stats(gate_probs: list, top_k: int = 1):
    """Compute per-expert load, mean probs, and the aux loss.

    gate_probs: list of per-token softmax distributions over experts.
    Returns (fractions, probs, aux_loss). The effective load of an expert is
    its mean gate probability (soft load), avoiding hard-dispatch tie ambiguity.
    """
    if not gate_probs:
        return [], [], 0.0
    n_tokens = len(gate_probs)
    n_experts = len(gate_probs[0])
    probs = [0.0] * n_experts
    for row in gate_probs:
        for j in range(n_experts):
            probs[j] += row[j] / n_tokens
    fractions = [p for p in probs]
    aux = n_experts * sum(fractions[j] * probs[j] for j in range(n_experts))
    return fractions, probs, aux
