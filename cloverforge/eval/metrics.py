# -*- coding: utf-8 -*-
"""Core evaluation metrics."""
from __future__ import annotations

import math

from ..utils.linalg import softmax


def token_accuracy(predicted: list, target: list) -> float:
    if not target:
        return 1.0
    return sum(1 for a, b in zip(predicted, target) if a == b) / len(target)


def cross_entropy(logits: list, target: list) -> float:
    total = 0.0
    for row, t in zip(logits, target):
        probs = softmax(row)
        total += -math.log(probs[t] + 1e-9)
    return total / len(target) if target else 0.0


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 100.0))
