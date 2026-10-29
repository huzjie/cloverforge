# -*- coding: utf-8 -*-
"""Confidence estimation: softmax probabilities + argmax prediction."""
from __future__ import annotations

from ..utils.linalg import softmax, argmax


def probs(logits: list, temperature: float = 1.0) -> list:
    if temperature and temperature != 1.0:
        logits = [x / temperature for x in logits]
    return softmax(logits)


def confidence(logits: list, temperature: float = 1.0) -> float:
    return max(probs(logits, temperature))


def predict_token(logits: list) -> int:
    return argmax(logits)


def top_confident(logits_map: dict, k: int):
    """Given {position: logits}, return the k positions with highest confidence."""
    scored = [(pos, confidence(logits)) for pos, logits in logits_map.items()]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [pos for pos, _ in scored[:k]]
