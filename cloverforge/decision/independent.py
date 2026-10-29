# -*- coding: utf-8 -*-
"""Independent (pointwise) judgment: each comparison answered in isolation."""
from __future__ import annotations

from ..utils.rng import stable_float


def judge_pair(i: int, j: int, skill: float, oracle) -> bool:
    """Answer "does item i outrank item j?" with skill-dependent accuracy.

    Accuracy = 0.5 + 0.5 * boost, so it is 0.5 (chance) at skill=0.5 and
    1.0 (perfect) at skill=1.0.
    """
    boost = (skill - 0.5) * 2.0
    truth = oracle.compare(i, j)
    noise = stable_float("judge:%d:%d" % (i, j), 0.0, 1.0)
    correct = noise < (0.5 + 0.5 * boost)
    return truth if correct else (not truth)


def build_adjacency(n_items: int, skill: float, oracle):
    """Judge each unordered pair once; enforce antisymmetry by construction."""
    adj = [[False] * n_items for _ in range(n_items)]
    for i in range(n_items):
        for j in range(i + 1, n_items):
            r = judge_pair(i, j, skill, oracle)
            adj[i][j] = r
            adj[j][i] = not r
    return adj


def pointwise_accuracy(n_items: int, skill: float, oracle) -> float:
    adj = build_adjacency(n_items, skill, oracle)
    correct = total = 0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            total += 1
            if adj[i][j] == oracle.compare(i, j):
                correct += 1
    return correct / total
