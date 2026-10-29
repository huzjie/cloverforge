# -*- coding: utf-8 -*-
"""Kendall tau rank correlation between a predicted and true total order."""
from __future__ import annotations


def kendall_tau(order: list, true_order: list) -> float:
    """order[i] and true_order[i] are item ids at rank i (best first)."""
    n = len(order)
    pos = {item: r for r, item in enumerate(order)}
    true_pos = {item: r for r, item in enumerate(true_order)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = order[i], order[j]
            if (true_pos[a] < true_pos[b]) == (pos[a] < pos[b]):
                concordant += 1
            else:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom else 0.0
