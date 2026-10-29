# -*- coding: utf-8 -*-
"""Joint (global-coordination) judgment: resolve pairwise votes into a total order.

The Copeland score (net wins) aggregates *all* pairwise evidence into a single
consistent ranking. This is exactly what "one-shot global coordination" means:
individual noisy judgments reference each other and get overruled by the
majority, so the result is conflict-free and typically more accurate.
"""
from __future__ import annotations


def copeland_order(adj, n_items: int):
    """Return items ordered best->worst by net pairwise wins (deterministic)."""
    score = [sum(1 for j in range(n_items) if adj[i][j]) for i in range(n_items)]
    return sorted(range(n_items), key=lambda i: (score[i], -i), reverse=True)
