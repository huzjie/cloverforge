# -*- coding: utf-8 -*-
"""Conflict detection over pairwise judgments (directed 3-cycles / violations)."""
from __future__ import annotations


def transitivity_violations(adj, n_items: int) -> int:
    """Count (i,j,k) where i beats j, j beats k, but i does NOT beat k."""
    cnt = 0
    for i in range(n_items):
        for j in range(n_items):
            if adj[i][j]:
                for k in range(n_items):
                    if adj[j][k] and not adj[i][k]:
                        cnt += 1
    return cnt


def cycle_count(adj, n_items: int) -> int:
    """Count directed 3-cycles (a stronger form of logical conflict)."""
    cnt = 0
    for i in range(n_items):
        for j in range(n_items):
            if adj[i][j]:
                for k in range(n_items):
                    if adj[j][k] and adj[k][i]:
                        cnt += 1
    return cnt // 3
