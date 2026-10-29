# -*- coding: utf-8 -*-
"""Bidirectional multi-head scaled dot-product attention (pure Python)."""
from __future__ import annotations

import math

from ..utils.linalg import matmul, softmax
from ..utils.rng import stable_matrix


def _transpose(M):
    return [list(row) for row in zip(*M)]


def _split_heads(X, n_heads, head_dim):
    L = len(X)
    return [[[X[i][h * head_dim + k] for k in range(head_dim)] for i in range(L)]
            for h in range(n_heads)]


def _merge_heads(heads, L, n_heads, head_dim, dim):
    out = [[0.0] * dim for _ in range(L)]
    for h in range(n_heads):
        for i in range(L):
            for k in range(head_dim):
                out[i][h * head_dim + k] = heads[h][i][k]
    return out


def _scaled_dot_attention(Q, K, V):
    L = len(Q)
    d = len(Q[0])
    Kt = _transpose(K)
    scores = matmul(Q, Kt)
    scale = 1.0 / math.sqrt(d)
    probs = [softmax([scores[i][j] * scale for j in range(L)]) for i in range(L)]
    return matmul(probs, V)


class MultiHeadAttention:
    def __init__(self, dim: int, n_heads: int, key: str):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.Wq = stable_matrix("%s:Wq" % key, dim, dim)
        self.Wk = stable_matrix("%s:Wk" % key, dim, dim)
        self.Wv = stable_matrix("%s:Wv" % key, dim, dim)
        self.Wo = stable_matrix("%s:Wo" % key, dim, dim)

    def forward(self, X):
        Q = matmul(X, self.Wq)
        K = matmul(X, self.Wk)
        V = matmul(X, self.Wv)
        qh = _split_heads(Q, self.n_heads, self.head_dim)
        kh = _split_heads(K, self.n_heads, self.head_dim)
        vh = _split_heads(V, self.n_heads, self.head_dim)
        oh = [_scaled_dot_attention(qh[h], kh[h], vh[h]) for h in range(self.n_heads)]
        O = _merge_heads(oh, len(X), self.n_heads, self.head_dim, self.dim)
        return matmul(O, self.Wo)
