# -*- coding: utf-8 -*-
"""Minimal dense linear-algebra helpers (pure Python, no numpy)."""
from __future__ import annotations

import math


def matvec(W: list, x: list) -> list:
    """Matrix-vector product W @ x.  W is [rows, cols], x is [cols]."""
    rows = len(W)
    cols = len(x)
    return [sum(W[i][j] * x[j] for j in range(cols)) for i in range(rows)]


def matmul(A: list, B: list) -> list:
    """Matrix-matrix product A @ B.  A [r, k], B [k, c] -> [r, c]."""
    r, k = len(A), len(B)
    c = len(B[0])
    return [[sum(A[i][m] * B[m][j] for m in range(k)) for j in range(c)] for i in range(r)]


def _is_matrix(v):
    return isinstance(v[0], list)


def add(a, b):
    if _is_matrix(a):
        return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    return [a[i] + b[i] for i in range(len(a))]


def sub(a, b):
    if _is_matrix(a):
        return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    return [a[i] - b[i] for i in range(len(a))]


def scale(a, s):
    if _is_matrix(a):
        return [[a[i][j] * s for j in range(len(a[0]))] for i in range(len(a))]
    return [a[i] * s for i in range(len(a))]


def relu(a):
    if _is_matrix(a):
        return [[max(0.0, v) for v in row] for row in a]
    return [max(0.0, v) for v in a]


def dot(a, b) -> float:
    return sum(a[i] * b[i] for i in range(len(a)))


def softmax(logits: list) -> list:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def argmax(vals: list) -> int:
    return max(range(len(vals)), key=lambda i: vals[i])


def l2_norm(a) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine(a, b) -> float:
    na, nb = l2_norm(a), l2_norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)
