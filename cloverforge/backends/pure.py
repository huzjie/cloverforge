# -*- coding: utf-8 -*-
"""Pure-python compute backend (default, zero dependency)."""
from __future__ import annotations

from ..utils import linalg
from .base import Backend


class PureBackend(Backend):
    def matvec(self, W, x):
        return linalg.matvec(W, x)

    def matmul(self, A, B):
        return linalg.matmul(A, B)

    def softmax(self, logits):
        return linalg.softmax(logits)

    def relu(self, x):
        return linalg.relu(x)

    def argmax(self, vals):
        return linalg.argmax(vals)
