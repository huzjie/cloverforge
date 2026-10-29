# -*- coding: utf-8 -*-
"""Compute backend interface (swap pure-python for numpy/torch without touching model code)."""
from __future__ import annotations


class Backend:
    def matvec(self, W, x):
        raise NotImplementedError

    def matmul(self, A, B):
        raise NotImplementedError

    def softmax(self, logits):
        raise NotImplementedError

    def relu(self, x):
        raise NotImplementedError

    def argmax(self, vals):
        raise NotImplementedError
