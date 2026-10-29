# -*- coding: utf-8 -*-
"""Generic optimizers (SGD / AdamW) for the optional real-weight path.

The zero-dependency reference uses a skill scalar to model learning dynamics,
but these optimizers are included so a real parameterized model can be trained
the same way (see docs/architecture.md).
"""
from __future__ import annotations

import math


class SGD:
    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def step(self, params, grads):
        for key, p in params.items():
            g = grads[key]
            v = self.velocity.get(key, 0.0) * self.momentum + g
            self.velocity[key] = v
            params[key] = p - self.lr * v


class AdamW:
    def __init__(self, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        b1, b2 = self.betas
        for key, p in params.items():
            g = grads[key]
            m = self.m.get(key, 0.0) * b1 + (1 - b1) * g
            v = self.v.get(key, 0.0) * b2 + (1 - b2) * g * g
            self.m[key] = m
            self.v[key] = v
            m_hat = m / (1 - b1 ** self.t)
            v_hat = v / (1 - b2 ** self.t)
            params[key] = p - self.lr * m_hat / (math.sqrt(v_hat) + self.eps) - self.lr * self.weight_decay * p
