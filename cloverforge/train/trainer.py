# -*- coding: utf-8 -*-
"""Trainer: raises the denoiser's skill monotonically toward 1 (learning)."""
from __future__ import annotations


class Trainer:
    def __init__(self, denoiser, lr: float = 0.25):
        self.denoiser = denoiser
        self.lr = lr
        self.skill = denoiser.skill

    def train_step(self) -> float:
        """One learning step. The skill delta is always positive (monotonic)."""
        self.skill = min(1.0, self.skill + self.lr * (1.0 - self.skill))
        self.denoiser.skill = self.skill
        return self.skill
