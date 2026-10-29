# -*- coding: utf-8 -*-
"""Evaluation: metrics, the full benchmark, and decode-step speedup analysis."""
from .metrics import token_accuracy, cross_entropy, perplexity
from .benchmark import run_all, summarize
from .speedup import parallel_speedup, semi_ar_speedup, speedup_curve

__all__ = [
    "token_accuracy", "cross_entropy", "perplexity",
    "run_all", "summarize", "parallel_speedup", "semi_ar_speedup", "speedup_curve",
]
