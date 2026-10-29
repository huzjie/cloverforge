# -*- coding: utf-8 -*-
"""Joint-mode decision head.

Two ways to answer a *group* of logically-related questions:
- independent mode: each question judged in isolation (fast, but can contradict)
- joint mode: global coordination resolves all questions into one consistent
  assignment in a single pass (no conflicts, often more accurate).
"""
from .oracle import DecisionOracle
from .independent import judge_pair, build_adjacency, pointwise_accuracy
from .joint import copeland_order
from .consistency import cycle_count, transitivity_violations
from .ranking import kendall_tau

__all__ = [
    "DecisionOracle", "judge_pair", "build_adjacency", "pointwise_accuracy",
    "copeland_order", "cycle_count", "transitivity_violations", "kendall_tau",
]
