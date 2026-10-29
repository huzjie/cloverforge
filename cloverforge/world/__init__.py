# -*- coding: utf-8 -*-
"""Deterministic ground-truth oracles that make training/benchmark falsifiable.

The sequence oracle defines the "correct" completion of a prompt; the rank
oracle defines the "true" total order used by the joint-decision demo. Both are
pure deterministic functions of a seed, so skill-driven metrics can be compared
across runs without any external data or randomness.
"""
from .sequence import SequenceOracle
from .rank import RankOracle

__all__ = ["SequenceOracle", "RankOracle"]
