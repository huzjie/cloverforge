# -*- coding: utf-8 -*-
"""Decode-step speedup: dLLM's constant #iterations vs. autoregressive length.

Autoregressive decoding needs L sequential steps (one per token). Diffusion
decoding needs a fixed number of refinement steps T regardless of length, so
the speedup grows linearly with sequence length.
"""
from __future__ import annotations


def parallel_speedup(gen_len: int, steps: int) -> float:
    """Speedup of fully-parallel (non-autoregressive) dLLM decoding."""
    return gen_len / steps if steps > 0 else float("inf")


def semi_ar_speedup(gen_len: int, steps: int, blocks: int) -> float:
    """Speedup with semi-autoregressive block decoding (blocks processed serially)."""
    if blocks <= 1:
        return parallel_speedup(gen_len, steps)
    return gen_len / (steps * blocks)


def speedup_curve(max_len: int, steps: int):
    """Return [(length, speedup), ...] for the fully-parallel setting."""
    return [(L, parallel_speedup(L, steps)) for L in range(8, max_len + 1, 8)]
