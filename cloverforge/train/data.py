# -*- coding: utf-8 -*-
"""Deterministic training prompts (no external corpus needed)."""
from __future__ import annotations

from ..utils.rng import stable_int
from ..tokenizer.vocab import SYMBOLS, N_SYMBOLS


def make_prompt(prompt_len: int, seed: int) -> str:
    """A deterministic prompt string of exactly ``prompt_len`` symbols."""
    return "".join(SYMBOLS[stable_int("prompt:%d:%d" % (seed, i), 0, N_SYMBOLS - 1)]
                   for i in range(prompt_len))


def training_prompts(n: int, prompt_len: int = 8, seed: int = 20261029) -> list:
    return [make_prompt(prompt_len, seed + i) for i in range(n)]
