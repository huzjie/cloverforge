# -*- coding: utf-8 -*-
"""Deterministic sequence oracle: the ground-truth completion of a prompt.

We model a "predictable world" -- a simple deterministic generator (a linear
congruential recurrence seeded by the prompt) -- so that a trained denoiser can
plausibly learn to reconstruct it, while an untrained one scores near chance.
"""
from __future__ import annotations

import hashlib

from ..utils.rng import _seed
from ..tokenizer.vocab import CONTENT_OFFSET, N_SYMBOLS


def _hash(prompt: str) -> int:
    return int(hashlib.md5(prompt.encode("utf-8")).hexdigest()[:12], 16)


class SequenceOracle:
    """Produces a deterministic target completion for a given prompt."""

    def __init__(self, seed: int = 20261029):
        self.seed = seed

    def completion(self, prompt: str, length: int) -> list:
        """Return ``length`` content-token ids (a deterministic function of prompt)."""
        state = _hash("%s|%d" % (prompt, self.seed))
        out = []
        for i in range(length):
            # linear congruential generator -> stable, cheap, learnable-looking
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            tok = CONTENT_OFFSET + (state % N_SYMBOLS)
            out.append(tok)
        return out

    def target(self, prompt: str, length: int) -> list:
        return self.completion(prompt, length)
