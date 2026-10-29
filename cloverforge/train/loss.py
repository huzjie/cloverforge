# -*- coding: utf-8 -*-
"""Masked-token cross-entropy loss (the diffusion / MLM training objective)."""
from __future__ import annotations

import math

from ..utils.linalg import softmax
from ..tokenizer.vocab import MASK_ID


def masked_cross_entropy(logits, x, target, prompt_len: int) -> float:
    """Mean cross-entropy over the currently-masked generated positions.

    logits: [L, vocab]; x: [L] token ids; target: [gen_len] true tokens;
    prompt_len: length of the observed prefix.
    """
    total = 0.0
    n = 0
    for i in range(prompt_len, len(x)):
        if x[i] == MASK_ID:
            probs = softmax(logits[i])
            t = target[i - prompt_len]
            total += -math.log(probs[t] + 1e-9)
            n += 1
    return total / n if n else 0.0
