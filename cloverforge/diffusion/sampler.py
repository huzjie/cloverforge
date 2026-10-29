# -*- coding: utf-8 -*-
"""Reverse diffusion: generate a completion via iterative mask-and-predict."""
from __future__ import annotations

from ..tokenizer.vocab import MASK_ID
from ..utils.linalg import softmax, argmax
from .schedule import alpha_bar, reveal_count


def generate(denoiser, prompt_ids: list, prompt: str, gen_len: int,
             steps: int, schedule: str = "cosine", remask: str = "confidence",
             temperature: float = 1.0):
    """Generate ``gen_len`` tokens conditioned on ``prompt_ids``.

    Returns the full sequence [prompt | generated] of token ids.
    """
    prompt_len = len(prompt_ids)
    L = prompt_len + gen_len
    x = list(prompt_ids) + [MASK_ID] * gen_len
    gen_positions = list(range(prompt_len, L))

    for t in range(1, steps + 1):
        logits = denoiser.logits(x, prompt, prompt_len)
        masked = [i for i in gen_positions if x[i] == MASK_ID]
        if not masked:
            break
        already = gen_len - len(masked)
        n_total = reveal_count(alpha_bar(t, steps, schedule), gen_len)
        n_new = max(0, n_total - already)
        if n_new == 0 and masked:
            n_new = 1  # guarantee forward progress
        preds = []
        for i in masked:
            p = softmax([v / temperature for v in logits[i]]) if temperature and temperature != 1.0                 else softmax(logits[i])
            preds.append((i, argmax(logits[i]), max(p)))
        if remask == "confidence":
            preds.sort(key=lambda t: t[2], reverse=True)
        else:
            import random as _r
            _r.Random(20261029).shuffle(preds)
        for (i, tok, _conf) in preds[:n_new]:
            x[i] = tok
    return x
