# -*- coding: utf-8 -*-
"""Discrete mask-and-predict diffusion for text generation.

The forward process *corrupts* tokens by masking; the reverse process
*denoises* by predicting all masked positions in parallel and revealing the
most-confident ones per a schedule -- the opposite of left-to-right
autoregressive decoding.
"""
from .schedule import alpha_bar, reveal_count
from .corruption import forward_mask
from .confidence import probs, confidence, predict_token, top_confident
from .remask import remask_greedy
from .denoiser import Denoiser
from .sampler import generate
from .semi_ar import semi_ar_blocks

__all__ = [
    "alpha_bar", "reveal_count", "forward_mask", "probs", "confidence",
    "predict_token", "top_confident", "remask_greedy", "Denoiser",
    "generate", "semi_ar_blocks",
]
