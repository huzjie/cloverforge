# -*- coding: utf-8 -*-
"""cloverforge: a diffusion language model (dLLM) reference implementation.

Unlike autoregressive LMs that emit tokens one-by-one from left to right, a
diffusion LM *masks* the whole output and then denoises it in a fixed number of
parallel refinement steps. This repository is a self-contained, zero-dependency
reference implementation of the core ideas:

- discrete mask-and-predict diffusion (forward corruption / reverse denoising)
- confidence-based remasking schedule
- sparse Mixture-of-Experts backbone with top-k routing + load-balancing loss
- a joint-mode decision head that resolves logically-related judgments globally
- decode-step speedup analysis (constant #iterations vs. sequence length)

Everything runs on the Python standard library (no numpy / torch / requests).
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
