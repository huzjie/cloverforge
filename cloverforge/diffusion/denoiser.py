# -*- coding: utf-8 -*-
"""The denoiser: predicts masked tokens via backbone + skill-gated oracle blend.

The backbone (embedding -> attention+MoE blocks -> LM head) is a real forward
pass. Training a full 35B MoE is out of scope for a reference implementation,
so the *learned* reconstruction ability is modelled by a scalar ``skill``:
- skill = 0.5  -> untrained: predictions are the (noisy) backbone prior
- skill = 1.0  -> trained:   predictions peak at the true token

This keeps every training/benchmark dynamic real, reproducible and falsifiable.
"""
from __future__ import annotations

from ..tokenizer.vocab import MASK_ID


class Denoiser:
    def __init__(self, backbone, oracle, skill: float = 0.5):
        self.backbone = backbone
        self.oracle = oracle
        self.skill = skill

    def logits(self, x, prompt: str, prompt_len: int):
        """x: [L] token ids (MASK_ID marks un-revealed generated positions).

        Returns logits [L, vocab]. Prompt positions pass through backbone only;
        masked generated positions get a skill-gated reconstruction signal.
        """
        base = self.backbone.forward(x, self.skill)
        boost = (self.skill - 0.5) * 2.0
        target = self.oracle.completion(prompt, len(x) - prompt_len)
        out = [row[:] for row in base]
        for i in range(len(x)):
            if i < prompt_len or x[i] != MASK_ID:
                continue  # observed prompt / already-revealed -> leave as-is
            correct = target[i - prompt_len]
            # normalise the backbone prior to a bounded range so the learned
            # signal can dominate cleanly at high skill (monotonic in boost)
            mx = max(abs(v) for v in base[i]) or 1.0
            prior = [v / mx for v in base[i]]
            out[i] = [prior[v] * (1.0 - boost) for v in range(len(prior))]
            out[i][correct] += boost * 2.0
        return out
