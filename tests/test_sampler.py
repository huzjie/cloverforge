# -*- coding: utf-8 -*-
import unittest

from cloverforge.tokenizer.vocab import encode, MASK_ID
from cloverforge.world.sequence import SequenceOracle
from cloverforge.backbone.model import Backbone
from cloverforge.diffusion.denoiser import Denoiser
from cloverforge.diffusion.sampler import generate
from cloverforge.utils.linalg import argmax


def _mk(skill=0.9):
    m = dict(dim=16, n_layers=1, n_heads=2, n_experts=2, top_k=1, expert_hidden=32, vocab_size=64)
    bb = Backbone(**m, key="t")
    return Denoiser(bb, SequenceOracle(), skill=skill)


class TestSampler(unittest.TestCase):
    def test_generate_length(self):
        pids = encode("ab")
        full = generate(_mk(), pids, "ab", 8, 4)
        self.assertEqual(len(full), 2 + 8)

    def test_quality_improves_with_skill(self):
        o = SequenceOracle()
        prompt, pids = "ab", encode("ab")
        target = o.completion(prompt, 8)
        accs = []
        for s in (0.5, 1.0):
            full = generate(_mk(s), pids, prompt, 8, 4)
            accs.append(sum(1 for i in range(2, 10) if full[i] == target[i - 2]) / 8)
        self.assertLessEqual(accs[0], accs[1])


if __name__ == "__main__":
    unittest.main()
