# -*- coding: utf-8 -*-
import unittest

from cloverforge.tokenizer.vocab import encode, MASK_ID
from cloverforge.world.sequence import SequenceOracle
from cloverforge.backbone.model import Backbone
from cloverforge.diffusion.denoiser import Denoiser
from cloverforge.utils.linalg import argmax


class TestDenoiser(unittest.TestCase):
    def test_skill_improves_reconstruction(self):
        m = dict(dim=16, n_layers=1, n_heads=2, n_experts=2, top_k=1, expert_hidden=32, vocab_size=64)
        bb = Backbone(**m, key="t")
        o = SequenceOracle()
        prompt, pids = "ab", encode("ab")
        target = o.completion(prompt, 6)
        x = list(pids) + [MASK_ID] * 6
        accs = []
        for s in (0.5, 1.0):
            dn = Denoiser(bb, o, skill=s)
            lg = dn.logits(x, prompt, 2)
            accs.append(sum(1 for i in range(2, len(x)) if argmax(lg[i]) == target[i - 2]) / 6)
        self.assertLess(accs[0], accs[1])

    def test_prompt_positions_not_overwritten(self):
        m = dict(dim=16, n_layers=1, n_heads=2, n_experts=2, top_k=1, expert_hidden=32, vocab_size=64)
        bb = Backbone(**m, key="t")
        dn = Denoiser(bb, SequenceOracle(), skill=0.5)
        pids = encode("ab")
        x = list(pids) + [MASK_ID] * 4
        lg = dn.logits(x, "ab", 2)
        self.assertEqual(len(lg), 6)


if __name__ == "__main__":
    unittest.main()
