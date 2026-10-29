# -*- coding: utf-8 -*-
import unittest

from cloverforge.diffusion.corruption import forward_mask
from cloverforge.tokenizer.vocab import MASK_ID


class TestCorruption(unittest.TestCase):
    def test_mask_ratio(self):
        tokens = list(range(20))
        out, pos = forward_mask(tokens, 0.5, MASK_ID, "k")
        self.assertEqual(len(pos), 10)
        for p in pos:
            self.assertEqual(out[p], MASK_ID)

    def test_deterministic(self):
        tokens = list(range(20))
        a, _ = forward_mask(tokens, 0.5, MASK_ID, "k")
        b, _ = forward_mask(tokens, 0.5, MASK_ID, "k")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
