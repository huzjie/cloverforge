# -*- coding: utf-8 -*-
import unittest

from cloverforge.backbone.attention import MultiHeadAttention
from cloverforge.utils.rng import stable_matrix


class TestAttention(unittest.TestCase):
    def test_shape(self):
        attn = MultiHeadAttention(16, 4, "t")
        O = attn.forward(stable_matrix("x", 6, 16))
        self.assertEqual(len(O), 6)
        self.assertEqual(len(O[0]), 16)


if __name__ == "__main__":
    unittest.main()
