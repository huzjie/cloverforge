# -*- coding: utf-8 -*-
import unittest

from cloverforge.tokenizer.vocab import encode, decode, SYMBOLS, N_SYMBOLS, VOCAB_SIZE


class TestVocab(unittest.TestCase):
    def test_roundtrip(self):
        s = "clover24"
        self.assertEqual(decode(encode(s)), s)

    def test_sizes(self):
        self.assertEqual(len(SYMBOLS), N_SYMBOLS)
        self.assertEqual(VOCAB_SIZE, 4 + N_SYMBOLS)


if __name__ == "__main__":
    unittest.main()
