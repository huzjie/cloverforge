# -*- coding: utf-8 -*-
import unittest

from cloverforge.world.sequence import SequenceOracle
from cloverforge.world.rank import RankOracle


class TestWorld(unittest.TestCase):
    def test_sequence_deterministic(self):
        o = SequenceOracle()
        self.assertEqual(o.completion("abc", 10), o.completion("abc", 10))

    def test_sequence_content_tokens(self):
        o = SequenceOracle()
        from cloverforge.tokenizer.vocab import CONTENT_OFFSET, VOCAB_SIZE
        for t in o.completion("abc", 10):
            self.assertGreaterEqual(t, CONTENT_OFFSET)
            self.assertLess(t, VOCAB_SIZE)

    def test_rank_is_permutation(self):
        o = RankOracle()
        self.assertEqual(sorted(o.ranks(8)), list(range(8)))

    def test_rank_transitive(self):
        o = RankOracle()
        items = list(range(8))
        for i in items:
            for j in items:
                for k in items:
                    if o.compare(i, j) and o.compare(j, k):
                        self.assertTrue(o.compare(i, k))


if __name__ == "__main__":
    unittest.main()
