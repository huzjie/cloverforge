# -*- coding: utf-8 -*-
import unittest

from cloverforge.diffusion.semi_ar import semi_ar_blocks


class TestSemiAr(unittest.TestCase):
    def test_single_block(self):
        self.assertEqual(semi_ar_blocks(16, 1), [list(range(16))])

    def test_partition_covers_all(self):
        blocks = semi_ar_blocks(16, 4)
        flat = [i for b in blocks for i in b]
        self.assertEqual(sorted(flat), list(range(16)))


if __name__ == "__main__":
    unittest.main()
