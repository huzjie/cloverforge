# -*- coding: utf-8 -*-
import unittest

from cloverforge.backends.pure import PureBackend


class TestBackends(unittest.TestCase):
    def test_pure_backend(self):
        b = PureBackend()
        self.assertEqual(b.matvec([[1, 2], [3, 4]], [1, 1]), [3, 7])
        self.assertAlmostEqual(sum(b.softmax([1.0, 2.0, 3.0])), 1.0)
        self.assertEqual(b.argmax([1.0, 5.0, 3.0]), 1)
        self.assertEqual(b.relu([-1.0, 2.0]), [0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
