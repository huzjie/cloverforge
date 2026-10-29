# -*- coding: utf-8 -*-
import unittest

from cloverforge.utils.linalg import matvec, matmul, add, scale, softmax, argmax


class TestLinalg(unittest.TestCase):
    def test_matvec(self):
        self.assertEqual(matvec([[1, 2], [3, 4]], [1, 1]), [3, 7])

    def test_matmul_shape(self):
        A = [[1, 2, 3], [4, 5, 6]]
        B = [[7, 8], [9, 10], [11, 12]]
        C = matmul(A, B)
        self.assertEqual(len(C), 2)
        self.assertEqual(len(C[0]), 2)

    def test_add_scale(self):
        self.assertEqual(add([1, 2], [3, 4]), [4, 6])
        self.assertEqual(scale([1, 2], 3), [3, 6])

    def test_softmax_sums_to_one(self):
        self.assertAlmostEqual(sum(softmax([1.0, 2.0, 3.0])), 1.0)

    def test_argmax(self):
        self.assertEqual(argmax([1.0, 5.0, 3.0]), 1)


if __name__ == "__main__":
    unittest.main()
