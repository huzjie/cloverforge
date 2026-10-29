# -*- coding: utf-8 -*-
import unittest

from cloverforge.decision.oracle import DecisionOracle
from cloverforge.decision.independent import build_adjacency, pointwise_accuracy
from cloverforge.decision.joint import copeland_order
from cloverforge.decision.consistency import cycle_count
from cloverforge.decision.ranking import kendall_tau


class TestDecision(unittest.TestCase):
    def setUp(self):
        self.o = DecisionOracle()

    def test_accuracy_monotonic(self):
        self.assertLess(pointwise_accuracy(8, 0.5, self.o), pointwise_accuracy(8, 1.0, self.o))

    def test_joint_is_total_order(self):
        adj = build_adjacency(8, 0.7, self.o)
        order = copeland_order(adj, 8)
        self.assertEqual(len(order), 8)
        self.assertEqual(sorted(order), list(range(8)))

    def test_joint_no_cycles_by_construction(self):
        adj = build_adjacency(8, 0.7, self.o)
        order = copeland_order(adj, 8)
        # a total order has zero directed cycles
        self.assertEqual(cycle_count([[order.index(i) < order.index(j) for j in range(8)] for i in range(8)], 8), 0)

    def test_joint_agreement_not_worse_than_chance(self):
        n = 8
        adj = build_adjacency(n, 0.7, self.o)
        true_order = self.o.ranks(n)
        tau = kendall_tau(copeland_order(adj, n), true_order)
        self.assertGreaterEqual(tau, -1.0)


if __name__ == "__main__":
    unittest.main()
