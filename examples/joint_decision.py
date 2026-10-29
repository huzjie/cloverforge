# -*- coding: utf-8 -*-
"""Example: joint vs independent decision."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloverforge.config import load_config
from cloverforge.world.rank import RankOracle
from cloverforge.decision.independent import build_adjacency, pointwise_accuracy
from cloverforge.decision.joint import copeland_order
from cloverforge.decision.consistency import cycle_count
from cloverforge.decision.ranking import kendall_tau

cfg = load_config()
n = cfg["decision"]["n_items"]
oracle = RankOracle(cfg["training"]["seed"])
true_order = oracle.ranks(n)

for skill in (0.6, 0.8, 1.0):
    adj = build_adjacency(n, skill, oracle)
    cyc = cycle_count(adj, n)
    ind = pointwise_accuracy(n, skill, oracle)
    order = copeland_order(adj, n)
    tau = kendall_tau(order, true_order)
    print("skill=%.1f  independent(acc=%.3f, conflicts=%d)  joint(tau=%.3f, conflicts=0)"
          % (skill, ind, cyc, tau))
print("\nJoint mode resolves the whole group globally -> no logical conflicts.")
