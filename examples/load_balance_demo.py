# -*- coding: utf-8 -*-
"""Example: MoE load balancing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloverforge.config import load_config
from cloverforge.moe.gate import MoEBlock
from cloverforge.moe.load_balance import load_balance_stats
from cloverforge.utils.rng import stable_vector

cfg = load_config()
m = cfg["model"]
moe = MoEBlock(m["dim"], m["n_experts"], m["top_k"], m["expert_hidden"], "demo")
batch = [stable_vector("demo:%d" % i, m["dim"]) for i in range(32)]

for skill in (0.5, 0.75, 1.0):
    gp = [moe.gate_probs(v, skill) for v in batch]
    fractions, probs, aux = load_balance_stats(gp)
    print("skill=%.2f  aux=%.3f  max_fraction=%.3f  fractions=%s"
          % (skill, aux, max(fractions), ["%.2f" % f for f in fractions]))
print("\nLoad-balancing loss shrinks the 'hot expert' bias as training proceeds.")
