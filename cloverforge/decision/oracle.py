# -*- coding: utf-8 -*-
"""Ground-truth total order used to generate the decision questions."""
from __future__ import annotations

from ..world.rank import RankOracle


class DecisionOracle(RankOracle):
    """A deterministic true ranking over n items (e.g. option quality)."""
