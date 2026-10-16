import numpy as np
from src.risk import assign_tier, expedite_decision


def test_tiers_at_boundaries():
    tiers = assign_tier([0.10, 0.50, 0.74, 0.75, 0.99])
    assert list(tiers) == ["Low", "Medium", "Medium", "High", "High"]


def test_expedite_only_above_break_even():
    # break-even p = 8 / (15 × 0.6) ≈ 0.889
    decisions = expedite_decision([0.85, 0.95], late_cost=15, expedite_cost=8, effectiveness=0.6)
    assert list(decisions) == [False, True]


def test_never_expedite_when_too_expensive():
    assert not expedite_decision([1.0], late_cost=10, expedite_cost=20, effectiveness=1.0)[0]