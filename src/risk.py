"""Business decision rules on top of the late-risk score."""

import numpy as np


def assign_tier(p, high=0.75, medium=0.50):
    """High if p >= high, Medium if p >= medium, else Low."""
    p = np.asarray(p, dtype=float)
    return np.select([p >= high, p >= medium], ["High", "Medium"], default="Low")


def expedite_decision(p, late_cost, expedite_cost, effectiveness):
    """Speed up an order only if the expected saving beats the cost:
       p × late_cost × effectiveness > expedite_cost"""
    p = np.asarray(p, dtype=float)
    return p * late_cost * effectiveness > expedite_cost