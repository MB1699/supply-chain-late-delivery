"""Week-by-week inventory simulation for one product category.
Demand is real (from DataCo orders); stock levels are SIMULATED."""

import numpy as np
import pandas as pd


def simulate_inventory(
    demand,
    expected,
    lead_time=1,
    safety_weeks=0.5,
    initial_cover_weeks=2,
    avg_window=4,
):
    """
    demand[t]   = real units ordered in week t
    expected[t] = what the planner expects to sell per week, decided at the END of week t
                  (naive policy: expected = this week's demand; later: the forecast)
    Unmet demand is treated as a lost sale (stockout).
    """
    demand = np.asarray(demand, dtype=float)
    expected = np.asarray(expected, dtype=float)
    n = len(demand)

    start_stock = np.zeros(n)
    arrivals = np.zeros(n)
    sold = np.zeros(n)
    lost = np.zeros(n)
    closing = np.zeros(n)
    orders = np.zeros(n)
    pipeline = np.zeros(n + lead_time + 1)   # deliveries scheduled per week

    stock = np.ceil(initial_cover_weeks * demand[:avg_window].mean())

    for t in range(n):
        start_stock[t] = stock
        arrivals[t] = pipeline[t]
        available = stock + arrivals[t]

        sold[t] = min(available, demand[t])
        lost[t] = demand[t] - sold[t]
        closing[t] = available - sold[t]

        # Reorder: cover the review week + lead time, plus a safety buffer
        recent_avg = demand[max(0, t - avg_window + 1): t + 1].mean()
        target = (lead_time + 1) * expected[t] + safety_weeks * recent_avg
        on_order = pipeline[t + 1: t + lead_time + 1].sum()
        orders[t] = np.ceil(max(0.0, target - closing[t] - on_order))
        pipeline[t + lead_time] += orders[t]

        stock = closing[t]

    return pd.DataFrame({
        "start_stock": start_stock,
        "arrivals": arrivals,
        "demand": demand,
        "units_sold": sold,
        "lost_sales": lost,
        "stockout_flag": (lost > 0).astype(int),
        "closing_stock": closing,
        "order_qty": orders,
    })