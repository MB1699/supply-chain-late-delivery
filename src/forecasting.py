"""Simple, explainable weekly forecasting methods.
Convention: nxt[t] = forecast for week t+1, made after seeing week t."""

import numpy as np
import pandas as pd


def naive_next(d):
    """Next week = this week."""
    return np.asarray(d, dtype=float).copy()


def moving_average_next(d, window=4):
    """Next week = average of the last `window` weeks."""
    return pd.Series(np.asarray(d, dtype=float)).rolling(window, min_periods=1).mean().values


def ses_next(d, alpha=0.3):
    """Simple exponential smoothing: recent weeks count more (alpha = weight on the latest week)."""
    d = np.asarray(d, dtype=float)
    out = np.empty(len(d))
    level = d[0]
    for t, x in enumerate(d):
        level = x if t == 0 else alpha * x + (1 - alpha) * level
        out[t] = level
    return out


def seasonal_naive_next(d, season=52):
    """Next week = same week last year (falls back to naive when there's no history yet)."""
    d = np.asarray(d, dtype=float)
    out = np.empty(len(d))
    for t in range(len(d)):
        k = t + 1 - season
        out[t] = d[k] if k >= 0 else d[t]
    return out


def one_step_errors(d, nxt, start):
    """Pair each actual week t (t >= start) with the forecast made at t-1."""
    d, nxt = np.asarray(d, dtype=float), np.asarray(nxt, dtype=float)
    idx = np.arange(max(start, 1), len(d))
    return d[idx], nxt[idx - 1]


def wape(actual, forecast):
    """Weighted Absolute Percentage Error = total |error| / total actual."""
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    return np.abs(a - f).sum() / a.sum() if a.sum() > 0 else np.nan


def bias(actual, forecast):
    """Positive = over-forecasting, negative = under-forecasting."""
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    return (f - a).sum() / a.sum() if a.sum() > 0 else np.nan


def tune_alpha(d_train, grid=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)):
    """Pick the smoothing weight with the lowest WAPE on training data only."""
    best_alpha, best_score = None, np.inf
    for a in grid:
        act, fc = one_step_errors(d_train, ses_next(d_train, a), 1)
        score = wape(act, fc)
        if score < best_score:
            best_alpha, best_score = a, score
    return best_alpha