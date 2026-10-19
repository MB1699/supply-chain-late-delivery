import numpy as np
from src.forecasting import (naive_next, moving_average_next, ses_next,
                             seasonal_naive_next, one_step_errors, wape, bias)


def test_naive_repeats_last_week():
    assert list(naive_next([5, 7, 9])) == [5, 7, 9]


def test_moving_average():
    assert list(moving_average_next([2, 4, 6], window=2)) == [2, 3, 5]


def test_ses_alpha_one_equals_naive():
    d = [3, 8, 1, 6]
    assert np.allclose(ses_next(d, alpha=1.0), naive_next(d))


def test_seasonal_naive_uses_last_season():
    assert list(seasonal_naive_next([1, 2, 3, 4], season=2)) == [1, 1, 2, 3]


def test_one_step_alignment():
    actual, forecast = one_step_errors([10, 20, 30], [11, 21, 31], start=1)
    assert list(actual) == [20, 30]
    assert list(forecast) == [11, 21]


def test_wape_and_bias():
    assert wape([10, 10], [10, 10]) == 0
    assert wape([10, 10], [12, 8]) == 0.2
    assert bias([10, 10], [12, 12]) == 0.2