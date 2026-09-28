from __future__ import annotations

import numpy as np
from scipy.stats import t


def holt_damped_forecast(
    y,
    alpha: float = 0.2,
    beta: float = 0.1,
    phi: float = 0.8,
):
    """One-step Holt with damped trend, no seasonality.

    Precision uses one-step errors from the 3rd observation onward.
    Returns (yhat_new, level, trend, errors, sigma, df).
    """
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        raise ValueError("Holt needs at least 2 historical points")

    n = y.size
    level = np.zeros(n)
    trend = np.zeros(n)
    yhat = np.full(n, np.nan)

    level[0] = y[0]
    trend[0] = y[1] - y[0]
    for i in range(1, n):
        yhat[i] = level[i - 1] + phi * trend[i - 1]
        level[i] = alpha * y[i] + (1.0 - alpha) * yhat[i]
        trend[i] = beta * (level[i] - level[i - 1]) + (1.0 - beta) * phi * trend[i - 1]

    errors = y[2:] - yhat[2:]
    if errors.size >= 2:
        sigma = float(np.std(errors, ddof=1))
        df = int(errors.size - 1)
    else:
        sigma = float("nan")
        df = 0

    yhat_new = float(level[-1] + phi * trend[-1])
    return yhat_new, level, trend, errors, sigma, df


def tolerance_bounds(yhat: float, sigma: float, df: int, pr: float = 0.99):
    """Return (lower, upper, t_crit). df < 1 yields NaNs."""
    if df < 1 or not np.isfinite(sigma):
        return float("nan"), float("nan"), float("nan")
    t_crit = float(t.ppf((1.0 + pr) / 2.0, df))
    half = t_crit * sigma
    return yhat - half, yhat + half, t_crit
