"""Fuller right-tail engine: exponential-tail outlier detection + winsorize-gap correction.

Reference: ``doc/_extracted/Item Inspection (Draft) 2.txt``, sections 2.8-2.13.

The right tail of a cross-section of sales rates is assumed exponential. Gaps between the
order statistics of the tail form two independent estimators of the exponential scale ``b``:

* ``b1`` = mean of the first ``n-2`` gaps (estimator of ``1/b``, ``2(n-2)`` degrees of freedom)
* ``b2`` = gap between the last two order statistics (``2`` degrees of freedom)

Under the exponential assumption the ratio ``b2 / b1 ~ F(2, 2(n-2) - 0.5)``; the ``-0.5`` makes the
denominator more tolerant for small ``n`` (per the SirVal spec). A point whose gap is significant at
``detect_limit`` is an outlier and is winsorized to ``second_to_last + expected_gap``, where
``expected_gap`` uses the F critical value at ``correct_limit`` (a probability level independent of
the detection level).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import f as _f_dist


def _clean(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _denom_df(n: int) -> float:
    """2(n-2) - 0.5 denominator degrees of freedom, floored at 1 to stay positive for tiny n."""
    return max(2 * (n - 2) - 0.5, 1.0)


def _expected_gap(b1: float, n: int, limit: float) -> float:
    """``b1 * F.ppf(limit, 2, 2(n-2) - 0.5)``: the gap expected at the tail under the exponential model."""
    return float(b1 * _f_dist.ppf(limit, 2.0, _denom_df(n)))


def fuller_right_tail(
    x,
    detect_limit: float = 0.995,
    correct_limit: float = 0.9,
    max_tail: int = 50,
    critical_ratio: float = 2.5,
    min_n: int = 5,
):
    """Detect and winsorize right-tail outliers.

    Returns ``(outlier_mask, corrected)``, both aligned to ``x`` (same shape/length). Non-finite
    entries and non-outliers keep their original value in ``corrected``.

    Steps (per spec):

    1. pre-check: too few observations / no variation -> skip
    2. Q0..Q4 (min / Q1 / median / Q3 / max), OCV=(Q3-Q1)/(1.34898*Q2)
    3. gate: only test the right tail when ``max > median * critical_ratio``
    4. tail (about 1/3 of the distribution, capped at ``max_tail``) assumed exponential;
       ``b1`` = mean of the first ``n-2`` gaps, ``b2`` = last gap, both shrunk toward the tail
       mean to avoid division by zero; statistic ``b2/b1 ~ F(2, 2(n-2)-0.5)``
    5. flag the last point when the statistic exceeds ``F.ppf(detect_limit, ...)``
    6. iterate: drop the flagged point and re-test (dynamic update of the tail)
    7. correction: ``corrected = second_to_last + expected_gap`` (``F.ppf(correct_limit, ...)``)
    """
    arr = np.asarray(x, dtype=float).ravel()
    n = len(arr)
    mask = np.zeros(n, dtype=bool)
    corrected = arr.copy()

    vals = _clean(arr)
    m = vals.size
    if m < min_n:
        return mask, corrected

    order = np.argsort(vals, kind="stable")
    sv = vals[order]
    orig_idx = np.flatnonzero(np.isfinite(arr))[order]  # sorted-finite position -> original index

    if sv[-1] <= sv[0]:  # no variation
        return mask, corrected

    median = float(np.median(sv))
    if median <= 0 or sv[-1] <= median * critical_ratio:
        return mask, corrected

    # tail = top ~1/3 of the distribution, capped at max_tail
    tail_size = max(3, min(int(max_tail), int(math.ceil(m / 3.0))))
    tail = sv[m - tail_size:]

    # ---- detection: iterate, flagging the top point while its gap is significant
    flagged: set[int] = set()
    working = tail.copy()
    while working.size >= 3:
        k = int(working.size)
        mean_t = float(working.mean())
        # b1: mean of the first (k-2) gaps = (x_{k-1} - x_1)/(k-2), shrunk toward the tail mean
        b1_raw = (working[k - 2] - working[0]) / (k - 2)
        b1 = (b1_raw * (k - 2) + mean_t) / (k - 1)
        # b2: gap between the last two order statistics, shrunk toward the tail mean
        b2 = ((working[-1] - working[-2]) + mean_t) / 2.0
        if b1 <= 0:
            break
        stat = b2 / b1
        crit = _f_dist.ppf(detect_limit, 2.0, _denom_df(k))
        if stat > crit:
            flagged.add(k - 1)
            working = working[:-1]
        else:
            break

    if not flagged:
        return mask, corrected

    # ---- correction: winsorize flagged points, stacking upward from the clean anchor
    clean = [v for i, v in enumerate(tail) if i not in flagged]
    anchor = float(clean[-1]) if clean else float(tail[0])
    kc = len(clean)
    if kc >= 3:
        b1_clean = (clean[-1] - clean[0]) / (kc - 2)
        mean_c = float(np.mean(clean))
        b1_clean = (b1_clean * (kc - 2) + mean_c) / (kc - 1)
        gap = _expected_gap(b1_clean, kc + 1, correct_limit)
    else:
        gap = _expected_gap(float(tail.mean()), len(tail), correct_limit)

    corrected_tail = tail.copy()
    prev = anchor
    for pos in sorted(flagged):
        corrected_tail[pos] = prev + gap
        prev = corrected_tail[pos]

    for pos in flagged:
        idx = orig_idx[m - tail_size + pos]
        mask[idx] = True
        corrected[idx] = corrected_tail[pos]

    return mask, corrected
