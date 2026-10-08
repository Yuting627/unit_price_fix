"""Fuller right-tail engine tests."""
from __future__ import annotations

import numpy as np
import pytest

from core.outlier_rules.fuller import fuller_right_tail


def test_full_right_tail_detects_known_outlier():
    # exponential-ish tail topped by a single spike
    x = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 50.0])
    mask, corrected = fuller_right_tail(x)
    assert mask.shape == x.shape
    assert int(mask.sum()) == 1
    assert mask[-1]  # the spike is flagged
    assert corrected[-1] < x[-1]  # winsorized down
    # non-flagged values are untouched
    assert np.array_equal(corrected[:-1], x[:-1])


def test_full_right_tail_skips_when_no_variation():
    x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    mask, corrected = fuller_right_tail(x)
    assert not mask.any()
    assert np.array_equal(corrected, x)


def test_full_right_tail_skips_below_critical_ratio():
    # smooth growth, no spike: max < median * critical_ratio
    x = np.linspace(1.0, 4.0, 20)
    mask, corrected = fuller_right_tail(x)
    assert not mask.any()
    assert np.array_equal(corrected, x)


def test_full_right_tail_skips_too_few_observations():
    x = np.array([1.0, 2.0, 100.0])
    mask, corrected = fuller_right_tail(x, min_n=5)
    assert not mask.any()
    assert np.array_equal(corrected, x)


def test_full_right_tail_preserves_nan_alignment():
    x = np.array([1, 1, 2, 2, 3, np.nan, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 50.0])
    mask, corrected = fuller_right_tail(x)
    assert np.isnan(corrected[5])
    assert not mask[5]
    assert corrected[-1] < x[-1]
