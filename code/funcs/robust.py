"""Robust statistics. All thresholds are passed in from PeerParams."""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import chi2, ks_2samp
from statsmodels.stats.stattools import medcouple as _sm_medcouple

HIGH, LOW, OK, NA = "high", "low", "ok", "na"
_FAST_MEDCOUPLE_MIN_N = 2000


def _clean(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def medcouple(x) -> float:
    arr = _clean(x)
    if arr.size < 3:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(_sm_medcouple(arr, use_fast=arr.size >= _FAST_MEDCOUPLE_MIN_N))


def adjbox_stats(x) -> tuple[float, float, float]:
    """(Q1, Q3, medcouple) of the finite values."""
    arr = _clean(x)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    return float(q1), float(q3), medcouple(arr)


def fence_from_stats(q1, q3, mc, a: float, b: float, c: float):
    """Adjusted boxplot fence (Hubert & Vandervieren); works on scalars or arrays.

    MC >= 0: [Q1 - c*exp(a*MC)*IQR, Q3 + c*exp(b*MC)*IQR]
    MC <  0: [Q1 - c*exp(-b*MC)*IQR, Q3 + c*exp(-a*MC)*IQR]
    """
    q1, q3, mc = (np.asarray(v, dtype=float) for v in (q1, q3, mc))
    iqr = q3 - q1
    pos = mc >= 0
    lo = np.where(pos, q1 - c * np.exp(a * mc) * iqr, q1 - c * np.exp(-b * mc) * iqr)
    hi = np.where(pos, q3 + c * np.exp(b * mc) * iqr, q3 + c * np.exp(-a * mc) * iqr)
    if lo.ndim == 0:
        return float(lo), float(hi)
    return lo, hi


def adjbox_fence(x, a: float, b: float, c: float) -> tuple[float, float]:
    q1, q3, mc = adjbox_stats(x)
    return fence_from_stats(q1, q3, mc, a, b, c)


def flag(x, lo, hi) -> np.ndarray:
    """high / low / ok per value; na where value or fence is missing."""
    x = np.asarray(x, dtype=float)
    lo = np.broadcast_to(np.asarray(lo, dtype=float), x.shape)
    hi = np.broadcast_to(np.asarray(hi, dtype=float), x.shape)
    out = np.full(x.shape, OK, dtype=object)
    out[x > hi] = HIGH
    out[x < lo] = LOW
    out[~np.isfinite(x) | ~np.isfinite(lo) | ~np.isfinite(hi)] = NA
    return out


def trim(x, pct: float) -> np.ndarray:
    """Drop ``pct`` of observations from each end of the sorted values."""
    arr = np.sort(_clean(x))
    k = int(math.floor(arr.size * pct))
    return arr[k : arr.size - k] if k > 0 else arr


def fova_longitude_limits(hist_by_period, coef: float, trim_pct: float) -> tuple[float, float, int]:
    """FOVA longitude test limits (LL, UL, periods used).

    For each historical period: drop zeros, trim ``trim_pct`` per side, compute Q1 / median / Q3,
    quartile skewness and skew-adjusted IQR. Across periods:
    LL/UL = Median(median) -/+ coef * Max(IQR); the skewed side uses Max(adj IQR).
    """
    medians, iqrs, adj_iqrs, skews = [], [], [], []
    for values in hist_by_period:
        arr = _clean(values)
        arr = trim(arr[arr != 0], trim_pct)
        if arr.size < 2:
            continue
        q1, med, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        skew = (q3 - 2 * med + q1) / iqr if iqr > 0 else 0.0
        adj_q1, adj_q3 = q1, q3
        if med > 0:
            if skew > 0:
                adj_q3 = q3 * q3 / med
            elif skew < 0:
                adj_q1 = q1 * q1 / med
        medians.append(med)
        iqrs.append(iqr)
        adj_iqrs.append(adj_q3 - adj_q1)
        skews.append(skew)
    if not medians:
        return math.nan, math.nan, 0
    center = float(np.median(medians))
    max_iqr, max_adj = float(np.max(iqrs)), float(np.max(adj_iqrs))
    skew_med = float(np.median(skews))
    ll, ul = center - coef * max_iqr, center + coef * max_iqr
    if skew_med > 0:
        ul = center + coef * max_adj
    elif skew_med < 0:
        ll = center - coef * max_adj
    return ll, ul, len(medians)


def sign_test(n_plus: int, n_minus: int, crit: float) -> tuple[float, bool, str]:
    """Normal-approximation sign test with continuity correction.

    stat = (|n+ - n-| - 1) / sqrt(n+ + n-); significant when stat > crit.
    """
    n = n_plus + n_minus
    if n == 0:
        return math.nan, False, ""
    stat = (abs(n_plus - n_minus) - 1) / math.sqrt(n)
    direction = "positive" if n_plus > n_minus else "negative" if n_minus > n_plus else ""
    return stat, bool(stat > crit and direction), direction


# ---------------------------------------------------------------- pooling heterogeneity
def variance_inflation(a, b) -> float:
    """Var(pooled) / weighted mean of within-group variances."""
    a, b = _clean(a), _clean(b)
    if a.size + b.size < 2:
        return math.nan
    va = a.var() if a.size > 1 else 0.0
    vb = b.var() if b.size > 1 else 0.0
    strict = (a.size * va + b.size * vb) / (a.size + b.size)
    pooled = np.concatenate([a, b]).var()
    if strict == 0:
        return 1.0 if pooled == 0 else math.inf
    return float(pooled / strict)


def ks_stat(a, b) -> float:
    a, b = _clean(a), _clean(b)
    if a.size == 0 or b.size == 0:
        return math.nan
    return float(ks_2samp(a, b).statistic)


def psi(a, b, bins: int, eps: float = 1e-4) -> float:
    """Population stability index of ``b`` against ``a`` on pooled quantile bins."""
    a, b = _clean(a), _clean(b)
    if a.size == 0 or b.size == 0:
        return math.nan
    edges = np.unique(np.quantile(np.concatenate([a, b]), np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / a.size
    pb = np.histogram(b, edges)[0] / b.size
    pa, pb = np.clip(pa, eps, None), np.clip(pb, eps, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def psi_bins_for(n_a: int, n_b: int, bins: int, min_bin_count: int) -> int:
    """Quantile bins so the smaller sample expects >= ``min_bin_count`` per bin (0 when fewer than 2 bins fit)."""
    k = min(bins, min(n_a, n_b) // max(min_bin_count, 1))
    return k if k >= 2 else 0


def heterogeneity(a, b, bins: int, alpha: float = 0.05, min_bin_count: int = 5) -> dict:
    """VI / KS / PSI of ``b`` against ``a`` with sample-size aware critical values.

    Under "same distribution" KS and PSI shrink only like 1/sqrt(n) and (k-1)(1/n_a+1/n_b), so fixed cut-offs
    reject almost every small sample. ``ks_p`` is the two-sample KS p-value; ``psi_crit`` is the chi-square
    critical value chi2_{1-alpha, k-1} * (1/n_a + 1/n_b) of PSI under the null. ``med_ratio`` is the median of
    ``b`` over the median of ``a`` on the original scale (inputs are log1p values).
    """
    a, b = _clean(a), _clean(b)
    k = psi_bins_for(a.size, b.size, bins, min_bin_count)
    if a.size and b.size:
        ks_res = ks_2samp(a, b)
        ks, ks_p = float(ks_res.statistic), float(ks_res.pvalue)
        med_a = np.expm1(np.median(a))
        med_ratio = float(np.expm1(np.median(b)) / med_a) if med_a > 0 else math.nan
    else:
        ks, ks_p, med_ratio = math.nan, math.nan, math.nan
    return {
        "med_ratio": med_ratio,
        "vi": variance_inflation(a, b),
        "ks": ks,
        "ks_p": ks_p,
        "psi": psi(a, b, k) if k else math.nan,
        "psi_bins": k,
        "psi_crit": float(chi2.ppf(1 - alpha, k - 1) * (1 / a.size + 1 / b.size)) if k else math.nan,
        "alpha": alpha,
    }


def passes(het: dict, vi_max: float, ks_max: float, psi_max: float, med_ratio_max: float = math.inf) -> bool:
    """VI below ``vi_max``; medians within ``med_ratio_max`` times of each other; KS and PSI either below the
    fixed cut-off or not significant at ``het['alpha']``.

    The median ratio is an effect size that does not depend on n: with a handful of stores KS / PSI are
    rarely significant, so a 7x level gap would otherwise pass. PSI is skipped when the samples are too
    small for 2 bins (``psi_bins`` = 0).
    """
    vi = het["vi"]
    if not (np.isfinite(vi) and vi < vi_max):
        return False
    ratio = het.get("med_ratio", 1.0)
    if np.isfinite(med_ratio_max) and not (np.isfinite(ratio) and 1 / med_ratio_max <= ratio <= med_ratio_max):
        return False
    ks_ok = het["ks"] < ks_max or het.get("ks_p", 0.0) >= het.get("alpha", 1.0)
    if not het.get("psi_bins", 1):
        return bool(ks_ok)
    psi_ok = het["psi"] < max(psi_max, het.get("psi_crit", -math.inf))
    return bool(ks_ok and psi_ok)


def cv(values) -> float:
    """Coefficient of variation (FBD reference, logged only)."""
    arr = _clean(values)
    if arr.size < 2 or arr.mean() == 0:
        return math.nan
    return float(arr.std(ddof=1) / arr.mean())
