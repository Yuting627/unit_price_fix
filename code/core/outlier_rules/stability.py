"""Stability checks: threshold-grid Jaccard and anomaly rate per period."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .peer_check import adjbox_flags_for_c
from funcs.robust import HIGH, LOW

ANOMALY_LEVELS = ("suspicious", "high_confidence")


def jaccard(a: set, b: set) -> float:
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def _flagged(detail: pd.DataFrame, flags) -> set:
    mask = np.isin(np.asarray(flags, dtype=object), [HIGH, LOW])
    return set(zip(detail.loc[mask, "store_id"], detail.loc[mask, "category"]))


def threshold_jaccard(detail: pd.DataFrame, params) -> pd.DataFrame:
    """Jaccard of flagged store x category sets between neighbouring grid values and vs the configured value."""
    grid = params.stability.grid
    fova = params.fova
    sets = {
        "adjbox_c": {c: _flagged(detail, adjbox_flags_for_c(detail, fova.adjbox_a, fova.adjbox_b, c)) for c in grid["adjbox_c"]},
    }
    current = {"adjbox_c": fova.adjbox_c}
    rows = []
    for detector, by_thr in sets.items():
        thrs = sorted(by_thr)
        pairs = list(zip(thrs[:-1], thrs[1:]))
        cur = current[detector]
        if cur not in by_thr:
            by_thr[cur] = _flagged(detail, adjbox_flags_for_c(detail, fova.adjbox_a, fova.adjbox_b, cur))
        pairs += [(cur, t) for t in thrs if t != cur and (cur, t) not in pairs and (t, cur) not in pairs]
        for x, y in pairs:
            rows.append(
                {
                    "check": "threshold_jaccard",
                    "detector": detector,
                    "thr_a": x,
                    "thr_b": y,
                    "n_a": len(by_thr[x]),
                    "n_b": len(by_thr[y]),
                    "jaccard": jaccard(by_thr[x], by_thr[y]),
                    "vs_config": x == cur or y == cur,
                }
            )
    return pd.DataFrame(rows)


def rate_by_period(level_summary: pd.DataFrame, spike_ratio: float) -> pd.DataFrame:
    """Rate per period and level, plus combined suspicious+; spike when rate > spike_ratio x median over periods."""
    if level_summary.empty:
        return pd.DataFrame(columns=["check", "period_id", "level", "rate", "median_rate", "spike"])
    s = level_summary[["period_id", "level", "rate"]].copy()
    combined = s[s["level"].isin(ANOMALY_LEVELS)].groupby("period_id", as_index=False)["rate"].sum()
    combined["level"] = "suspicious+"
    s = pd.concat([s, combined], ignore_index=True)
    s = s[s["level"] != "normal"]
    s["median_rate"] = s.groupby("level")["rate"].transform("median")
    s["spike"] = (s["median_rate"] > 0) & (s["rate"] > spike_ratio * s["median_rate"])
    s["check"] = "rate_by_period"
    return s[["check", "period_id", "level", "rate", "median_rate", "spike"]].sort_values(["level", "period_id"]).reset_index(drop=True)


def log_stability(logger, jac: pd.DataFrame, rates: pd.DataFrame, spike_ratio: float) -> None:
    for r in jac.itertuples():
        logger.info(
            "[STABILITY] %s Jaccard %s vs %s = %.2f (n %d / %d)%s",
            r.detector, r.thr_a, r.thr_b, r.jaccard, r.n_a, r.n_b, " [vs config]" if r.vs_config else "",
        )
    for r in rates[rates["spike"]].itertuples():
        logger.warning(
            "[STABILITY] WARNING %s level=%s rate=%.3f > %.1f x median %.3f",
            r.period_id, r.level, r.rate, spike_ratio, r.median_rate,
        )
    combined = rates[rates["level"] == "suspicious+"]
    if not combined.empty:
        logger.info("[STABILITY] suspicious+ rate by period: %s", {int(p): round(float(v), 4) for p, v in zip(combined["period_id"], combined["rate"])})
