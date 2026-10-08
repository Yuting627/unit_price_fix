"""Evidence grading: count anomalous detectors per store x category; assign auto_action."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import LEVELS
from funcs.robust import HIGH, LOW, OK

EVIDENCE_COLS = ("adjbox_value", "share_adjbox", "item_adjbox", "fova_final")
TIE_BREAK = ("fova_final", "adjbox_value", "share_adjbox", "item_adjbox")
# Rows at/above this level with a clear direction are eligible for automatic unit_fix.
FIX_MIN_LEVEL = "suspicious"
AUTO_FIX = "fix"
AUTO_FLAG = "flag_only"
AUTO_SKIP = "skip"


def level_for(n: int, high_min: int) -> str:
    if n <= 0:
        return "normal"
    if n >= high_min:
        return "high_confidence"
    if n == 1:
        return "watch"
    return "suspicious"


def level_rank(level: str) -> int:
    return LEVELS.index(level)


def auto_action_for(level: str, direction: str, peer_level: str) -> str:
    """Map graded level to an automated action: fix / flag_only / skip."""
    if peer_level == "na" or level == "normal" or direction not in (HIGH, LOW):
        return AUTO_SKIP
    if level_rank(level) >= level_rank(FIX_MIN_LEVEL):
        return AUTO_FIX
    return AUTO_FLAG


def demote_zero_sales_from_fix(graded: pd.DataFrame) -> pd.DataFrame:
    """本期销量为 0 / 补 0 的店×品类只告警，不允许 unit_fix 改商品。"""
    out = graded.copy()
    if out.empty or "auto_action" not in out.columns:
        return out
    zero = pd.Series(False, index=out.index)
    if "zero_filled" in out.columns:
        zero = zero | out["zero_filled"].fillna(False).astype(bool)
    if "sales_value" in out.columns:
        zero = zero | (out["sales_value"].fillna(0) <= 0)
    fix_zero = zero & out["auto_action"].eq(AUTO_FIX)
    out.loc[fix_zero, "auto_action"] = AUTO_FLAG
    return out


def grade(detail: pd.DataFrame, high_min: int) -> pd.DataFrame:
    """Add n_evidence, n_high, n_low, direction, level, evidence_list, auto_action."""
    out = detail.copy()
    for c in EVIDENCE_COLS:
        if c not in out.columns:
            out[c] = OK
    flags = out[list(EVIDENCE_COLS)]
    is_high, is_low = flags.eq(HIGH), flags.eq(LOW)
    out["n_high"], out["n_low"] = is_high.sum(axis=1), is_low.sum(axis=1)
    out["n_evidence"] = out["n_high"] + out["n_low"]
    direction = np.where(out["n_high"] > out["n_low"], HIGH, np.where(out["n_low"] > out["n_high"], LOW, "")).astype(object)
    tie = (out["n_evidence"] > 0) & (out["n_high"] == out["n_low"])
    for idx in out.index[tie]:
        chosen = next((out.at[idx, c] for c in TIE_BREAK if out.at[idx, c] in (HIGH, LOW)), "mixed")
        direction[out.index.get_loc(idx)] = chosen
    out["direction"] = direction
    out["level"] = [level_for(n, high_min) for n in out["n_evidence"]]
    names = np.array(EVIDENCE_COLS)
    anomalous = (is_high | is_low).to_numpy()
    out["evidence_list"] = [
        ",".join(f"{n}:{v}" for n, v, m in zip(names, row_vals, row) if m)
        for row_vals, row in zip(flags.to_numpy(), anomalous)
    ]
    peer = out["peer_level"] if "peer_level" in out.columns else pd.Series("ibd", index=out.index)
    out["auto_action"] = [
        auto_action_for(lv, d, pl) for lv, d, pl in zip(out["level"], out["direction"], peer)
    ]
    return demote_zero_sales_from_fix(out)


def summarize_levels(graded: pd.DataFrame) -> pd.DataFrame:
    """Counts and rates per period and level (rows with peer_level=na excluded from the rate base)."""
    base = graded[graded["peer_level"] != "na"]
    rows = []
    for period, g in base.groupby("period_id"):
        total = len(g)
        for level in LEVELS:
            sub = g[g["level"] == level]
            rows.append(
                {
                    "period_id": period,
                    "level": level,
                    "n": len(sub),
                    "rate": len(sub) / total if total else np.nan,
                    "n_high": int((sub["direction"] == HIGH).sum()),
                    "n_low": int((sub["direction"] == LOW).sum()),
                    "n_fix": int((sub["auto_action"] == AUTO_FIX).sum()) if "auto_action" in sub.columns else 0,
                }
            )
    return pd.DataFrame(rows)


def log_evidence(logger, period: int, summary: pd.DataFrame) -> None:
    s = summary[summary["period_id"] == period].set_index("level")
    if s.empty:
        return
    logger.info(
        "[PEER] %s: high_confidence %d (%.1f%%), suspicious %d, watch %d, normal %d",
        period, s.loc["high_confidence", "n"], 100 * s.loc["high_confidence", "rate"],
        s.loc["suspicious", "n"], s.loc["watch", "n"], s.loc["normal", "n"],
    )
