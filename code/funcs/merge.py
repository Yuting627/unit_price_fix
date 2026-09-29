from __future__ import annotations

import pandas as pd

MARK_COLS = [
    "n_hist",
    "outlier_n0",
    "outlier_n1_5",
    "outlier_n6_11",
    "outlier_n8",
    "agree",
    "n_methods",
    "note",
    "yhat_n0",
    "yhat_n1_5",
    "yhat_n6_11",
    "yhat_n8",
]


def attach_july_marks(
    original_df: pd.DataFrame,
    marks_df: pd.DataFrame,
    current_period: int,
) -> pd.DataFrame:
    """Keep historical sales/share; attach outlier columns only on current rows."""
    key = ["period_id", "province", "category"]
    hist = original_df[original_df["period_id"] != current_period].copy()
    current = original_df[original_df["period_id"] == current_period].copy()

    marks = marks_df.copy()
    marks["period_id"] = current_period
    mark_only = [c for c in MARK_COLS if c in marks.columns]
    marks = marks[key + mark_only]

    current = current.merge(marks, on=key, how="left")

    missing = marks.merge(current[key], on=key, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    if not missing.empty:
        extra = pd.DataFrame(
            {
                "period_id": current_period,
                "province": missing["province"],
                "category": missing["category"],
                "sales_value": 0.0,
                "share": 0.0,
            }
        )
        extra = extra.merge(marks, on=key, how="left")
        current = pd.concat([current, extra], ignore_index=True)

    ordered = list(original_df.columns) + [c for c in MARK_COLS if c not in original_df.columns]
    current = current.reindex(columns=ordered)
    out = pd.concat([hist, current], ignore_index=True)
    return out.reindex(columns=ordered)
