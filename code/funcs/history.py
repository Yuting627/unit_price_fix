from __future__ import annotations

import pandas as pd


def split_current_hist(df: pd.DataFrame, current_period: int):
    current = df[df["period_id"] == current_period].copy()
    hist = df[df["period_id"] < current_period].copy()
    return current, hist


def count_hist(hist_df: pd.DataFrame, province: str, category: str) -> int:
    mask = (hist_df["province"] == province) & (hist_df["category"] == category)
    return int(hist_df.loc[mask, "period_id"].nunique())


def hist_share_series(
    hist_df: pd.DataFrame,
    province: str,
    category: str,
    fill_calendar: bool = True,
) -> list[float]:
    """Historical shares in period order. Current period must already be excluded.

    If fill_calendar, start from the first observed period and fill inner gaps with 0
    up to the last historical period present in hist_df.
    """
    mask = (hist_df["province"] == province) & (hist_df["category"] == category)
    sub = hist_df.loc[mask, ["period_id", "share"]].drop_duplicates("period_id")
    if sub.empty:
        return []
    sub = sub.sort_values("period_id")
    if not fill_calendar:
        return sub["share"].astype(float).tolist()

    all_periods = sorted(hist_df["period_id"].unique())
    first = int(sub["period_id"].iloc[0])
    calendar = [p for p in all_periods if p >= first]
    lookup = dict(zip(sub["period_id"].astype(int), sub["share"].astype(float)))
    return [float(lookup.get(p, 0.0)) for p in calendar]
