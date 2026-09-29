from __future__ import annotations

import numpy as np
import pandas as pd


def add_national_category_share(df: pd.DataFrame) -> pd.DataFrame:
    """national_share = this category / all categories nationwide in the same period."""
    out = df.copy()
    cat_total = out.groupby(["period_id", "category"], sort=False)["sales_value"].transform("sum")
    national_total = out.groupby("period_id", sort=False)["sales_value"].transform("sum")
    out["national_share"] = np.where(national_total == 0, 0.0, cat_total / national_total)
    return out


def add_national_share_loo(df: pd.DataFrame) -> pd.DataFrame:
    """national_share_loo = category mix of all other provinces this period."""
    out = df.copy()
    cat_total = out.groupby(["period_id", "category"], sort=False)["sales_value"].transform("sum")
    national_total = out.groupby("period_id", sort=False)["sales_value"].transform("sum")
    province_total = out.groupby(["period_id", "province"], sort=False)["sales_value"].transform(
        "sum"
    )
    inclusive = np.where(national_total == 0, 0.0, cat_total / national_total)
    denom = national_total - province_total
    numer = cat_total - out["sales_value"]
    out["national_share_loo"] = np.where(denom == 0, inclusive, numer / denom)
    return out


def add_province_weight(df: pd.DataFrame) -> pd.DataFrame:
    """province_weight = province total / national total in the same period."""
    out = df.copy()
    province_total = out.groupby(["period_id", "province"], sort=False)["sales_value"].transform(
        "sum"
    )
    national_total = out.groupby("period_id", sort=False)["sales_value"].transform("sum")
    out["province_weight"] = np.where(national_total == 0, 0.0, province_total / national_total)
    return out


def to_category_share(df: pd.DataFrame) -> pd.DataFrame:
    """share = this category / all categories in the same province and period."""
    out = add_province_weight(df)
    out = add_national_category_share(out)
    out = add_national_share_loo(out)
    province_total = out.groupby(["period_id", "province"], sort=False)["sales_value"].transform(
        "sum"
    )
    out["share"] = np.where(province_total == 0, 0.0, out["sales_value"] / province_total)
    return out


def add_short_term_share_change(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Add period-by-period short-term share changes for every province-category.

    share_chg_n1_7: vs mean of the last min(window, available) previous periods.
    share_pp_*: same in share points (current - baseline).
    First observed period of a key is left as NA.
    """

    def _rel(current: pd.Series, baseline: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=current.index, dtype=float)
        both_zero = (baseline == 0) & (current == 0)
        base_zero = (baseline == 0) & (current != 0)
        valid = baseline != 0
        out.loc[both_zero] = 0.0
        out.loc[base_zero] = np.inf
        out.loc[valid] = current.loc[valid] / baseline.loc[valid] - 1.0
        out.loc[baseline.isna()] = np.nan
        return out

    out = df.copy().sort_values(["province", "category", "period_id"])
    share = out["share"].astype(float)
    grouped = out.groupby(["province", "category"], sort=False)["share"]
    base_n17 = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    out["share_chg_n1_7"] = _rel(share, base_n17)
    out["share_pp_n1_7"] = share - base_n17
    return out.reset_index(drop=True)
