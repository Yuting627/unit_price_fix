from __future__ import annotations

import numpy as np
import pandas as pd


def category_share(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """share = category sales / total sales within group_cols (excluding category)."""
    work = df.copy()
    keys = [c for c in group_cols if c != "category"]
    if keys:
        total = work.groupby(keys, sort=False)["sales_value"].transform("sum")
    else:
        total = work["sales_value"].sum()
    work["share"] = np.where(total == 0, 0.0, work["sales_value"] / total)
    return work


def _collapse_category(df: pd.DataFrame, extra_groups: list[str] | None = None) -> pd.DataFrame:
    groups = list(extra_groups or []) + ["category"]
    return df.groupby(groups, sort=False)["sales_value"].sum().reset_index()


def compare_dwh_platform_mix(
    aug: pd.DataFrame,
    july: pd.DataFrame,
    july_period: int = 20261407,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each August dwh platform's category share to July share."""
    dwh = aug[aug["shoptype"].astype(str) == "dwh"].copy()
    if dwh.empty:
        raise ValueError("no dwh rows in August table")
    july_cur = july[july["period_id"] == july_period].copy()
    overlap_provinces = sorted(dwh["province"].astype(str).unique())

    july_all = _collapse_category(july_cur)
    july_all = category_share(july_all, ["category"])[["category", "share", "sales_value"]].rename(
        columns={"share": "share_july", "sales_value": "sales_july"}
    )

    dwh_platform = _collapse_category(dwh, ["platform"])
    dwh_platform = category_share(dwh_platform, ["platform", "category"])
    dwh_platform = dwh_platform.rename(columns={"share": "share_dwh", "sales_value": "sales_dwh"})
    by_platform = dwh_platform.merge(july_all, on="category", how="outer")
    by_platform["platform"] = by_platform["platform"].fillna("")
    by_platform["share_dwh"] = by_platform["share_dwh"].fillna(0.0)
    by_platform["share_july"] = by_platform["share_july"].fillna(0.0)
    by_platform["sales_dwh"] = by_platform["sales_dwh"].fillna(0.0)
    by_platform = by_platform[by_platform["platform"] != ""]
    by_platform["pp"] = by_platform["share_dwh"] - by_platform["share_july"]
    by_platform["abs_pp"] = by_platform["pp"].abs()
    by_platform = (
        by_platform.sort_values(["platform", "abs_pp", "category"], ascending=[True, False, True])
        .drop(columns="abs_pp")
        .reset_index(drop=True)
    )

    dwh_prov = _collapse_category(dwh, ["province", "platform"])
    dwh_prov = category_share(dwh_prov, ["province", "platform", "category"])
    dwh_prov = dwh_prov.rename(columns={"share": "share_dwh", "sales_value": "sales_dwh"})

    july_prov = _collapse_category(july_cur[july_cur["province"].isin(overlap_provinces)], ["province"])
    july_prov = category_share(july_prov, ["province", "category"])
    july_prov = july_prov.rename(columns={"share": "share_july", "sales_value": "sales_july"})

    by_province = dwh_prov.merge(
        july_prov[["province", "category", "share_july", "sales_july"]],
        on=["province", "category"],
        how="left",
    )
    by_province["share_july"] = by_province["share_july"].fillna(0.0)
    by_province["pp"] = by_province["share_dwh"] - by_province["share_july"]
    by_province["abs_pp"] = by_province["pp"].abs()
    by_province = (
        by_province.sort_values(
            ["platform", "abs_pp", "province", "category"],
            ascending=[True, False, True, True],
        )
        .drop(columns="abs_pp")
        .reset_index(drop=True)
    )
    return by_platform, by_province
