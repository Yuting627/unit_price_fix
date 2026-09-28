from __future__ import annotations

from pathlib import Path

import pandas as pd

BEFORE_MP_COLS = ("period_id", "store_id", "category", "sales_value", "sales_unit")
TARGET_COLS = ("STOREID", "PROVINCE", "SHOPTYPE", "PLATFORMNAME")


def read_before_mp(path, encoding: str = "gb18030") -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding, low_memory=False)
    missing = [c for c in BEFORE_MP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing} in {path}")
    out = df[list(BEFORE_MP_COLS)].copy()
    out["period_id"] = out["period_id"].astype(int)
    out["store_id"] = out["store_id"].astype(str)
    out["category"] = out["category"].astype(str)
    out["sales_value"] = out["sales_value"].astype(float)
    out["sales_unit"] = out["sales_unit"].astype(float)
    return out


def read_store_target(path, encoding: str = "gb18030") -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding)
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing} in {path}")
    out = df[list(TARGET_COLS)].copy()
    out["STOREID"] = out["STOREID"].astype(str)
    out["PROVINCE"] = out["PROVINCE"].astype(str)
    out["SHOPTYPE"] = out["SHOPTYPE"].astype(str)
    out["PLATFORMNAME"] = out["PLATFORMNAME"].astype(str)
    return out.drop_duplicates("STOREID")


def merge_before_mp_target(before_mp: pd.DataFrame, store_target: pd.DataFrame) -> pd.DataFrame:
    return before_mp.merge(
        store_target,
        left_on="store_id",
        right_on="STOREID",
        how="inner",
    )


def agg_province_shoptype(df: pd.DataFrame) -> pd.DataFrame:
    """Sum sales and unique stores by shoptype × province × category × platform."""
    out = (
        df.groupby(["period_id", "SHOPTYPE", "PROVINCE", "category", "PLATFORMNAME"], sort=False)
        .agg(
            sales_value=("sales_value", "sum"),
            sales_unit=("sales_unit", "sum"),
            store_count=("store_id", "nunique"),
        )
        .reset_index()
        .rename(
            columns={
                "PROVINCE": "province",
                "SHOPTYPE": "shoptype",
                "PLATFORMNAME": "platform",
            }
        )
    )
    return out.sort_values(["shoptype", "province", "category", "platform"]).reset_index(drop=True)


def write_province_shoptype(df: pd.DataFrame, path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(dest, index=False)
    return dest
