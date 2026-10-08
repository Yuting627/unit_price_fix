"""Data reading and processing steps shared by setup and production.

read -> match by period -> aggregate -> write. No detection logic here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

BEFORE_MP_REQUIRED = ("period_id", "store_id", "category", "sales_value", "sales_unit")
BEFORE_MP_OPTIONAL = (
    "prod_id", "prod_desc_raw", "nankey",
    "brand", "packsize", "imdb_packsize", "score", "price_promo_factor",
)
TARGET_COLS = ("PERIODCODE", "STOREID", "PROVINCE", "SHOPTYPE", "PLATFORMNAME")
ATTR_COLS = ("PLATFORMNAME", "SHOPTYPE", "PROVINCE")

_BMP_PATTERN = re.compile(r"O2O_itemcoding_output_(\d{8})_.*\.csv(\.gz)?$")
_TARGET_PATTERN = re.compile(r"target_shop_qc_(\d{8})\.csv$")


# ---------------------------------------------------------------- read
def read_before_mp(path, encoding: str = "gb18030", keep_desc: bool = True) -> pd.DataFrame:
    header = pd.read_csv(path, encoding=encoding, nrows=0).columns
    missing = [c for c in BEFORE_MP_REQUIRED if c not in header]
    if missing:
        raise ValueError(f"missing columns {missing} in {path}")
    optional = [c for c in BEFORE_MP_OPTIONAL if c in header]
    if not keep_desc:
        optional = [c for c in optional if c != "prod_desc_raw"]
    df = pd.read_csv(path, encoding=encoding, usecols=list(BEFORE_MP_REQUIRED) + optional, low_memory=False)
    df["period_id"] = df["period_id"].astype(int)
    df["store_id"] = df["store_id"].astype(str)
    df["category"] = df["category"].astype(str)
    df["sales_value"] = pd.to_numeric(df["sales_value"], errors="coerce").astype(float)
    df["sales_unit"] = pd.to_numeric(df["sales_unit"], errors="coerce").astype(float)
    if "prod_id" in df.columns:
        df["prod_id"] = df["prod_id"].astype("int64")
    if "nankey" in df.columns:
        df["nankey"] = df["nankey"].astype("int64")
    if "prod_desc_raw" in df.columns:
        df["prod_desc_raw"] = df["prod_desc_raw"].astype(str)
    for col in ("brand", "packsize", "imdb_packsize"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    for col in ("score", "price_promo_factor"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def list_before_mp_files(directory) -> dict[int, Path]:
    """One file per period; a ``_fixed`` file wins over other files of the same period."""
    chosen: dict[int, Path] = {}
    for path in sorted(Path(directory).iterdir()):
        m = _BMP_PATTERN.match(path.name)
        if not m:
            continue
        period = int(m.group(1))
        prev = chosen.get(period)
        if prev is None or ("_fixed" in path.name and "_fixed" not in prev.name):
            chosen[period] = path
    return dict(sorted(chosen.items()))


def read_before_mp_periods(directory, periods=None, encoding: str = "gb18030", keep_desc: bool = False) -> pd.DataFrame:
    files = list_before_mp_files(directory)
    wanted = files.keys() if periods is None else [p for p in periods if p in files]
    frames = []
    for period in wanted:
        df = read_before_mp(files[period], encoding=encoding, keep_desc=keep_desc)
        frames.append(df[df["period_id"] == period])
    if not frames:
        raise FileNotFoundError(f"no before_mp files for periods {periods} in {directory}")
    return pd.concat(frames, ignore_index=True)


def read_store_target(path, encoding: str = "gb18030") -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding)
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing} in {path}")
    out = df[list(TARGET_COLS)].copy()
    out["PERIODCODE"] = out["PERIODCODE"].astype(int)
    for col in ("STOREID", "PROVINCE", "SHOPTYPE", "PLATFORMNAME"):
        out[col] = out[col].astype(str)
    return out.drop_duplicates(["PERIODCODE", "STOREID"]).reset_index(drop=True)


def list_store_target_files(directory) -> dict[int, Path]:
    out = {}
    for path in sorted(Path(directory).iterdir()):
        m = _TARGET_PATTERN.match(path.name)
        if m:
            out[int(m.group(1))] = path
    return dict(sorted(out.items()))


def read_store_target_periods(directory, periods=None, encoding: str = "gb18030") -> pd.DataFrame:
    files = list_store_target_files(directory)
    wanted = files.keys() if periods is None else [p for p in periods if p in files]
    frames = [read_store_target(files[p], encoding=encoding) for p in wanted]
    if not frames:
        raise FileNotFoundError(f"no store_target files for periods {periods} in {directory}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- match
def merge_by_period(bmp: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner join on (period_id, store_id) = (PERIODCODE, STOREID).

    Returns the merged rows and per-period unmatched store counts.
    """
    tgt = target.rename(columns={"PERIODCODE": "period_id", "STOREID": "store_id"})
    tgt = tgt[["period_id", "store_id", *ATTR_COLS]]
    merged = bmp.merge(tgt, on=["period_id", "store_id"], how="inner")
    return merged, unmatched_stores(bmp, target)


def unmatched_stores(bmp: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Per period: matched / before_mp-only / target-only store counts."""
    tgt = target.rename(columns={"PERIODCODE": "period_id", "STOREID": "store_id"})
    bmp_stores = bmp[["period_id", "store_id"]].drop_duplicates()
    tgt_stores = tgt[["period_id", "store_id"]].drop_duplicates()
    tgt_stores = tgt_stores[tgt_stores["period_id"].isin(bmp_stores["period_id"].unique())]
    both = bmp_stores.merge(tgt_stores, how="outer", indicator=True)
    unmatched = (
        both.groupby("period_id")["_merge"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["both", "left_only", "right_only"], fill_value=0)
        .rename(columns={"both": "matched_stores", "left_only": "bmp_only_stores", "right_only": "target_only_stores"})
        .reset_index()
    )
    unmatched.columns.name = None
    return unmatched


# ---------------------------------------------------------------- aggregate
def agg_store_category(df: pd.DataFrame, item_key: str = "prod_id") -> pd.DataFrame:
    """period x store x category totals with n_item, avg_price and cat_share."""
    keys = ["period_id", "store_id", "category", *ATTR_COLS]
    out = (
        df.groupby(keys, sort=False, observed=True)
        .agg(
            sales_value=("sales_value", "sum"),
            sales_unit=("sales_unit", "sum"),
            n_item=(item_key, "nunique"),
        )
        .reset_index()
    )
    unit = out["sales_unit"].where(out["sales_unit"] > 0)
    out["avg_price"] = out["sales_value"] / unit
    store_total = out.groupby(["period_id", "store_id"])["sales_value"].transform("sum")
    out["cat_share"] = np.where(store_total > 0, out["sales_value"] / store_total.where(store_total > 0), np.nan)
    return out.sort_values(["period_id", "store_id", "category"]).reset_index(drop=True)


def zero_fill_categories(sc: pd.DataFrame, store_ibd: pd.Series) -> pd.DataFrame:
    """Complete store x category rows inside each IBD: stores not selling a category get 0.

    ``sc`` is one period of store x category rows; ``store_ibd`` maps store_id -> ibd_id.
    Categories come from those sold by at least one store of the IBD. Adds ``zero_filled``.
    """
    df = sc.assign(ibd_id=sc["store_id"].map(store_ibd)).dropna(subset=["ibd_id"])
    stores = df[["period_id", "store_id", "ibd_id", *ATTR_COLS]].drop_duplicates("store_id")
    cats = df[["ibd_id", "category"]].drop_duplicates()
    grid = stores.merge(cats, on="ibd_id").drop(columns="ibd_id")
    out = grid.merge(sc, on=["period_id", "store_id", "category", *ATTR_COLS], how="left")
    out["zero_filled"] = out["sales_value"].isna()
    out[["sales_value", "sales_unit", "n_item", "cat_share"]] = out[["sales_value", "sales_unit", "n_item", "cat_share"]].fillna(0)
    return out.sort_values(["store_id", "category"]).reset_index(drop=True)


def agg_province_shoptype(df: pd.DataFrame) -> pd.DataFrame:
    """Sum sales and unique stores by shoptype x province x category x platform."""
    out = (
        df.groupby(["period_id", "SHOPTYPE", "PROVINCE", "category", "PLATFORMNAME"], sort=False)
        .agg(
            sales_value=("sales_value", "sum"),
            sales_unit=("sales_unit", "sum"),
            store_count=("store_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"PROVINCE": "province", "SHOPTYPE": "shoptype", "PLATFORMNAME": "platform"})
    )
    return out.sort_values(["shoptype", "province", "category", "platform"]).reset_index(drop=True)


def agg_items(df: pd.DataFrame, item_key: str) -> pd.DataFrame:
    """period x store x category x item totals (for drilldown / unit_fix)."""
    keys = ["period_id", "store_id", "category", item_key]
    agg = {"sales_value": ("sales_value", "sum"), "sales_unit": ("sales_unit", "sum")}
    for col in ("prod_desc_raw", "brand", "packsize", "imdb_packsize", "score", "price_promo_factor", "nankey"):
        if col in df.columns and col not in keys:
            agg[col] = (col, "first")
    return df.groupby(keys, sort=False, observed=True).agg(**agg).reset_index()


# ---------------------------------------------------------------- panel
@dataclass
class Panel:
    """Multi-period inputs after quality and matching."""

    sc: pd.DataFrame
    items: pd.DataFrame
    dq: pd.DataFrame
    periods: list[int]


def load_panel(
    before_mp_dir,
    store_target_dir,
    item_key: str,
    periods=None,
    item_periods=(),
    encoding: str = "gb18030",
    quality_fn: Callable | None = None,
    logger=None,
) -> Panel:
    """Read each period, run quality, match with store_target, aggregate.

    Item-level rows are kept only for ``item_periods`` (drilldown needs current and previous).
    """
    bmp_files = list_before_mp_files(before_mp_dir)
    tgt_files = list_store_target_files(store_target_dir)
    wanted = [p for p in (bmp_files if periods is None else periods) if p in bmp_files]
    sc_frames, item_frames, dq_frames, loaded = [], [], [], []
    for period in wanted:
        if period not in tgt_files:
            if logger:
                logger.warning("[LOAD] %s: no store_target file, period skipped", period)
            continue
        keep_items = period in item_periods
        bmp = read_before_mp(bmp_files[period], encoding=encoding, keep_desc=keep_items)
        bmp = bmp[bmp["period_id"] == period]
        target = read_store_target(tgt_files[period], encoding=encoding)
        target = target[target["PERIODCODE"] == period]
        unmatched = unmatched_stores(bmp, target)
        if quality_fn is not None:
            bmp, dq = quality_fn(bmp, unmatched)
            dq_frames.append(dq)
        merged, _ = merge_by_period(bmp, target)
        sc_frames.append(agg_store_category(merged, item_key=item_key))
        if keep_items:
            items = agg_items(merged, item_key)
            attrs = merged[["store_id", *ATTR_COLS]].drop_duplicates("store_id")
            item_frames.append(items.merge(attrs, on="store_id", how="left"))
        loaded.append(period)
        if logger:
            logger.info("[LOAD] %s: file=%s rows=%d matched_stores=%d", period, bmp_files[period].name, len(merged), merged["store_id"].nunique())
        del bmp, merged
    if not sc_frames:
        raise FileNotFoundError("no period could be loaded")
    return Panel(
        sc=pd.concat(sc_frames, ignore_index=True),
        items=pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame(),
        dq=pd.concat(dq_frames, ignore_index=True) if dq_frames else pd.DataFrame(),
        periods=loaded,
    )


# ---------------------------------------------------------------- write
def write_excel(sheets: dict[str, pd.DataFrame], path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(dest, engine="openpyxl") as writer:
        for name, df in sheets.items():
            frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return dest
