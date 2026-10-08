"""Drill anomalous store x category combinations down to the items that drive them."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evidence import level_rank
from funcs.robust import HIGH, LOW

SRC_PEER, SRC_PREV, SRC_NEW = "peer", "own_prev", "new_item"
FLAG_NEW, FLAG_MISSING = "new_item", "missing_item"


def drill_columns(item_key: str) -> list[str]:
    return [
        "period_id", "store_id", "category", "peer_group_id", "level", "direction", "rank", item_key, "prod_desc_raw",
        "sales_value", "sales_unit", "unit_price", "expected_sales", "expected_source", "delta", "contribution",
        "item_flag",
    ]


def _unit_price(value, unit):
    value, unit = np.asarray(value, dtype=float), np.asarray(unit, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(unit > 0, value / unit, np.nan)


def run_drilldown(
    graded: pd.DataFrame,
    items_cur: pd.DataFrame,
    items_prev: pd.DataFrame,
    seasonal_table: pd.DataFrame,
    item_key: str,
    drill,
    logger=None,
) -> pd.DataFrame:
    """Top items per anomalous (level >= drill.min_level) store x category.

    ``items_cur`` / ``items_prev`` are item-level rows with ``ibd_id`` attached (current store map).
    """
    cols = drill_columns(item_key)
    min_rank = level_rank(drill.min_level)
    targets = graded[graded["level"].map(level_rank) >= min_rank]
    targets = targets[targets["direction"].isin([HIGH, LOW]) & (targets["peer_level"] != "na")]
    if targets.empty:
        if logger is not None:
            logger.info("[DRILLDOWN] 0 组合下钻")
        return pd.DataFrame(columns=cols)

    t_sv = seasonal_table[seasonal_table["metric"] == "sales_value"].set_index(["peer_group_id", "category"])["T_final"] if not seasonal_table.empty else pd.Series(dtype=float)
    cur = items_cur[items_cur["sales_value"] > 0].copy()
    cur["unit_price"] = _unit_price(cur["sales_value"], cur["sales_unit"])
    cur_by = {k: g for k, g in cur.groupby(["ibd_id", "category"])}
    own_by = {k: g for k, g in cur.groupby(["store_id", "category"])}
    prev_by = {k: g for k, g in items_prev[items_prev["sales_value"] > 0].groupby(["store_id", "category"])} if not items_prev.empty else {}
    desc_col = "prod_desc_raw" in cur.columns

    out = []
    for tgt in targets.itertuples(index=False):
        members = str(tgt.member_ibds).split(";")
        peers = [cur_by[(m, tgt.category)] for m in members if (m, tgt.category) in cur_by]
        peers = pd.concat(peers) if peers else cur.iloc[0:0]
        peers = peers[peers["store_id"] != tgt.store_id]
        pstats = peers.groupby(item_key).agg(n_peer=("store_id", "nunique"), peer_med=("sales_value", "median"))

        own = own_by.get((tgt.store_id, tgt.category), cur.iloc[0:0])
        prev = prev_by.get((tgt.store_id, tgt.category))
        t = t_sv.get((tgt.peer_group_id, tgt.category), 1.0)
        t = t if np.isfinite(t) and t > 0 else 1.0
        prev_sales = prev.groupby(item_key)["sales_value"].sum() / t if prev is not None else pd.Series(dtype=float)

        df = own[[item_key, "sales_value", "sales_unit", "unit_price"] + (["prod_desc_raw"] if desc_col else [])].copy()
        df = df.join(pstats, on=item_key)
        use_peer = df["n_peer"].fillna(0) >= drill.min_peer_stores
        prev_val = df[item_key].map(prev_sales)
        df["expected_sales"] = np.where(use_peer, df["peer_med"], np.where(prev_val.notna(), prev_val, 0.0))
        df["expected_source"] = np.where(use_peer, SRC_PEER, np.where(prev_val.notna(), SRC_PREV, SRC_NEW))
        df["item_flag"] = np.where(df["expected_source"] == SRC_NEW, FLAG_NEW, "")

        if tgt.direction == LOW and prev is not None:
            gone = prev[~prev[item_key].isin(df[item_key])]
            if not gone.empty:
                agg = {"expected_sales": ("sales_value", "sum")}
                if "prod_desc_raw" in gone.columns:
                    agg["prod_desc_raw"] = ("prod_desc_raw", "first")
                g = gone.groupby(item_key).agg(**agg).reset_index()
                g["expected_sales"] = g["expected_sales"] / t
                g["sales_value"], g["sales_unit"], g["unit_price"] = 0.0, 0.0, np.nan
                g["expected_source"], g["item_flag"] = SRC_PREV, FLAG_MISSING
                df = pd.concat([df, g], ignore_index=True)

        df["delta"] = df["sales_value"] - df["expected_sales"]
        df = df[df["delta"] > 0] if tgt.direction == HIGH else df[df["delta"] < 0]
        if df.empty:
            continue
        df["contribution"] = df["delta"] / df["delta"].sum()
        df = df.reindex(df["delta"].abs().sort_values(ascending=False).index).head(drill.top_n)

        df["rank"] = np.arange(1, len(df) + 1)
        df["period_id"], df["store_id"], df["category"] = tgt.period_id, tgt.store_id, tgt.category
        df["peer_group_id"], df["level"], df["direction"] = tgt.peer_group_id, tgt.level, tgt.direction
        out.append(df)

    result = pd.concat(out, ignore_index=True).reindex(columns=cols) if out else pd.DataFrame(columns=cols)
    if logger is not None:
        n_combo = int(targets[["store_id", "category"]].drop_duplicates().shape[0])
        logger.info(
            "[DRILLDOWN] %s: %d 组合下钻, 带出 %d 个商品; new_item %d; missing_item %d",
            int(targets["period_id"].iloc[0]), n_combo, len(result),
            int((result["item_flag"] == FLAG_NEW).sum()), int((result["item_flag"] == FLAG_MISSING).sum()),
        )
    return result
