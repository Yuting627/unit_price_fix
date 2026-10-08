"""FOVA seasonality check: runtime T_final per peer group x category x metric.

Same-store chained index (robust to store-panel changes): each link is sum(x_t) / sum(x_t-1)
over stores selling in both periods; T_raw = I_t / mean(I_{t-1..t-B}); sign test on the current link.
T_final = 1 when few common stores, not significant, or direction conflicts with T_raw.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from funcs.robust import sign_test

METRICS = ("sales_value", "n_item", "avg_price")
GROUP_KEYS = ["peer_group_id", "category"]
OUT_COLUMNS = [
    "period_id", "peer_group_id", "category", "metric", "link_t", "n_common", "source", "idx_t", "idx_base",
    "base_periods_used", "n_links_borrowed", "n_links_missing", "T_raw",
    "n_plus", "n_minus", "sign_stat", "significant", "direction", "T_final", "reason",
]


def group_members(assigned: pd.DataFrame) -> pd.DataFrame:
    """(peer_group_id, category) -> ibd_id rows, one per member IBD (na groups dropped)."""
    groups = assigned.loc[assigned["peer_level"] != "na", ["peer_group_id", "category", "member_ibds"]].drop_duplicates()
    groups = groups.assign(ibd_id=groups["member_ibds"].str.split(";")).explode("ibd_id")
    return groups[["peer_group_id", "category", "ibd_id"]].drop_duplicates().reset_index(drop=True)


def peer_panel(sc: pd.DataFrame, store_map: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """All-period store x category rows expanded to every peer set they belong to.

    Stores take their current-period IBD for all history (FOVA: IBD of the last period applies to prior weeks).
    """
    panel = sc.merge(store_map[["store_id", "ibd_id"]], on="store_id", how="inner")
    return panel.merge(members, on=["ibd_id", "category"], how="inner")


LINK_COLS = ["link", "n_common", "n_plus", "n_minus"]


def _links(pos: pd.DataFrame, keys: list[str], metric: str, periods: list[int]) -> pd.DataFrame:
    """Same-store links per consecutive period pair: link = sum(x_t) / sum(x_t-1) over stores with x > 0 in both."""
    d = pos[[*keys, "store_id", "period_id", metric]].drop_duplicates([*keys, "store_id", "period_id"])
    out = []
    for p0, p1 in zip(periods[:-1], periods[1:]):
        m = d[d["period_id"] == p1].merge(d[d["period_id"] == p0], on=[*keys, "store_id"], suffixes=("_t", "_p"))
        if m.empty:
            continue
        m["plus"] = m[f"{metric}_t"] > m[f"{metric}_p"]
        m["minus"] = m[f"{metric}_t"] < m[f"{metric}_p"]
        g = m.groupby(keys).agg(
            s_t=(f"{metric}_t", "sum"), s_p=(f"{metric}_p", "sum"), n_common=("store_id", "size"),
            n_plus=("plus", "sum"), n_minus=("minus", "sum"),
        )
        g["link"] = g["s_t"] / g["s_p"]
        out.append(g[LINK_COLS].reset_index().assign(period_id=p1))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=[*keys, *LINK_COLS, "period_id"])


def compute_seasonal(panel: pd.DataFrame, period: int, seasonal, metrics=METRICS, base_cell=None) -> pd.DataFrame:
    """Same-store chained index per peer group x category x metric.

    I_first = 1, I_k = I_k-1 x link_k; T_raw = I_t / mean(I over the last ``baseline_periods`` before t).
    A link with fewer than ``min_common_stores`` common stores borrows the base_cell x category link
    (``seasonal.borrow_base_cell``), else it is set to 1 (``source = none``).
    """
    periods = sorted(p for p in panel["period_id"].unique() if p <= period)
    base_periods = [p for p in periods if p < period][-seasonal.baseline_periods :]
    cell_keys = [*base_cell, "category"] if base_cell and seasonal.borrow_base_cell and set(base_cell) <= set(panel.columns) else None
    out = []
    for metric in metrics:
        pos = panel[(panel["period_id"] <= period) & (panel[metric] > 0)]
        groups = pos.loc[pos["period_id"] == period, GROUP_KEYS + (list(base_cell) if cell_keys else [])].drop_duplicates(GROUP_KEYS)
        if groups.empty:
            continue
        grid = groups.merge(pd.DataFrame({"period_id": periods[1:]}), how="cross") if len(periods) > 1 else groups.assign(period_id=np.nan).iloc[0:0]
        g_links = _links(pos, GROUP_KEYS, metric, periods)
        grid = grid.merge(g_links, on=[*GROUP_KEYS, "period_id"], how="left")
        grid["n_common"] = grid["n_common"].fillna(0)
        grid["source"] = np.where(grid["n_common"] >= seasonal.min_common_stores, "group", "none")
        if cell_keys:
            c_links = _links(pos, cell_keys, metric, periods).rename(columns={c: f"c_{c}" for c in LINK_COLS})
            grid = grid.merge(c_links, on=[*cell_keys, "period_id"], how="left")
            use = (grid["source"] == "none") & (grid["c_n_common"].fillna(0) >= seasonal.min_common_stores)
            for c in LINK_COLS:
                grid.loc[use, c] = grid.loc[use, f"c_{c}"]
            grid.loc[use, "source"] = "base_cell"
        none = grid["source"] == "none"
        grid.loc[none, "link"] = 1.0
        grid.loc[none, ["n_plus", "n_minus"]] = 0
        grid = grid.sort_values([*GROUP_KEYS, "period_id"])
        grid["idx"] = grid.groupby(GROUP_KEYS)["link"].cumprod()

        res = groups[GROUP_KEYS].copy()
        idx = grid.set_index([*GROUP_KEYS, "period_id"])["idx"]
        first = pd.DataFrame({"period_id": [periods[0]]}).merge(res, how="cross").assign(idx=1.0).set_index([*GROUP_KEYS, "period_id"])["idx"]
        idx = pd.concat([first, idx])
        keys = list(zip(res["peer_group_id"], res["category"]))
        res["idx_t"] = [idx.get((g, c, period), np.nan) for g, c in keys]
        base_vals = [[idx.get((g, c, p), np.nan) for p in base_periods] for g, c in keys]
        res["idx_base"] = [np.nanmean(v) if len(v) and np.isfinite(v).any() else np.nan for v in base_vals]
        res["base_periods_used"] = len(base_periods)
        cur = grid.loc[grid["period_id"] == period, [*GROUP_KEYS, "link", "n_common", "n_plus", "n_minus", "source"]]
        counts = grid.assign(
            n_links_borrowed=grid["source"] == "base_cell", n_links_missing=grid["source"] == "none"
        ).groupby(GROUP_KEYS)[["n_links_borrowed", "n_links_missing"]].sum().reset_index()
        res = res.merge(cur.rename(columns={"link": "link_t"}), on=GROUP_KEYS, how="left").merge(counts, on=GROUP_KEYS, how="left")
        res[["n_links_borrowed", "n_links_missing"]] = res[["n_links_borrowed", "n_links_missing"]].fillna(0).astype(int)
        res["source"] = res["source"].fillna("none")
        res[["n_common", "n_plus", "n_minus"]] = res[["n_common", "n_plus", "n_minus"]].fillna(0).astype(int)
        res["T_raw"] = res["idx_t"] / res["idx_base"]
        res["metric"] = metric
        out.append(res)
    table = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=OUT_COLUMNS)
    table["period_id"] = period
    if table.empty:
        return table.reindex(columns=OUT_COLUMNS)
    decided = [_decide(r, seasonal) for r in table[["T_raw", "n_plus", "n_minus"]].itertuples(index=False)]
    table[["sign_stat", "significant", "direction", "T_final", "reason"]] = pd.DataFrame(decided, index=table.index)
    return table[OUT_COLUMNS]


def _decide(row, seasonal) -> tuple:
    t_raw, n_plus, n_minus = row
    if n_plus + n_minus < seasonal.min_common_stores or not np.isfinite(t_raw):
        return np.nan, False, "", 1.0, "few_common_stores"
    stat, sig, direction = sign_test(int(n_plus), int(n_minus), seasonal.sign_crit)
    if not sig:
        return stat, False, direction, 1.0, "not_significant"
    if (direction == "positive" and t_raw < 1) or (direction == "negative" and t_raw > 1):
        return stat, True, direction, 1.0, "direction_conflict"
    return stat, True, direction, float(t_raw), "ok"


def t_final_lookup(table: pd.DataFrame, metric: str) -> pd.Series:
    sub = table[table["metric"] == metric]
    return sub.set_index(GROUP_KEYS)["T_final"]


def log_seasonal(logger, period: int, table: pd.DataFrame) -> None:
    sv = table[table["metric"] == "sales_value"]
    if sv.empty:
        logger.info("[SEASONAL] %s: no groups", period)
        return
    q = np.quantile(sv["T_final"], [0.1, 0.5, 0.9])
    reasons = sv["reason"].value_counts().to_dict()
    logger.info(
        "[SEASONAL] %s: %d 组 (sales_value, 同店合计链式); significant %d; T_final P10/P50/P90 = %.2f/%.2f/%.2f; reasons %s; 当期环比来源 %s",
        period, len(sv), int(sv["significant"].sum()), q[0], q[1], q[2], reasons, sv["source"].value_counts().to_dict(),
    )


def seasonal_records(table: pd.DataFrame) -> list[dict]:
    recs = table.replace({np.nan: None}).to_dict(orient="records")
    for r in recs:
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                r[k] = int(v)
            elif isinstance(v, (np.floating,)):
                r[k] = float(v)
            elif isinstance(v, np.bool_):
                r[k] = bool(v)
    return recs
