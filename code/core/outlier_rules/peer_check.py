"""Peer detectors on the current period (peer set = peer_group_id x category x period).

Cross-sectional: adjbox_value on log1p(sales_value), share_adjbox on cat_share,
item_adjbox on log1p(n_item) (counts toward evidence).
FOVA: longitude (history limits vs de-seasonalized current value), trend (de-seasonalized
log change vs cross-sectional adjbox), final (FOVA combination table).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from funcs.robust import HIGH, LOW, NA, OK, adjbox_stats, fence_from_stats, flag, trim
from .seasonal import t_final_lookup

KEYS = ["peer_group_id", "category"]
DETAIL_COLUMNS = [
    "period_id", "store_id", "category", "PLATFORMNAME", "SHOPTYPE", "PROVINCE", "ibd_id", "peer_group_id", "peer_level",
    "member_ibds", "n_peer", "n_sell", "zero_filled", "sales_value", "sales_unit", "n_item", "avg_price", "cat_share", "T_final",
    "lv_q1", "lv_q3", "lv_mc", "lv_lo", "lv_hi", "adjbox_basis", "adjbox_value",
    "share_adjbox", "item_adjbox", "adjbox_shrink_w",
    "n_hist", "long_ll", "long_ul", "fova_longitude", "trend_d", "fova_trend", "fova_final",
    "td_q1", "td_q3", "td_mc", "td_n", "c_td_q1", "c_td_q3", "c_td_mc", "c_td_n", "td_fence_src",
    "n_hist_cell", "n_hist_eff", "lon_src",
    "soft_kept", "final_ok",
]


def fova_combine(longitude: str, trend: str) -> str:
    """Combine longitude and trend: a flag survives an ``ok`` from the other test; conflict -> longitude.

    Deviates from FOVA ("any ok -> ok"): longitude limits span the whole peer group's level range,
    so a store jumping many-fold inside that range would otherwise have its trend flag vetoed.
    """
    if longitude in (NA, OK):
        return trend if trend != NA else longitude
    return longitude


def _cross_stats(cur: pd.DataFrame) -> pd.DataFrame:
    """Value / share stats use every peer store (non-sellers carry 0); items use sellers only."""
    rows = []
    for key, g in cur.groupby(KEYS, sort=False):
        pos = g[g["sales_value"] > 0]
        lv = np.log1p(g["sales_value"].clip(lower=0).to_numpy(dtype=float))
        stats = {"n_peer": len(g), "n_sell": len(pos), "lv_med": float(np.median(lv)) if len(lv) else np.nan}
        for name, vals in (
            ("lv", lv),
            ("sh", g["cat_share"].to_numpy(dtype=float)),
            ("it", np.log1p(pos["n_item"].to_numpy())),
        ):
            stats[f"{name}_q1"], stats[f"{name}_q3"], stats[f"{name}_mc"] = adjbox_stats(vals)
        rows.append({"peer_group_id": key[0], "category": key[1], **stats})
    return pd.DataFrame(rows)


def _seller_stats(cur: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    """Sellers-only (sales_value > 0) value / share stats, used when the all-store distribution degenerates."""
    names = [f"{prefix}{s}" for s in ("n", "lv_q1", "lv_q3", "lv_mc", "sh_q1", "sh_q3", "sh_mc")]
    rows = []
    for key, g in cur[cur["sales_value"] > 0].groupby(keys, sort=False):
        lv = np.log1p(g["sales_value"].to_numpy(dtype=float))
        vals = [len(g), *adjbox_stats(lv), *adjbox_stats(g["cat_share"].to_numpy(dtype=float))]
        rows.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))), **dict(zip(names, vals))})
    return pd.DataFrame(rows, columns=[*keys, *names])


def _fallback(out: pd.DataFrame, degenerate: np.ndarray, targets: list[str], sources: list[str], min_sell: int) -> np.ndarray:
    """Replace degenerate stats with group-sellers ("g_") then cell-sellers ("c_") stats; return the basis per row."""
    basis = np.where(degenerate, "none", "all").astype(object)
    for label, prefix in (("sellers", "g_"), ("cell_sellers", "c_")):
        ok = (out[f"{prefix}n"].fillna(0) >= min_sell).to_numpy()
        for s in sources:
            ok &= np.isfinite(out[f"{prefix}{s}"].to_numpy(dtype=float))
        if sources[0].endswith("_q1"):
            ok &= (out[f"{prefix}{sources[0]}"] != out[f"{prefix}{sources[1]}"]).to_numpy()
        else:
            ok &= (out[f"{prefix}{sources[1]}"] > 0).to_numpy()
        use = (basis == "none") & ok
        for t, s in zip(targets, sources):
            out.loc[use, t] = out.loc[use, f"{prefix}{s}"]
        basis[use] = label
    return basis


def _trend_values(panel: pd.DataFrame, period: int, prev: int | None, t_sv: pd.Series) -> pd.DataFrame:
    cols = [*KEYS, "store_id", "sales_value"]
    if prev is None:
        return pd.DataFrame(columns=[*KEYS, "store_id", "trend_d"])
    a = panel.loc[panel["period_id"] == period, cols]
    b = panel.loc[panel["period_id"] == prev, cols]
    both = a.merge(b, on=[*KEYS, "store_id"], suffixes=("_t", "_p"))
    both = both[(both["sales_value_t"] > 0) & (both["sales_value_p"] > 0)]
    t = both.set_index(KEYS).index.map(t_sv).to_numpy(dtype=float)
    t = np.where(np.isfinite(t) & (t > 0), t, 1.0)
    both["trend_d"] = np.log1p(both["sales_value_t"]) - np.log1p(both["sales_value_p"]) - np.log(t)
    return both[[*KEYS, "store_id", "trend_d"]]


def _stat_table(d: pd.DataFrame, keys: list[str], value: str, prefix: str) -> pd.DataFrame:
    """Q1 / Q3 / medcouple / finite-n of ``value`` per key combination (fence building block)."""
    names = [f"{prefix}{s}" for s in ("q1", "q3", "mc", "n")]
    rows = []
    for key, g in d.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        q1, q3, mc = adjbox_stats(g[value].to_numpy(dtype=float))
        n = float(np.isfinite(g[value].to_numpy(dtype=float)).sum())
        rows.append({**dict(zip(keys, key)), **dict(zip(names, (q1, q3, mc, n)))})
    return pd.DataFrame(rows, columns=[*keys, *names])


def _apply_adjbox_shrink(out: pd.DataFrame, min_n: int, scale: float, prefix: str = "lv",
                         update_basis: bool = False) -> np.ndarray:
    """Blend group adjbox stats toward cell-seller stats when the peer set is thin or a bad IBD.

    Trigger: ``n_peer < min_n``, or ``soft_kept``, or ``final_ok`` is False. Weight
    ``w = n_peer / (n_peer + scale)`` so small groups lean on the cell. Returns per-row weights
    (1.0 = no shrink). Only touches rows whose current basis is not ``none``.
    """
    n = pd.to_numeric(out["n_peer"], errors="coerce").fillna(0).to_numpy(dtype=float)
    soft = out["soft_kept"].fillna(False).astype(bool).to_numpy() if "soft_kept" in out.columns else np.zeros(len(out), dtype=bool)
    bad = (~out["final_ok"].fillna(True).astype(bool).to_numpy()) if "final_ok" in out.columns else np.zeros(len(out), dtype=bool)
    cell_n = pd.to_numeric(out["c_n"] if "c_n" in out.columns else 0, errors="coerce")
    if not isinstance(cell_n, np.ndarray):
        cell_n = cell_n.fillna(0).to_numpy(dtype=float) if hasattr(cell_n, "fillna") else np.zeros(len(out))
    c_q1 = f"c_{prefix}_q1"
    if c_q1 not in out.columns:
        return np.ones(len(out), dtype=float)
    ok_c = (cell_n >= 1) & np.isfinite(out[c_q1].to_numpy(dtype=float))
    take = ((n < min_n) | soft | bad) & ok_c
    if "adjbox_basis" in out.columns:
        take &= out["adjbox_basis"].to_numpy() != "none"
    w = np.ones(len(out), dtype=float)
    w[take] = np.where(n[take] > 0, n[take] / (n[take] + scale), 0.0)
    for s in ("q1", "q3", "mc"):
        gcol, ccol = f"{prefix}_{s}", f"c_{prefix}_{s}"
        if ccol not in out.columns:
            continue
        g = out[gcol].to_numpy(dtype=float)
        c = out[ccol].to_numpy(dtype=float)
        out[gcol] = np.where(take, w * np.nan_to_num(g) + (1 - w) * np.nan_to_num(c), g)
    if update_basis and take.any() and "adjbox_basis" in out.columns:
        basis = out["adjbox_basis"].to_numpy(copy=True)
        basis[take] = "shrink"
        out["adjbox_basis"] = basis
    return w


def _apply_fence_fallback(ref: pd.DataFrame, min_n: int, min_peer: int) -> pd.DataFrame:
    """T1: a peer group too thin to own a fence borrows the base_cell x category distribution.

    ``ref`` carries the group stats (``td_*``) and, when borrowing is enabled, the cell stats
    (``c_td_*``). Adds ``td_fence_src``:

    * ``group`` - the group has enough peers, its own stats stand;
    * ``cell``  - the group is thin and the cell is well observed, so cell stats replace them;
    * ``thin``  - the group is thin and no cell was available: the (unreliable) group stats stand.

    Only the *reference* is substituted; the store's own ``trend_d`` is never touched.
    """
    ref = ref.copy()
    n_g = pd.to_numeric(ref["td_n"], errors="coerce").fillna(0).to_numpy(dtype=float)
    src = np.where(n_g < min_n, "thin", "group").astype(object)
    if "c_td_n" in ref.columns:
        n_c = pd.to_numeric(ref["c_td_n"], errors="coerce").fillna(0).to_numpy(dtype=float)
        take = (n_g < min_n) & (n_c >= min_peer)
        for target, source in (("td_q1", "c_td_q1"), ("td_q3", "c_td_q3"), ("td_mc", "c_td_mc")):
            ref.loc[take, target] = ref.loc[take, source]
        ref.loc[take, "td_n"] = ref.loc[take, "c_td_n"]
        src = np.where(take, "cell", src).astype(object)
    ref["td_fence_src"] = src
    return ref


def _longitude_parts(arrays: list[np.ndarray], coef: float, trim_pct: float) -> tuple[float, float, float, int]:
    """(center, lo_width, hi_width, n_periods); limits = center -/+ coef * width.

    Same statistics as ``funcs.robust.fova_longitude_limits``, but the two widths are exposed
    separately so a short group history can be blended with the (far better observed) cell history.
    """
    medians, iqrs, adj_iqrs, skews = [], [], [], []
    for values in arrays:
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        arr = trim(arr[arr != 0], trim_pct)
        if arr.size < 2:
            continue
        q1, med, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        skew = (q3 - 2 * med + q1) / iqr if iqr > 0 else 0.0
        adj_q1, adj_q3 = q1, q3
        if med > 0:
            if skew > 0:
                adj_q3 = q3 * q3 / med
            elif skew < 0:
                adj_q1 = q1 * q1 / med
        medians.append(med)
        iqrs.append(iqr)
        adj_iqrs.append(adj_q3 - adj_q1)
        skews.append(skew)
    if not medians:
        return math.nan, math.nan, math.nan, 0
    center = float(np.median(medians))
    max_iqr, max_adj = float(np.max(iqrs)), float(np.max(adj_iqrs))
    skew_med = float(np.median(skews))
    return center, (max_adj if skew_med < 0 else max_iqr), (max_adj if skew_med > 0 else max_iqr), len(medians)


def _longitude_table(panel: pd.DataFrame, hist_periods: list[int], keys: list[str], fova) -> pd.DataFrame:
    """Historical level limits per key, split into center / widths / periods used."""
    cols = [*keys, "center", "lo_w", "hi_w", "n_hist"]
    hist = panel[panel["period_id"].isin(hist_periods) & (panel["sales_value"] > 0)]
    if hist.empty:
        return pd.DataFrame(columns=cols)
    hist = hist.assign(lv=np.log1p(hist["sales_value"].to_numpy(dtype=float)))
    rows = []
    for key, g in hist.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        arrays = [sub["lv"].to_numpy() for _, sub in g.groupby("period_id", sort=True)]
        center, lo_w, hi_w, n = _longitude_parts(arrays, fova.longitude_iqr_coef, fova.trim_pct)
        rows.append({**dict(zip(keys, key)), "center": center, "lo_w": lo_w, "hi_w": hi_w, "n_hist": n})
    return pd.DataFrame(rows, columns=cols)


def run_peer_check(assigned: pd.DataFrame, panel: pd.DataFrame, seasonal_table: pd.DataFrame, period: int, params, logger=None) -> pd.DataFrame:
    """Detector flags for each current store x category row in ``assigned``."""
    fova, peer = params.fova, params.peer
    a, b, c = fova.adjbox_a, fova.adjbox_b, fova.adjbox_c
    periods = sorted(p for p in panel["period_id"].unique() if p < period)
    prev = periods[-1] if periods else None
    # The global date cut on the group's history window is optional (fova.hist_start_filter);
    # the cell window always uses the latest periods, since a cell has no panel-replacement cliff.
    hist = [p for p in periods if p >= fova.hist_start_period] if fova.hist_start_filter else list(periods)
    hist_periods = hist[-fova.recommended_hist_periods :]
    cell_hist_periods = periods[-fova.recommended_hist_periods :]
    cell_keys = [*params.base_cell, "category"]

    cur = panel[panel["period_id"] == period]
    t_sv = t_final_lookup(seasonal_table, "sales_value") if not seasonal_table.empty else pd.Series(dtype=float)

    out = assigned.copy()
    out = out.merge(_cross_stats(cur), on=KEYS, how="left")
    out = out.merge(
        _longitude_table(panel, hist_periods, KEYS, fova).rename(columns={"center": "g_center", "lo_w": "g_lo", "hi_w": "g_hi"}),
        on=KEYS, how="left",
    )
    trend = _trend_values(panel, period, prev, t_sv)
    out = out.merge(_stat_table(trend, KEYS, "trend_d", "td_"), on=KEYS, how="left")
    if fova.trend_fence_fallback:
        g2cell = out[["peer_group_id", *params.base_cell]].drop_duplicates("peer_group_id")
        out = out.merge(
            _stat_table(trend.merge(g2cell, on="peer_group_id", how="inner"), cell_keys, "trend_d", "c_td_"),
            on=cell_keys, how="left",
        )
    out = out.merge(trend, on=[*KEYS, "store_id"], how="left")
    out = _apply_fence_fallback(out, fova.fence_fallback_min_n, fova.borrow_min_peer)

    t = out.set_index(KEYS).index.map(t_sv).to_numpy(dtype=float)
    out["T_final"] = np.where(np.isfinite(t) & (t > 0), t, 1.0)
    lv = np.log1p(out["sales_value"].clip(lower=0).to_numpy(dtype=float))

    out = out.merge(_seller_stats(cur, KEYS, "g_"), on=KEYS, how="left")
    out = out.merge(_seller_stats(cur.drop_duplicates(["store_id", "category"]), cell_keys, "c_"), on=cell_keys, how="left")
    zero = (out["sales_value"] <= 0).to_numpy()
    q = ["q1", "q3", "mc"]

    out["adjbox_basis"] = _fallback(out, (out["lv_q1"] == out["lv_q3"]).to_numpy(), [f"lv_{s}" for s in q], [f"lv_{s}" for s in q], peer.fallback_min_sell)
    share_basis = _fallback(out, (out["sh_q1"] == out["sh_q3"]).to_numpy(), [f"sh_{s}" for s in q], [f"sh_{s}" for s in q], peer.fallback_min_sell)
    out["adjbox_shrink_w"] = 1.0
    if fova.adjbox_cell_shrink:
        out["adjbox_shrink_w"] = _apply_adjbox_shrink(
            out, fova.adjbox_shrink_min_n, fova.adjbox_shrink_scale, "lv", update_basis=True,
        )
        _apply_adjbox_shrink(out, fova.adjbox_shrink_min_n, fova.adjbox_shrink_scale, "sh", update_basis=False)

    out["lv_lo"], out["lv_hi"] = fence_from_stats(out["lv_q1"], out["lv_q3"], out["lv_mc"], a, b, c)
    adj = flag(lv, out["lv_lo"], out["lv_hi"])
    basis = out["adjbox_basis"].to_numpy()
    # evaluate zeros only against an all-store (or shrink-blended) fence; seller-only fences skip zeros
    adj[(basis == "none") | (zero & ~np.isin(basis, ["all", "shrink"]))] = NA
    out["adjbox_value"] = adj

    for name, col, values in (
        ("sh", "share_adjbox", out["cat_share"].to_numpy(dtype=float)),
        ("it", "item_adjbox", np.log1p(out["n_item"].to_numpy(dtype=float))),
    ):
        lo, hi = fence_from_stats(out[f"{name}_q1"], out[f"{name}_q3"], out[f"{name}_mc"], a, b, c)
        flags = flag(values, lo, hi)
        if name == "sh":
            sh_ok = (share_basis == "all") | ((basis == "shrink") & (share_basis != "none"))
            flags[(share_basis == "none") | (zero & ~sh_ok)] = NA
        out[col] = flags

    lv_adj = np.log1p(out["sales_value"].clip(lower=0).to_numpy(dtype=float) / out["T_final"].to_numpy())
    ng = pd.to_numeric(out["n_hist"], errors="coerce").fillna(0).to_numpy(dtype=float)
    ok_g = np.isfinite(out["g_center"].to_numpy(dtype=float))
    center = np.nan_to_num(out["g_center"].to_numpy(dtype=float))
    lo_w = np.nan_to_num(out["g_lo"].to_numpy(dtype=float))
    hi_w = np.nan_to_num(out["g_hi"].to_numpy(dtype=float))
    c_center = np.full(len(out), np.nan)
    c_lo, c_hi = np.zeros(len(out)), np.zeros(len(out))
    n_cell = np.zeros(len(out), dtype=float)
    if fova.longitude_borrow_cell:
        cell_lon = _longitude_table(panel, cell_hist_periods, cell_keys, fova).rename(
            columns={"center": "c_center", "lo_w": "c_lo", "hi_w": "c_hi", "n_hist": "n_hist_cell"}
        )
        out = out.merge(cell_lon, on=cell_keys, how="left")
        c_center = out["c_center"].to_numpy(dtype=float)
        c_lo = np.nan_to_num(out["c_lo"].to_numpy(dtype=float))
        c_hi = np.nan_to_num(out["c_hi"].to_numpy(dtype=float))
        n_cell = pd.to_numeric(out["n_hist_cell"], errors="coerce").fillna(0).to_numpy(dtype=float)
        out["n_hist_cell"] = n_cell.astype(int)
        # weight the group by how much history it actually has; a group with no history is all cell
        w = np.where(ok_g & np.isfinite(c_center),
                     np.where(ng > 0, ng / (ng + fova.longitude_borrow_hist_scale), 0.0),
                     np.where(np.isfinite(c_center), 0.0, 1.0))
    else:
        out["n_hist_cell"] = 0
        w = np.ones(len(out), dtype=float)
    ok_c = np.isfinite(c_center)
    center = w * center + (1 - w) * np.nan_to_num(c_center)
    lo_w = w * lo_w + (1 - w) * c_lo
    hi_w = w * hi_w + (1 - w) * c_hi
    n_hist_eff = np.where(ok_g, np.nan_to_num(ng), 0.0) + np.where(ok_c, np.nan_to_num(n_cell), 0.0)
    usable = ok_g | ok_c
    long_ll = np.where(usable, center - fova.longitude_iqr_coef * lo_w, np.nan)
    long_ul = np.where(usable, center + fova.longitude_iqr_coef * hi_w, np.nan)
    longitude = flag(lv_adj, long_ll, long_ul)
    longitude[n_hist_eff < fova.min_hist_periods] = NA
    out["fova_longitude"] = longitude
    out["n_hist"] = ng.astype(int)
    out["n_hist_eff"] = n_hist_eff.astype(int)
    out["long_ll"], out["long_ul"] = long_ll, long_ul
    out["lon_src"] = np.where(ok_g & ok_c, "blend", np.where(ok_c, "cell", np.where(ok_g, "group", "none")))

    td_lo, td_hi = fence_from_stats(out["td_q1"], out["td_q3"], out["td_mc"], a, b, c)
    out["fova_trend"] = flag(out["trend_d"].to_numpy(dtype=float), td_lo, td_hi)
    out["fova_final"] = [fova_combine(l, r) for l, r in zip(out["fova_longitude"], out["fova_trend"])]

    na_rows = out["peer_level"] == "na"
    for col in ("adjbox_value", "share_adjbox", "item_adjbox", "fova_longitude", "fova_trend", "fova_final"):
        out.loc[na_rows, col] = NA
    out["period_id"] = period
    if "zero_filled" in out.columns:
        out["zero_filled"] = out["zero_filled"].fillna(False).astype(bool)
        if not peer.evaluate_zero_rows:
            out = out[~out["zero_filled"]].reset_index(drop=True)

    if logger is not None:
        max_hist = int(out["n_hist_eff"].max()) if len(out) else 0
        if max_hist < fova.min_hist_periods:
            logger.warning(
                "[PEER] %s: longitude 历史 %d 期 < %d (组 %d 期 + cell %d 期), fova_longitude 全部为 na",
                period, max_hist, fova.min_hist_periods, int(out["n_hist"].max()), int(out["n_hist_cell"].max()),
            )
        elif max_hist < fova.recommended_hist_periods:
            logger.warning("[PEER] %s: longitude 历史 %d 期 < 推荐 %d 期 (hist_periods < 13), 界限波动可能偏大", period, max_hist, fova.recommended_hist_periods)
        logger.info(
            "[PEER] %s: 参考来源 trend fence %s; longitude %s; longitude 可用行 %d/%d (%.1f%%)",
            period, out["td_fence_src"].value_counts().to_dict(), out["lon_src"].value_counts().to_dict(),
            int((out["n_hist_eff"] >= fova.min_hist_periods).sum()), len(out),
            100 * float((out["n_hist_eff"] >= fova.min_hist_periods).mean()) if len(out) else 0.0,
        )
        counts = {col: out[col].value_counts().to_dict() for col in ("adjbox_value", "share_adjbox", "item_adjbox", "fova_longitude", "fova_trend", "fova_final")}
        for col, cnt in counts.items():
            logger.info("[PEER] %s: %s %s", period, col, {k: cnt.get(k, 0) for k in (HIGH, LOW, OK, NA)})
    keep = [c_ for c_ in DETAIL_COLUMNS if c_ in out.columns] + ["lv_med"]
    return out[keep]


MISSING_COLUMNS = [
    "period_id", "store_id", "category", "PLATFORMNAME", "SHOPTYPE", "PROVINCE", "ibd_id", "peer_group_id", "peer_level",
    "n_peer", "n_sell", "sell_ratio", "peer_median_sales", "prev_sales_value", "store_n_cat", "level",
]


STORE_DROP_COLUMNS = [
    "period_id", "store_id", "PLATFORMNAME", "SHOPTYPE", "PROVINCE", "ibd_id", "n_cat", "prev_n_cat", "cat_ratio",
    "sales_value", "prev_sales_value", "sales_ratio", "n_missing",
]


def store_drop(sc: pd.DataFrame, store_map: pd.DataFrame, period: int, peer, logger=None) -> pd.DataFrame:
    """Whole-store drop: selling categories fell below store_drop_ratio x previous period (likely data gap / closure)."""
    prevs = sorted(p for p in sc["period_id"].unique() if p < period)
    if not prevs:
        return pd.DataFrame(columns=STORE_DROP_COLUMNS)
    pos = sc[sc["sales_value"] > 0]

    def per_store(p):
        g = pos[pos["period_id"] == p].groupby("store_id")
        return g.agg(n_cat=("category", "nunique"), sales_value=("sales_value", "sum"))

    j = per_store(period).join(per_store(prevs[-1]), rsuffix="_prev", how="inner").rename(
        columns={"n_cat_prev": "prev_n_cat", "sales_value_prev": "prev_sales_value"}
    )
    j = j[(j["prev_n_cat"] >= peer.store_drop_min_prev_cat) & (j["n_cat"] < peer.store_drop_ratio * j["prev_n_cat"])]
    j = j[j.index.isin(store_map["store_id"])].reset_index()
    attrs = sc[sc["period_id"] == period].drop_duplicates("store_id").set_index("store_id")
    for col in ("PLATFORMNAME", "SHOPTYPE", "PROVINCE"):
        j[col] = j["store_id"].map(attrs[col])
    j["ibd_id"] = j["store_id"].map(store_map.set_index("store_id")["ibd_id"])
    j["cat_ratio"] = j["n_cat"] / j["prev_n_cat"]
    j["sales_ratio"] = j["sales_value"] / j["prev_sales_value"]
    j["n_missing"] = 0
    j["period_id"] = period
    out = j.sort_values("prev_sales_value", ascending=False).reset_index(drop=True)[STORE_DROP_COLUMNS]
    if logger is not None:
        logger.info(
            "[STORE_DROP] %s: 整店品类数 < 上期 %.0f%% (上期 >= %d 类) %d 店",
            period, 100 * peer.store_drop_ratio, peer.store_drop_min_prev_cat, len(out),
        )
    return out


def missing_sales(assigned: pd.DataFrame, panel: pd.DataFrame, period: int, peer, logger=None, exclude_stores=()) -> pd.DataFrame:
    """该卖没卖: zero-filled rows that dropped a previously meaningful category.

    Filters (all required):
    - same-group sell coverage ``n_sell / n_peer >= missing_sell_ratio``
    - previous period sold the category with
      ``prev_sales >= missing_min_prev_ratio * peer_median_sales`` (long-term non-sellers ignored)
    - not in ``exclude_stores`` (whole-store drops go to store_drop)

    All remaining rows are ``level=suspicious``.
    """
    rows = assigned[assigned["peer_level"] != "na"]
    if rows.empty or "zero_filled" not in rows.columns:
        return pd.DataFrame(columns=MISSING_COLUMNS)
    zero = rows["zero_filled"].fillna(False).astype(bool)
    sellers = rows[~zero]
    stats = rows.groupby(KEYS).agg(n_peer=("store_id", "nunique")).join(
        sellers.groupby(KEYS).agg(n_sell=("store_id", "nunique"), peer_median_sales=("sales_value", "median"))
    ).reset_index()
    stats["n_sell"] = stats["n_sell"].fillna(0).astype(int)
    stats["sell_ratio"] = stats["n_sell"] / stats["n_peer"]

    miss = rows[zero].drop(columns=["n_peer", "n_sell"], errors="ignore").merge(stats, on=KEYS)
    miss = miss[(miss["sell_ratio"] >= peer.missing_sell_ratio) & ~miss["store_id"].isin(set(exclude_stores))]

    prev_periods = [p for p in panel["period_id"].unique() if p < period]
    if prev_periods:
        prev = panel.loc[panel["period_id"] == max(prev_periods), [*KEYS, "store_id", "sales_value"]]
        prev = prev.groupby([*KEYS, "store_id"], as_index=False)["sales_value"].sum().rename(columns={"sales_value": "prev_sales_value"})
        miss = miss.merge(prev, on=[*KEYS, "store_id"], how="left")
    else:
        miss = miss.assign(prev_sales_value=np.nan)
    miss["prev_sales_value"] = miss["prev_sales_value"].fillna(0.0)
    # Drop long-term non-sellers; require previous sales to be material vs current peer sellers.
    min_ratio = float(getattr(peer, "missing_min_prev_ratio", 0.5))
    peer_med = miss["peer_median_sales"].fillna(0.0)
    min_prev = np.where(peer_med > 0, min_ratio * peer_med, 0.0)
    miss = miss[miss["prev_sales_value"] > min_prev]
    miss["store_n_cat"] = miss["store_id"].map(sellers.groupby("store_id")["category"].nunique()).fillna(0).astype(int)
    miss["level"] = "suspicious"
    miss["period_id"] = period
    miss = miss.sort_values(["peer_median_sales"], ascending=False).reset_index(drop=True)
    out = miss[MISSING_COLUMNS]
    if logger is not None:
        logger.warning(
            "[MISSING] WARNING %s: 该卖没卖 %d 行 / %d 店（售卖比例≥%.0f%% 且上期销≥%.0f%%×同组中位）"
            "——本期为 0，仅告警，不修正商品级; 整店下降排除 %d 店",
            period, len(out), out["store_id"].nunique() if len(out) else 0,
            100 * peer.missing_sell_ratio, 100 * min_ratio, len(set(exclude_stores)),
        )
    return out


def adjbox_flags_for_c(detail: pd.DataFrame, a: float, b: float, c: float) -> np.ndarray:
    """Recompute adjbox_value flags for another fence constant (stability grid)."""
    lo, hi = fence_from_stats(detail["lv_q1"], detail["lv_q3"], detail["lv_mc"], a, b, c)
    flags = flag(np.log1p(detail["sales_value"].clip(lower=0).to_numpy(dtype=float)), lo, hi)
    flags[(detail["peer_level"] == "na").to_numpy()] = NA
    if "adjbox_basis" in detail.columns:
        basis = detail["adjbox_basis"].to_numpy()
        flags[(basis == "none") | ((detail["sales_value"] <= 0).to_numpy() & ~np.isin(basis, ["all", "shrink"]))] = NA
    return flags
