"""Standalone backtest: how the FOVA trend / seasonal *reference* should borrow when the store
panel is replaced (20261407) and the chained index has to cross the break.

Question answered with numbers instead of theory: "趋势上能不能借鉴 C 收缩的做法?"

The production peer definition (IBD + ibd_check fallback) is held fixed. Only the *reference*
side varies, because the observation side -- a store's own ``trend_d`` -- must never be shrunk:
doing so would erase exactly the anomalies the detector is looking for.

Arms
----
* ``T0现状``        production-faithful: hard ``min_common_stores`` switch for the seasonal link
                    (no partial borrowing); trend fence from the peer group only (no fallback);
                    longitude ``na`` when ``n_hist < min_hist_periods``.
* ``T1fence回退``   T0 + conditional fallback for the trend fence: peer group -> base_cell x
                    category -> na, triggered only where the group reference is too thin.
* ``T2a/b/c重叠``   T0 + overlap-weighted seasonal link, with ``kappa_link`` = 3 / 10 / 30:

                        w = n_common / (n_common + kappa_link)
                            * n_common / min(n_sell_t, n_sell_t-1)

                    so the reference slides continuously from the peer group to the rest of the
                    base cell as the panel is replaced. The cell contribution uses ``cell \\ group``
                    (on same-store sums, so the store composition stays single). ``n_plus`` /
                    ``n_minus`` are never borrowed: the cell must not vote on a group-level move.
* ``T3long借cell``  T0 + longitude limits borrowed from base_cell x category history, weight from
                    ``n_hist`` (``w = n_hist / (n_hist + K_HIST)``), cell history unfiltered by
                    ``hist_start_period``.
* ``T4组合``        T1 + T2b + T3.

Metrics per arm: how much of the population gets a usable trend reference at all, where the
reference came from, how much the seasonal link moves versus the hard switch, flag stability when
the reference is refreshed one period late, x3 detection, and the two clustering numbers from the
README comparison (``systematic_share`` / ``province_lift_max``) that decide whether a borrow is
safe at all.

Speed notes (both are exact, not approximations):
* the fence multiplier is solved analytically instead of by a 50-step bisection over every row:
  each observation has a multiplier threshold ``c_i`` and ``rate(c) = #{c_i > c} / N``, so the
  smallest ``c`` reaching the target is the ``floor(target * N) + 1``-th largest ``c_i``.
* the trend distribution's Q1/Q3/medcouple are shift-invariant *within* a group (``T`` is constant
  inside ``peer_group_id x category``), so they are computed once per period on the raw log ratio
  and only shifted by ``-log T`` per arm. Medcouple itself is skipped by the shift.
* the base-cell fallback stats are only computed for the cells that actually contain a thin group.

Read-only with respect to production code: production modules are imported, never written to.

Usage:
    python backtest/trend_borrow_backtest.py --periods 20261406 20261407 20261408
    python backtest/trend_borrow_backtest.py --quick
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
for _p in (CODE, ROOT / "backtest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import group_scheme_backtest as B  # noqa: E402  shared harness: fence / flag / calibration / metrics

from core.config import load_params  # noqa: E402
from core.data_prep.data_prep import list_before_mp_files, load_panel, zero_fill_categories  # noqa: E402
from core.ibd_define.ibd_check import check_ibd, peer_assignment  # noqa: E402
from core.ibd_define.ibd_define import define_ibd  # noqa: E402
from core.outlier_rules.seasonal import _decide, compute_seasonal, group_members, peer_panel  # noqa: E402
from funcs.robust import flag, trim  # noqa: E402

ARMS = ("T0现状", "T1fence回退", "T2a重叠w3", "T2b重叠w10", "T2c重叠w30", "T3long借cell", "T4组合")
KAPPA_LINK = {"T2a重叠w3": 3.0, "T2b重叠w10": 10.0, "T2c重叠w30": 30.0}
T4_KAPPA = 10.0
K_HIST = 3.0                       # longitude history-borrow scale, in periods
MIN_PEER = B.MIN_PEER              # 5: a fence needs at least this many observations
MC_CAP = 1500                      # medcouple subsample cap (a robust skewness needs no more)
HEAD = "sales_value"
GROUP_KEYS = ["peer_group_id", "category"]
CELL_KEYS = ["cell_key", "category"]
LINK_COLS = ["s_t", "s_p", "n_common", "n_plus", "n_minus"]
SEAS_COLS = [*GROUP_KEYS, "link_t", "n_common", "n_plus", "n_minus", "source", "w", "c_n_common",
             "idx_t", "idx_base", "T_raw", "sign_stat", "significant", "direction", "T_final", "reason"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- statistics
def mc_of(v: np.ndarray) -> float:
    """Medcouple with a large-group subsample and the O(n log n) fast path where it works."""
    from statsmodels.stats.stattools import medcouple

    arr = np.asarray(v, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 0.0
    if arr.size > MC_CAP:
        arr = np.random.default_rng(arr.size).choice(arr, MC_CAP, replace=False)
    if arr.size >= 200:
        try:
            return float(medcouple(arr, use_fast=True))
        except Exception:  # the fast sweep can fail on heavily tied data
            pass
    return float(medcouple(arr, use_fast=False))


def group_stats(frame: pd.DataFrame, keys: list[str], value: str = "td") -> pd.DataFrame:
    """Q1 / Q3 / medcouple / n_peer per key (vectorised quantile, one medcouple per group)."""
    cols = [*keys, "q1", "q3", "mc", "n_peer"]
    if frame.empty:
        return pd.DataFrame(columns=cols)
    d = frame[[*keys, value]].dropna(subset=[value])
    if d.empty:
        return pd.DataFrame(columns=cols)
    q = d.groupby(keys, sort=False)[value].quantile([0.25, 0.75]).unstack()
    q.columns = ["q1", "q3"]
    cnt = d.groupby(keys, sort=False)[value].size().rename("n_peer")
    out = q.join(cnt).reset_index()
    out["mc"] = [mc_of(g.to_numpy(dtype=float)) for _, g in d.groupby(keys, sort=False)[value]]
    return out[cols]


def shift_stats(stats: pd.DataFrame, seas: pd.DataFrame) -> pd.DataFrame:
    """Move a per-group distribution by ``-log T`` for this arm (medcouple is shift invariant)."""
    out = stats.merge(seas[[*GROUP_KEYS, "T_final"]], on=GROUP_KEYS, how="left")
    t = out["T_final"].to_numpy(dtype=float)
    logt = np.zeros(t.shape, dtype=float)
    good = np.isfinite(t) & (t > 0)
    logt[good] = np.log(t[good])
    out["q1"] = out["q1"].to_numpy(dtype=float) - logt
    out["q3"] = out["q3"].to_numpy(dtype=float) - logt
    return out[[*GROUP_KEYS, "q1", "q3", "mc", "n_peer"]]


def logt_of(trend: pd.DataFrame, seas: pd.DataFrame) -> np.ndarray:
    """``log T`` per trend row (0 where the factor is missing or non-positive)."""
    t = trend[GROUP_KEYS].merge(seas[[*GROUP_KEYS, "T_final"]], on=GROUP_KEYS, how="left")["T_final"]
    t = t.to_numpy(dtype=float)
    out = np.zeros(t.shape, dtype=float)
    good = np.isfinite(t) & (t > 0)
    out[good] = np.log(t[good])
    return out


def fence_thresholds(td, q1, q3, mc, a: float, b: float) -> np.ndarray:
    """Per-observation multiplier threshold ``c_i``: the row is flagged iff the calibrated c < c_i.

    Adjusted boxplot fence for row i is ``[q1 - c*kl, q3 + c*kh]`` with ``kl/kh`` fixed per peer
    group, so ``td_i > hi`` and ``td_i < lo`` each give a linear bound on ``c``; the row is
    anomalous iff ``c`` is below the larger of the two.
    """
    td = np.asarray(td, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q3 = np.asarray(q3, dtype=float)
    mc = np.asarray(mc, dtype=float)
    iqr = q3 - q1
    pos = mc >= 0
    kl = np.where(pos, np.exp(a * mc) * iqr, np.exp(-b * mc) * iqr)
    kh = np.where(pos, np.exp(b * mc) * iqr, np.exp(-a * mc) * iqr)
    upper = np.where(kh > 0, (td - q3) / np.where(kh > 0, kh, 1.0),
                     np.where(td > q3, np.inf, -np.inf))
    lower = np.where(kl > 0, (q1 - td) / np.where(kl > 0, kl, 1.0),
                     np.where(td < q1, np.inf, -np.inf))
    return np.maximum(upper, lower)


def calibrate_analytic(thr: np.ndarray, target: float, lo: float = 0.01, hi: float = 500.0) -> float:
    """Smallest ``c`` whose anomaly rate ``#{thr_i > c} / N`` is at or below ``target``."""
    thr = np.asarray(thr, dtype=float)
    n = thr.size
    if n == 0:
        return lo
    m = int(math.floor(target * n))
    if m >= n:
        return lo
    c = float(np.sort(thr)[::-1][m])
    if not np.isfinite(c):
        return lo
    return float(min(max(c, lo), hi))


# ---------------------------------------------------------------- link sums
def link_sums(pos: pd.DataFrame, keys: list[str], metric: str, periods: list[int]) -> pd.DataFrame:
    """Per key x consecutive period pair: same-store sums ``s_t`` / ``s_p``, ``n_common``, sign counts.

    The same construction as production ``seasonal._links``, but returning the *sums*: the cell
    complement (``cell \\ group``) is ``cell_sums - group_sums``, which needs sums, not ratios.
    """
    d = pos[[*keys, "store_id", "period_id", metric]].drop_duplicates([*keys, "store_id", "period_id"])
    out = []
    for p0, p1 in zip(periods[:-1], periods[1:]):
        m = d[d["period_id"] == p1].merge(d[d["period_id"] == p0], on=[*keys, "store_id"], suffixes=("_t", "_p"))
        if m.empty:
            continue
        m["plus"] = m[f"{metric}_t"] > m[f"{metric}_p"]
        m["minus"] = m[f"{metric}_t"] < m[f"{metric}_p"]
        g = m.groupby(keys, sort=False).agg(
            s_t=(f"{metric}_t", "sum"), s_p=(f"{metric}_p", "sum"),
            n_common=("store_id", "size"), n_plus=("plus", "sum"), n_minus=("minus", "sum"),
        )
        out.append(g.reset_index().assign(period_id=p1))
    if not out:
        return pd.DataFrame(columns=[*keys, *LINK_COLS, "period_id"])
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- seasonal, with borrowing
def seasonal_table(panel: pd.DataFrame, period: int, seasonal, base_cell, mode: str = "hard",
                   kappa_link: float = 10.0, links: dict | None = None) -> pd.DataFrame:
    """Production ``compute_seasonal`` with a ``mode`` switch.

    ``mode="hard"`` reproduces the production chain (verified against ``compute_seasonal`` at run
    time). ``mode="weighted"`` replaces the threshold switch with the overlap-weighted blend.
    """
    links = links or {}
    periods = sorted(p for p in panel["period_id"].unique() if p <= period)
    base_periods = [p for p in periods if p < period][-seasonal.baseline_periods:]
    raw_cell_keys = [*base_cell, "category"]
    pos = panel[(panel["period_id"] <= period) & (panel[HEAD] > 0)]
    cols = SEAS_COLS

    groups = pos.loc[pos["period_id"] == period, [*GROUP_KEYS, *base_cell]].drop_duplicates(GROUP_KEYS)
    if groups.empty or len(periods) < 2:
        return pd.DataFrame(columns=cols)
    grid = groups.merge(pd.DataFrame({"period_id": periods[1:]}), how="cross")

    g = links.get("group")
    if g is None:
        g = link_sums(pos, GROUP_KEYS, HEAD, periods)
    grid = grid.merge(g, on=[*GROUP_KEYS, "period_id"], how="left")
    grid["n_common"] = grid["n_common"].fillna(0).astype(int)
    grid[["n_plus", "n_minus"]] = grid[["n_plus", "n_minus"]].fillna(0).astype(int)
    grid["link"] = np.where(grid["s_p"] > 0, grid["s_t"] / grid["s_p"], np.nan)

    c = links.get("cell")
    if c is None:
        c = link_sums(pos, raw_cell_keys, HEAD, periods)
    c = c.rename(columns={k: f"c_{k}" for k in LINK_COLS})
    grid = grid.merge(c, on=[*raw_cell_keys, "period_id"], how="left")
    for k in LINK_COLS:
        grid[f"c_{k}"] = grid[f"c_{k}"].fillna(0)

    if mode == "hard":
        grid["source"] = np.where(grid["n_common"] >= seasonal.min_common_stores, "group", "none")
        csp = grid["c_s_p"].to_numpy(dtype=float)
        link_c = np.where(csp > 0, grid["c_s_t"].to_numpy(dtype=float) / np.where(csp > 0, csp, 1.0), np.nan)
        use = (grid["source"] == "none") & (grid["c_n_common"] >= seasonal.min_common_stores)
        grid.loc[use, "link"] = link_c[use]
        for k in LINK_COLS:
            grid.loc[use, k] = grid.loc[use, f"c_{k}"]
        grid.loc[use, "source"] = "base_cell"
        none = grid["source"] == "none"
        grid.loc[none, "link"] = 1.0
        grid.loc[none, ["n_plus", "n_minus"]] = 0
        grid["w"] = np.where(grid["source"] == "group", 1.0, 0.0)
    else:
        s = links.get("sellers")
        if s is None:
            s = (pos.groupby([*GROUP_KEYS, "period_id"], sort=False)["store_id"].nunique()
                    .rename("n_sell").reset_index())
        prev_pair = {p1: p0 for p0, p1 in zip(periods[:-1], periods[1:])}
        grid["_p0"] = grid["period_id"].map(prev_pair)
        grid = grid.merge(s.rename(columns={"n_sell": "n_sell_g"}), on=[*GROUP_KEYS, "period_id"], how="left")
        grid = grid.merge(s.rename(columns={"n_sell": "n_sell_p", "period_id": "_p0"}),
                          on=[*GROUP_KEYS, "_p0"], how="left").drop(columns=["_p0"])
        grid["n_sell_g"] = grid["n_sell_g"].fillna(grid["n_common"])
        grid["n_sell_p"] = grid["n_sell_p"].fillna(grid["n_common"])

        n_c = grid["n_common"].to_numpy(dtype=float)
        min_sell = np.minimum(grid["n_sell_g"].to_numpy(dtype=float), grid["n_sell_p"].to_numpy(dtype=float))
        overlap = np.where(min_sell > 0, n_c / np.where(min_sell > 0, min_sell, 1.0), 0.0)
        w = (n_c / (n_c + kappa_link)) * np.clip(overlap, 0.0, 1.0)
        grid["w"] = w

        s_p = grid["s_p"].fillna(0).to_numpy(dtype=float)
        s_t = grid["s_t"].fillna(0).to_numpy(dtype=float)
        rest_p = grid["c_s_p"].to_numpy(dtype=float) - s_p
        rest_t = grid["c_s_t"].to_numpy(dtype=float) - s_t
        rest_n = grid["c_n_common"].to_numpy(dtype=float) - n_c
        link_rest = np.where(rest_p > 0, rest_t / np.where(rest_p > 0, rest_p, 1.0), np.nan)
        has_rest = np.isfinite(link_rest) & (rest_n > 0)
        link_g = grid["link"].to_numpy(dtype=float)
        both = np.isfinite(link_g) & has_rest
        grid["link"] = np.where(both, w * np.nan_to_num(link_g) + (1.0 - w) * np.nan_to_num(link_rest),
                                np.where(np.isfinite(link_g), link_g, link_rest))
        grid["source"] = np.where(np.isfinite(link_g), "group", np.where(has_rest, "base_cell", "none"))
        grid.loc[grid["source"].to_numpy() != "group", ["n_plus", "n_minus"]] = 0
        none = ~np.isfinite(grid["link"].to_numpy(dtype=float))
        grid.loc[none, "link"] = 1.0
        grid.loc[none, "source"] = "none"

    grid = grid.sort_values([*GROUP_KEYS, "period_id"])
    grid["idx"] = grid.groupby(GROUP_KEYS)["link"].cumprod()

    res = groups[GROUP_KEYS].copy()
    idx = grid.set_index([*GROUP_KEYS, "period_id"])["idx"]
    first = pd.DataFrame({"period_id": [periods[0]]}).merge(res, how="cross").assign(idx=1.0)
    idx = pd.concat([first.set_index([*GROUP_KEYS, "period_id"])["idx"], idx])
    keys = list(zip(res["peer_group_id"], res["category"]))
    res["idx_t"] = [idx.get((gid, cat, period), np.nan) for gid, cat in keys]
    base_vals = [[idx.get((gid, cat, p), np.nan) for p in base_periods] for gid, cat in keys]
    res["idx_base"] = [np.nanmean(v) if len(v) and np.isfinite(v).any() else np.nan for v in base_vals]
    res["T_raw"] = res["idx_t"] / res["idx_base"]

    cur = grid.loc[grid["period_id"] == period,
                   [*GROUP_KEYS, "link", "n_common", "n_plus", "n_minus", "source", "w", "c_n_common"]]
    res = res.merge(cur.rename(columns={"link": "link_t"}), on=GROUP_KEYS, how="left")
    res["source"] = res["source"].fillna("none")
    for k in ("n_common", "n_plus", "n_minus", "c_n_common"):
        res[k] = res[k].fillna(0).astype(int)
    res["w"] = res["w"].fillna(0.0)
    decided = [_decide(r, seasonal) for r in res[["T_raw", "n_plus", "n_minus"]].itertuples(index=False)]
    res[["sign_stat", "significant", "direction", "T_final", "reason"]] = pd.DataFrame(decided, index=res.index)
    return res[cols]


# ---------------------------------------------------------------- longitude parts
def longitude_parts(arrays: list[np.ndarray], coef: float, trim_pct: float) -> tuple[float, float, float, int]:
    """(center, lo_width, hi_width, n_periods); limits = center -/+ coef * width.

    Mirrors ``funcs.robust.fova_longitude_limits`` but exposes the two widths separately, so a
    short group history can be blended with the (far better observed) base-cell history.
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


def longitude_table(panel: pd.DataFrame, hist_periods: list[int], keys: list[str],
                    coef: float, trim_pct: float) -> pd.DataFrame:
    hist = panel[panel["period_id"].isin(hist_periods) & (panel[HEAD] > 0)]
    cols = [*keys, "center", "lo_w", "hi_w", "n_hist"]
    if hist.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for key, g in hist.groupby(keys, sort=False):
        arrays = [np.log1p(sub[HEAD].to_numpy(dtype=float)) for _, sub in g.groupby("period_id", sort=True)]
        center, lo_w, hi_w, n = longitude_parts(arrays, coef, trim_pct)
        rows.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                     "center": center, "lo_w": lo_w, "hi_w": hi_w, "n_hist": n})
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- per-period artifacts
class Artifacts:
    __slots__ = ("period", "ppanel", "assigned", "trend", "trend_inj", "inj_mask", "g_base", "g_inj",
                 "needed_cells", "g2cell", "seas_hard", "seas_w", "lon_group", "lon_cell")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def build_artifacts(sc, period, params, item_key, cache: dict, rng, inject_frac: float) -> Artifacts:
    if period in cache:
        return cache[period]
    seasonal, fova, ibd = params.seasonal, params.fova, params.ibd
    base_cell = params.base_cell
    sc_cur = sc[sc["period_id"] == period]

    define = define_ibd(sc, period, base_cell, ibd)
    empty_items = pd.DataFrame(columns=["period_id", "store_id", "category", item_key, "sales_value"])
    merge, _ = check_ibd(sc_cur, empty_items, define, item_key, ibd)
    store_map = define.store_map
    assigned_grid = peer_assignment(sc_cur, store_map, merge)
    members = group_members(assigned_grid)

    ppanel = peer_panel(sc, store_map, members)
    ppanel["cell_key"] = ppanel[list(base_cell)].astype(str).agg("|".join, axis=1)
    ppanel = ppanel[ppanel["period_id"] <= period]

    filled = zero_fill_categories(sc_cur, store_map.set_index("store_id")["ibd_id"])
    assigned = peer_assignment(filled, store_map, merge)
    assigned = assigned[assigned["ibd_id"].notna()].copy()
    assigned["cell_key"] = assigned[list(base_cell)].astype(str).agg("|".join, axis=1)
    assigned["prov_key"] = assigned["cell_key"] + "|" + assigned["PROVINCE"].astype(str)
    assigned = assigned.reset_index(drop=True)

    periods = sorted(p for p in ppanel["period_id"].unique() if p <= period)
    prev = periods[-2] if len(periods) > 1 else None
    pos = ppanel[ppanel[HEAD] > 0]

    links = {
        "group": link_sums(pos, GROUP_KEYS, HEAD, periods),
        "cell": link_sums(pos, [*base_cell, "category"], HEAD, periods),
        "sellers": (pos.groupby([*GROUP_KEYS, "period_id"], sort=False)["store_id"].nunique()
                       .rename("n_sell").reset_index()),
    }
    seas_hard = seasonal_table(ppanel, period, seasonal, base_cell, "hard", links=links)
    seas_w = {k: seasonal_table(ppanel, period, seasonal, base_cell, "weighted", kappa_link=k, links=links)
              for k in set(KAPPA_LINK.values()) | {T4_KAPPA}}

    # ``base`` is the raw log ratio, arm-independent; each arm applies its own T on top.
    trend = pd.DataFrame(columns=[*GROUP_KEYS, "store_id", "base", "cur", "prev"])
    if prev is not None:
        a = pos.loc[pos["period_id"] == period, [*GROUP_KEYS, "store_id", HEAD]]
        b = pos.loc[pos["period_id"] == prev, [*GROUP_KEYS, "store_id", HEAD]]
        both = a.merge(b, on=[*GROUP_KEYS, "store_id"], suffixes=("_t", "_p"))
        both = both.rename(columns={f"{HEAD}_t": "cur", f"{HEAD}_p": "prev"})
        both["base"] = (np.log1p(both["cur"].to_numpy(dtype=float))
                        - np.log1p(both["prev"].to_numpy(dtype=float)))
        trend = both[[*GROUP_KEYS, "store_id", "base", "cur", "prev"]].reset_index(drop=True)

    g_base = group_stats(trend.assign(td=trend["base"]), GROUP_KEYS)
    inj_mask = rng.random(len(trend)) < inject_frac
    trend_inj = trend.copy()
    cur = trend["cur"].to_numpy(dtype=float)
    trend_inj["base"] = np.where(inj_mask, np.log1p(np.nan_to_num(cur) * B.INJECT_MULT)
                                 - np.log1p(np.nan_to_num(cur)), trend["base"].to_numpy(dtype=float))
    g_inj = group_stats(trend_inj.assign(td=trend_inj["base"]), GROUP_KEYS)

    g2cell = assigned.drop_duplicates("peer_group_id").set_index("peer_group_id")["cell_key"].to_dict()
    thin = g_base.loc[g_base["n_peer"] < seasonal.min_common_stores, GROUP_KEYS].copy()
    thin["cell_key"] = thin["peer_group_id"].map(g2cell)
    needed_cells = thin[[*CELL_KEYS]].drop_duplicates() if len(thin) else None

    hist_periods = [p for p in periods if p >= fova.hist_start_period][-fova.recommended_hist_periods:]
    lon_group = longitude_table(ppanel, hist_periods, GROUP_KEYS, fova.longitude_iqr_coef, fova.trim_pct)
    lon_cell = longitude_table(ppanel, periods[-fova.recommended_hist_periods:], CELL_KEYS,
                               fova.longitude_iqr_coef, fova.trim_pct)

    out = Artifacts(period=period, ppanel=ppanel, assigned=assigned, trend=trend, trend_inj=trend_inj,
                    inj_mask=inj_mask, g_base=g_base, g_inj=g_inj, needed_cells=needed_cells,
                    g2cell=g2cell, seas_hard=seas_hard, seas_w=seas_w, lon_group=lon_group, lon_cell=lon_cell)
    cache[period] = out
    return out


def cell_stats(trend: pd.DataFrame, g2cell: dict, needed: pd.DataFrame, keys: list[str] = CELL_KEYS) -> pd.DataFrame:
    """Base-cell distribution of ``td``, restricted to the cells that hold a thin group."""
    if needed is None or needed.empty:
        return pd.DataFrame(columns=[*keys, "q1", "q3", "mc", "n_peer"])
    d = trend.copy()
    d["cell_key"] = d["peer_group_id"].map(g2cell)
    return group_stats(d.merge(needed, on=keys, how="inner"), keys)


def build_reference(base: pd.DataFrame, group_stats_: pd.DataFrame, cell_stats_: pd.DataFrame,
                    use_fb: bool, min_fence_n: int) -> pd.DataFrame:
    """Merge the group (and optionally base-cell) fence stats onto ``base``."""
    ref = base.merge(group_stats_, on=GROUP_KEYS, how="left")
    ref = ref.merge(cell_stats_.rename(columns={"q1": "c_q1", "q3": "c_q3", "mc": "c_mc", "n_peer": "c_n"}),
                    on=CELL_KEYS, how="left")
    n_g = ref["n_peer"].fillna(0).to_numpy(dtype=float)
    n_c = ref["c_n"].fillna(0).to_numpy(dtype=float)
    src = np.where(n_g < min_fence_n, "thin", "group").astype(object)
    if use_fb:
        take = (n_g < min_fence_n) & (n_c >= MIN_PEER)
        for x, y in (("q1", "c_q1"), ("q3", "c_q3"), ("mc", "c_mc")):
            ref.loc[take, x] = ref.loc[take, y]
        ref.loc[take, "n_peer"] = ref.loc[take, "c_n"]
        src = np.where(take, "cell", np.where(n_g < MIN_PEER, "thin", "group")).astype(object)
    ref["fence_src"] = src
    ref["n_peer"] = ref["n_peer"].fillna(0.0)
    return ref


# ---------------------------------------------------------------- arm evaluation
def evaluate(period, params, art, prev_art, arm, target_rate):
    fova, seasonal = params.fova, params.seasonal
    a, b = fova.adjbox_a, fova.adjbox_b
    fence = B.make_fence("adjbox", a, b)
    use_fb = arm in ("T1fence回退", "T4组合")
    use_lon = arm in ("T3long借cell", "T4组合")
    min_fence_n = seasonal.min_common_stores
    seas = art.seas_w[KAPPA_LINK.get(arm, T4_KAPPA)] if (arm in KAPPA_LINK or arm == "T4组合") else art.seas_hard

    base = art.assigned[[*GROUP_KEYS, "cell_key", "store_id", "PROVINCE", "sales_value"]].copy()
    j = base[["store_id", "category"]].merge(art.trend[["store_id", "category", "base"]],
                                             on=["store_id", "category"], how="left")
    b_cur = j["base"].to_numpy(dtype=float)

    def t_of(seas_: pd.DataFrame) -> np.ndarray:
        t = base[GROUP_KEYS].merge(seas_[[*GROUP_KEYS, "T_final"]], on=GROUP_KEYS, how="left")["T_final"]
        t = t.to_numpy(dtype=float)
        return np.where(np.isfinite(t) & (t > 0), t, 1.0)

    def reference(fr: pd.DataFrame, stats: pd.DataFrame):
        return build_reference(base, shift_stats(stats, seas),
                               cell_stats(fr, art.g2cell, art.needed_cells), use_fb, min_fence_n)

    t_fresh = t_of(seas)
    logt_tr = logt_of(art.trend, seas)
    td = b_cur - np.log(t_fresh)
    ref = reference(art.trend.assign(td=art.trend["base"].to_numpy(dtype=float) - logt_tr), art.g_base)
    q1 = ref["q1"].to_numpy(dtype=float)
    q3 = ref["q3"].to_numpy(dtype=float)
    mc = ref["mc"].to_numpy(dtype=float)
    n_peer = ref["n_peer"].to_numpy(dtype=float)
    usable = np.isfinite(td) & np.isfinite(q1) & np.isfinite(q3) & (n_peer >= MIN_PEER)
    thr = np.full(len(ref), -np.inf)
    thr[usable] = fence_thresholds(td[usable], q1[usable], q3[usable], mc[usable], a, b)
    cal = calibrate_analytic(thr[usable], target_rate)
    lo_c, hi_c = fence(ref, cal)
    flags = flag(td, lo_c, hi_c)
    flags[~usable] = "na"

    # ---- x3 injected into the current-period value only; the whole reference is recomputed
    fr_i = art.trend_inj.assign(td=art.trend_inj["base"].to_numpy(dtype=float) - logt_tr)
    ref_i = reference(fr_i, art.g_inj)
    tt = base[["store_id", "category"]].merge(fr_i[["store_id", "category", "td"]],
                                              on=["store_id", "category"], how="left")["td"].to_numpy(dtype=float)
    fl_i = flag(tt, *fence(ref_i, cal))
    fl_i[~np.isfinite(tt) | (ref_i["n_peer"].to_numpy(dtype=float) < MIN_PEER)] = "na"
    inj = base[["store_id", "category"]].merge(
        art.trend[["store_id", "category"]].assign(inj=art.inj_mask),
        on=["store_id", "category"], how="left")["inj"].fillna(False).to_numpy(dtype=bool)
    hit = inj & (fl_i != "na")
    detect = float(np.mean(fl_i[hit] == "high")) if hit.any() else np.nan

    # ---- stability: the reference as it looked one period earlier -- its own stores, its own base
    # values, its own seasonal factor -- applied to this period's rows.
    if prev_art is None:
        stab = np.nan
    else:
        seas_l = (prev_art.seas_w[KAPPA_LINK.get(arm, T4_KAPPA)]
                  if (arm in KAPPA_LINK or arm == "T4组合") else prev_art.seas_hard)
        fr_l = prev_art.trend.assign(td=prev_art.trend["base"].to_numpy(dtype=float)
                                     - logt_of(prev_art.trend, seas_l))
        needed_l = None
        thin_l = prev_art.g_base.loc[prev_art.g_base["n_peer"] < min_fence_n, GROUP_KEYS].copy()
        if len(thin_l):
            thin_l["cell_key"] = thin_l["peer_group_id"].map(prev_art.g2cell)
            needed_l = thin_l[[*CELL_KEYS]].drop_duplicates()

    # ---- longitude: level test against per-key historical limits
    lv_adj = np.log1p(art.assigned["sales_value"].clip(lower=0).to_numpy(dtype=float) / t_fresh)
    lg = art.assigned[GROUP_KEYS + ["cell_key"]].merge(art.lon_group, on=GROUP_KEYS, how="left")
    ng = lg["n_hist"].to_numpy(dtype=float)
    ok_g = np.isfinite(lg["center"].to_numpy(dtype=float))
    if use_lon:
        lc = art.assigned[GROUP_KEYS + ["cell_key"]].merge(
            art.lon_cell.rename(columns={"center": "c_center", "lo_w": "c_lo", "hi_w": "c_hi", "n_hist": "c_nhist"}),
            on=CELL_KEYS, how="left")
        ok_c = np.isfinite(lc["c_center"].to_numpy(dtype=float))
        w = np.where(np.isfinite(ng) & (ng > 0), ng / (ng + K_HIST), 0.0)
        w = np.where(ok_g & ok_c, w, np.where(ok_c, 0.0, 1.0))
        center = (w * np.nan_to_num(lg["center"].to_numpy(dtype=float))
                  + (1 - w) * np.nan_to_num(lc["c_center"].to_numpy(dtype=float)))
        lo_w = (w * np.nan_to_num(lg["lo_w"].to_numpy(dtype=float))
                + (1 - w) * np.nan_to_num(lc["c_lo"].to_numpy(dtype=float)))
        hi_w = (w * np.nan_to_num(lg["hi_w"].to_numpy(dtype=float))
                + (1 - w) * np.nan_to_num(lc["c_hi"].to_numpy(dtype=float)))
        n_hist_eff = (np.where(ok_g, np.nan_to_num(ng), 0.0)
                      + np.where(ok_c, np.nan_to_num(lc["c_nhist"].to_numpy(dtype=float)), 0.0))
        bad = ~(ok_g | ok_c)
    else:
        center, lo_w, hi_w = (lg["center"].to_numpy(dtype=float), lg["lo_w"].to_numpy(dtype=float),
                              lg["hi_w"].to_numpy(dtype=float))
        n_hist_eff = np.nan_to_num(ng)
        bad = ~ok_g
    enough = n_hist_eff >= fova.min_hist_periods
    usable_lon = ~bad & enough
    lon = flag(lv_adj, np.where(usable_lon, center - fova.longitude_iqr_coef * lo_w, np.nan),
               np.where(usable_lon, center + fova.longitude_iqr_coef * hi_w, np.nan))

    # ---- metrics
    ok = flags != "na"
    rate = float(np.mean(flags[ok] != "ok")) if ok.any() else 0.0
    sys_share, sys_n, sys_base = B.systematic_share(ref, flags)
    lift = B.province_lift(ref, flags)
    cal5 = calibrate_analytic(thr[usable], 0.05)
    lo5, hi5 = fence(ref, cal5)
    f5 = flag(td, lo5, hi5)
    f5[~usable] = "na"
    sys5, _, _ = B.systematic_share(ref, f5)
    lift5 = B.province_lift(ref, f5)
    fl_i5 = flag(tt, *fence(ref_i, cal5))
    fl_i5[~np.isfinite(tt) | (ref_i["n_peer"].to_numpy(dtype=float) < MIN_PEER)] = "na"
    detect5 = float(np.mean(fl_i5[hit] == "high")) if hit.any() else np.nan

    # ---- stability: same observations, reference as of the previous period. Because T is constant
    # inside a group, the T mismatch cancels exactly for group-referenced rows, so this isolates
    # peer-set renewal (and, for the fallback arms, the cell's mixed-T reference).
    if prev_art is None:
        stab = np.nan
    else:
        seas_l = (prev_art.seas_w[KAPPA_LINK.get(arm, T4_KAPPA)]
                  if (arm in KAPPA_LINK or arm == "T4组合") else prev_art.seas_hard)
        fr_l = prev_art.trend.assign(td=prev_art.trend["base"].to_numpy(dtype=float)
                                     - logt_of(prev_art.trend, seas_l))
        needed_l = None
        thin_l = prev_art.g_base.loc[prev_art.g_base["n_peer"] < min_fence_n, GROUP_KEYS].copy()
        if len(thin_l):
            thin_l["cell_key"] = thin_l["peer_group_id"].map(prev_art.g2cell)
            needed_l = thin_l[[*CELL_KEYS]].drop_duplicates()
        ref_l = build_reference(base, shift_stats(prev_art.g_base, seas_l),
                                cell_stats(fr_l, prev_art.g2cell, needed_l), use_fb, min_fence_n)
        lo_l, hi_l = fence(ref_l, cal)
        flags_l = flag(td, lo_l, hi_l)
        flags_l[~usable | (ref_l["n_peer"].to_numpy(dtype=float) < MIN_PEER)] = "na"
        both = usable & (flags_l != "na")
        stab = B.jaccard(B.flag_set(ref, flags, both), B.flag_set(ref, flags_l, both))
        stab_n = int(both.sum())
        stab_flags = int(((flags != "ok") & (flags != "na") & both).sum())
        stab_flags_l = int(((flags_l != "ok") & (flags_l != "na") & both).sum())
    if prev_art is None:
        stab_n = stab_flags = stab_flags_l = 0
    srcmix = seas["source"].value_counts(normalize=True).to_dict()
    nc = art.seas_hard["n_common"].to_numpy(dtype=float)
    sw = seas["w"].to_numpy(dtype=float) if "w" in seas.columns else np.ones(len(seas))
    metrics = {
        "period_id": period, "arm": arm, "n_rows": int(len(ref)),
        "seller_rate": float((art.assigned["sales_value"] > 0).mean()),
        "trend_value_rate": float(np.isfinite(td).mean()),
        "trend_usable_rate": float(usable.mean()),
        "fence_group_share": float(np.mean(ref["fence_src"] == "group")),
        "fence_cell_share": float(np.mean(ref["fence_src"] == "cell")),
        "fence_thin_share": float(np.mean(ref["fence_src"] == "thin")),
        "calibrated_c": cal, "anomaly_rate": rate,
        "stability_jaccard": stab,
        "stability_n": stab_n, "stability_flags_fresh": stab_flags, "stability_flags_late": stab_flags_l,
        "x3_detect_rate": detect, "x3_detect_rate_5pct": detect5,
        "systematic_share": sys_share, "systematic_n": sys_n, "systematic_base": sys_base,
        "province_lift_max": lift,
        "systematic_share_5pct": sys5, "province_lift_max_5pct": lift5,
        "lon_available_rate": float(np.mean(lon != "na")),
        "lon_high_rate": float(np.mean(lon == "high")),
        "lon_hist_median": float(np.median(n_hist_eff)),
        "seasonal_group_share": float(srcmix.get("group", 0.0)),
        "seasonal_cell_share": float(srcmix.get("base_cell", 0.0)),
        "seasonal_none_share": float(srcmix.get("none", 0.0)),
        "n_common_p10": float(np.quantile(nc, 0.10)), "n_common_p50": float(np.median(nc)),
        "n_common_lt5_rate": float(np.mean(nc < 5)), "n_common_lt10_rate": float(np.mean(nc < 10)),
        "blend_w_median": float(np.median(sw)),
        "blend_w_positive_rate": float(np.mean(sw > 0)),
    }
    ref["_flags"] = flags
    ref["_usable"] = usable
    ref["_lon"] = lon
    ref["_n"] = n_peer
    ref["_td"] = td
    return metrics, ref


# ---------------------------------------------------------------- run
def run(periods, params_path, out_path, arms, target_rate=B.TARGET_RATE,
        inject_frac=B.INJECT_FRAC, check_calibration: bool = False) -> Path:
    params = load_params(params_path)
    bmp_dir = params.path("before_mp_dir", ROOT)
    tgt_dir = params.path("store_target_dir", ROOT)
    item_key = params.data.item_key
    wanted = [p for p in list_before_mp_files(bmp_dir) if p <= max(periods)]
    log(f"loading panel periods={wanted}")
    panel = load_panel(bmp_dir, tgt_dir, item_key, periods=wanted, encoding=params.data.encoding)
    sc = panel.sc
    log(f"sc rows={len(sc)} periods={panel.periods}")

    rng = np.random.default_rng(B.SEED)
    cache: dict[int, Artifacts] = {}
    for p in sorted({q for q in periods} | {q - 1 for q in periods}):
        if p in panel.periods:
            t0 = time.time()
            build_artifacts(sc, p, params, item_key, cache, rng, inject_frac)
            a0 = cache[p]
            log(f"  artifacts {p}: {time.time() - t0:.1f}s assigned={len(a0.assigned)} "
                f"trend={len(a0.trend)} groups={len(a0.seas_hard)} "
                f"thin_cells={0 if a0.needed_cells is None else len(a0.needed_cells)}")

    # self-check: the hard-mode reimplementation must reproduce production compute_seasonal
    for p in [q for q in periods if q in cache]:
        a0 = cache[p]
        prod = compute_seasonal(a0.ppanel, p, params.seasonal, base_cell=params.base_cell)
        prod = prod[prod["metric"] == HEAD][["peer_group_id", "category", "T_final", "link_t", "n_common"]]
        mine = a0.seas_hard[["peer_group_id", "category", "T_final", "link_t", "n_common"]]
        j = prod.merge(mine, on=GROUP_KEYS, suffixes=("_p", "_m"), how="outer")
        dt = np.abs(j["T_final_p"].to_numpy(dtype=float) - j["T_final_m"].to_numpy(dtype=float))
        dl = np.abs(np.log(j["link_t_p"].to_numpy(dtype=float)) - np.log(j["link_t_m"].to_numpy(dtype=float)))
        same = int((j["n_common_p"].fillna(-1).to_numpy() == j["n_common_m"].fillna(-2).to_numpy()).sum())
        log(f"  selfcheck {p}: prod={len(prod)} mine={len(mine)} joined={len(j)} "
            f"n_common match={same}/{len(j)} | T_final mismatch={int(np.nansum(dt > 1e-9))} "
            f"| link mismatch={int(np.nansum(dl > 1e-9))}")

    rows, by_n, link_diag = [], [], []
    for period in periods:
        art, prev_art = cache[period], cache.get(period - 1)
        for arm in arms:
            t0 = time.time()
            metrics, ref = evaluate(period, params, art, prev_art, arm, target_rate)
            rows.append(metrics)
            log(f"  {arm:<12s} usable={metrics['trend_usable_rate']:.3f} c={metrics['calibrated_c']:7.2f} "
                f"rate={metrics['anomaly_rate']:.4f} stab={metrics['stability_jaccard']:.3f}"
                f"(n={metrics['stability_n']}) "
                f"x3={metrics['x3_detect_rate']:.3f}/{metrics['x3_detect_rate_5pct']:.3f} "
                f"sys={metrics['systematic_share']:.4f} lift={metrics['province_lift_max']:.2f} "
                f"lon={metrics['lon_available_rate']:.3f} none={metrics['seasonal_none_share']:.3f} "
                f"({time.time() - t0:.1f}s)")
            if check_calibration and arm == arms[0]:
                log(f"    calibration self-check: target={target_rate} observed={metrics['anomaly_rate']:.4f} "
                    f"(denominator = rows with a finite trend value, i.e. testable rows only)")
            fl, us, nn, ln = (ref["_flags"].to_numpy(), ref["_usable"].to_numpy(),
                              ref["_n"].to_numpy(), ref["_lon"].to_numpy())
            for (lo, hi), lab in zip(B.SIZE_BUCKETS, B.SIZE_LABELS):
                m = us & (nn >= lo) & (nn < hi)
                if not m.any():
                    continue
                by_n.append({"period_id": period, "arm": arm, "fence_n_bucket": lab, "n_rows": int(m.sum()),
                             "anomaly_rate": float(np.mean(fl[m] != "ok")),
                             "lon_available_rate": float(np.mean(ln[m] != "na"))})
        # arm-independent: how far the weighted link moves away from the hard switch
        m = art.seas_w[T4_KAPPA][[*GROUP_KEYS, "link_t", "w", "n_common"]].merge(
            art.seas_hard[[*GROUP_KEYS, "link_t"]].rename(columns={"link_t": "link_hard"}),
            on=GROUP_KEYS, how="left")
        d = np.abs(np.log(m["link_t"].to_numpy(dtype=float)) - np.log(m["link_hard"].to_numpy(dtype=float)))
        near = ((m["n_common"] >= 5) & (m["n_common"] < 15)).to_numpy()
        link_diag.append({
            "period_id": period, "n_groups": int(len(m)),
            "w_median": float(m["w"].median()), "w_zero_share": float((m["w"] == 0).mean()),
            "n_common_lt10_share": float((m["n_common"] < 10).mean()),
            "groups_near_threshold": int(near.sum()),
            "link_shift_all_median": float(np.nanmedian(d)),
            "link_shift_near_threshold_median": float(np.nanmedian(d[near])) if near.any() else np.nan,
            "link_shift_p90": float(np.nanquantile(d, 0.90)),
        })

    out = pd.DataFrame(rows)
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    settings = [
        ("periods", str(periods)), ("arms", str(arms)),
        ("target_rate", target_rate), ("inject_frac", inject_frac), ("inject_mult", B.INJECT_MULT),
        ("min_peer", MIN_PEER), ("min_fence_n", params.seasonal.min_common_stores),
        ("kappa_link", str(KAPPA_LINK)), ("k_hist", K_HIST), ("mc_cap", MC_CAP),
        ("adjbox_a_b", f"{params.fova.adjbox_a}/{params.fova.adjbox_b}"),
        ("hist_start_period", params.fova.hist_start_period),
        ("note_obs", "the store's own trend_d is never shrunk; only the reference (T and fence) borrows"),
        ("note_usable", f"usable = finite trend_d AND finite fence AND fence_n >= {MIN_PEER}"),
        ("note_detect", "x3 injected into the current-period value only; reference recomputed on injected data"),
        ("note_lon", "T3/T4 blend group limits toward base_cell x category limits over all history, w = n_hist/(n_hist+K_HIST)"),
        ("note_peerset", "production IBD + ibd_check fallback, recomputed each period; not an arm"),
        ("note_stability", "same observations, reference as of the previous period; T mismatch cancels for group-referenced rows"),
        ("note_calibration", "c solved analytically on rows with a finite trend value (testable rows only); the shared harness would dilute the denominator with rows that have no trend value"),
        ("note_t_neutral", "T is constant inside peer_group_id x category, so under a group reference it shifts observation and fence alike and cancels exactly in the trend test"),
    ]
    with pd.ExcelWriter(dest, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="summary", index=False)
        for metric in ("trend_usable_rate", "fence_cell_share", "stability_jaccard",
                       "x3_detect_rate", "x3_detect_rate_5pct",
                       "systematic_share", "province_lift_max", "lon_available_rate",
                       "seasonal_none_share", "blend_w_median", "n_common_lt10_rate"):
            out.pivot_table(index=["period_id"], columns="arm", values=metric).to_excel(w, sheet_name=metric[:31])
        pd.DataFrame(by_n).to_excel(w, sheet_name="by_fence_n", index=False)
        pd.DataFrame(link_diag).to_excel(w, sheet_name="seasonal_link", index=False)
        pd.DataFrame(settings, columns=["key", "value"]).to_excel(w, sheet_name="settings", index=False)
    log(f"wrote {dest}")
    return dest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="trend / seasonal reference-borrowing backtest")
    p.add_argument("--periods", type=int, nargs="*", default=[20261406, 20261407, 20261408])
    p.add_argument("--params", default=str(ROOT / "config" / "peer_params.json"))
    p.add_argument("--out", default=None)
    p.add_argument("--arms", nargs="*", default=list(ARMS))
    p.add_argument("--target-rate", type=float, default=B.TARGET_RATE)
    p.add_argument("--inject-frac", type=float, default=B.INJECT_FRAC)
    p.add_argument("--check-calibration", action="store_true",
                   help="cross-check the analytic fence multiplier against the bisection")
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    periods, arms = list(args.periods), list(args.arms)
    if args.quick:
        periods, arms = periods[-2:], ["T0现状", "T1fence回退"]
    stem = f"trend_borrow_backtest_{'_'.join(str(p) for p in periods)}"
    out = args.out or str(ROOT / "data" / "backtest" / f"{stem}.xlsx")
    run(periods, args.params, out, arms, target_rate=args.target_rate, inject_frac=args.inject_frac,
        check_calibration=args.check_calibration)


if __name__ == "__main__":
    main()
