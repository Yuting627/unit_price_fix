"""Standalone backtest: peer-group schemes x adjbox detection core.

Answers "if the peer-group definition changes, how does adjbox detection behave?" using the
three metrics from the README comparison, but with **adjbox** as the detection core instead of
the median/MAD robust z used in the original run:

* 稳定性    Jaccard(flags at t under a freshly refreshed definition, flags at t under the
            definition refreshed one period late) -- same period, two definitions.
* x3 检出率  inject x3 into a random share of seller rows, re-run detection with the same
            calibrated fence constant, measure the share of injected rows flagged high.
* 系统性异常  share of (province x category) cells where > 50% of stores are flagged in the
            same direction.

Schemes (README "同组范围的回测评估"):

* 现状IBD       define_ibd on the latest available period (production-faithful).
* A冻结         define_ibd on the pooled previous 3 periods, applied to the current period.
* B相似店       inside the base cell, the 20 most similar stores that sell the category
                (same province first); similarity = previous-3-period category mix + size.
* C收缩         no province merging: Q1/Q3/MC = w * province + (1-w) * base cell,
                w = n_province / (n_province + kappa).
* C2两级收缩    province -> current IBD -> base cell (two nested shrinks).
* C3相似省收缩  base cell is replaced by category-mix-similar provinces.

Read-only with respect to production code: production modules are imported, never written to.

Usage:
    python backtest/group_scheme_backtest.py --periods 20261405 20261406 20261408
    python backtest/group_scheme_backtest.py --quick
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
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from core.config import load_params
from core.data_prep.data_prep import list_before_mp_files, load_panel
from core.ibd_define.ibd_define import define_ibd
from funcs.robust import fence_from_stats

DETECTORS = ("log_sales", "log_share", "mom")
VALUE_COL = {"log_sales": "v_sales", "log_share": "v_share", "mom": "v_mom"}
SCHEMES = ("现状IBD", "A冻结", "B相似店", "C收缩", "C2两级收缩", "C3相似省")
SHRINK_SCHEMES = ("C收缩", "C2两级收缩", "C3相似省")

PI_2 = math.pi / 2.0          # kappa scale, per the README formula
MIN_PEER = 5                  # minimum peer stores for a fence to be usable
KAPPA_MIN, KAPPA_MAX = 0.25, 1e5
TARGET_RATE = 0.01
INJECT_FRAC = 0.05
INJECT_MULT = 3.0
SEED = 20261408
SIZE_BUCKETS = ((0, 10), (10, 25), (25, 100), (100, 500), (500, math.inf))
SIZE_LABELS = ("<10", "10-24", "25-99", "100-499", ">=500")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- small statistics
MC_SUBSAMPLE = 3000


def medcouple_of(v: np.ndarray) -> float:
    """Medcouple.

    ``use_fast=True`` is buggy for some sizes in statsmodels 0.15 (IndexError inside the
    median-of-h-kernel sweep) and, because of ``apply_along_axis`` overhead, is not actually
    faster here, so the O(n^2) vectorised path is used throughout, with very large groups
    subsampled (a robust skewness estimate does not need every point).
    """
    from statsmodels.stats.stattools import medcouple

    arr = np.asarray(v, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 0.0
    if arr.size > MC_SUBSAMPLE:
        arr = np.random.default_rng(arr.size).choice(arr, MC_SUBSAMPLE, replace=False)
    return float(medcouple(arr, use_fast=False))


def _loo_quantile(sv: np.ndarray, ranks: np.ndarray, q: float) -> np.ndarray:
    """Leave-one-out quantile ``q`` for every element of the sorted array ``sv``.

    ``ranks[i]`` is the sorted position of original element ``i``. Removing an element shifts
    everything after it one slot left, so the interpolating neighbours are a constant pair of
    sorted indices offset by 0/1 depending on the rank.
    """
    m = sv.size
    mm = m - 1
    if mm < 1:
        return np.full(m, np.nan)
    pos = q * (mm - 1)
    i = int(math.floor(pos))
    frac = pos - i
    j = min(i + 1, mm - 1)
    return sv[np.where(ranks > i, i, i + 1)] * (1.0 - frac) + sv[np.where(ranks > j, j, j + 1)] * frac


STAT_KEYS = ("q1", "q3", "mc", "med", "mad")


def _blank(n: int) -> dict:
    out = {k: np.full(n, np.nan) for k in STAT_KEYS}
    out["n_peer"] = np.zeros(n)
    return out


def _mad(v: np.ndarray) -> float:
    """Median absolute deviation about the group median (not leave-one-out)."""
    return float(np.median(np.abs(v - float(np.median(v)))))


def _codes(keys) -> np.ndarray:
    """Integer group codes for the combination of one or several label arrays."""
    arrs = [np.asarray(k) for k in (keys if isinstance(keys, (list, tuple)) else [keys])]
    if len(arrs) == 1:
        return pd.factorize(arrs[0])[0]
    return pd.factorize(pd.Index(list(zip(*arrs))))[0]


def group_stats(keys, value: np.ndarray) -> pd.DataFrame:
    """Leave-one-out Q1/Q3 plus the full-group medcouple, grouped by the key combination.

    ``keys`` is one label array or a list of them (the peer set is e.g. IBD x category).
    """
    gid = _codes(keys)
    acc = _blank(gid.size)
    order = np.argsort(gid, kind="stable")
    gid_s, val_s = gid[order], value[order]
    edges = np.flatnonzero(np.r_[True, gid_s[1:] != gid_s[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        if a == b:
            continue
        v, idx = val_s[a:b], order[a:b]
        keep = np.isfinite(v)
        if keep.sum() < 3:
            continue
        v, idx = v[keep], idx[keep]
        pos = np.argsort(v, kind="stable")
        sv = v[pos]
        ranks = np.empty(v.size, dtype=np.int64)
        ranks[pos] = np.arange(v.size)
        acc["q1"][idx] = _loo_quantile(sv, ranks, 0.25)
        acc["q3"][idx] = _loo_quantile(sv, ranks, 0.75)
        acc["med"][idx] = _loo_quantile(sv, ranks, 0.50)
        acc["mc"][idx] = medcouple_of(v)
        acc["mad"][idx] = _mad(v)
        acc["n_peer"][idx] = v.size
    return pd.DataFrame(acc)


def stats_from_long(n: int, ri: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    """Q1/Q3/MC per target row from a long table of (target row, peer value) pairs."""
    acc = _blank(n)
    if ri.size == 0:
        return pd.DataFrame(acc)
    order = np.argsort(ri, kind="stable")
    ri_s, val_s = ri[order], values[order]
    edges = np.flatnonzero(np.r_[True, ri_s[1:] != ri_s[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        v = val_s[a:b]
        keep = np.isfinite(v)
        if keep.sum() < 3:
            continue
        v = np.sort(v[keep])
        acc["q1"][ri_s[a]] = float(np.quantile(v, 0.25))
        acc["q3"][ri_s[a]] = float(np.quantile(v, 0.75))
        acc["med"][ri_s[a]] = float(np.median(v))
        acc["mad"][ri_s[a]] = _mad(v)
        acc["mc"][ri_s[a]] = medcouple_of(v)
        acc["n_peer"][ri_s[a]] = v.size
    return pd.DataFrame(acc)


def blend(parts: list[tuple[pd.DataFrame, np.ndarray]]) -> pd.DataFrame:
    """Weighted blend of stat frames (same row order), renormalised over finite parts."""
    n = len(parts[0][0])
    num = {k: np.zeros(n) for k in STAT_KEYS}
    den = np.zeros(n)
    n_peer = np.zeros(n)
    for st, w in parts:
        w = np.asarray(w, dtype=float)
        q1 = st["q1"].to_numpy(dtype=float)
        ok = np.isfinite(q1) & np.isfinite(w) & (w > 0)
        if not ok.any():
            continue
        for k in STAT_KEYS:
            num[k][ok] += w[ok] * st[k].to_numpy(dtype=float)[ok]
        den[ok] += w[ok]
        n_peer = np.maximum(n_peer, st["n_peer"].to_numpy(dtype=float))
    out = {k: np.where(den > 0, num[k] / np.where(den > 0, den, 1.0), np.nan) for k in STAT_KEYS}
    out["n_peer"] = n_peer
    return pd.DataFrame(out)


# ---------------------------------------------------------------- detection
def make_fence(core: str, a: float, b: float):
    """Fence builder: adjusted boxplot, or the plain median +/- c * MAD robust z core."""
    if core == "adjbox":
        return lambda st, c: fence_from_stats(st["q1"].to_numpy(dtype=float),
                                              st["q3"].to_numpy(dtype=float),
                                              st["mc"].to_numpy(dtype=float), a, b, c)
    if core == "robustz":
        return lambda st, c: (st["med"].to_numpy(dtype=float) - c * st["mad"].to_numpy(dtype=float),
                              st["med"].to_numpy(dtype=float) + c * st["mad"].to_numpy(dtype=float))
    raise ValueError(f"unknown core {core!r}")


def flag_values(value: np.ndarray, stats: pd.DataFrame, fence, c: float,
                min_peer: int = MIN_PEER) -> np.ndarray:
    lo, hi = fence(stats, c)
    usable = np.isfinite(lo) & np.isfinite(hi) & (stats["n_peer"].to_numpy() >= min_peer)
    out = np.where(value > hi, "high", np.where(value < lo, "low", "ok")).astype(object)
    out[~usable] = "na"
    return out


def anomaly_rate(value: np.ndarray, stats: pd.DataFrame, fence, c: float) -> float:
    flags = flag_values(value, stats, fence, c)
    ok = flags != "na"
    return float(np.mean(flags[ok] != "ok")) if ok.any() else 0.0


def calibrate(value: np.ndarray, stats: pd.DataFrame, fence,
              target: float = TARGET_RATE) -> float:
    """Binary-search the fence multiplier so the anomaly rate equals ``target``."""
    lo_c, hi_c = 0.01, 500.0
    if anomaly_rate(value, stats, fence, lo_c) <= target:
        return lo_c
    for _ in range(50):
        mid = 0.5 * (lo_c + hi_c)
        if anomaly_rate(value, stats, fence, mid) > target:
            lo_c = mid
        else:
            hi_c = mid
    return 0.5 * (lo_c + hi_c)


# ---------------------------------------------------------------- frames / features
def build_frame(sc: pd.DataFrame, period: int, prev: int | None, base_cell) -> pd.DataFrame:
    """Seller rows of ``period`` with the three detection values and the grouping keys."""
    a = sc[sc["period_id"] == period]
    a = a[a["sales_value"] > 0][["store_id", "category", "PLATFORMNAME", "SHOPTYPE",
                                 "PROVINCE", "sales_value", "cat_share"]].copy()
    a["v_sales"] = np.log1p(a["sales_value"].to_numpy(dtype=float))
    a["v_share"] = np.log1p(1000.0 * a["cat_share"].fillna(0).to_numpy(dtype=float))
    a["v_mom"] = np.nan
    if prev is not None:
        b = sc.loc[sc["period_id"] == prev, ["store_id", "category", "sales_value"]]
        b = b.groupby(["store_id", "category"], as_index=False)["sales_value"].sum()
        a = a.merge(b.rename(columns={"sales_value": "prev"}), on=["store_id", "category"], how="left")
        a["v_mom"] = np.log1p(a["sales_value"].to_numpy(dtype=float)) - np.log1p(a["prev"].to_numpy(dtype=float))
        a = a.drop(columns=["prev"])
    a["cell_key"] = a[list(base_cell)].astype(str).agg("|".join, axis=1)
    a["prov_key"] = a["cell_key"] + "|" + a["PROVINCE"].astype(str)
    return a.reset_index(drop=True)


def feature_matrix(sc: pd.DataFrame, hist_periods) -> pd.DataFrame:
    """Store x feature matrix: category mix over ``hist_periods`` plus log store size."""
    sub = sc[sc["period_id"].isin(hist_periods)]
    if sub.empty:
        return pd.DataFrame()
    mix = (sub.assign(x=np.log1p(1000.0 * sub["cat_share"].fillna(0)))
              .groupby(["store_id", "category"])["x"].mean().unstack("category"))
    size = sub.groupby(["period_id", "store_id"])["sales_value"].sum().groupby("store_id").mean()
    out = mix.fillna(0.0)
    out["__size__"] = np.log1p(size.reindex(out.index).fillna(0.0))
    return out


# ---------------------------------------------------------------- scheme: IBD / frozen
def _map_from_ibd_map(define, base_cell) -> pd.Series:
    m = define.ibd_map
    idx = pd.MultiIndex.from_frame(m[[*base_cell, "PROVINCE"]])
    return pd.Series(m["ibd_id"].to_numpy(), index=idx).rename("gid")


def province_ibd_map(sc: pd.DataFrame, params, hist_end: int, cache: dict) -> pd.Series:
    key = ("ibd", hist_end)
    if key not in cache:
        sub = sc[sc["period_id"] <= hist_end]
        cache[key] = _map_from_ibd_map(define_ibd(sub, hist_end, params.base_cell, params.ibd), params.base_cell)
    return cache[key]


def frozen_ibd_map(sc: pd.DataFrame, params, hist_periods, cache: dict) -> pd.Series:
    key = ("frozen", tuple(hist_periods))
    if key not in cache:
        sub = sc[sc["period_id"].isin(hist_periods)].copy()
        if sub.empty:
            raise ValueError(f"no data for frozen window {hist_periods}")
        stamp = max(hist_periods)
        sub["period_id"] = stamp
        cache[key] = _map_from_ibd_map(define_ibd(sub, stamp, params.base_cell, params.ibd), params.base_cell)
    return cache[key]


def map_ibd(frame: pd.DataFrame, mapping: pd.Series, base_cell) -> np.ndarray:
    idx = pd.MultiIndex.from_frame(frame[[*base_cell, "PROVINCE"]])
    g = mapping.reindex(idx).to_numpy()
    return np.where(pd.isna(g), frame["prov_key"].to_numpy(), g).astype(object)


# ---------------------------------------------------------------- scheme: similar stores
def similar_peer_pairs(frame: pd.DataFrame, sc: pd.DataFrame, params, hist_periods,
                       top_n: int = 20, k_search: int = 150):
    """(target row, peer store_id, category) for the top-N most similar sellers of each row."""
    from sklearn.neighbors import NearestNeighbors

    base_cell = list(params.base_cell)
    feats = feature_matrix(sc, hist_periods)
    if feats.empty:
        return np.array([], dtype=int), np.array([], dtype=object), np.array([], dtype=object)
    store_arr = frame["store_id"].to_numpy()
    cat_arr = frame["category"].to_numpy()
    prov_arr = frame["PROVINCE"].to_numpy()
    ri_out, sid_out, cat_out = [], [], []
    for _, sub in frame.groupby(base_cell, sort=False):
        rows_idx = sub.index.to_numpy()
        stores = pd.Index(pd.unique(store_arr[rows_idx]))
        if len(stores) < MIN_PEER + 1:
            continue
        x = feats.reindex(stores).fillna(0.0).to_numpy(dtype=float)
        sd = x.std(axis=0)
        sd[sd == 0] = 1.0
        x = (x - x.mean(axis=0)) / sd
        nbr = NearestNeighbors(n_neighbors=min(len(stores) - 1, k_search)).fit(x).kneighbors(x)[1]
        pos = {s: i for i, s in enumerate(stores)}
        s_arr = stores.to_numpy()
        prov_of = dict(zip(store_arr[rows_idx], prov_arr[rows_idx]))
        p_arr = np.array([prov_of[s] for s in s_arr], dtype=object)
        masks = {}
        for cat, grp in sub.groupby("category", sort=False):
            m = np.zeros(len(stores), dtype=bool)
            m[[pos[s] for s in pd.unique(grp["store_id"])]] = True
            masks[cat] = m
        for row in rows_idx:
            mask = masks.get(cat_arr[row])
            if mask is None:
                continue
            cand = nbr[pos[store_arr[row]]]
            sel = cand[mask[cand] & (s_arr[cand] != store_arr[row])]
            if sel.size == 0:
                continue
            order = np.argsort(~(p_arr[sel] == prov_arr[row]), kind="stable")
            chosen = sel[order][:top_n]
            ri_out.append(np.full(chosen.size, row))
            sid_out.append(s_arr[chosen])
            cat_out.append(np.full(chosen.size, cat_arr[row], dtype=object))
    if not ri_out:
        return np.array([], dtype=int), np.array([], dtype=object), np.array([], dtype=object)
    return np.concatenate(ri_out), np.concatenate(sid_out), np.concatenate(cat_out)


def peer_pair_stats(n: int, ri: np.ndarray, peer_sid: np.ndarray, peer_cat: np.ndarray,
                    frame: pd.DataFrame, value: np.ndarray) -> pd.DataFrame:
    look = frame[["store_id", "category"]].copy()
    look["v"] = value
    keys = pd.DataFrame({"store_id": peer_sid, "category": peer_cat})
    v = keys.merge(look, on=["store_id", "category"], how="left")["v"].to_numpy(dtype=float)
    return stats_from_long(n, ri, v)


# ---------------------------------------------------------------- scheme: shrink
def anova_kappa(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Variance-component kappa per (cell, category): (pi/2) * sigma2_within / tau2_between."""
    d = frame[["cell_key", "category", "prov_key", value_col]].dropna()
    d = d[np.isfinite(d[value_col].to_numpy(dtype=float))]
    if d.empty:
        return pd.DataFrame(columns=["cell_key", "category", "kappa"])
    d = d.assign(_y2=d[value_col].to_numpy(dtype=float) ** 2)
    gk = ["cell_key", "category", "prov_key"]
    g = d.groupby(gk, sort=False).agg(n=(value_col, "size"), s=(value_col, "sum"), ss=("_y2", "sum")).reset_index()
    g["nmean2"] = g["s"] ** 2 / g["n"]
    agg = g.groupby(["cell_key", "category"], sort=False).agg(
        N=("n", "sum"), K=("n", "size"), sy=("s", "sum"), ss=("ss", "sum"), snmean2=("nmean2", "sum"))
    N, K = agg["N"].to_numpy(dtype=float), agg["K"].to_numpy(dtype=float)
    grand = agg["sy"].to_numpy(dtype=float) / N
    ssb = agg["snmean2"].to_numpy(dtype=float) - N * grand ** 2
    ssw = np.maximum(agg["ss"].to_numpy(dtype=float) - agg["snmean2"].to_numpy(dtype=float), 0.0)
    good = (K >= 2) & (N - K >= 1)
    msb = np.where(good, ssb / np.where(K - 1 > 0, K - 1, 1.0), np.nan)
    msw = np.where(good, ssw / np.where(N - K > 0, N - K, 1.0), np.nan)
    tau2 = np.where(good, np.maximum((msb - msw) / np.where(N > 0, N / np.where(K > 0, K, 1.0), 1.0), 0.0), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = np.where(tau2 > 0, PI_2 * msw / tau2, KAPPA_MAX)
        kappa = np.where((msw <= 0) & (tau2 > 0), KAPPA_MIN, kappa)
    kappa = np.clip(kappa, KAPPA_MIN, KAPPA_MAX)
    out = pd.DataFrame({"cell_key": agg.index.get_level_values(0),
                        "category": agg.index.get_level_values(1),
                        "kappa": kappa, "n_obs": N, "n_prov": K})
    return out


def kappa_lookup(kappa: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    if kappa is None or kappa.empty:
        return np.full(len(frame), KAPPA_MAX)
    k = kappa.drop_duplicates(["cell_key", "category"]).set_index(["cell_key", "category"])["kappa"]
    idx = pd.MultiIndex.from_frame(frame[["cell_key", "category"]])
    v = k.reindex(idx).to_numpy(dtype=float)
    return np.where(np.isfinite(v), v, KAPPA_MAX)


def shrink_weights(stats: pd.DataFrame, kappa: np.ndarray) -> np.ndarray:
    n = stats["n_peer"].to_numpy(dtype=float)
    return n / (n + kappa)


def non_loo_province_stats(frame: pd.DataFrame, value: np.ndarray) -> pd.DataFrame:
    """Q1/Q3/MC per (cell, province, category) over every seller (leave-one-out is not needed
    because the province being described is never the one the target store belongs to)."""
    d = pd.DataFrame({"cell_key": frame["cell_key"].to_numpy(), "prov_key": frame["prov_key"].to_numpy(),
                      "category": frame["category"].to_numpy(), "v": value})
    d = d[np.isfinite(d["v"].to_numpy(dtype=float))]
    keys, rows = [], []
    for (ck, pk, cat), g in d.groupby(["cell_key", "prov_key", "category"], sort=False):
        v = g["v"].to_numpy(dtype=float)
        if v.size < 3:
            continue
        keys.append((ck, pk, cat))
        rows.append([float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75)), medcouple_of(v),
                     float(np.median(v)), _mad(v), float(v.size)])
    if not rows:
        return pd.DataFrame(columns=[*STAT_KEYS, "n_peer"])
    return pd.DataFrame(rows, columns=[*STAT_KEYS, "n_peer"],
                        index=pd.MultiIndex.from_tuples(keys, names=["cell_key", "prov_key", "category"]))


# ---------------------------------------------------------------- scheme: C3 weights
def province_similarity(sc: pd.DataFrame, params, hist_periods, cache) -> dict:
    """Category-mix similarity weights between the provinces of each base cell."""
    key = ("simprov", tuple(hist_periods))
    if key in cache:
        return cache[key]
    feats = feature_matrix(sc, hist_periods)
    out = {}
    if not feats.empty:
        meta = sc[["store_id", *params.base_cell, "PROVINCE"]].drop_duplicates("store_id").reset_index(drop=True)
        f = feats.reindex(meta["store_id"]).fillna(0.0).to_numpy(dtype=float)
        sd = f.std(axis=0)
        sd[sd == 0] = 1.0
        f = (f - f.mean(axis=0)) / sd
        for cell, sub in meta.groupby(list(params.base_cell), sort=False):
            cell_key = "|".join(str(v) for v in (cell if isinstance(cell, tuple) else (cell,)))
            provs = sorted(sub["PROVINCE"].unique())
            if len(provs) < 2:
                continue
            cent = np.vstack([f[sub.index[sub["PROVINCE"].to_numpy() == p].to_numpy()].mean(axis=0) for p in provs])
            dist = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2))
            off = dist[~np.eye(len(provs), dtype=bool)]
            scale = float(np.median(off)) if off.size else 0.0
            scale = scale if scale > 0 else 1.0
            w = np.exp(-dist / scale)
            np.fill_diagonal(w, 0.0)
            row_sum = w.sum(axis=1, keepdims=True)
            w = w / np.where(row_sum > 0, row_sum, 1.0)
            out[cell_key] = {p: dict(zip(provs, w[i])) for i, p in enumerate(provs)}
    cache[key] = out
    return out


def c3_target_stats(frame: pd.DataFrame, value: np.ndarray, sim: dict) -> pd.DataFrame:
    """Similarity-weighted blend of the other provinces' distributions, per (cell, province, category)."""
    ps = non_loo_province_stats(frame, value)
    n = len(frame)
    acc = {k: np.full(n, np.nan) for k in STAT_KEYS}
    acc["n_peer"] = np.zeros(n)
    if ps.empty:
        return pd.DataFrame(acc)
    pk_arr = frame["prov_key"].to_numpy()
    cat_arr = frame["category"].to_numpy()
    for ck, grp in frame.groupby("cell_key", sort=False):
        weights = sim.get(ck)
        if not weights:
            continue
        base = grp.index.to_numpy()
        pairs = pd.Index(list(zip(pk_arr[base], cat_arr[base])))
        for pk, cat in pairs.unique():
            wmap = weights.get(pk.split("|")[-1])
            if not wmap:
                continue
            parts = []
            for other, w in wmap.items():
                key = (ck, f"{ck}|{other}", cat)
                if w > 0 and key in ps.index:
                    parts.append((ps.loc[key], w))
            if not parts:
                continue
            tot = sum(w for _, w in parts)
            sel = base[(pk_arr[base] == pk) & (cat_arr[base] == cat)]
            for k in STAT_KEYS:
                acc[k][sel] = sum(float(row[k]) * w for row, w in parts) / tot
            acc["n_peer"][sel] = max(float(row["n_peer"]) for row, _ in parts)
    return pd.DataFrame(acc)


# ---------------------------------------------------------------- metrics
def flag_set(frame: pd.DataFrame, flags: np.ndarray, mask: np.ndarray | None = None) -> set:
    m = (flags != "ok") & (flags != "na")
    if mask is not None:
        m &= mask
    return set(zip(frame["store_id"].to_numpy()[m], frame["category"].to_numpy()[m]))


def jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else np.nan


def province_lift(frame: pd.DataFrame, flags: np.ndarray, min_rows: int = 50) -> float:
    """Max over provinces of (flag rate in province / overall flag rate): regional clustering."""
    ok = flags != "na"
    if not ok.any():
        return np.nan
    hit = ((flags != "ok") & ok).astype(float)
    overall = float(hit[ok].mean())
    if overall <= 0:
        return np.nan
    s = pd.Series(hit[ok]).groupby(pd.Series(frame["PROVINCE"].to_numpy()[ok]), sort=False)
    rate, n = s.mean(), s.size()
    rate = rate[n >= min_rows]
    return float(rate.max() / overall) if len(rate) else np.nan


def systematic_share(frame: pd.DataFrame, flags: np.ndarray, min_stores: int = 3):
    d = pd.DataFrame({"prov": frame["PROVINCE"].to_numpy(), "cat": frame["category"].to_numpy(),
                      "hi": (flags == "high").astype(float), "lo": (flags == "low").astype(float)})
    d = d[flags != "na"]
    if d.empty:
        return np.nan, 0, 0
    g = d.groupby(["prov", "cat"], sort=False).agg(n=("hi", "size"), hi=("hi", "sum"), lo=("lo", "sum"))
    g = g[g["n"] >= min_stores]
    if g.empty:
        return np.nan, 0, 0
    frac = np.maximum(g["hi"].to_numpy() / g["n"].to_numpy(), g["lo"].to_numpy() / g["n"].to_numpy())
    bad = frac > 0.5
    return float(bad.mean()), int(bad.sum()), int(len(g))


# ---------------------------------------------------------------- main
def run(periods, params_path, out_path, detectors, schemes, core="adjbox",
        target_rate=TARGET_RATE, inject_frac=INJECT_FRAC) -> Path:
    params = load_params(params_path)
    bmp_dir = params.path("before_mp_dir", ROOT)
    tgt_dir = params.path("store_target_dir", ROOT)
    wanted = [p for p in list_before_mp_files(bmp_dir) if p <= max(periods)]
    log(f"loading panel periods={wanted}")
    panel = load_panel(bmp_dir, tgt_dir, params.data.item_key, periods=wanted,
                       encoding=params.data.encoding)
    sc = panel.sc
    log(f"sc rows={len(sc)} periods={panel.periods}")

    a, b = params.fova.adjbox_a, params.fova.adjbox_b
    fence = make_fence(core, a, b)
    frames, cache, kappa_cache, pair_cache = {}, {}, {}, {}
    summary, membership, by_size = [], [], []
    rng = np.random.default_rng(SEED)

    def frame_for(p: int) -> pd.DataFrame:
        if p not in frames:
            prevs = [q for q in panel.periods if q < p]
            frames[p] = build_frame(sc, p, prevs[-1] if prevs else None, params.base_cell)
        return frames[p]

    def history_kappa(mode: str, det: str, hist) -> pd.DataFrame:
        key = (mode, det)
        if key not in kappa_cache:
            hf = [frame_for(p) for p in hist if p in panel.periods]
            if hf:
                merged = pd.concat(hf, ignore_index=True)
                merged = merged[merged[VALUE_COL[det]].notna()]
                kappa_cache[key] = anova_kappa(merged, VALUE_COL[det])
            else:
                kappa_cache[key] = pd.DataFrame(columns=["cell_key", "category", "kappa"])
        return kappa_cache[key]

    for period in periods:
        prevs = [q for q in panel.periods if q < period]
        frame = frame_for(period)
        win = {"fresh": prevs[-3:], "late": prevs[-4:-1]}
        log(f"--- period {period}: {len(frame)} seller rows; windows fresh={win['fresh']} late={win['late']}")
        inj = rng.random(len(frame)) < inject_frac
        pair_cache.clear()          # the peer lists are the memory-heavy part

        for scheme in schemes:
            gids, pairs, ibd_frames, sims = {}, {}, {}, {}
            for mode, hist in win.items():
                if scheme == "现状IBD":
                    hist_end = period if mode == "fresh" else (prevs[-1] if prevs else period)
                    gids[mode] = map_ibd(frame, province_ibd_map(sc, params, hist_end, cache), params.base_cell)
                elif scheme == "A冻结":
                    gids[mode] = map_ibd(frame, frozen_ibd_map(sc, params, hist, cache), params.base_cell)
                elif scheme == "B相似店":
                    ck = ("sim", tuple(hist))
                    if ck not in pair_cache:
                        pair_cache[ck] = similar_peer_pairs(frame, sc, params, hist)
                    pairs[mode] = pair_cache[ck]
                elif scheme == "C2两级收缩":
                    hist_end = period if mode == "fresh" else (prevs[-1] if prevs else period)
                    col = map_ibd(frame, province_ibd_map(sc, params, hist_end, cache), params.base_cell)
                    ibd_frames[mode] = frame.assign(ibd_key=col)
                elif scheme == "C3相似省":
                    sims[mode] = province_similarity(sc, params, hist, cache)

            cat_arr = frame["category"].to_numpy()

            def stats_for(value, mode, kap):
                if scheme in ("现状IBD", "A冻结"):
                    return group_stats([gids[mode], cat_arr], value)
                if scheme == "B相似店":
                    ri, psid, pcat = pairs[mode]
                    return peer_pair_stats(len(frame), ri, psid, pcat, frame, value)
                prov = group_stats([frame["prov_key"].to_numpy(), cat_arr], value)
                if scheme == "C收缩":
                    cell = group_stats([frame["cell_key"].to_numpy(), cat_arr], value)
                    w = shrink_weights(prov, kappa_lookup(kap, frame))
                    return blend([(prov, w), (cell, 1.0 - w)])
                if scheme == "C2两级收缩":
                    ibd = group_stats([ibd_frames[mode]["ibd_key"].to_numpy(), cat_arr], value)
                    cell = group_stats([frame["cell_key"].to_numpy(), cat_arr], value)
                    kl = kappa_lookup(kap, frame)
                    w1, w2 = shrink_weights(prov, kl), shrink_weights(ibd, kl)
                    return blend([(blend([(prov, w1), (ibd, 1.0 - w1)]), w2), (cell, 1.0 - w2)])
                tgt = c3_target_stats(frame, value, sims[mode])
                w = shrink_weights(prov, kappa_lookup(kap, frame))
                return blend([(prov, w), (tgt, 1.0 - w)])

            for det in detectors:
                value = frame[VALUE_COL[det]].to_numpy(dtype=float)
                shrink = scheme in SHRINK_SCHEMES
                stats_by_mode = {
                    mode: stats_for(value, mode, history_kappa(mode, det, hist) if shrink else None)
                    for mode, hist in win.items()
                }
                fresh = stats_by_mode["fresh"]
                c = calibrate(value, fresh, fence, target_rate)
                flags_fresh = flag_values(value, fresh, fence, c)
                flags_late = flag_values(value, stats_by_mode["late"], fence, c)
                stab = jaccard(flag_set(frame, flags_fresh), flag_set(frame, flags_late))

                v_inj = np.where(inj & np.isfinite(value), value + math.log(INJECT_MULT), value)
                kap_f = history_kappa("fresh", det, win["fresh"]) if shrink else None
                fl_inj = flag_values(v_inj, stats_for(v_inj, "fresh", kap_f), fence, c)
                detected = float(np.mean(fl_inj[inj] == "high")) if inj.any() else np.nan

                rate = float(np.mean(flags_fresh[flags_fresh != "na"] != "ok"))
                sys_share, sys_n, sys_base = systematic_share(frame, flags_fresh)
                lift = province_lift(frame, flags_fresh)
                # the same two clustering numbers at a looser budget: at a 1% rate the
                # ">50% of a province x category flagged" test has almost no resolution.
                c5 = calibrate(value, fresh, fence, 0.05)
                f5 = flag_values(value, fresh, fence, c5)
                sys5, _, _ = systematic_share(frame, f5)
                lift5 = province_lift(frame, f5)
                n_peer = fresh["n_peer"].to_numpy(dtype=float)
                usable = np.isfinite(fresh["q1"].to_numpy(dtype=float)) & (n_peer >= MIN_PEER)
                summary.append({
                    "period_id": period, "detector": det, "scheme": scheme,
                    "n_rows": int(len(frame)), "n_usable": int(usable.sum()),
                    "coverage": float(usable.mean()),
                    "median_n_peer": float(np.median(n_peer[usable])) if usable.any() else np.nan,
                    "anomaly_rate": rate, "calibrated_c": c,
                    "stability_jaccard": stab, "x3_detect_rate": detected,
                    "systematic_share": sys_share, "systematic_n": sys_n, "systematic_base": sys_base,
                    "province_lift_max": lift,
                    "systematic_share_5pct": sys5, "province_lift_max_5pct": lift5,
                })
                for (lo, hi), lab in zip(SIZE_BUCKETS, SIZE_LABELS):
                    m = usable & (n_peer >= lo) & (n_peer < hi)
                    if not m.any():
                        continue
                    hit = inj & m
                    by_size.append({
                        "period_id": period, "detector": det, "scheme": scheme,
                        "peer_bucket": lab, "n_rows": int(m.sum()),
                        "anomaly_rate": float(np.mean(flags_fresh[m] != "ok")),
                        "stability_jaccard": jaccard(flag_set(frame, flags_fresh, m),
                                                     flag_set(frame, flags_late, m)),
                        "x3_detect_rate": float(np.mean(fl_inj[hit] == "high")) if hit.any() else np.nan,
                    })
                log(f"  {scheme:<10s} {det:<9s} c={c:7.2f} rate={rate:.4f} "
                    f"stab={stab:.3f} x3={detected:.3f} sys={sys_share:.4f} lift={lift:.2f} "
                    f"sys5={sys5:.4f} lift5={lift5:.2f} "
                    f"cov={usable.mean():.3f} npeer={np.median(n_peer[usable]) if usable.any() else float('nan'):.0f}")

            if scheme in ("现状IBD", "A冻结"):
                gf, gl = np.asarray(gids["fresh"]), np.asarray(gids["late"])
                m = pd.notna(gf) & pd.notna(gl)
                churn = float(np.mean(gf[m] != gl[m])) if m.any() else np.nan
            else:
                churn = np.nan
            membership.append({"period_id": period, "scheme": scheme, "member_churn": churn})

    out = pd.DataFrame(summary)
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    setting_rows = [
        ("periods", str(periods)), ("detectors", str(detectors)), ("schemes", str(schemes)),
        ("target_rate", target_rate), ("inject_frac", inject_frac), ("inject_mult", INJECT_MULT),
        ("min_peer", MIN_PEER), ("adjbox_a", a), ("adjbox_b", b),
        ("kappa_scale", PI_2), ("kappa_bounds", f"[{KAPPA_MIN}, {KAPPA_MAX}]"),
        ("detection_core", core),
        ("peer_scope", "sellers only (zero-filled rows excluded)"),
        ("notes", "kappa estimated by one-way ANOVA over province on the same detector value"),
        ("fence_adjbox", f"Q1/Q3 LOO, medcouple on the full group; a={a}, b={b}"),
        ("fence_robustz", "leave-one-out median +/- c * full-group MAD, c calibrated to the target rate"),
    ]
    with pd.ExcelWriter(dest, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(membership).to_excel(writer, sheet_name="membership_churn", index=False)
        size_df = pd.DataFrame(by_size)
        if not size_df.empty:
            size_df.to_excel(writer, sheet_name="by_peer_size", index=False)
            for metric in ("stability_jaccard", "x3_detect_rate"):
                size_df.pivot_table(index=["detector", "peer_bucket"], columns="scheme",
                                    values=metric, aggfunc="mean") \
                       .to_excel(writer, sheet_name=f"size_{metric}"[:31])
        for metric in ("stability_jaccard", "x3_detect_rate", "systematic_share",
                       "province_lift_max", "systematic_share_5pct", "province_lift_max_5pct"):
            out.pivot_table(index=["period_id", "detector"], columns="scheme", values=metric) \
               .to_excel(writer, sheet_name=metric[:31])
        pd.DataFrame(setting_rows, columns=["key", "value"]).to_excel(writer, sheet_name="settings", index=False)
    log(f"wrote {dest}")
    return dest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="peer-group scheme backtest over an adjbox or robust-z core")
    p.add_argument("--periods", type=int, nargs="*", default=[20261405, 20261406, 20261408])
    p.add_argument("--params", default=str(ROOT / "config" / "peer_params.json"))
    p.add_argument("--out", default=None)
    p.add_argument("--detectors", nargs="*", default=list(DETECTORS))
    p.add_argument("--schemes", nargs="*", default=list(SCHEMES))
    p.add_argument("--core", choices=("adjbox", "robustz"), default="adjbox",
                   help="fence core; robustz reproduces the README's median/MAD comparison")
    p.add_argument("--target-rate", type=float, default=TARGET_RATE)
    p.add_argument("--inject-frac", type=float, default=INJECT_FRAC)
    p.add_argument("--quick", action="store_true", help="one period, one detector, two schemes")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    periods, detectors, schemes = list(args.periods), list(args.detectors), list(args.schemes)
    if args.quick:
        periods, detectors, schemes = periods[:1], detectors[:1], ["现状IBD", "C收缩"]
    stem = f"group_scheme_backtest_{args.core}_{'_'.join(str(p) for p in periods)}"
    out = args.out or str(ROOT / "data" / "backtest" / f"{stem}.xlsx")
    run(periods, args.params, out, detectors, schemes, core=args.core,
        target_rate=args.target_rate, inject_frac=args.inject_frac)


if __name__ == "__main__":
    main()
