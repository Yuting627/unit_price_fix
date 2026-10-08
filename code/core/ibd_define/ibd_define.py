"""Layer 1 pooling: merge similar provinces inside each base cell into IBDs.

Province pooling is this framework's own design. The distance orders merge candidates; the acceptance test
(store sales distribution) decides. ``ibd.distance``:
- "category_mix" (default): RMS difference of province category-mix profiles (mean log share per category),
  so provinces that sell a similar mix (regional taste) are pooled.
- "trend": same-store log links of the province, Euclidean(D_p - D_q) / n_overlap (formula borrowed from the
  IBD Shop Similarity doc, which ranks shops within an IBD).
``ibd.method = "size"`` instead builds FBD-style size bands inside each base cell.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from funcs.robust import cv, heterogeneity, passes

MERGE_ACCEPTED = "accepted"
MERGE_FORCED = "forced_nearest"
SOFT_KEPT = "soft_kept"


@dataclass
class IbdDefineResult:
    ibd_map: pd.DataFrame
    store_map: pd.DataFrame
    merges: pd.DataFrame
    summary: pd.DataFrame
    distances: dict = field(default_factory=dict)
    base_cell: tuple = ()

    @property
    def forced_ratio(self) -> float:
        if self.merges.empty:
            return 0.0
        return float((self.merges["pooling_reason"] == MERGE_FORCED).mean())


def cell_label(values) -> str:
    return "|".join(str(v) for v in values)


def store_totals(sc: pd.DataFrame, base_cell) -> pd.DataFrame:
    keys = ["period_id", "store_id", *base_cell, "PROVINCE"]
    return sc.groupby(keys, observed=True, sort=False)["sales_value"].sum().reset_index()


def province_links(totals: pd.DataFrame, base_cell, min_common: int) -> pd.DataFrame:
    """Same-store log links per province: D_t = log(sum_C x_t / sum_C x_{t-1}) over stores C selling in both periods.

    Index (cell..., PROVINCE), columns = period t of the t-1 -> t link; NaN when fewer than ``min_common``
    common stores. Unlike the change of per-store averages, this is not moved by panel replacement.
    """
    keys = [*base_cell, "PROVINCE"]
    sell = totals[totals["sales_value"] > 0]
    periods = sorted(sell["period_id"].unique())
    out = []
    for prev, cur in zip(periods[:-1], periods[1:]):
        a = sell[sell["period_id"] == prev][["store_id", *keys, "sales_value"]]
        b = sell[sell["period_id"] == cur][["store_id", *keys, "sales_value"]]
        both = a.merge(b, on=["store_id", *keys], suffixes=("_prev", "_cur"))
        g = both.groupby(keys, observed=True).agg(n=("store_id", "size"), prev=("sales_value_prev", "sum"), cur=("sales_value_cur", "sum"))
        link = np.log(g["cur"] / g["prev"]).where(g["n"] >= min_common)
        out.append(link.rename(cur))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, axis=1).sort_index(axis=1)


def mix_lookback_periods(all_periods, period: int, lookback: int, start_period: int = 0) -> list[int]:
    """Last ``lookback`` periods at or before ``period``, optionally cut at ``start_period`` (0 = off).

    If the cut leaves the window empty (e.g. current period is before the cut), fall back to ``[period]``.
    """
    avail = sorted(int(p) for p in all_periods if int(p) <= period)
    if start_period:
        cut = [p for p in avail if p >= start_period]
        if cut:
            avail = cut
    window = avail[-max(1, lookback):]
    return window or [period]


def mix_period_weights(n: int, weights) -> np.ndarray:
    """Oldest-to-newest weights of length ``n``; empty ``weights`` -> recency 2^t, then L1-normalize."""
    if n <= 0:
        return np.array([], dtype=float)
    if weights:
        w = np.asarray(list(weights)[-n:], dtype=float)
        if w.size < n:
            w = np.concatenate([np.zeros(n - w.size), w])
    else:
        w = np.power(2.0, np.arange(n, dtype=float))
    total = float(w.sum())
    return w / total if total > 0 else np.full(n, 1.0 / n)


def _mix_one(sc_cur: pd.DataFrame, base_cell) -> pd.DataFrame:
    """Category mix for one period: mean over stores of log1p(1000 * cat_share)."""
    keys = [*base_cell, "PROVINCE"]
    stores = sc_cur[["store_id", *keys]].drop_duplicates("store_id")
    cats = sc_cur[[*base_cell, "category"]].drop_duplicates()
    grid = stores.merge(cats, on=list(base_cell))
    share = sc_cur.groupby(["store_id", "category"], observed=True)["cat_share"].sum()
    grid["sh"] = np.log1p(1000 * grid.set_index(["store_id", "category"]).index.map(share).fillna(0).to_numpy(dtype=float))
    return grid.groupby([*keys, "category"], observed=True)["sh"].mean().unstack("category")


def province_mix_profile(sc: pd.DataFrame, base_cell, ibd=None, period: int | None = None) -> pd.DataFrame:
    """Category mix per province: mean over stores of log1p(1000 * cat_share), index (cell..., PROVINCE), columns category.

    Every store gets every category sold in its base cell (0 when not sold), so "does not sell coffee" counts.
    With ``ibd.mix_lookback`` > 1, period vectors are averaged (weights oldest-to-newest) before RMS;
    a province missing a period drops that period and renormalizes. ``mix_start_period`` > 0 drops earlier
    periods; ``mix_common_stores`` keeps stores present in every period of the common window.
    ``mix_common_lookback`` n>0 uses only the last n periods of the mix window (e.g. 2 = current+previous).
    """
    if "period_id" not in sc.columns:
        return _mix_one(sc, base_cell)
    if period is None:
        period = int(sc["period_id"].max())
    lookback = int(ibd.get("mix_lookback", 1)) if ibd is not None else 1
    start = int(ibd.get("mix_start_period", 0) or 0) if ibd is not None else 0
    common = bool(ibd.get("mix_common_stores", False)) if ibd is not None else False
    common_n = int(ibd.get("mix_common_lookback", 0) or 0) if ibd is not None else 0
    raw_w = list(ibd.get("mix_weights", []) or []) if ibd is not None else []
    window = mix_lookback_periods(sc["period_id"].unique(), period, lookback, start)
    sub = sc[sc["period_id"].isin(window)]
    if common and len(window) > 1:
        common_periods = window[-common_n:] if common_n else window
        if len(common_periods) < 2:
            common_periods = window[-2:] if len(window) >= 2 else window
        sets = [set(sub.loc[sub["period_id"] == p, "store_id"]) for p in common_periods]
        keep = set.intersection(*sets) if sets else set()
        if keep:
            sub = sub[sub["store_id"].isin(keep)]
    if len(window) == 1:
        return _mix_one(sub[sub["period_id"] == window[0]], base_cell)
    weights = mix_period_weights(len(window), raw_w)
    keys = [*base_cell, "PROVINCE"]
    parts = []
    for w, p in zip(weights, window):
        one = _mix_one(sub[sub["period_id"] == p], base_cell)
        if one.empty:
            continue
        stacked = one.stack()
        stacked = stacked.rename("sh").reset_index()
        stacked["w"] = w
        parts.append(stacked)
    if not parts:
        return _mix_one(sc[sc["period_id"] == period], base_cell)
    allp = pd.concat(parts, ignore_index=True)
    allp = allp[np.isfinite(allp["sh"])]
    allp["wx"] = allp["sh"] * allp["w"]
    g = allp.groupby([*keys, "category"], observed=True, sort=False)
    out = (g["wx"].sum() / g["w"].sum()).unstack("category")
    return out


def mix_distance(profile: pd.DataFrame) -> pd.DataFrame:
    """Root mean squared difference of category mix profiles over categories present in both rows."""
    x = profile.to_numpy(dtype=float)
    valid = np.isfinite(x)
    n = len(profile)
    out = np.full((n, n), np.nan)
    for i in range(n):
        out[i, i] = 0.0
        for j in range(i + 1, n):
            both = valid[i] & valid[j]
            if both.any():
                out[i, j] = out[j, i] = math.sqrt(float(((x[i, both] - x[j, both]) ** 2).mean()))
    return pd.DataFrame(out, index=profile.index, columns=profile.index)


def _cell_rows(frame: pd.DataFrame, cell: tuple, provinces) -> pd.DataFrame:
    try:
        sub = frame.xs(cell, level=list(range(len(cell))), drop_level=True)
    except KeyError:
        sub = pd.DataFrame(columns=frame.columns, dtype=float)
    return sub.reindex(provinces)


def _scaled(dist: pd.DataFrame) -> pd.DataFrame:
    """Divide by the median finite off-diagonal distance so different distances share one scale."""
    v = dist.to_numpy(dtype=float)
    off = v[~np.eye(len(v), dtype=bool) & np.isfinite(v)]
    med = float(np.median(off)) if off.size else 0.0
    return dist / med if med > 0 else dist


def combine_distances(mix: pd.DataFrame, trend: pd.DataFrame, weight_mix: float) -> pd.DataFrame:
    """weight_mix * mix + (1 - weight_mix) * trend on median-scaled distances; mix alone where trend is unknown."""
    m, t = _scaled(mix), _scaled(trend)
    return m.where(~np.isfinite(t), weight_mix * m + (1 - weight_mix) * t)


def cell_distance(cell: tuple, provinces, profile, links, ibd) -> pd.DataFrame:
    """Province distance matrix of one base cell for ``ibd.distance`` = category_mix / trend / combined."""
    mix = mix_distance(_cell_rows(profile, cell, provinces)) if profile is not None else None
    trend = trend_distance(_cell_rows(links, cell, provinces), ibd.min_overlap_periods) if links is not None else None
    if mix is None:
        return trend
    if trend is None:
        return mix
    return combine_distances(mix, trend, ibd.distance_mix_weight)


def trend_distance(D: pd.DataFrame, min_overlap: int) -> pd.DataFrame:
    """Normalized Shop Similarity distance between rows of the link matrix ``D`` (NaN = no link)."""
    d = D.to_numpy(dtype=float)
    valid = np.isfinite(d)
    return _pairwise(np.where(valid, d, 0.0), valid, D.index, min_overlap)


def province_distance(series: pd.DataFrame, min_overlap: int) -> pd.DataFrame:
    """Normalized Shop Similarity distance between rows of a level ``series`` (periods as columns)."""
    x = series.to_numpy(dtype=float)
    observed = np.isfinite(x)
    L = np.log1p(np.where(observed, x, 0.0))
    D = L[:, 1:] - L[:, :-1]
    valid = observed[:, 1:] & observed[:, :-1]
    return _pairwise(D, valid, series.index, min_overlap)


def _pairwise(D: np.ndarray, valid: np.ndarray, index, min_overlap: int) -> pd.DataFrame:
    n = len(index)
    out = np.full((n, n), np.nan)
    for i in range(n):
        out[i, i] = 0.0
        for j in range(i + 1, n):
            both = valid[i] & valid[j]
            k = int(both.sum())
            if k >= min_overlap:
                d = math.sqrt(float(((D[i, both] - D[j, both]) ** 2).sum())) / k
                out[i, j] = out[j, i] = d
    return pd.DataFrame(out, index=index, columns=index)


def group_distance(dist: pd.DataFrame, members_a, members_b) -> float:
    """Median province distance between two member lists (inf when unknown)."""
    sub = dist.loc[list(members_a), list(members_b)].to_numpy(dtype=float)
    sub = sub[np.isfinite(sub)]
    return float(np.median(sub)) if sub.size else math.inf


def distance_cap(dist: pd.DataFrame, quantile: float) -> float:
    """``quantile`` of finite province-pair distances in a cell; inf when disabled (>= 1) or unknown."""
    v = dist.to_numpy(dtype=float)
    pairs = v[np.triu_indices(len(v), k=1)]
    pairs = pairs[np.isfinite(pairs)]
    if quantile >= 1 or not pairs.size:
        return math.inf
    return float(np.quantile(pairs, quantile))


def _merge_cell(provinces: list[str], values: dict, dist: pd.DataFrame, ibd) -> tuple[dict, list, dict, set]:
    """Greedy merge of provinces; returns (groups, merges, prov_info, soft_kept group keys).

    A candidate farther than ``distance_cap`` (the ``max_distance_quantile`` of this cell's province distances)
    cannot be accepted: small samples pass the size test too easily, so distance alone must bound the merge.
    A group below ``min_store`` but with at least ``min_store_soft`` stores is kept on its own
    (``soft_kept``) when no candidate passes, instead of being forced into the nearest. With ``soft_stop``
    such a group does not look for a merge at all: ``min_store`` is the target, ``min_store_soft`` the floor,
    so a 28-store group is not folded into a 43-store one just to reach 30.
    """
    cap = distance_cap(dist, ibd.max_distance_quantile)
    target = ibd.min_store_soft if ibd.soft_stop else ibd.min_store
    groups = {p: [p] for p in provinces}
    prov_info = {p: {"pooling_reason": "none", "vi": np.nan, "ks": np.nan, "psi": np.nan, "distance": np.nan} for p in provinces}
    merges = []
    soft_kept: set = set()

    def size(g):
        return sum(len(values[p]) for p in groups[g])

    def vals(g):
        return np.concatenate([values[p] for p in groups[g]])

    while len(groups) > 1:
        small = [g for g in groups if size(g) < target and g not in soft_kept]
        if not small:
            break
        g = min(small, key=lambda k: (size(k), k))
        cands = sorted(
            (h for h in groups if h != g),
            key=lambda h: (group_distance(dist, groups[g], groups[h]), -size(h), h),
        )
        chosen, reason, het = None, MERGE_FORCED, None
        for h in cands:
            if group_distance(dist, groups[g], groups[h]) > cap:
                break
            h_het = heterogeneity(vals(h), vals(g), ibd.psi_bins, ibd.het_alpha, ibd.psi_min_bin_count)
            if passes(h_het, ibd.vi_max, ibd.ks_max, ibd.psi_max, ibd.max_median_ratio):
                chosen, reason, het = h, MERGE_ACCEPTED, h_het
                break
        # Soft-keep when acceptance fails and either the group already meets the soft floor,
        # or forced merges are disabled (allow_forced_nearest=false): never fold into a neighbour.
        if chosen is None and (size(g) >= ibd.min_store_soft or not ibd.allow_forced_nearest):
            soft_kept.add(g)
            nearest = heterogeneity(vals(cands[0]), vals(g), ibd.psi_bins, ibd.het_alpha, ibd.psi_min_bin_count)
            for p in groups[g]:
                if prov_info[p]["pooling_reason"] == "none":
                    prov_info[p] = {"pooling_reason": SOFT_KEPT, "distance": group_distance(dist, groups[g], groups[cands[0]]), **nearest}
            continue
        if chosen is None:
            chosen = cands[0]
            het = heterogeneity(vals(chosen), vals(g), ibd.psi_bins, ibd.het_alpha, ibd.psi_min_bin_count)
        d = group_distance(dist, groups[g], groups[chosen])
        merges.append(
            {
                "from_members": "+".join(groups[g]),
                "into_members": "+".join(groups[chosen]),
                "n_from": size(g),
                "n_into": size(chosen),
                "pooling_reason": reason,
                "distance": d,
                "distance_cap": cap,
                **het,
            }
        )
        for p in groups[g]:
            prov_info[p] = {"pooling_reason": reason, "distance": d, **het}
        groups[chosen] = groups[chosen] + groups.pop(g)
        if chosen in soft_kept and size(chosen) >= ibd.min_store:
            soft_kept.discard(chosen)
    for g in groups:
        if size(g) < ibd.min_store and size(g) >= ibd.min_store_soft and g not in soft_kept:
            soft_kept.add(g)
            for p in groups[g]:
                if prov_info[p]["pooling_reason"] == "none":
                    prov_info[p]["pooling_reason"] = SOFT_KEPT
    return groups, merges, prov_info, soft_kept


def recheck_members(groups: dict, values: dict, ibd) -> dict:
    """Each province of a multi-province group against the rest of its final group (same acceptance test).

    Greedy merging compares against the group as it was at merge time; later merges can drift away.
    """
    out = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        for p in members:
            rest = np.concatenate([values[q] for q in members if q != p])
            het = heterogeneity(rest, values[p], ibd.psi_bins, ibd.het_alpha, ibd.psi_min_bin_count)
            med = np.median(rest) if rest.size else np.nan
            out[p] = {
                "final_vi": het["vi"], "final_ks": het["ks"], "final_ks_p": het["ks_p"],
                "final_med_ratio": float(np.expm1(np.median(values[p])) / np.expm1(med)) if rest.size and np.expm1(med) > 0 else np.nan,
                "final_ok": passes(het, ibd.vi_max, ibd.ks_max, ibd.psi_max, ibd.max_median_ratio),
            }
    return out


def store_size(totals: pd.DataFrame, period: int, hist_periods: int) -> pd.DataFrame:
    """Store size for banding: mean total sales over its last ``hist_periods`` selling periods before ``period``.

    Stores without history fall back to the current period (``size_source = current``); banding on current
    sales would let a spiking store move into a larger band and look normal, so history is preferred.
    """
    sell = totals[totals["sales_value"] > 0]
    prior = sell[sell["period_id"] < period].sort_values("period_id")
    hist = prior.groupby("store_id").tail(hist_periods).groupby("store_id")["sales_value"].agg(["mean", "size"])
    cur = sell[sell["period_id"] == period].set_index("store_id")["sales_value"]
    out = pd.DataFrame({"size": hist["mean"], "size_periods": hist["size"]}).reindex(cur.index)
    out["size_source"] = np.where(out["size"].notna(), "history", "current")
    out["size"] = out["size"].fillna(cur)
    out["size_periods"] = out["size_periods"].fillna(0).astype(int)
    return out.reset_index()


def _band_labels(k: int) -> list[str]:
    names = {1: ["All"], 2: ["Small", "Large"], 3: ["Small", "Medium", "Large"], 4: ["Small", "SmallMedium", "LargeMedium", "Large"]}
    return names.get(k, [f"B{i + 1}" for i in range(k)])


def define_ibd_by_size(sc: pd.DataFrame, period: int, base_cell, ibd, logger=None) -> IbdDefineResult:
    """FBD-style IBDs: inside each base cell, equal-count bands of historical store size.

    Bands = min(n_store // min_store, size_max_splits), so every band has >= min_store stores; one band keeps
    the whole cell. CV_value (CV of current store sales) of the cell and of the bands is reported, as in FBD.
    """
    base_cell = tuple(base_cell)
    totals = store_totals(sc[sc["period_id"] <= period], base_cell)
    cur = totals[totals["period_id"] == period]
    sizes = store_size(totals, period, ibd.size_hist_periods).set_index("store_id")
    map_rows, store_rows, summary_rows, distances = [], [], [], {}
    for cell, cell_cur in cur.groupby(list(base_cell), observed=True, sort=True):
        cell = cell if isinstance(cell, tuple) else (cell,)
        label = cell_label(cell)
        s = cell_cur[["store_id", "PROVINCE", "sales_value"]].copy()
        s = s.join(sizes, on="store_id")
        s["size"] = s["size"].fillna(s["sales_value"])
        s["size_source"] = s["size_source"].fillna("current")
        k = int(max(1, min(len(s) // ibd.min_store, ibd.size_max_splits)))
        names = _band_labels(k)
        band = np.minimum((s["size"].rank(method="first", pct=True).to_numpy() * k - 1e-9).astype(int), k - 1)
        s["band"] = [names[b] for b in band]
        s["ibd_id"] = [f"{label}|{names[b]}" for b in band]
        distances[cell] = pd.DataFrame(np.abs(np.subtract.outer(range(k), range(k))).astype(float), index=names, columns=names)
        for b, name in enumerate(names):
            g = s[s["band"] == name]
            map_rows.append({
                **dict(zip(base_cell, cell)),
                "ibd_id": f"{label}|{name}", "member_provinces": name, "band": name, "n_band": k,
                "n_store": int(len(g)), "n_province": int(g["PROVINCE"].nunique()),
                "size_min": float(g["size"].min()), "size_median": float(g["size"].median()), "size_max": float(g["size"].max()),
                "n_size_from_current": int((g["size_source"] == "current").sum()),
                "cv_value": cv(g["sales_value"].to_numpy()), "cv_size": cv(g["size"].to_numpy()),
                "pooling_reason": "size_band", "soft_kept": False, "final_ok": True,
            })
        stores = s[["store_id", "PROVINCE", "ibd_id"]].copy()
        for col, v in zip(base_cell, cell):
            stores[col] = v
        store_rows.append(stores)
        cvs = [cv(g["sales_value"].to_numpy()) for _, g in s.groupby("band")]
        weights = s.groupby("band").size().to_numpy()
        summary_rows.append({
            "cell": label, "n_province": int(s["PROVINCE"].nunique()), "n_ibd": k, "n_merge": 0, "accepted": 0, "forced_nearest": 0,
            "soft_kept": 0, "recheck_fail": 0, "n_store": int(len(s)), "min_ibd_store": int(weights.min()),
            "cv_value_cell": cv(s["sales_value"].to_numpy()), "cv_value_bands": float(np.average(cvs, weights=weights)),
            "n_size_from_current": int((s["size_source"] == "current").sum()),
        })
    ibd_map = pd.DataFrame(map_rows)
    store_map = pd.concat(store_rows, ignore_index=True)[["store_id", *base_cell, "PROVINCE", "ibd_id"]] if store_rows else pd.DataFrame()
    result = IbdDefineResult(ibd_map, store_map, pd.DataFrame(), pd.DataFrame(summary_rows), distances, base_cell)
    if logger is not None:
        for row in result.summary.itertuples():
            logger.info(
                "[IBD_DEFINE] %s: %d 店 -> %d 档 (最小档 %d 店); CV_value 整体 %.2f -> 分档加权 %.2f; 无历史按当期定档 %d 店",
                row.cell, row.n_store, row.n_ibd, row.min_ibd_store, row.cv_value_cell, row.cv_value_bands, row.n_size_from_current,
            )
    return result


def define_ibd(sc: pd.DataFrame, period: int, base_cell, ibd, logger=None) -> IbdDefineResult:
    """Build IBDs for ``period``: ``ibd.method`` = "size" (FBD size bands) or "province" (similar-province pooling)."""
    if ibd.method == "size":
        return define_ibd_by_size(sc, period, base_cell, ibd, logger=logger)
    base_cell = tuple(base_cell)
    hist = sc[sc["period_id"] <= period]
    totals = store_totals(hist, base_cell)
    profile = province_mix_profile(hist, base_cell, ibd=ibd, period=period) if ibd.distance != "trend" else None
    links = province_links(totals, base_cell, ibd.trend_min_common_stores) if ibd.distance != "category_mix" else None
    cur = totals[totals["period_id"] == period]

    map_rows, store_rows, merge_rows, summary_rows, distances = [], [], [], [], {}
    for cell, cell_cur in cur.groupby(list(base_cell), observed=True, sort=True):
        cell = cell if isinstance(cell, tuple) else (cell,)
        label = cell_label(cell)
        provinces = sorted(cell_cur["PROVINCE"].unique())
        dist = cell_distance(cell, provinces, profile, links, ibd)
        distances[cell] = dist
        values = {p: np.log1p(cell_cur.loc[cell_cur["PROVINCE"] == p, "sales_value"].clip(lower=0).to_numpy()) for p in provinces}

        groups, merges, prov_info, soft_kept = _merge_cell(provinces, values, dist, ibd)
        final = recheck_members(groups, values, ibd)
        soft_members = {p for g in soft_kept for p in groups[g]}
        ordered = sorted(groups.values(), key=lambda m: (-sum(len(values[p]) for p in m), m[0]))
        prov_to_ibd = {}
        for idx, members in enumerate(ordered, start=1):
            ibd_id = f"{label}|G{idx:02d}"
            group_vals = np.concatenate([values[p] for p in members])
            for p in members:
                prov_to_ibd[p] = ibd_id
                map_rows.append(
                    {
                        **dict(zip(base_cell, cell)),
                        "PROVINCE": p,
                        "ibd_id": ibd_id,
                        "member_provinces": "+".join(sorted(members)),
                        "n_store": int(group_vals.size),
                        "n_store_province": int(values[p].size),
                        "soft_kept": p in soft_members,
                        "cv_value": cv(np.expm1(group_vals)),
                        **prov_info[p],
                        **final.get(p, {"final_ok": True}),
                    }
                )
        for m in merges:
            merge_rows.append({**dict(zip(base_cell, cell)), **m})
        stores = cell_cur[["store_id", "PROVINCE"]].copy()
        stores["ibd_id"] = stores["PROVINCE"].map(prov_to_ibd)
        for col, v in zip(base_cell, cell):
            stores[col] = v
        store_rows.append(stores)
        n_forced = sum(m["pooling_reason"] == MERGE_FORCED for m in merges)
        summary_rows.append(
            {
                "cell": label,
                "n_province": len(provinces),
                "n_ibd": len(groups),
                "n_merge": len(merges),
                "accepted": len(merges) - n_forced,
                "forced_nearest": n_forced,
                "soft_kept": len(soft_kept),
                "recheck_fail": sum(not v["final_ok"] for v in final.values()),
                "n_store": int(len(cell_cur)),
                "min_ibd_store": int(min(sum(len(values[p]) for p in m) for m in groups.values())),
            }
        )

    ibd_map = pd.DataFrame(map_rows)
    store_map = pd.concat(store_rows, ignore_index=True)[["store_id", *base_cell, "PROVINCE", "ibd_id"]] if store_rows else pd.DataFrame()
    if not store_map.empty and not ibd_map.empty and {"soft_kept", "final_ok"} <= set(ibd_map.columns):
        keys = [*base_cell, "PROVINCE", "ibd_id"]
        store_map = store_map.merge(ibd_map[keys + ["soft_kept", "final_ok"]].drop_duplicates(keys), on=keys, how="left")
    merges_df = pd.DataFrame(merge_rows)
    result = IbdDefineResult(ibd_map, store_map, merges_df, pd.DataFrame(summary_rows), distances, base_cell)
    if logger is not None:
        for row in result.summary.itertuples():
            logger.info(
                "[IBD_DEFINE] %s: %d 省 -> %d IBD; 合并 %d 次 (accepted %d, forced_nearest %d); soft_kept %d; 复查不通过 %d",
                row.cell, row.n_province, row.n_ibd, row.n_merge, row.accepted, row.forced_nearest, row.soft_kept, row.recheck_fail,
            )
        warn_forced_nearest(result, ibd.forced_nearest_warn_ratio, logger)
        warn_recheck(result, logger)
    return result


def warn_recheck(result: IbdDefineResult, logger) -> bool:
    """Warn listing provinces that no longer pass the acceptance test against the rest of their final IBD."""
    m = result.ibd_map
    bad = m[~m["final_ok"].astype(bool)] if "final_ok" in m else m.iloc[0:0]
    if bad.empty:
        logger.info("[IBD_DEFINE] 合并后复查: 全部省份通过")
        return False
    logger.warning(
        "[IBD_DEFINE] WARNING 合并后复查 %d 省不通过 (与所在 IBD 其余省份分布不一致): %s",
        len(bad), "; ".join(
            f"{r.ibd_id} {r.PROVINCE} n={r.n_store_province} 中位数比={r.final_med_ratio:.2f}"
            for r in bad.sort_values("final_ks_p").itertuples()
        ),
    )
    return True


def warn_forced_nearest(result: IbdDefineResult, warn_ratio: float, logger) -> bool:
    """Warn when forced merges dominate; returns True when warned."""
    ratio = result.forced_ratio
    if not result.merges.empty and ratio >= warn_ratio:
        logger.warning(
            "[IBD_DEFINE] WARNING forced_nearest_ratio=%.2f >= %.2f -> 省份相似度门槛或 base_cell 可能不合适，请检查 ibd / base_cell 配置",
            ratio, warn_ratio,
        )
        return True
    logger.info("[IBD_DEFINE] forced_nearest_ratio=%.2f (< %.2f)", ratio, warn_ratio)
    return False
