"""Backtest: category-mix lookback weights for IBD province pooling.

Arms W0–W4 change only the mix *distance* window (acceptance still uses current-period size).
Phase 1: cluster Jaccard / membership churn / nearest-neighbour flips.
Phase 2: on 20261408 sales, adjbox flags under each arm's 07 vs 08 IBD map.

Usage:
    python backtest/ibd_mix_window.py
    python backtest/ibd_mix_window.py --quick
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from core.config import load_params, params_from_dict
from core.data_prep.data_prep import list_before_mp_files, load_panel
from core.ibd_define.ibd_define import define_ibd, mix_distance, province_mix_profile
from funcs.robust import HIGH, LOW, adjbox_stats, fence_from_stats, flag

PERIODS = (20261406, 20261407, 20261408)
OUT_XLSX = ROOT / "data" / "backtest" / "ibd_mix_window.xlsx"
A, B, C_FENCE = -4.0, 3.0, 7.0

ARMS = {
    "W0": dict(mix_lookback=1, mix_weights=[], mix_common_stores=False, mix_start_period=0),
    "W1": dict(mix_lookback=3, mix_weights=[1 / 3, 1 / 3, 1 / 3], mix_common_stores=False, mix_start_period=0),
    "W2": dict(mix_lookback=3, mix_weights=[0.2, 0.3, 0.5], mix_common_stores=False, mix_start_period=0),
    "W3": dict(mix_lookback=3, mix_weights=[0.2, 0.3, 0.5], mix_common_stores=True, mix_common_lookback=0, mix_start_period=0),
    "W3_2": dict(mix_lookback=3, mix_weights=[0.2, 0.3, 0.5], mix_common_stores=True, mix_common_lookback=2, mix_start_period=0),
    "W4": dict(mix_lookback=3, mix_weights=[0.2, 0.3, 0.5], mix_common_stores=False, mix_start_period=20261407),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def override_ibd(base_raw: dict, **ibd_kw):
    raw = copy.deepcopy(base_raw)
    raw["ibd"].update(ibd_kw)
    return params_from_dict(raw, source="config/peer_params.json")


def pkey(m: pd.DataFrame, base_cell) -> pd.Series:
    return m[list(base_cell)].astype(str).agg("|".join, axis=1) + "|" + m["PROVINCE"].astype(str)


def clusters(ibd_map: pd.DataFrame, base_cell) -> dict[str, frozenset]:
    out = {}
    for r in ibd_map.itertuples(index=False):
        key = "|".join(str(getattr(r, c)) for c in base_cell) + "|" + str(r.PROVINCE)
        members = str(r.member_provinces).split("+") if pd.notna(r.member_provinces) else [str(r.PROVINCE)]
        out[key] = frozenset(members)
    return out


def cluster_compare(prev: dict, cur: dict) -> dict:
    common = set(prev) & set(cur)
    if not common:
        return dict(n_overlap=0, exact_same=0, exact_same_pct=np.nan, jac_mean=np.nan, jac_p50=np.nan, jac_ge50=np.nan)
    jacs = []
    same = 0
    for k in common:
        a, b = prev[k], cur[k]
        uni = len(a | b)
        jacs.append((len(a & b) / uni) if uni else 1.0)
        if a == b:
            same += 1
    j = np.asarray(jacs, dtype=float)
    return dict(
        n_overlap=len(common),
        exact_same=same,
        exact_same_pct=same / len(common),
        jac_mean=float(j.mean()),
        jac_p50=float(np.median(j)),
        jac_ge50=float((j >= 0.5).mean()),
    )


def store_churn(prev_map: pd.DataFrame, cur_map: pd.DataFrame, base_cell) -> float:
    """Share of overlapping stores whose co-member store set changed."""
    if prev_map.empty or cur_map.empty:
        return np.nan
    a = prev_map[["store_id", "ibd_id", *base_cell]].drop_duplicates("store_id")
    b = cur_map[["store_id", "ibd_id", *base_cell]].drop_duplicates("store_id")
    both = set(a["store_id"]) & set(b["store_id"])
    if not both:
        return np.nan
    members_a = a.groupby("ibd_id")["store_id"].apply(set).to_dict()
    members_b = b.groupby("ibd_id")["store_id"].apply(set).to_dict()
    ia = a.set_index("store_id")
    ib = b.set_index("store_id")
    changed = 0
    for sid in both:
        sa = members_a.get(ia.loc[sid, "ibd_id"], set()) - {sid}
        sb = members_b.get(ib.loc[sid, "ibd_id"], set()) - {sid}
        if sa != sb:
            changed += 1
    return changed / len(both)


def nearest_by_cell(profile: pd.DataFrame) -> dict:
    if profile.empty:
        return {}
    nn = {}
    cells = profile.index.droplevel(-1).unique() if isinstance(profile.index, pd.MultiIndex) else [()]
    for cell in cells:
        try:
            sub = profile.xs(cell, drop_level=True)
        except KeyError:
            continue
        if sub.empty or len(sub) < 2:
            continue
        d = mix_distance(sub)
        for p in sub.index:
            row = d.loc[p].drop(labels=[p], errors="ignore")
            row = row[np.isfinite(row)]
            if row.empty:
                continue
            nn[(cell if isinstance(cell, tuple) else (cell,), str(p))] = str(row.idxmin())
    return nn


def nn_flip(a: dict, b: dict) -> tuple[int, int]:
    keys = set(a) & set(b)
    if not keys:
        return 0, 0
    return sum(1 for k in keys if a[k] != b[k]), len(keys)


def ibd_summary(define, period: int, arm: str) -> dict:
    m = define.ibd_map
    s = define.summary
    cl = m.drop_duplicates([c for c in m.columns if c in ("ibd_id", "PLATFORMNAME", "SHOPTYPE", "member_provinces")])
    nprov = cl["member_provinces"].astype(str).str.split("+").str.len() if not cl.empty else pd.Series(dtype=float)
    return dict(
        arm=arm,
        period_id=period,
        n_ibd=int(m["ibd_id"].nunique()) if not m.empty else 0,
        n_province=int(len(m)),
        n_store=int(len(define.store_map.drop_duplicates("store_id"))) if not define.store_map.empty else 0,
        soft_kept=int(m["soft_kept"].astype(bool).sum()) if "soft_kept" in m else 0,
        accepted=int(s["accepted"].sum()) if not s.empty and "accepted" in s else 0,
        n_ge4_prov=int((nprov >= 4).sum()) if len(nprov) else 0,
        max_n_prov=int(nprov.max()) if len(nprov) else 0,
    )


def adjbox_flags(values: np.ndarray, gids: np.ndarray) -> np.ndarray:
    out = np.full(len(values), "ok", dtype=object)
    df = pd.DataFrame({"v": values, "g": gids})
    for g, sub in df.groupby("g", sort=False):
        q1, q3, mc = adjbox_stats(sub["v"].to_numpy())
        lo, hi = fence_from_stats(q1, q3, mc, A, B, C_FENCE)
        out[sub.index.to_numpy()] = flag(sub["v"].to_numpy(), lo, hi)
    return out


def anom(s) -> np.ndarray:
    return np.isin(np.asarray(s, dtype=object), [HIGH, LOW])


def jaccard_sets(a: set, b: set) -> float:
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def map_gid(frame: pd.DataFrame, ibd_map: pd.DataFrame, base_cell) -> pd.Series:
    m = ibd_map.drop_duplicates([*base_cell, "PROVINCE"])
    key = list(base_cell) + ["PROVINCE"]
    hit = frame.merge(m[key + ["ibd_id"]], on=key, how="left")
    cell = hit[list(base_cell)].astype(str).agg("|".join, axis=1)
    gid = cell + "||" + hit["ibd_id"].fillna("NA").astype(str) + "||" + hit["category"].astype(str)
    return gid


def detect_on_08(sc: pd.DataFrame, maps: dict, base_cell, period=20261408) -> list[dict]:
    cur = sc[sc["period_id"] == period].copy()
    cur = cur[cur["peer_ok"]].copy() if "peer_ok" in cur.columns else cur
    cur = cur.reset_index(drop=True)
    lv = np.log1p(cur["sales_value"].clip(lower=0).to_numpy(dtype=float))
    sh = cur["cat_share"].fillna(0).to_numpy(dtype=float)
    it = np.log1p(cur["n_item"].clip(lower=0).to_numpy(dtype=float)) if "n_item" in cur.columns else np.zeros(len(cur))
    rows = []
    for arm, by_p in maps.items():
        m08 = by_p[period].ibd_map
        m07 = by_p.get(20261407)
        g08 = map_gid(cur, m08, base_cell)
        fa = adjbox_flags(lv, g08.to_numpy())
        fs = adjbox_flags(sh, g08.to_numpy())
        fi = adjbox_flags(it, g08.to_numpy())
        n08 = anom(fa).astype(int) + anom(fs).astype(int) + anom(fi).astype(int)
        keys08 = set(zip(cur["store_id"], cur["category"], np.where(n08 >= 2, 1, 0)))
        sus08 = {(s, c) for s, c, k in keys08 if k}
        adj08 = set(zip(cur.loc[anom(fa), "store_id"], cur.loc[anom(fa), "category"]))
        rec = dict(arm=arm, n_adj_08=len(adj08), n_sus_08=len(sus08))
        if m07 is None:
            rec.update(adj_jaccard_late=np.nan, sus_jaccard_late=np.nan, sus_only_08=np.nan, sus_only_07map=np.nan, sus_both=np.nan)
        else:
            g07 = map_gid(cur, m07.ibd_map, base_cell)
            fa7 = adjbox_flags(lv, g07.to_numpy())
            fs7 = adjbox_flags(sh, g07.to_numpy())
            fi7 = adjbox_flags(it, g07.to_numpy())
            n07 = anom(fa7).astype(int) + anom(fs7).astype(int) + anom(fi7).astype(int)
            sus07 = set(zip(cur.loc[n07 >= 2, "store_id"], cur.loc[n07 >= 2, "category"]))
            adj07 = set(zip(cur.loc[anom(fa7), "store_id"], cur.loc[anom(fa7), "category"]))
            rec["adj_jaccard_late"] = jaccard_sets(adj08, adj07)
            rec["sus_jaccard_late"] = jaccard_sets(sus08, sus07)
            rec["sus_only_08"] = len(sus08 - sus07)
            rec["sus_only_07map"] = len(sus07 - sus08)
            rec["sus_both"] = len(sus08 & sus07)
            rec["n_adj_07map"] = len(adj07)
            rec["n_sus_07map"] = len(sus07)
        rows.append(rec)
    return rows


def load_sc(params, periods, quick: bool) -> pd.DataFrame:
    cache = ROOT / "data" / "backtest" / "_panel_cache.pkl"
    bmp = params.path("before_mp_dir", ROOT)
    if cache.exists() and not quick:
        sc = pd.read_pickle(cache)
        log(f"panel cache rows={len(sc)}")
        return sc[sc["period_id"].isin(periods)].copy()
    wanted = [p for p in list_before_mp_files(bmp) if p in periods]
    if quick:
        wanted = wanted[-2:]
    log(f"loading panel {wanted}")
    return load_panel(
        bmp, params.path("store_target_dir", ROOT), params.data.item_key,
        periods=wanted, encoding=params.data.encoding,
    ).sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--params", default=str(ROOT / "config" / "peer_params.json"))
    args = ap.parse_args()
    with Path(args.params).open(encoding="utf-8") as fh:
        raw = json.load(fh)
    params0 = load_params(args.params)
    periods = PERIODS[-2:] if args.quick else PERIODS
    sc = load_sc(params0, periods, args.quick)
    base_cell = params0.base_cell
    if args.arms:
        unknown = [a for a in args.arms if a not in ARMS]
        if unknown:
            raise SystemExit(f"unknown arms {unknown}; choose from {list(ARMS)}")
        arms = list(args.arms)
        if "W0" not in arms:
            arms = ["W0", *arms]
    else:
        arms = ["W0", "W2"] if args.quick else list(ARMS)

    defines: dict[str, dict[int, object]] = {}
    summaries, transitions, nn_rows = [], [], []
    nn_cache: dict[tuple, dict] = {}

    for arm in arms:
        p_arm = override_ibd(raw, **ARMS[arm])
        defines[arm] = {}
        prev_cl, prev_store, prev_nn = None, None, None
        prev_p = None
        for period in periods:
            log(f"{arm} define_ibd {period}")
            dfn = define_ibd(sc[sc["period_id"] <= period], period, base_cell, p_arm.ibd)
            defines[arm][period] = dfn
            summaries.append(ibd_summary(dfn, period, arm))
            cl = clusters(dfn.ibd_map, base_cell)
            prof = province_mix_profile(sc[sc["period_id"] <= period], base_cell, ibd=p_arm.ibd, period=period)
            nn = nearest_by_cell(prof)
            nn_cache[(arm, period)] = nn
            if arm != "W0" and ("W0", period) in nn_cache:
                flips0, nnn0 = nn_flip(nn_cache[("W0", period)], nn)
                nn_rows.append(dict(arm=arm, period_id=period, vs="W0", nn_flip=flips0, nn_n=nnn0,
                                    nn_flip_pct=(flips0 / nnn0) if nnn0 else np.nan))
            if prev_cl is not None:
                cmp_ = cluster_compare(prev_cl, cl)
                flips, nnn = nn_flip(prev_nn, nn)
                trans = dict(
                    arm=arm, from_period=prev_p, to_period=period,
                    member_churn=store_churn(prev_store, dfn.store_map, base_cell),
                    nn_flip=flips, nn_n=nnn, nn_flip_pct=(flips / nnn) if nnn else np.nan,
                    **cmp_,
                )
                transitions.append(trans)
                log(f"  {prev_p}->{period} jac_mean={cmp_['jac_mean']:.3f} exact={cmp_['exact_same_pct']:.1%} churn={trans['member_churn']:.3f}")
            prev_cl, prev_store, prev_nn, prev_p = cl, dfn.store_map, nn, period

    log("phase 2 adjbox on 08")
    detect = detect_on_08(sc, defines, base_cell) if 20261408 in periods else []

    # pick winners: 07->08 jac_mean vs W0
    trans_df = pd.DataFrame(transitions)
    pick = []
    if not trans_df.empty and trans_df["to_period"].eq(20261408).any():
        sub = trans_df[trans_df["to_period"] == 20261408]
        w0 = float(sub.loc[sub["arm"] == "W0", "jac_mean"].mean()) if (sub["arm"] == "W0").any() else np.nan
        ranked = sub.sort_values("jac_mean", ascending=False)
        pick = ranked["arm"].tolist()
        log(f"07->08 cluster jac W0={w0:.3f}; ranked {list(zip(ranked.arm, ranked.jac_mean.round(3)))}")

    verdict_rows = []
    if detect:
        det = pd.DataFrame(detect).set_index("arm")
        w0j = det.loc["W0", "sus_jaccard_late"] if "W0" in det.index else np.nan
        for arm in det.index:
            better = det.loc[arm, "sus_jaccard_late"] - w0j if np.isfinite(w0j) else np.nan
            verdict_rows.append({
                "arm": arm,
                "sus_jaccard_late": det.loc[arm, "sus_jaccard_late"],
                "delta_vs_W0": better,
                "sus_only_08": det.loc[arm, "sus_only_08"],
                "sus_only_07map": det.loc[arm, "sus_only_07map"],
            })

    dest = OUT_XLSX
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheets = {
        "summary": pd.DataFrame(summaries),
        "transitions": trans_df,
        "nn_vs_W0": pd.DataFrame(nn_rows),
        "detect_08": pd.DataFrame(detect),
        "verdict": pd.DataFrame(verdict_rows),
        "settings": pd.DataFrame(
            [{"arm": k, **{kk: str(vv) for kk, vv in v.items()}} for k, v in ARMS.items() if k in arms]
            + [{"arm": "_pick_07to08", "note": ",".join(pick[:3])}]
        ),
    }
    with pd.ExcelWriter(dest, engine="openpyxl") as w:
        for name, df in sheets.items():
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_excel(w, sheet_name=name[:31], index=False)
    log(f"wrote {dest}")


if __name__ == "__main__":
    main()
