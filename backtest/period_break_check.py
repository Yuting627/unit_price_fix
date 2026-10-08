"""判定实验：06->07 到底是「水平跳变（口径切换）」还是「构成变化（换样本）」？

只读，不改生产。这个问题决定 `fova.hist_start_period=20261407` 这个全局日期截断该不该保留：
- 构成变化 -> 店群整体换了人，但留下的店历史仍连续，历史可用，只要控住构成本身；
- 水平跳变 -> 同一批店自己在 06->07 之间发生了离散位移，`Median(各期中位数)` 与 `Max(各期 IQR)`
             双双把两段不同刻度的数据混在一起，历史才有毒。

三个口径：

A 同店固定集  取 06 和 07 都在卖同一 (store, category) 的集合，用 07 期的分组固定住，
              看这些店的水平序列 m01..m08。把 |m07-m06| 除以该店自己在其它相邻期上的典型跳幅，
              得到归一化跳变 z。z 的中位数 ~1 -> 只是构成变化；z 整体 >>1 -> 水平跳变。
B 全成员对照  每期用当期自己的 IBD 分组与成员（生产口径），算每个 (组, 品类) 的同店合计环比。
              若 06->07 的环比整体偏向一侧 -> 水平；若只是离散变大而中位数不动 -> 构成。
C 窗口影响    直接算 fova longitude 中心在「全窗口」「截断窗口 [>=hist_start]」「断点前」三种下的差，
              量化那一刀砍掉了多少信息，并复现 07/08 期历史期数不足 3 的死锁。

Usage:
    python backtest/period_break_check.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "code", ROOT / "backtest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core.config import load_params  # noqa: E402
from core.data_prep.data_prep import list_before_mp_files, load_panel  # noqa: E402
from core.ibd_define.ibd_check import check_ibd, peer_assignment  # noqa: E402
from core.ibd_define.ibd_define import define_ibd  # noqa: E402

PAIRS = [(20261402, 20261401), (20261403, 20261402), (20261404, 20261403),
         (20261405, 20261404), (20261406, 20261405), (20261407, 20261406),
         (20261408, 20261407)]
FOCUS = (20261406, 20261407)
BREAK_START = 20261407
FOCUS_PAIR = f"{FOCUS[0]}->{FOCUS[1]}"
MIN_N = 10
MIN_STORE_PER_PERIOD = 3
LAST_PERIOD = 20261408
HEAD = "sales_value"
KEYS = ["peer_group_id", "category"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def assign_for(sc, period, params, item_key, cache: dict) -> pd.DataFrame:
    """(store_id, category) -> peer_group_id for one period, production-faithful.

    check_level == "ibd" 时 ``need_pooling = low_store`` 恒为 False（define_ibd 保证 min_store），
    因此用空 items 调用 check_ibd 与生产结果完全一致。
    """
    if period in cache:
        return cache[period]
    sc_cur = sc[sc["period_id"] == period]
    define = define_ibd(sc, period, params.base_cell, params.ibd)
    empty = pd.DataFrame(columns=["period_id", "store_id", "category", item_key, "sales_value"])
    merge, _ = check_ibd(sc_cur, empty, define, item_key, params.ibd)
    a = peer_assignment(sc_cur, define.store_map, merge)
    a = a[a["peer_group_id"].notna()][["store_id", "category", "peer_group_id"]].drop_duplicates()
    cache[period] = a
    return a


def ratio_of(cur: pd.DataFrame, prev: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """同店合计环比 log r（每 key），附 n_common 与门店方向计数。"""
    cols = [*keys, "store_id", HEAD]
    m = cur[cols].merge(prev[cols], on=[*keys, "store_id"], suffixes=("_t", "_p"))
    m["up"] = m[f"{HEAD}_t"] > m[f"{HEAD}_p"]
    m["dn"] = m[f"{HEAD}_t"] < m[f"{HEAD}_p"]
    g = m.groupby(keys, sort=False).agg(s_t=(f"{HEAD}_t", "sum"), s_p=(f"{HEAD}_p", "sum"),
                                        n=("store_id", "size"), up=("up", "sum"), dn=("dn", "sum")).reset_index()
    ok = (g["s_p"] > 0) & (g["s_t"] > 0)
    g["logr"] = np.where(ok, np.log(g["s_t"] / g["s_p"]), np.nan)
    return g


def describe(delta: np.ndarray) -> dict:
    d = delta[np.isfinite(delta)]
    if d.size == 0:
        return {"n": 0}
    return {
        "n": int(d.size),
        "med_abs": float(np.median(np.abs(d))),
        "p90_abs": float(np.quantile(np.abs(d), 0.90)),
        "med_signed": float(np.median(d)),
        "mean_signed": float(np.mean(d)),
        "share_up": float(np.mean(d > 0)),
        "share_down": float(np.mean(d < 0)),
        "share_abs_gt0.1": float(np.mean(np.abs(d) > 0.1)),
        "share_abs_gt0.2": float(np.mean(np.abs(d) > 0.2)),
        "share_abs_gt0.5": float(np.mean(np.abs(d) > 0.5)),
    }


def main() -> None:
    params = load_params(ROOT / "config" / "peer_params.json")
    bmp = params.path("before_mp_dir", ROOT)
    item_key = params.data.item_key

    cache_pkl = ROOT / "data" / "backtest" / "_panel_cache.pkl"
    if cache_pkl.exists():
        sc = pd.read_pickle(cache_pkl)
        log(f"panel from cache rows={len(sc)}")
    else:
        wanted = [p for p in list_before_mp_files(bmp) if p <= LAST_PERIOD]
        log(f"loading panel periods={wanted}")
        sc = load_panel(bmp, params.path("store_target_dir", ROOT), item_key,
                        periods=wanted, encoding=params.data.encoding).sc
        cache_pkl.parent.mkdir(parents=True, exist_ok=True)
        sc.to_pickle(cache_pkl)
        log(f"sc rows={len(sc)}")

    pos = sc.loc[sc[HEAD] > 0, ["period_id", "store_id", "category", HEAD]].copy()
    pos["lv"] = np.log1p(pos[HEAD].to_numpy(dtype=float))
    periods = sorted(pos["period_id"].unique())
    log(f"periods={periods} seller_rows={len(pos)}")

    # ------------------------------------------------ 每期分组
    a_cache: dict[int, pd.DataFrame] = {}
    for p in sorted({p for pair in PAIRS for p in pair}):
        if p in periods:
            a_cache[p] = assign_for(sc, p, params, item_key, a_cache)
            log(f"  assigned {p}: groups={a_cache[p]['peer_group_id'].nunique()} rows={len(a_cache[p])}")

    # ------------------------------------------------ A0 同一篮子 vs 全量（构成效应的直接证据）
    basket = []
    all_pairs = {p: set(zip(pos.loc[pos["period_id"] == p, "store_id"],
                            pos.loc[pos["period_id"] == p, "category"])) for p in periods}
    s67 = all_pairs[FOCUS[0]] & all_pairs[FOCUS[1]]
    s_all8 = set.intersection(*[all_pairs[p] for p in periods])
    for p in periods:
        a = pos[pos["period_id"] == p]
        m67 = a[[(s, c) in s67 for s, c in zip(a["store_id"], a["category"])]]
        m8 = a[[(s, c) in s_all8 for s, c in zip(a["store_id"], a["category"])]]
        basket.append({
            "period": int(p),
            "n_row_all": len(a), "n_store_all": a["store_id"].nunique(), "med_all": float(a["lv"].median()),
            "n_row_s67": len(m67), "n_store_s67": m67["store_id"].nunique(),
            "med_s67": float(m67["lv"].median()) if len(m67) else np.nan,
            "n_row_s_all8": len(m8), "n_store_s_all8": m8["store_id"].nunique(),
            "med_s_all8": float(m8["lv"].median()) if len(m8) else np.nan,
        })
    df_basket = pd.DataFrame(basket)
    df_basket["d_all"] = df_basket["med_all"].diff()
    df_basket["d_s67"] = df_basket["med_s67"].diff()
    df_basket["d_s_all8"] = df_basket["med_s_all8"].diff()

    # ------------------------------------------------ A 同店固定集
    t0, t1 = FOCUS
    p0 = pos.loc[pos["period_id"] == t0, ["store_id", "category"]].drop_duplicates()
    p1 = pos.loc[pos["period_id"] == t1, ["store_id", "category"]].drop_duplicates()
    fixed = p0.merge(p1, on=["store_id", "category"], how="inner")
    fixed = fixed.merge(a_cache[t1], on=["store_id", "category"], how="inner").dropna(subset=["peer_group_id"])
    log(f"A fixed set: {len(fixed)} (store,cat) pairs in {fixed['peer_group_id'].nunique()} groups "
        f"out of {len(p1)} sold in {t1} and {len(p0)} sold in {t0}")

    sub = pos.merge(fixed, on=["store_id", "category"], how="inner")
    log(f"A rows kept across periods: {len(sub)}")

    med = sub.groupby([*KEYS, "period_id"], sort=False)["lv"].median().unstack("period_id")
    cnt = sub.groupby([*KEYS, "period_id"], sort=False)["store_id"].size().unstack("period_id")
    q1 = sub.groupby([*KEYS, "period_id"], sort=False)["lv"].quantile(0.25).unstack("period_id")
    q3 = sub.groupby([*KEYS, "period_id"], sort=False)["lv"].quantile(0.75).unstack("period_id")
    iqr = q3 - q1

    cols = [c for c in sorted(med.columns)]
    big = cnt.reindex(columns=cols).min(axis=1).fillna(0) >= MIN_STORE_PER_PERIOD
    med_b, iqr_b = med.loc[big, cols], iqr.loc[big, cols]
    log(f"A groups with >={MIN_STORE_PER_PERIOD} stores each period: {len(med_b)} of {len(med)}")

    rows_a, rows_norm, rows_iqr = [], [], []
    norm_z = {}
    for (p_1, p_0) in PAIRS:
        if p_1 not in med_b.columns or p_0 not in med_b.columns:
            continue
        d = (med_b[p_1] - med_b[p_0]).to_numpy(dtype=float)
        di = (iqr_b[p_1] - iqr_b[p_0]).to_numpy(dtype=float)
        rows_a.append({"pair": f"{p_0}->{p_1}", **describe(d)})
        rows_iqr.append({"pair": f"{p_0}->{p_1}", **describe(di)})
        norm_z[(p_0, p_1)] = np.abs(d)

    # 归一化：|Δ(该期)| / 该店在其它相邻期上的中位跳幅
    for (p_1, p_0) in PAIRS:
        if (p_0, p_1) not in norm_z:
            continue
        others = [v for k, v in norm_z.items() if k != (p_0, p_1)]
        base = np.nanmedian(np.vstack(others), axis=0)
        z = norm_z[(p_0, p_1)] / np.where(base > 1e-6, base, np.nan)
        zc = z[np.isfinite(z)]
        rows_norm.append({"pair": f"{p_0}->{p_1}", "n": int(zc.size),
                          "med_z": float(np.median(zc)),
                          "p75_z": float(np.quantile(zc, 0.75)),
                          "p90_z": float(np.quantile(zc, 0.90)),
                          "share_z_gt2": float(np.mean(zc > 2)),
                          "share_z_gt3": float(np.mean(zc > 3)),
                          "share_z_lt1": float(np.mean(zc < 1))})
    df_a, df_norm, df_iqr = pd.DataFrame(rows_a), pd.DataFrame(rows_norm), pd.DataFrame(rows_iqr)

    # ------------------------------------------------ A6 门店级配对（彻底剔除构成效应）
    fs = fixed[["store_id", "category"]]
    rows_pair, rows_pair_all = [], []
    pair_detail = {}
    for (p_1, p_0) in PAIRS:
        cur = pos.loc[pos["period_id"] == p_1, ["store_id", "category", "lv"]]
        prv = pos.loc[pos["period_id"] == p_0, ["store_id", "category", "lv"]]
        m = cur.merge(prv, on=["store_id", "category"], suffixes=("_t", "_p"))
        for tag, mm, out in (("fixed", m.merge(fs, on=["store_id", "category"], how="inner"), rows_pair),
                             ("all", m, rows_pair_all)):
            d = (mm["lv_t"] - mm["lv_p"]).to_numpy(dtype=float)
            s = d[np.isfinite(d)]
            n = s.size
            se = (1.2533 * np.std(s, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            out.append({"pair": f"{p_0}->{p_1}", "scope": tag, "n_stores": n,
                        "median": float(np.median(s)), "mean": float(np.mean(s)),
                        "se_median": float(se), "t_like": float(np.median(s) / se) if se else np.nan,
                        "q10": float(np.quantile(s, 0.10)), "q25": float(np.quantile(s, 0.25)),
                        "q75": float(np.quantile(s, 0.75)), "q90": float(np.quantile(s, 0.90)),
                        "share_up": float(np.mean(s > 0)), "share_down": float(np.mean(s < 0)),
                        "share_abs_gt0.2": float(np.mean(np.abs(s) > 0.2)),
                        "share_abs_gt0.5": float(np.mean(np.abs(s) > 0.5))})
            if tag == "fixed":
                pair_detail[f"{p_0}->{p_1}"] = pd.Series(
                    d, index=pd.MultiIndex.from_frame(mm[["store_id", "category"]])
                )
    df_pair = pd.DataFrame(rows_pair)
    df_pair_all = pd.DataFrame(rows_pair_all)

    # 归一化：门店级 |Δ| 相对其它相邻期的中位 |Δ|（门店集合不同，按 (store,cat) 对齐）
    mat = pd.DataFrame({k: s.abs() for k, s in pair_detail.items()})
    rows_pair_norm = []
    for k in mat.columns:
        base = mat.drop(columns=k).median(axis=1).to_numpy(dtype=float)
        z = mat[k].to_numpy(dtype=float) / np.where(base > 1e-6, base, np.nan)
        zc = z[np.isfinite(z)]
        rows_pair_norm.append({"pair": k, "n_store_cat": int(zc.size), "med_z": float(np.median(zc)),
                               "share_z_gt2": float(np.mean(zc > 2)), "share_z_gt3": float(np.mean(zc > 3))})
    df_pair_norm = pd.DataFrame(rows_pair_norm)

    # ------------------------------------------------ B 全成员对照
    rows_b, rows_dir = [], []
    for (p_1, p_0) in PAIRS:
        if p_1 not in a_cache or p_0 not in a_cache:
            continue
        cur = pos.loc[pos["period_id"] == p_1].merge(a_cache[p_1], on=["store_id", "category"], how="inner")
        prv = pos.loc[pos["period_id"] == p_0].merge(a_cache[p_0], on=["store_id", "category"], how="inner")
        r = ratio_of(cur, prv, KEYS)
        r = r[r["n"] >= MIN_N]
        rows_b.append({"pair": f"{p_0}->{p_1}", "n_groups": int(len(r)),
                       **describe(r["logr"].to_numpy(dtype=float))})
        tot = r["up"].sum() + r["dn"].sum()
        rows_dir.append({"pair": f"{p_0}->{p_1}", "n_groups": int(len(r)),
                         "share_groups_up": float(np.mean(r["logr"] > 0)),
                         "share_stores_up": float(r["up"].sum() / max(tot, 1)),
                         "med_n_common": float(r["n"].median())})
    df_b, df_dir = pd.DataFrame(rows_b), pd.DataFrame(rows_dir)

    # ------------------------------------------------ C 窗口影响
    last_ph = 20261401
    all_hist = [p for p in cols if last_ph <= p < LAST_PERIOD]
    pre_hist = [p for p in all_hist if p < BREAK_START]
    tr_hist = [p for p in all_hist if p >= BREAK_START]

    def center(hist_periods: list[int]) -> pd.Series:
        h = sub[sub["period_id"].isin(hist_periods)]
        if h.empty:
            return pd.Series(dtype=float)
        return (h.groupby([*KEYS, "period_id"], sort=False)["lv"].median()
                .groupby(level=list(range(len(KEYS)))).median())

    impl = pd.concat([center(all_hist).rename("center_all"),
                      center(tr_hist).rename("center_trunc"),
                      center(pre_hist).rename("center_pre")], axis=1).reset_index()
    impl["n_periods_all"] = len(all_hist)
    impl["n_periods_trunc"] = len(tr_hist)
    impl["d_trunc_vs_all"] = impl["center_trunc"] - impl["center_all"]
    impl["d_pre_vs_trunc"] = impl["center_pre"] - impl["center_trunc"]

    avail = (sub[sub["period_id"] >= BREAK_START]
             .groupby([*KEYS, "period_id"])["store_id"].size().reset_index()
             .groupby(KEYS)["period_id"].nunique().rename("n_periods_in_trunc_window").reset_index())
    avail["passes_min_hist"] = avail["n_periods_in_trunc_window"] >= params.fova.min_hist_periods

    # ---- C3/C4 用【当期自己的分组 + 回溯全部历史】（=生产 peer_panel 的口径）看历史可用性
    gm = pos.merge(a_cache[LAST_PERIOD], on=["store_id", "category"], how="inner")   # 08 期成员集回溯到各期
    per_cnt = (gm[gm["period_id"] < LAST_PERIOD]
               .groupby([*KEYS, "period_id"])["store_id"].size().unstack("period_id").fillna(0))
    avail2 = pd.DataFrame({
        "n_periods_any": (per_cnt > 0).sum(axis=1),
        "n_periods_ge5": (per_cnt >= 5).sum(axis=1),
        "n_periods_ge10": (per_cnt >= MIN_N).sum(axis=1),
    }).reset_index()
    for col, name in (("n_periods_any", "any"), ("n_periods_ge5", "ge5"), ("n_periods_ge10", "ge10")):
        avail2[f"{name}_passes_min_hist"] = avail2[col] >= params.fova.min_hist_periods
        avail2[f"{name}_passes_recommended"] = avail2[col] >= params.fova.recommended_hist_periods

    med08 = gm.groupby([*KEYS, "period_id"], sort=False)["lv"].median().unstack("period_id")
    med08 = med08.reindex(columns=cols)
    big08 = per_cnt.reindex(columns=[c for c in cols if c < LAST_PERIOD]).min(axis=1).fillna(0) >= MIN_STORE_PER_PERIOD
    rows_norm08 = []
    dz = {}
    for (p_1, p_0) in PAIRS:
        if p_1 not in med08.columns or p_0 not in med08.columns:
            continue
        dz[(p_0, p_1)] = np.abs(med08[p_1] - med08[p_0]).to_numpy(dtype=float)
    for k, v in dz.items():
        others = [x for kk, x in dz.items() if kk != k]
        base = np.nanmedian(np.vstack(others), axis=0)
        z = v / np.where(base > 1e-6, base, np.nan)
        zc = z[np.isfinite(z)]
        rows_norm08.append({"pair": f"{k[0]}->{k[1]}", "n_groups": int(zc.size),
                            "med_z": float(np.median(zc)), "p90_z": float(np.quantile(zc, 0.90)),
                            "share_z_gt2": float(np.mean(zc > 2)), "share_z_gt3": float(np.mean(zc > 3))})
    df_norm08 = pd.DataFrame(rows_norm08)
    df_med08 = med08.reset_index()

    # ------------------------------------------------ 输出
    dest = ROOT / "data" / "backtest" / "period_break_check.xlsx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    settings = pd.DataFrame([
        ("focus_pair", FOCUS_PAIR),
        ("pairs", ";".join(f"{a}->{b}" for b, a in PAIRS)),
        ("A_fixed_set", "06&07 都在卖的 (store,cat)，按 07 期分组固定；看相邻期组中位数 log1p 变化"),
        ("A_norm", "z = |Δ(该期)| / 该店在其他相邻期上的中位跳幅；z~1 说明这期是普通期过渡"),
        ("B_all_member", "每期用当期自己的 IBD 分组与成员，同店合计环比 log r（生产口径）"),
        ("C_window", "fova longitude 中心：全窗口 / 截断[>=hist_start] / 断点前"),
        ("min_n_groups_B", MIN_N),
        ("min_store_per_period_A", MIN_STORE_PER_PERIOD),
        ("fova.hist_start_period", params.fova.hist_start_period),
        ("fova.min_hist_periods", params.fova.min_hist_periods),
        ("base_cell", ",".join(params.base_cell)),
        ("ibd.method", params.ibd.method),
        ("ibd.distance", params.ibd.distance),
        ("ibd.check_level", params.ibd.check_level),
        ("n_store_cat_pairs_fixed", len(fixed)),
        ("n_groups_fixed_a_used", len(med_b)),
        ("n_rows_kept_fixed", len(sub)),
    ], columns=["key", "value"])
    with pd.ExcelWriter(dest, engine="openpyxl") as w:
        settings.to_excel(w, sheet_name="settings", index=False)
        df_basket.to_excel(w, sheet_name="A0_basket_level", index=False)
        df_a.to_excel(w, sheet_name="A1_pairs", index=False)
        df_norm.to_excel(w, sheet_name="A2_normalized_jump", index=False)
        df_iqr.to_excel(w, sheet_name="A3_iqr_pairs", index=False)
        med_b.reset_index().to_excel(w, sheet_name="A4_group_median_series", index=False)
        cnt.loc[big, cols].reset_index().to_excel(w, sheet_name="A5_group_counts", index=False)
        df_pair.to_excel(w, sheet_name="A6_store_paired_fixed", index=False)
        df_pair_all.to_excel(w, sheet_name="A6b_store_paired_all", index=False)
        df_pair_norm.to_excel(w, sheet_name="A7_store_paired_normz", index=False)
        df_b.to_excel(w, sheet_name="B1_pairs", index=False)
        df_dir.to_excel(w, sheet_name="B2_direction", index=False)
        impl.to_excel(w, sheet_name="C1_window_impact", index=False)
        avail.to_excel(w, sheet_name="C2_periods_available", index=False)
        avail2.to_excel(w, sheet_name="C3_hist_usable_08groups", index=False)
        df_med08.to_excel(w, sheet_name="C4_08group_median_series", index=False)
        df_norm08.to_excel(w, sheet_name="C5_08group_normz", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("\n===== A0 同一篮子 vs 全量：中位数 log1p(sales_value) =====")
    print(df_basket[["period", "n_store_all", "n_store_s67", "med_all", "med_s67",
                     "d_all", "d_s67", "d_s_all8"]].to_string(index=False))
    print("S67 pairs=%d stores=%d | S_all8 pairs=%d stores=%d"
          % (len(s67), len({s for s, c in s67}), len(s_all8), len({s for s, c in s_all8})))
    print("\n===== A1 同店固定集：组中位数 log1p 的相邻期变化 =====")
    print(df_a.to_string(index=False))
    print("\n===== A2 归一化跳变 z（06->07 如果只是构成变化，med_z 应接近 1）=====")
    print(df_norm.to_string(index=False))
    print("\n===== A3 同店固定集：组 IQR 的相邻期变化 =====")
    print(df_iqr.to_string(index=False))
    print("\n===== B1 全成员（每期自己的分组/成员）：同店合计环比 log r =====")
    print(df_b.to_string(index=False))
    print("\n===== B2 方向一致性 =====")
    print(df_dir.to_string(index=False))
    print("\n===== C1 08 期 longitude 中心：截断 vs 全窗口 =====")
    dd = impl["d_trunc_vs_all"].dropna()
    print(dd.describe().to_string())
    print(f"share |d_trunc_vs_all| > 0.10: {np.mean(np.abs(dd) > 0.10):.3f}")
    print(f"share |d_trunc_vs_all| > 0.20: {np.mean(np.abs(dd) > 0.20):.3f}")
    print("\n===== C2 截断窗口内每组的可用历史期数 =====")
    print(avail["n_periods_in_trunc_window"].value_counts().sort_index().to_string())
    print(f"passes_min_hist share: {avail['passes_min_hist'].mean():.3f}")

    print("\n===== A6 门店级配对（固定集，彻底无构成效应）：log(sales_t/sales_p) =====")
    print(df_pair[df_pair["scope"] == "fixed"].drop(columns="scope").to_string(index=False))
    print("\n===== A7 门店级归一化跳变 z =====")
    print(df_pair_norm.to_string(index=False))

    print("\n===== C3 08 期分组回溯全历史：每组可用历史期数（不是截断窗口）=====")
    for col in ("n_periods_any", "n_periods_ge5", "n_periods_ge10"):
        s = avail2[col]
        name = col.replace("n_periods_", "")
        print(f"{col}: median={s.median():.0f} mean={s.mean():.2f} "
              f"share>=3={float((s >= 3).mean()):.3f} share>=13={float((s >= 13).mean()):.3f}")
    print(f"n_groups={len(avail2)}")

    print("\n===== C5 08 期分组的归一化跳变 z（应看 07->08 是否特殊）=====")
    print(df_norm08.to_string(index=False))
    log(f"wrote {dest}")


if __name__ == "__main__":
    main()
