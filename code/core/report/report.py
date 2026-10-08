"""Reporting: raw rows with issue flags, adjustment estimates, and summary statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.data_prep.data_prep import ATTR_COLS
from core.outlier_rules.evidence import AUTO_FIX, AUTO_FLAG, AUTO_SKIP, EVIDENCE_COLS, level_rank
from core.data_prep.quality import tag_quality
from funcs.robust import HIGH, LOW

SC_KEYS = ["period_id", "store_id", "category"]
DQ_DROP = ("dup_period_store_prod", "negative_value")
RAW_FLAG_COLS = [
    "dq_flags", "dq_dropped", "store_matched", *ATTR_COLS, "ibd_id", "peer_group_id", "peer_level",
    "level", "direction", "n_evidence", "evidence_list", "auto_action", "adjbox_value", "share_adjbox",
    "item_adjbox", "fova_final", "adj_to_fence", "adj_to_median", "store_drop", "drill_rank",
    "drill_contribution", "item_flag", "issue",
]
ADJ_COLUMNS = [
    "period_id", "store_id", "category", *ATTR_COLS, "peer_group_id", "level", "direction", "evidence_list",
    "sales_value", "fence", "peer_median_sales", "expected_fence", "expected_median", "adj_to_fence", "adj_to_median",
    "adj_pct_fence", "adj_pct_median",
]


# ---------------------------------------------------------------- adjustments
def adjustments(graded: pd.DataFrame, min_level: str) -> pd.DataFrame:
    """Estimated adjustment for flagged store x category rows (level >= min_level, direction high/low).

    expected_fence: winsorize to the adjbox fence of the peer set (0 adjustment when inside the fence).
    expected_median: peer sellers' median sales (upper bound of the adjustment).
    adjustment = expected - actual (negative for high, positive for low).
    """
    sellers = graded[graded["sales_value"] > 0]
    med = sellers.groupby(["peer_group_id", "category"])["sales_value"].median().rename("peer_median_sales")
    rows = graded[
        (graded["level"].map(level_rank) >= level_rank(min_level))
        & graded["direction"].isin([HIGH, LOW])
        & (graded["peer_level"] != "na")
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=ADJ_COLUMNS)
    rows = rows.join(med, on=["peer_group_id", "category"])
    high = (rows["direction"] == HIGH).to_numpy()
    lo, hi = (np.expm1(np.minimum(rows[c].to_numpy(dtype=float), 700.0)) for c in ("lv_lo", "lv_hi"))
    x = rows["sales_value"].to_numpy(dtype=float)
    fence = np.where(high, hi, np.maximum(lo, 0.0))
    exp_fence = np.where(high, np.minimum(x, hi), np.maximum(x, np.maximum(lo, 0.0)))
    exp_fence = np.where(np.isfinite(exp_fence), exp_fence, x)
    rows["fence"] = fence
    rows["expected_fence"] = exp_fence
    rows["expected_median"] = rows["peer_median_sales"].fillna(rows["sales_value"])
    rows["adj_to_fence"] = rows["expected_fence"] - rows["sales_value"]
    rows["adj_to_median"] = rows["expected_median"] - rows["sales_value"]
    with np.errstate(divide="ignore", invalid="ignore"):
        rows["adj_pct_fence"] = np.where(x > 0, rows["adj_to_fence"] / x, np.nan)
        rows["adj_pct_median"] = np.where(x > 0, rows["adj_to_median"] / x, np.nan)
    rows = rows.reindex(rows["adj_to_median"].abs().sort_values(ascending=False).index)
    return rows[ADJ_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------- summary
def _share(n, d):
    return float(n) / float(d) if d else np.nan


def _level_rule_note(name: str, high_min: int) -> str:
    """Human-readable n_evidence → level rule for the summary note column."""
    rules = "、".join(EVIDENCE_COLS)
    if name == "normal":
        return f"n_evidence=0（{rules} 均非 high/low）"
    if name == "watch":
        return "n_evidence=1（三项证据中恰好 1 项 high/low）"
    if name == "suspicious":
        if high_min <= 2:
            return f"n_evidence∈(1, {high_min})（当前 high_min={high_min} 时通常为空）"
        return f"n_evidence∈[2, {high_min})（两项证据异常，未达 high_min={high_min}）"
    if name == "high_confidence":
        return f"n_evidence≥{high_min}（evidence.high_min；四项证据中至少 {high_min} 项异常）"
    if name == "suspicious+":
        return f"level∈{{suspicious, high_confidence}}，即 n_evidence≥2（high_min={high_min} 时含全部≥2）"
    return ""


def summary_overview(graded, adj, missing, drops, dq, drill, raw_stats: dict, min_level: str,
                     unit=None, ibd_map=None, high_min: int = 3) -> pd.DataFrame:
    """One table: funnel counts, detection shares, level rules, and impact by issue type."""
    base = graded[graded["peer_level"] != "na"]
    n_rows, n_stores, sales = len(graded), graded["store_id"].nunique(), float(graded["sales_value"].sum())
    geo_cols = [c for c in ("PROVINCE", "SHOPTYPE", "PLATFORMNAME", "category") if c in graded.columns]
    n_geo = int(graded.drop_duplicates(geo_cols).shape[0]) if geo_cols else np.nan
    if ibd_map is not None and len(ibd_map) and "ibd_id" in ibd_map.columns:
        n_ibd = int(ibd_map["ibd_id"].nunique())
        n_soft = int(ibd_map["soft_kept"].sum()) if "soft_kept" in ibd_map.columns else np.nan
        n_forced = int((ibd_map["pooling_reason"] == "forced_nearest").sum()) if "pooling_reason" in ibd_map.columns else np.nan
    else:
        n_ibd = int(graded["ibd_id"].nunique()) if "ibd_id" in graded.columns else np.nan
        n_soft = n_forced = np.nan
    fova_flag = graded[graded["fova_final"].isin([HIGH, LOW])] if "fova_final" in graded.columns else graded.iloc[0:0]
    raw_rows = raw_stats.get("rows") if raw_stats else None
    funnel = [
        {"section": "监测漏斗", "item": "省×店型×平台×品类", "n": n_geo,
         "note": "peer_detail 上 PROVINCE×SHOPTYPE×PLATFORMNAME×category 去重"},
        {"section": "监测漏斗", "item": "IBD 数", "n": n_ibd,
         "note": (f"soft_kept 省行 {n_soft}; forced_nearest {n_forced}" if pd.notna(n_soft) else "ibd_map.ibd_id 去重")},
        {"section": "监测漏斗", "item": "门店×品类 行数", "n": n_rows, "note": "peer_detail 行（含补 0）"},
        {"section": "监测漏斗", "item": "可检测行 (peer_level≠na)", "n": len(base), "pct_rows": _share(len(base), n_rows)},
        {"section": "监测漏斗", "item": "fova_final 告警", "n": len(fova_flag),
         "pct_rows": _share(len(fova_flag), len(base) if len(base) else n_rows),
         "n_stores": int(fova_flag["store_id"].nunique()) if len(fova_flag) else 0,
         "sales": float(fova_flag["sales_value"].sum()) if len(fova_flag) else 0.0,
         "pct_sales": _share(float(fova_flag["sales_value"].sum()) if len(fova_flag) else 0.0, sales),
         "note": "fova_final∈{high,low}；由 longitude∪trend 合并，计入证据时只算 1 项"},
    ]
    if len(drill):
        n_drill_combo = int(drill[["store_id", "category"]].drop_duplicates().shape[0])
        n_new = int((drill["item_flag"] == "new_item").sum()) if "item_flag" in drill.columns else 0
        n_miss = int((drill["item_flag"] == "missing_item").sum()) if "item_flag" in drill.columns else 0
        drill_sales = float(drill["sales_value"].sum()) if "sales_value" in drill.columns else np.nan
        funnel.append({"section": "监测漏斗", "item": "下钻商品行", "n": len(drill),
                       "pct_rows": _share(len(drill), raw_rows) if raw_rows else np.nan,
                       "sales": drill_sales, "pct_sales": _share(drill_sales, sales) if pd.notna(drill_sales) else np.nan,
                       "note": f"覆盖店×品类 {n_drill_combo}; new_item {n_new}; missing_item {n_miss}"
                               + (f"; 占原始商品行 {len(drill) / raw_rows:.4%}" if raw_rows else "")})
    if unit is not None:
        corr0 = unit.corrections
        before0 = float(corr0["original_value"].sum()) if len(corr0) and "original_value" in corr0.columns else 0.0
        after0 = float(corr0["corrected_value"].sum()) if len(corr0) and "corrected_value" in corr0.columns else 0.0
        net0 = float(corr0["impact_value"].sum()) if len(corr0) and "impact_value" in corr0.columns else after0 - before0
        funnel.append({"section": "监测漏斗", "item": "修正商品", "n": len(corr0),
                       "pct_rows": _share(len(corr0), raw_rows) if raw_rows else _share(len(corr0), n_rows),
                       "n_stores": corr0["store_id"].nunique() if len(corr0) else 0,
                       "sales": before0, "pct_sales": _share(before0, sales),
                       "adj_to_median": net0, "pct_adj_median": _share(net0, sales),
                       "note": f"修正前 sales {before0:,.2f} → 修正后 {after0:,.2f}; 净影响 {net0:,.2f}"})
        if getattr(unit, "funnel", None) is not None and len(unit.funnel):
            for r in unit.funnel.itertuples(index=False):
                funnel.append({
                    "section": f"修正漏斗/{r.stage}", "item": r.reason, "n": r.n,
                    "n_stores": r.n_store_cat, "note": r.note,
                })
    rows = funnel + [
        {"section": "总体", "item": "门店×品类 行数", "n": n_rows},
        {"section": "总体", "item": "门店数", "n": n_stores},
        {"section": "总体", "item": "销售额", "sales": sales},
        {"section": "总体", "item": "可检测行 (peer_level≠na)", "n": len(base), "pct_rows": _share(len(base), n_rows)},
        {"section": "分级规则", "item": "计入证据的规则", "n": len(EVIDENCE_COLS),
         "note": "、".join(EVIDENCE_COLS) + "；单独的 longitude、trend 已并入 fova_final，不另计"},
        {"section": "分级规则", "item": "等级门槛",
         "note": f"0→normal; 1→watch; 2..{high_min - 1}→suspicious; ≥{high_min}→high_confidence"
                 f"（evidence.high_min={high_min}）"},
        {"section": "分级规则", "item": "方向 direction",
         "note": "取 high/low 较多一方；平局依次看 fova_final、adjbox_value、share_adjbox、item_adjbox，否则 mixed"},
        {"section": "分级规则", "item": "auto_action",
         "note": f"suspicious+ 且有方向 → {AUTO_FIX}（唯一允许 unit_fix 改数）；"
                 f"watch → {AUTO_FLAG}；normal/na → {AUTO_SKIP}"},
    ]
    if "auto_action" in graded.columns:
        for act, label in ((AUTO_FIX, "fix"), (AUTO_FLAG, "flag_only"), (AUTO_SKIP, "skip")):
            sub = graded[graded["auto_action"] == act]
            rows.append({"section": "自动动作", "item": label, "n": len(sub),
                         "pct_rows": _share(len(sub), n_rows),
                         "n_stores": int(sub["store_id"].nunique()) if len(sub) else 0,
                         "sales": float(sub["sales_value"].sum()) if len(sub) else 0.0,
                         "pct_sales": _share(float(sub["sales_value"].sum()) if len(sub) else 0.0, sales),
                         "note": "仅 fix 进入 unit_fix 写回" if act == AUTO_FIX else ""})
    levels = [("watch", ["watch"]), ("suspicious", ["suspicious"]), ("high_confidence", ["high_confidence"]),
              ("suspicious+", ["suspicious", "high_confidence"])]
    adj_by = adj.set_index(["store_id", "category"]) if not adj.empty else None
    for name, lv in levels:
        sub = base[base["level"].isin(lv)]
        r = {
            "section": "异常分级", "item": name, "n": len(sub), "pct_rows": _share(len(sub), len(base)),
            "n_stores": sub["store_id"].nunique(), "pct_stores": _share(sub["store_id"].nunique(), n_stores),
            "sales": float(sub["sales_value"].sum()), "pct_sales": _share(sub["sales_value"].sum(), sales),
            "n_high": int((sub["direction"] == HIGH).sum()), "n_low": int((sub["direction"] == LOW).sum()),
            "note": _level_rule_note(name, high_min),
        }
        if adj_by is not None and level_rank(lv[0]) >= level_rank(min_level):
            a = adj[adj["level"].isin(lv)]
            r.update(adj_to_fence=float(a["adj_to_fence"].sum()), pct_adj_fence=_share(a["adj_to_fence"].sum(), sales),
                     adj_to_median=float(a["adj_to_median"].sum()), pct_adj_median=_share(a["adj_to_median"].sum(), sales))
        rows.append(r)
    if not adj.empty:
        for d, label in ((HIGH, "偏高"), (LOW, "偏低")):
            a = adj[adj["direction"] == d]
            rows.append({"section": f"调整估算 (level≥{min_level})", "item": label, "n": len(a), "n_stores": a["store_id"].nunique(),
                         "sales": float(a["sales_value"].sum()), "pct_sales": _share(a["sales_value"].sum(), sales),
                         "adj_to_fence": float(a["adj_to_fence"].sum()), "pct_adj_fence": _share(a["adj_to_fence"].sum(), sales),
                         "adj_to_median": float(a["adj_to_median"].sum()), "pct_adj_median": _share(a["adj_to_median"].sum(), sales)})
        rows.append({"section": f"调整估算 (level≥{min_level})", "item": "净调整", "n": len(adj),
                     "adj_to_fence": float(adj["adj_to_fence"].sum()), "pct_adj_fence": _share(adj["adj_to_fence"].sum(), sales),
                     "adj_to_median": float(adj["adj_to_median"].sum()), "pct_adj_median": _share(adj["adj_to_median"].sum(), sales)})
    for lv in ("suspicious", "watch", None):
        m = missing if lv is None else missing[missing["level"] == lv]
        exp = float(m["peer_median_sales"].sum()) if len(m) else 0.0
        rows.append({"section": "该卖没卖", "item": lv or "合计", "n": len(m), "n_stores": m["store_id"].nunique(),
                     "pct_stores": _share(m["store_id"].nunique(), n_stores), "adj_to_median": exp, "pct_adj_median": _share(exp, sales)})
    if len(drops):
        prev, now = float(drops["prev_sales_value"].sum()), float(drops["sales_value"].sum())
        rows.append({"section": "整店下降", "item": "门店", "n_stores": len(drops), "pct_stores": _share(len(drops), n_stores),
                     "sales": now, "pct_sales": _share(now, sales), "adj_to_median": prev - now, "pct_adj_median": _share(prev - now, sales),
                     "note": f"上期销售额 {prev:,.0f}; 缺失品类 {int(drops['n_missing'].sum())} 个"})
    else:
        rows.append({"section": "整店下降", "item": "门店", "n_stores": 0})
    for key, label in (("rows", "原始行数"), ("dropped_rows", "剔除行 (重复/负值)"), ("dropped_sales", "剔除销售额"),
                       ("unmatched_rows", "未匹配门店行"), ("unmatched_sales", "未匹配门店销售额")):
        if key in raw_stats:
            v = raw_stats[key]
            r = {"section": "数据质量", "item": label}
            if "sales" in key:
                r.update(sales=v, pct_sales=_share(v, raw_stats.get("sales", 0)))
            else:
                r.update(n=v, pct_rows=_share(v, raw_stats.get("rows", 0)))
            rows.append(r)
    for r in dq[dq["check"].isin(["bmp_only_stores", "target_only_stores"])].itertuples():
        rows.append({"section": "数据质量", "item": r.check, "n_stores": int(r.n_stores)})
    if len(drill):
        n_drill_combo = int(drill[["store_id", "category"]].drop_duplicates().shape[0])
        n_new = int((drill["item_flag"] == "new_item").sum()) if "item_flag" in drill.columns else 0
        n_miss = int((drill["item_flag"] == "missing_item").sum()) if "item_flag" in drill.columns else 0
        rows.append({"section": "商品下钻", "item": "带出商品的组合", "n": n_drill_combo,
                     "note": f"商品 {len(drill)}; new_item {n_new}; missing_item {n_miss}"})
    if unit is not None:
        corr = unit.corrections
        recheck = unit.recheck
        net = float(corr["impact_value"].sum()) if len(corr) else 0.0
        vol = float(corr["impact_value"].abs().sum()) if len(corr) else 0.0
        resolved = int((recheck["resolved"] == True).sum()) if len(recheck) else 0
        before = float(corr["original_value"].sum()) if len(corr) and "original_value" in corr.columns else 0.0
        after = float(corr["corrected_value"].sum()) if len(corr) and "corrected_value" in corr.columns else 0.0
        rows.append({"section": "商品修正", "item": "修正商品", "n": len(corr),
                     "n_stores": corr["store_id"].nunique() if len(corr) else 0,
                     "sales": before, "pct_sales": _share(before, sales),
                     "adj_to_median": net, "pct_adj_median": _share(net, sales),
                     "note": f"修正前 {before:,.2f} → 修正后 {after:,.2f}; 总修正量 {vol:,.0f}; 复检 resolved {resolved}/{len(recheck)}"})
        rows.append({"section": "商品修正", "item": "整店普涨豁免",
                     "n_stores": len(unit.store_shift), "pct_stores": _share(len(unit.store_shift), n_stores)})
    cols = ["section", "item", "n", "pct_rows", "n_stores", "pct_stores", "sales", "pct_sales", "n_high", "n_low",
            "adj_to_fence", "pct_adj_fence", "adj_to_median", "pct_adj_median", "note"]
    return pd.DataFrame(rows).reindex(columns=cols)


def summary_by_dim(graded, adj, missing, drops, dims=None) -> pd.DataFrame:
    """Detection share and impact by platform, shoptype, platform x shoptype, province, category."""
    dims = dims or [["PLATFORMNAME"], ["SHOPTYPE"], ["PLATFORMNAME", "SHOPTYPE"], ["PROVINCE"], ["category"]]
    g = graded.assign(susp=graded["level"].isin(["suspicious", "high_confidence"]))
    g["susp_sales"] = np.where(g["susp"], g["sales_value"], 0.0)
    total_sales = float(g["sales_value"].sum())
    drop_stores = set(drops["store_id"]) if len(drops) else set()
    out = []
    for dim in dims:
        agg = g.groupby(dim).agg(
            n_rows=("store_id", "size"), n_stores=("store_id", "nunique"), sales=("sales_value", "sum"),
            n_watch=("level", lambda s: int((s == "watch").sum())), n_susp_plus=("susp", "sum"),
            susp_stores=("store_id", lambda s: s[g.loc[s.index, "susp"]].nunique()), susp_sales=("susp_sales", "sum"),
        )
        agg["pct_rows_susp"] = agg["n_susp_plus"] / agg["n_rows"]
        agg["pct_sales_susp"] = agg["susp_sales"] / agg["sales"]
        agg["pct_of_total_sales"] = agg["sales"] / total_sales
        if not adj.empty:
            a = adj.groupby(dim)[["adj_to_fence", "adj_to_median"]].sum()
            agg = agg.join(a)
            agg["pct_adj_fence"] = agg["adj_to_fence"] / agg["sales"]
            agg["pct_adj_median"] = agg["adj_to_median"] / agg["sales"]
        if len(missing) and set(dim) <= set(missing.columns):
            agg = agg.join(missing.groupby(dim).agg(n_missing=("store_id", "size"), missing_expected=("peer_median_sales", "sum")))
        stores = g[[*dim, "store_id"]].drop_duplicates()
        agg = agg.join(stores[stores["store_id"].isin(drop_stores)].groupby(dim).size().rename("n_store_drop"))
        agg = agg.reset_index()
        agg.insert(0, "dimension", "x".join(dim))
        agg.insert(1, "value", agg[dim].astype(str).agg(" | ".join, axis=1))
        out.append(agg.drop(columns=dim))
    res = pd.concat(out, ignore_index=True)
    fill = [c for c in ("n_missing", "missing_expected", "n_store_drop", "adj_to_fence", "adj_to_median") if c in res.columns]
    res[fill] = res[fill].fillna(0)
    return res


# ---------------------------------------------------------------- raw
def _join_flags(masks: dict, sep: str) -> np.ndarray:
    """Vectorized join of the names whose mask is True, per row."""
    n = len(next(iter(masks.values())))
    out = np.full(n, "", dtype=object)
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        out = np.where(mask & (out != ""), out + sep + name, np.where(mask, name, out))
    return out


def flag_raw(raw: pd.DataFrame, target: pd.DataFrame, graded: pd.DataFrame, adj: pd.DataFrame, drops: pd.DataFrame,
             drill: pd.DataFrame, item_key: str, min_level: str, corrections=None) -> tuple[pd.DataFrame, dict]:
    """All raw before_mp rows (original columns kept) with issue columns appended; returns (rows, stats)."""
    keys = pd.DataFrame({
        "period_id": pd.to_numeric(raw["period_id"], errors="coerce").astype("Int64"),
        "store_id": raw["store_id"].astype(str),
        "category": raw["category"].astype(str),
        "sales_value": pd.to_numeric(raw["sales_value"], errors="coerce"),
        "sales_unit": pd.to_numeric(raw["sales_unit"], errors="coerce"),
    }, index=raw.index)
    if "prod_id" in raw.columns:
        keys["prod_id"] = raw["prod_id"]
    if item_key in raw.columns:
        keys[item_key] = pd.to_numeric(raw[item_key], errors="coerce").astype("Int64")
    tags = tag_quality(keys)
    out = raw.copy()
    out["dq_flags"] = _join_flags({c: tags[c].to_numpy() for c in tags.columns}, ",")
    out["dq_dropped"] = tags[list(DQ_DROP)].any(axis=1).to_numpy()

    tgt = target.rename(columns={"PERIODCODE": "period_id", "STOREID": "store_id"})[["period_id", "store_id", *ATTR_COLS]]
    tgt["period_id"] = tgt["period_id"].astype("Int64")
    k = keys[["period_id", "store_id", "category"]].merge(tgt, on=["period_id", "store_id"], how="left")
    out["store_matched"] = k["PLATFORMNAME"].notna().to_numpy()
    for c in ATTR_COLS:
        out[c] = k[c].to_numpy()

    sc_cols = ["ibd_id", "peer_group_id", "peer_level", "level", "direction", "n_evidence", "evidence_list",
               "adjbox_value", "share_adjbox", "fova_final"]
    sc = graded[[*SC_KEYS, *sc_cols]].copy()
    sc["period_id"] = sc["period_id"].astype("Int64")
    if not adj.empty:
        a = adj[SC_KEYS + ["adj_to_fence", "adj_to_median"]].copy()
        a["period_id"] = a["period_id"].astype("Int64")
        sc = sc.merge(a, on=SC_KEYS, how="left")
    else:
        sc["adj_to_fence"] = np.nan
        sc["adj_to_median"] = np.nan
    j = k[SC_KEYS].merge(sc, on=SC_KEYS, how="left")
    for c in [*sc_cols, "adj_to_fence", "adj_to_median"]:
        out[c] = j[c].to_numpy()
    out["store_drop"] = keys["store_id"].isin(set(drops["store_id"]) if len(drops) else set()).to_numpy()

    if len(drill) and item_key in keys.columns:
        d = drill[drill["item_flag"] != "missing_item"][["store_id", "category", item_key, "rank", "contribution", "item_flag"]].copy()
        d[item_key] = pd.to_numeric(d[item_key], errors="coerce").astype("Int64")
        d = d.drop_duplicates(["store_id", "category", item_key])
        jd = keys[["store_id", "category", item_key]].merge(d, on=["store_id", "category", item_key], how="left")
        out["drill_rank"] = jd["rank"].to_numpy()
        out["drill_contribution"] = jd["contribution"].to_numpy()
        out["item_flag"] = jd["item_flag"].to_numpy()
    else:
        for c in ("drill_rank", "drill_contribution", "item_flag"):
            out[c] = np.nan

    for c in ("corrected_sales_value", "corrected_sales_unit", "correction_flag", "match_confidence"):
        out[c] = np.nan
    if corrections is not None and len(corrections) and item_key in keys.columns:
        corr = corrections[["store_id", "category", "item_key", "corrected_value", "corrected_unit",
                            "match_level", "confidence"]].copy()
        corr["item_key"] = pd.to_numeric(corr["item_key"], errors="coerce").astype("Int64")
        corr = corr.drop_duplicates(["store_id", "category", "item_key"]).rename(columns={
            "item_key": item_key, "corrected_value": "corrected_sales_value",
            "corrected_unit": "corrected_sales_unit", "match_level": "correction_flag",
            "confidence": "match_confidence"})
        jc = keys[["store_id", "category", item_key]].merge(
            corr, on=["store_id", "category", item_key], how="left")
        for c in ("corrected_sales_value", "corrected_sales_unit", "correction_flag", "match_confidence"):
            out[c] = jc[c].to_numpy()

    ranks = {lv: level_rank(lv) for lv in graded["level"].dropna().unique()}
    lv_ok = (out["level"].map(ranks).fillna(-1) >= level_rank(min_level)).to_numpy()
    out["issue"] = _join_flags({
        "数据质量剔除": out["dq_dropped"].to_numpy(),
        "门店未匹配": ~out["store_matched"].to_numpy(),
        "peer异常": lv_ok,
        "整店下降": out["store_drop"].to_numpy(),
    }, ";")

    sv = keys["sales_value"].fillna(0)
    stats = {
        "rows": len(out), "sales": float(sv.sum()),
        "dropped_rows": int(out["dq_dropped"].sum()), "dropped_sales": float(sv[out["dq_dropped"]].sum()),
        "unmatched_rows": int((~out["store_matched"]).sum()), "unmatched_sales": float(sv[~out["store_matched"]].sum()),
        "issue_rows": int((out["issue"] != "").sum()), "issue_sales": float(sv[out["issue"] != ""].sum()),
    }
    return out, stats


def write_csv_gz(df: pd.DataFrame, path) -> None:
    from pathlib import Path

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False, encoding="utf-8-sig", compression="gzip")
