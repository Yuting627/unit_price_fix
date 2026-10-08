"""Item-level correction orchestrator (unit_fix).

Runs after detection + drilldown. For each anomalous store x category it:

* **HIGH** — right-tail Fuller detection + winsorize-gap correction on a cross-section of
  store-size-normalized sales rates (``item sales_value / store total sales``, an ACV proxy).
* **LOW** — three-way triage: missing sales (zero-filled), under-selling (non-zero but below the
  seller-only left fence), or whole-store drop.
* **store_shift** — a store whose selling categories are systematically HIGH (>= ``store_shift_ratio``)
  is exempted from item-level carving and routed to a whole-store shift table.
* **recheck** — re-test the corrected store x category totals against clean peers (excluding this
  round's flagged stores) to confirm the point is no longer an outlier.
* **impact** — per-item original / corrected / correction / confidence plus a net-impact summary.

The matching of the cross-store peer set uses a fallback chain (``nankey x brand x packsize`` ->
``nankey x brand`` -> ``nankey`` -> ``brand x category x packsize`` -> category -> own_prev -> tag only)
because ``prod_id`` is unique per store x month and cannot serve as a cross-store key.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.data_prep.data_prep import ATTR_COLS
from core.outlier_rules.evidence import FIX_MIN_LEVEL, level_rank
from core.outlier_rules.fuller import fuller_right_tail
from funcs.parse import packsize_volume, parse_packsize, split_brand
from funcs.robust import HIGH, LOW, adjbox_stats, fence_from_stats

# match levels, strongest to weakest
LV_NB_PK = "nankey_brand_packsize"
LV_NB = "nankey_brand"
LV_NK = "nankey"
LV_BC_PK = "brand_category_packsize"
LV_CAT = "category"
LV_PREV = "own_prev"
LV_TAG = "tag_only"

_LEVEL_CONFIDENCE = {
    LV_NB_PK: 1.0, LV_NB: 0.9, LV_NK: 0.8, LV_BC_PK: 0.6, LV_CAT: 0.4, LV_PREV: 0.5, LV_TAG: 0.0,
}

CORRECTION_COLS = [
    "period_id", "store_id", "category", "item_key", "prod_desc_raw", "brand", "packsize",
    "direction", "match_level", "n_peer", "confidence", "original_value", "original_unit",
    "corrected_value", "corrected_unit", "impact_value", "impact_unit", "score",
]

STORE_SHIFT_COLS = [
    "period_id", "store_id", "PLATFORMNAME", "SHOPTYPE", "PROVINCE", "ibd_id",
    "n_cat", "n_high", "n_low", "high_ratio", "sales_value", "direction",
]

RECHECK_COLS = [
    "period_id", "store_id", "category", "original_value", "corrected_value", "n_peer",
    "lv_lo", "lv_hi", "resolved", "recheck_fail",
]

IMPACT_COLS = ["dimension", "level", "n_items", "net_impact", "total_volume"]
FUNNEL_COLS = ["stage", "reason", "n", "n_store_cat", "note"]

# Human-readable notes for every skip / pass reason in the fix funnel.
FUNNEL_NOTES = {
    "suspicious_plus": "level≥suspicious 且 direction∈{high,low}（auto_action=fix 候选）",
    "fix_action": "auto_action=fix 的店×品类",
    "flag_only": "watch：只告警不改数",
    "drill_with_items": "下钻至少带出 1 个商品的店×品类",
    "drill_no_delta": "suspicious+ 但无可归因商品（delta 过滤后为空）",
    "drill_item_rows": "下钻商品行数",
    "skip_not_fix_action": "下钻行所属店×品类不是 auto_action=fix",
    "skip_store_shift": "整店普涨豁免（store_shift）",
    "skip_no_peer_stores": "同组无其他 peer 门店",
    "skip_match_empty": "HIGH：peer 匹配落到 tag_only 或无 peer 商品",
    "skip_peer_rate_lt_min": "HIGH：有效 rate 样本 < min_peer_stores",
    "skip_fuller_not_flagged": "HIGH：Fuller 右尾未把该店标为异常",
    "skip_store_drop": "LOW：整店下降门店，不按品类改数",
    "skip_missing_sales_warn": "该卖没卖/本期为0：只告警，禁止商品级修正",
    "skip_missing_upstream": "LOW：缺货/零销，由 missing_sales 告警（不改数）",
    "skip_zero_sales": "LOW：销售额≤0，跳过（不改数）",
    "skip_peer_sales_lt_min": "LOW：同组卖家样本 < min_peer_stores",
    "skip_above_left_fence": "LOW：未低于卖家左界，不改",
    "corrected": "写出 unit_fix 修正的商品行",
    "corrected_store_cat": "至少修正 1 个商品的店×品类",
    "store_shift_stores": "整店普涨豁免门店数",
    "recheck_rows": "复检店×品类行",
    "recheck_resolved": "复检仍落在同行界内",
    "recheck_fail": "复检仍出界",
}


@dataclass
class UnitFixResult:
    corrections: pd.DataFrame
    store_shift: pd.DataFrame
    recheck: pd.DataFrame
    impact: pd.DataFrame
    funnel: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=FUNNEL_COLS))


# ---------------------------------------------------------------- helpers
def _brand_key(brand) -> str:
    """Middle (brand) level of the hierarchical brand string, falling back to the manufacturer."""
    mfg, b, _ = split_brand(brand)
    return b or mfg


def _volume(packsize) -> float:
    return packsize_volume(parse_packsize(packsize))


def _same_packsize(va, vb, tol: float) -> bool:
    if not (math.isfinite(va) and math.isfinite(vb)):
        return False
    m = max(abs(va), abs(vb))
    return m <= 0 or abs(va - vb) / m <= tol


def store_totals(items_cur: pd.DataFrame) -> pd.Series:
    """Store total sales (ACV proxy) per store for the current period."""
    return items_cur.groupby("store_id")["sales_value"].sum().rename("store_total")


def attach_rate(items: pd.DataFrame, totals: pd.Series) -> pd.DataFrame:
    """Add ``store_total`` and store-size-normalized ``rate`` = sales_value / store_total."""
    out = items.copy()
    out["store_total"] = out["store_id"].map(totals)
    out["rate"] = np.where(out["store_total"].fillna(0) > 0, out["sales_value"] / out["store_total"], np.nan)
    return out


def match_peers(target: pd.Series, candidates: pd.DataFrame, params) -> tuple[pd.DataFrame, str]:
    """Return (peer rows, match level) for ``target`` using the fallback chain.

    ``target`` needs ``nankey``, ``brand``, ``packsize``, ``category``, ``store_id``;
    ``candidates`` is the item-level frame of other stores already restricted to the peer group.
    """
    uf = params.unit_fix
    tol = uf.packsize_tolerance
    t_key = target.get("nankey")
    t_brand = _brand_key(target.get("brand", ""))
    t_vol = _volume(target.get("packsize", ""))
    t_cat = target.get("category")

    if candidates.empty:
        return candidates.copy(), LV_TAG

    cand = candidates.copy()
    cand["_brand"] = cand["brand"].map(_brand_key) if "brand" in cand.columns else ""
    cand["_vol"] = cand["packsize"].map(_volume) if "packsize" in cand.columns else np.nan

    def take(mask, level):
        return cand[mask].drop(columns=["_brand", "_vol"], errors="ignore"), level

    # 1. nankey x brand x packsize
    mask = (cand["nankey"] == t_key) & (cand["_brand"] == t_brand)
    mask &= cand["_vol"].map(lambda v: _same_packsize(v, t_vol, tol))
    if mask.any():
        return take(mask, LV_NB_PK)

    # 2. nankey x brand
    mask = (cand["nankey"] == t_key) & (cand["_brand"] == t_brand)
    if mask.any():
        return take(mask, LV_NB)

    # 3. nankey
    mask = cand["nankey"] == t_key
    if mask.any():
        return take(mask, LV_NK)

    # 4. brand x category x packsize
    mask = (cand["_brand"] == t_brand) & (cand["category"] == t_cat)
    mask &= cand["_vol"].map(lambda v: _same_packsize(v, t_vol, tol))
    if mask.any():
        return take(mask, LV_BC_PK)

    # 5. category only
    mask = cand["category"] == t_cat
    if mask.any():
        return take(mask, LV_CAT)

    return cand.iloc[0:0].drop(columns=["_brand", "_vol"], errors="ignore"), LV_TAG


def _peer_rates(target: pd.Series, peer_items: pd.DataFrame, totals: pd.Series) -> pd.DataFrame:
    """One rate per peer store (summing multiple matched items), plus the target store's rate."""
    if peer_items.empty:
        return pd.DataFrame(columns=["store_id", "rate"])
    peers = peer_items.groupby("store_id", as_index=False).agg(sales_value=("sales_value", "sum"))
    peers["store_total"] = peers["store_id"].map(totals)
    peers["rate"] = np.where(peers["store_total"].fillna(0) > 0, peers["sales_value"] / peers["store_total"], np.nan)
    t_total = totals.get(target["store_id"], np.nan)
    t_rate = target["sales_value"] / t_total if (np.isfinite(t_total) and t_total > 0) else np.nan
    target_row = pd.DataFrame({"store_id": [target["store_id"]], "rate": [t_rate]})
    return pd.concat([peers[["store_id", "rate"]], target_row], ignore_index=True)


def _confidence(level: str, score) -> float:
    conf = _LEVEL_CONFIDENCE.get(level, 0.0)
    if level in (LV_NB_PK, LV_NB, LV_NK) and np.isfinite(score):
        conf *= float(score)
    return round(float(conf), 4)


def _round_units(corrected_value, original_value, original_unit) -> float:
    """Translate a corrected value back to integer units, preserving the old unit price."""
    if not np.isfinite(corrected_value) or not np.isfinite(original_unit):
        return np.nan
    if not np.isfinite(original_value) or original_value <= 0:
        return float(round(corrected_value)) if np.isfinite(corrected_value) else np.nan
    unit_price = original_value / original_unit if original_unit > 0 else np.nan
    if np.isfinite(unit_price) and unit_price > 0:
        return float(round(corrected_value / unit_price))
    return float(round(corrected_value))


def detect_store_shift(graded: pd.DataFrame, store_map: pd.DataFrame, params) -> pd.DataFrame:
    """Stores with >= store_shift_ratio of selling categories flagged HIGH (whole-store boom).

    Only ``auto_action=fix`` (suspicious+) highs/lows count toward the ratio; watch is ignored.
    """
    uf = params.unit_fix
    g = graded[graded["peer_level"] != "na"].copy()
    if g.empty:
        return pd.DataFrame(columns=STORE_SHIFT_COLS)
    if "auto_action" in g.columns:
        countable = g["auto_action"].eq("fix")
    elif "level" in g.columns:
        countable = g["level"].isin(["suspicious", "high_confidence"])
    else:
        countable = g["direction"].isin([HIGH, LOW])
    g["_cnt_high"] = (g["direction"] == HIGH) & countable
    g["_cnt_low"] = (g["direction"] == LOW) & countable
    per = g.groupby("store_id").agg(
        n_cat=("category", "nunique"),
        n_high=("_cnt_high", "sum"),
        n_low=("_cnt_low", "sum"),
        sales_value=("sales_value", "sum"),
    )
    per = per[per["n_cat"] >= uf.store_shift_min_cat]
    per["high_ratio"] = per["n_high"] / per["n_cat"]
    shift = per[per["high_ratio"] >= uf.store_shift_ratio].reset_index()
    if shift.empty:
        return pd.DataFrame(columns=STORE_SHIFT_COLS)
    attrs = g.drop_duplicates("store_id").set_index("store_id")
    for col in ATTR_COLS:
        shift[col] = shift["store_id"].map(attrs[col])
    shift["ibd_id"] = shift["store_id"].map(store_map.set_index("store_id")["ibd_id"])
    shift["direction"] = HIGH
    shift["period_id"] = int(g["period_id"].iloc[0])
    return shift[STORE_SHIFT_COLS].reset_index(drop=True)


def triage_low(row, drops: pd.DataFrame) -> str:
    """Classify a LOW store x category row: store_drop / missing / undersell."""
    if drops is not None and len(drops) and row["store_id"] in set(drops["store_id"]):
        return "store_drop"
    if row.get("zero_filled", False) or (row["sales_value"] <= 0):
        return "missing"
    return "undersell"


def _funnel_row(stage: str, reason: str, n: int, n_store_cat: int = 0, note: str | None = None) -> dict:
    return {
        "stage": stage, "reason": reason, "n": int(n), "n_store_cat": int(n_store_cat),
        "note": note if note is not None else FUNNEL_NOTES.get(reason, ""),
    }


def drill_action_funnel(graded: pd.DataFrame, drill: pd.DataFrame, min_level: str = FIX_MIN_LEVEL) -> list[dict]:
    """Counts from suspicious+ candidates through drilldown coverage."""
    if graded is None or graded.empty:
        return [_funnel_row("候选", "suspicious_plus", 0)]
    min_rank = level_rank(min_level)
    targets = graded[
        (graded["level"].map(level_rank) >= min_rank)
        & graded["direction"].isin([HIGH, LOW])
        & (graded["peer_level"] != "na")
    ]
    target_keys = set(zip(targets["store_id"], targets["category"]))
    drilled_keys = (
        set(zip(drill["store_id"], drill["category"])) if drill is not None and len(drill) else set()
    )
    with_items = target_keys & drilled_keys
    no_delta = target_keys - drilled_keys
    flag_only = 0
    if "auto_action" in graded.columns:
        flag_only = int((graded["auto_action"] == "flag_only").sum())
    fix_n = int((graded["auto_action"] == "fix").sum()) if "auto_action" in graded.columns else len(target_keys)
    rows = [
        _funnel_row("候选", "suspicious_plus", len(target_keys), len(target_keys)),
        _funnel_row("候选", "fix_action", fix_n, fix_n),
        _funnel_row("候选", "flag_only", flag_only, flag_only),
        _funnel_row("下钻", "drill_with_items", len(with_items), len(with_items)),
        _funnel_row("下钻", "drill_no_delta", len(no_delta), len(no_delta)),
        _funnel_row("下钻", "drill_item_rows", len(drill) if drill is not None else 0, len(drilled_keys)),
    ]
    return rows


# ---------------------------------------------------------------- orchestrator
def run_unit_fix(graded, drill, items_cur, items_prev, store_map, params, logger=None,
                 drops=None) -> UnitFixResult:
    """Correct item-level outliers and measure impact. Returns a UnitFixResult.

    ``drops`` (store_drop table, optional) excludes whole-store-drop stores from LOW item carving.
    """
    uf = params.unit_fix
    item_key = params.data.item_key
    period = int(graded["period_id"].iloc[0]) if len(graded) else 0
    totals = store_totals(items_cur) if not items_cur.empty else pd.Series(dtype=float)

    shift = detect_store_shift(graded, store_map, params)
    shift_stores = set(shift["store_id"]) if not shift.empty else set()
    drop_stores = set(drops["store_id"]) if drops is not None and len(drops) else set()

    # peer group membership: peer_group_id -> set of member IBD ids
    members_map = {}
    if not graded.empty and "member_ibds" in graded.columns:
        for _, r in graded.drop_duplicates(["peer_group_id", "member_ibds"]).iterrows():
            members_map[r["peer_group_id"]] = {s for s in str(r["member_ibds"]).split(";") if s}
    ibd_to_stores = store_map.groupby("ibd_id")["store_id"].apply(set).to_dict() if not store_map.empty else {}

    def peer_store_ids(gr, sid) -> set:
        if gr is None:
            return set()
        gid = gr.peer_group_id if isinstance(gr, pd.Series) else gr.get("peer_group_id")
        if gid is None:
            return set()
        out = set()
        for ibd in members_map.get(gid, set()):
            out |= set(ibd_to_stores.get(ibd, set()))
        out.discard(sid)
        return out

    graded_index = graded.set_index(["store_id", "category"]) if not graded.empty else pd.DataFrame()
    items_r = attach_rate(items_cur, totals) if not items_cur.empty else pd.DataFrame()
    items_by_store = {s: g for s, g in items_r.groupby("store_id")} if not items_r.empty else {}

    def item_info(sid, key) -> pd.Series:
        g = items_by_store.get(sid)
        if g is None:
            return pd.Series(dtype=object)
        m = g[g[item_key] == key]
        return m.iloc[0] if len(m) else pd.Series(dtype=object)

    if "auto_action" in graded.columns:
        fix_keys = set(zip(graded.loc[graded["auto_action"] == "fix", "store_id"],
                           graded.loc[graded["auto_action"] == "fix", "category"]))
    else:
        fix_keys = set(zip(graded.loc[graded["level"].isin(["suspicious", "high_confidence"]), "store_id"],
                           graded.loc[graded["level"].isin(["suspicious", "high_confidence"]), "category"]))

    item_n: dict[str, int] = defaultdict(int)
    store_cat: dict[str, set] = defaultdict(set)

    def bump(reason: str, sid: str, cat: str) -> None:
        item_n[reason] += 1
        store_cat[reason].add((sid, cat))

    corrections = []
    if not drill.empty:
        for tgt in drill.itertuples(index=False):
            direction = tgt.direction
            sid, cat, key = tgt.store_id, tgt.category, getattr(tgt, item_key, None)
            if (sid, cat) not in fix_keys:
                bump("skip_not_fix_action", sid, cat)
                continue
            if sid in shift_stores:
                bump("skip_store_shift", sid, cat)
                continue
            gkey = (sid, cat)
            gr = graded_index.loc[gkey] if gkey in graded_index.index else None
            if isinstance(gr, pd.DataFrame):
                gr = gr.iloc[0]
            # 本期店×品类销量为 0 / 补 0：该卖没卖只告警，绝不改商品
            g_sales = float(gr["sales_value"]) if gr is not None and "sales_value" in gr.index else np.nan
            g_zero = bool(gr["zero_filled"]) if gr is not None and "zero_filled" in gr.index else False
            if g_zero or (np.isfinite(g_sales) and g_sales <= 0) or float(tgt.sales_value) <= 0:
                bump("skip_missing_sales_warn", sid, cat)
                continue
            peer_ids = peer_store_ids(gr, sid)
            if not peer_ids:
                bump("skip_no_peer_stores", sid, cat)
                continue
            info = item_info(sid, key)
            target = pd.Series({
                "store_id": sid, "category": cat, "nankey": key,
                "brand": info.get("brand", "") if len(info) else getattr(tgt, "brand", ""),
                "packsize": info.get("packsize", "") if len(info) else getattr(tgt, "packsize", ""),
                "prod_desc_raw": info.get("prod_desc_raw", "") if len(info) else getattr(tgt, "prod_desc_raw", ""),
                "sales_value": float(tgt.sales_value), "sales_unit": float(tgt.sales_unit),
                "score": info.get("score", np.nan) if len(info) else np.nan,
            })
            candidates = pd.concat([items_by_store[s] for s in peer_ids if s in items_by_store], ignore_index=True) if items_by_store else pd.DataFrame()
            peers, level = match_peers(target, candidates, params)

            if direction == HIGH:
                if level == LV_TAG or peers.empty:
                    bump("skip_match_empty", sid, cat)
                    continue
                rates = _peer_rates(target, peers, totals)
                n_rate = int(rates["rate"].notna().sum())
                if n_rate < uf.min_peer_stores:
                    bump("skip_peer_rate_lt_min", sid, cat)
                    continue
                mask, corrected = fuller_right_tail(
                    rates["rate"].to_numpy(), detect_limit=uf.fuller_detect_limit,
                    correct_limit=uf.fuller_correct_limit, max_tail=uf.fuller_max_tail,
                    critical_ratio=uf.critical_ratio, min_n=uf.min_peer_stores,
                )
                t_pos = int((rates["store_id"] == sid).to_numpy().argmax())
                if not mask[t_pos]:
                    bump("skip_fuller_not_flagged", sid, cat)
                    continue
                corrected_value = corrected[t_pos] * totals.get(sid, np.nan) if np.isfinite(corrected[t_pos]) else np.nan
                corrections.append(_row(
                    period, sid, cat, key, target, HIGH, level, n_rate,
                    float(tgt.sales_value), float(tgt.sales_unit), corrected_value,
                ))
                bump("corrected", sid, cat)

            else:  # LOW
                if sid in drop_stores:
                    bump("skip_store_drop", sid, cat)
                    continue
                if level in (LV_TAG,) and peers.empty and tgt.sales_value <= 0:
                    bump("skip_missing_upstream", sid, cat)
                    continue
                if tgt.sales_value <= 0:
                    bump("skip_zero_sales", sid, cat)
                    continue
                peer_sales = _seller_category_sales(peer_ids, cat, graded)
                if peer_sales.size < uf.min_peer_stores:
                    bump("skip_peer_sales_lt_min", sid, cat)
                    continue
                lv = np.log1p(peer_sales.to_numpy(dtype=float))
                q1, q3, mc = adjbox_stats(lv)
                lo, _ = fence_from_stats(q1, q3, mc, params.fova.adjbox_a, params.fova.adjbox_b, params.fova.adjbox_c)
                cur_lv = float(np.log1p(tgt.sales_value))
                if cur_lv >= lo:
                    bump("skip_above_left_fence", sid, cat)
                    continue
                corrected_value = max(float(np.expm1(lo)), 0.0)
                corrections.append(_row(
                    period, sid, cat, key, target, LOW, LV_CAT, int(peer_sales.size),
                    float(tgt.sales_value), float(tgt.sales_unit), corrected_value,
                ))
                bump("corrected", sid, cat)

    corrections_df = pd.DataFrame(corrections, columns=CORRECTION_COLS) if corrections else pd.DataFrame(columns=CORRECTION_COLS)
    recheck_df = _recheck(graded, corrections_df, params, shift_stores)
    impact_df = _impact_summary(graded, corrections_df)

    funnel_rows = drill_action_funnel(graded, drill if drill is not None else pd.DataFrame())
    funnel_rows.append(_funnel_row("豁免", "store_shift_stores", len(shift_stores), len(shift_stores)))
    skip_order = [
        "skip_not_fix_action", "skip_store_shift", "skip_missing_sales_warn", "skip_no_peer_stores",
        "skip_match_empty", "skip_peer_rate_lt_min", "skip_fuller_not_flagged", "skip_store_drop",
        "skip_missing_upstream", "skip_zero_sales", "skip_peer_sales_lt_min", "skip_above_left_fence",
        "corrected",
    ]
    for reason in skip_order:
        funnel_rows.append(_funnel_row(
            "修正", reason, item_n.get(reason, 0), len(store_cat.get(reason, ())),
        ))
    corr_sc = store_cat.get("corrected", set())
    funnel_rows.append(_funnel_row("修正", "corrected_store_cat", len(corr_sc), len(corr_sc)))
    funnel_rows.append(_funnel_row("复检", "recheck_rows", len(recheck_df), len(recheck_df)))
    if len(recheck_df):
        funnel_rows.append(_funnel_row(
            "复检", "recheck_resolved", int(recheck_df["resolved"].sum()),
            int(recheck_df.loc[recheck_df["resolved"], ["store_id", "category"]].drop_duplicates().shape[0])
            if "resolved" in recheck_df.columns else int(recheck_df["resolved"].sum()),
        ))
        funnel_rows.append(_funnel_row(
            "复检", "recheck_fail", int((~recheck_df["resolved"]).sum()),
            int(recheck_df.loc[~recheck_df["resolved"], ["store_id", "category"]].drop_duplicates().shape[0])
            if "resolved" in recheck_df.columns else int((~recheck_df["resolved"]).sum()),
        ))
    else:
        funnel_rows.append(_funnel_row("复检", "recheck_resolved", 0))
        funnel_rows.append(_funnel_row("复检", "recheck_fail", 0))
    funnel_df = pd.DataFrame(funnel_rows, columns=FUNNEL_COLS)

    if logger is not None:
        logger.info(
            "[UNIT_FIX] %s: auto_action=fix 店×品类 %d; 下钻行 %d; 修正 %d 条 / %d 店×品类; "
            "整店普涨豁免 %d 店; 复检 resolved %d / %d; 净 impact %.2f",
            period, len(fix_keys), len(drill) if drill is not None else 0, len(corrections_df), len(corr_sc),
            len(shift),
            int((recheck_df["resolved"] == True).sum()) if len(recheck_df) else 0,
            len(recheck_df),
            float(corrections_df["impact_value"].sum()) if len(corrections_df) else 0.0,
        )
        for r in funnel_df.itertuples(index=False):
            if r.n:
                logger.info("[UNIT_FIX][漏斗] %s / %s: n=%d store×cat=%d (%s)", r.stage, r.reason, r.n, r.n_store_cat, r.note)

    return UnitFixResult(
        corrections=corrections_df, store_shift=shift, recheck=recheck_df, impact=impact_df, funnel=funnel_df,
    )


def _row(period, sid, cat, key, target, direction, level, n_peer, orig_v, orig_u, corr_v) -> dict:
    """Build one correction row with unit rounding and impact."""
    score = target.get("score", np.nan)
    corrected_unit = _round_units(corr_v, orig_v, orig_u)
    return {
        "period_id": period, "store_id": sid, "category": cat, "item_key": key,
        "prod_desc_raw": target.get("prod_desc_raw", ""), "brand": target.get("brand", ""),
        "packsize": target.get("packsize", ""), "direction": direction, "match_level": level,
        "n_peer": n_peer, "confidence": _confidence(level, score),
        "original_value": float(orig_v), "original_unit": float(orig_u),
        "corrected_value": float(corr_v) if np.isfinite(corr_v) else np.nan,
        "corrected_unit": corrected_unit,
        "impact_value": float(corr_v - orig_v) if np.isfinite(corr_v) else np.nan,
        "impact_unit": float(corrected_unit - orig_u) if np.isfinite(corrected_unit) else np.nan,
        "score": float(score) if np.isfinite(score) else np.nan,
    }


def _seller_category_sales(peer_ids, category, graded) -> pd.Series:
    sub = graded[(graded["store_id"].isin(peer_ids)) & (graded["category"] == category)]
    return sub[sub["sales_value"] > 0]["sales_value"]


def _recheck(graded, corrections, params, shift_stores) -> pd.DataFrame:
    """Re-test corrected store x category totals against clean peers (excluding flagged stores)."""
    if corrections.empty:
        return pd.DataFrame(columns=RECHECK_COLS)
    a, b, c = params.fova.adjbox_a, params.fova.adjbox_b, params.fova.adjbox_c
    corr_agg = corrections.groupby(["store_id", "category"], as_index=False).agg(impact=("impact_value", "sum"))
    graded_idx = graded.set_index(["store_id", "category"])
    flagged_stores = set(corrections["store_id"])
    rows = []
    for _, r in corr_agg.iterrows():
        sid, cat = r["store_id"], r["category"]
        if sid in shift_stores:
            continue
        try:
            base = graded_idx.loc[(sid, cat)]
            if isinstance(base, pd.DataFrame):
                base = base.iloc[0]
            orig = float(base["sales_value"])
            gid = base.get("peer_group_id")
            period = int(base["period_id"])
        except (KeyError, TypeError):
            continue
        corrected = orig + float(r["impact"])
        peers = graded[(graded["peer_group_id"] == gid) & (graded["category"] == cat) & (graded["store_id"] != sid)]
        peers = peers[~peers["store_id"].isin(flagged_stores)]  # clean peers
        lv = np.log1p(peers["sales_value"].clip(lower=0).to_numpy(dtype=float))
        if lv.size < params.unit_fix.min_peer_stores:
            continue
        q1, q3, mc = adjbox_stats(lv)
        lo, hi = fence_from_stats(q1, q3, mc, a, b, c)
        clv = float(np.log1p(corrected))
        resolved = bool(lo <= clv <= hi)
        rows.append({
            "period_id": period, "store_id": sid, "category": cat,
            "original_value": orig, "corrected_value": corrected, "n_peer": int(lv.size),
            "lv_lo": lo, "lv_hi": hi, "resolved": resolved, "recheck_fail": not resolved,
        })
    return pd.DataFrame(rows, columns=RECHECK_COLS)


def _impact_summary(graded, corrections) -> pd.DataFrame:
    """Net impact and total correction volume by dimension."""
    if corrections.empty:
        return pd.DataFrame(columns=IMPACT_COLS)
    corr = corrections.copy()
    corr["brand_key"] = corr["brand"].map(_brand_key)
    store_attrs = graded.drop_duplicates("store_id").set_index("store_id")[list(ATTR_COLS)] if len(graded) else pd.DataFrame()
    rows = [{"dimension": "overall", "level": "overall", "n_items": len(corr),
             "net_impact": corr["impact_value"].sum(), "total_volume": corr["impact_value"].abs().sum()}]
    for dim, col in (("platform", "PLATFORMNAME"), ("shoptype", "SHOPTYPE"),
                     ("province", "PROVINCE"), ("category", "category"), ("brand", "brand_key")):
        if col in corr.columns:
            vals = corr[col]
        elif col in ATTR_COLS and len(store_attrs):
            vals = corr["store_id"].map(store_attrs[col])
        else:
            continue
        for level, g in corr.groupby(vals.astype(str), observed=False):
            rows.append({"dimension": dim, "level": level, "n_items": len(g),
                         "net_impact": g["impact_value"].sum(), "total_volume": g["impact_value"].abs().sum()})
    return pd.DataFrame(rows, columns=IMPACT_COLS).sort_values(["dimension", "level"]).reset_index(drop=True)
