"""Layer 2 pooling: IBD x category sample check, merge with the nearest IBD, else fall back."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ibd_define import IbdDefineResult, cell_label, group_distance
from funcs.robust import heterogeneity, passes

LEVEL_IBD, LEVEL_MERGED, LEVEL_BASE, LEVEL_NA = "ibd", "ibd_merged", "base_cell", "na"
REASON_NONE, REASON_MERGE, REASON_BASE, REASON_NA = "none", "merge_nearest_ibd", "fallback_base_cell", "insufficient"

MERGE_COLUMNS = [
    "period_id", "cell", "ibd_id_raw", "category", "ibd_id_merged", "peer_level", "pooling_reason",
    "member_provinces_raw", "member_provinces_merged", "member_ibds",
    "n_store_raw", "n_store_merged", "n_store_cat_raw", "n_store_cat_merged", "n_item_raw", "n_item_merged",
    "low_store", "low_store_cat", "low_item", "vi", "ks", "psi", "distance",
]


def _member_provinces(ibd_map: pd.DataFrame) -> dict:
    return ibd_map.drop_duplicates("ibd_id").set_index("ibd_id")["member_provinces"].to_dict()


def check_ibd(
    sc_cur: pd.DataFrame,
    items_cur: pd.DataFrame,
    define: IbdDefineResult,
    item_key: str,
    ibd,
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ibd_merge table, summary) for the current period.

    ``sc_cur`` holds the current period store x category rows; ``items_cur`` its item rows.
    """
    base_cell = define.base_cell
    period = int(sc_cur["period_id"].iloc[0])
    store_ibd = define.store_map.set_index("store_id")["ibd_id"]
    ibd_cell = define.store_map.drop_duplicates("ibd_id").set_index("ibd_id")[list(base_cell)].apply(tuple, axis=1).to_dict()
    ibd_n_store = define.store_map.groupby("ibd_id")["store_id"].nunique().to_dict()
    soft_ibds = set(define.ibd_map.loc[define.ibd_map["soft_kept"].astype(bool), "ibd_id"]) if "soft_kept" in define.ibd_map else set()
    provinces = _member_provinces(define.ibd_map)
    ibd_provs = {k: v.split("+") for k, v in provinces.items()}

    sell = sc_cur[sc_cur["sales_value"] > 0].copy()
    sell["ibd_id"] = sell["store_id"].map(store_ibd)
    sell = sell.dropna(subset=["ibd_id"])
    values = {k: np.log1p(g["sales_value"].to_numpy()) for k, g in sell.groupby(["ibd_id", "category"])}
    n_cat = sell.groupby(["ibd_id", "category"])["store_id"].nunique().to_dict()

    it = items_cur[items_cur["sales_value"] > 0][["store_id", "category", item_key]].copy()
    it["ibd_id"] = it["store_id"].map(store_ibd)
    item_sets = {k: np.unique(g[item_key].to_numpy()) for k, g in it.dropna(subset=["ibd_id"]).groupby(["ibd_id", "category"])}
    empty = np.array([], dtype="int64")

    cell_cat_store = {}
    for (ibd_id, cat), n in n_cat.items():
        key = (ibd_cell[ibd_id], cat)
        cell_cat_store[key] = cell_cat_store.get(key, 0) + n

    def adequate(n_store_cat, n_item):
        return n_store_cat > ibd.min_store_cat and n_item >= ibd.min_item

    rows = []
    for (ibd_id, cat), n_sc in sorted(n_cat.items()):
        cell = ibd_cell[ibd_id]
        items = item_sets.get((ibd_id, cat), empty)
        row = {
            "period_id": period,
            "cell": cell_label(cell),
            "ibd_id_raw": ibd_id,
            "category": cat,
            "member_provinces_raw": provinces[ibd_id],
            "n_store_raw": ibd_n_store[ibd_id],
            "n_store_cat_raw": n_sc,
            "n_item_raw": len(items),
            "low_store": ibd_n_store[ibd_id] < (ibd.min_store_soft if ibd_id in soft_ibds else ibd.min_store),
            "low_store_cat": n_sc <= ibd.min_store_cat,
            "low_item": len(items) < ibd.min_item,
            "vi": np.nan, "ks": np.nan, "psi": np.nan, "distance": np.nan,
        }
        same = {
            "ibd_id_merged": ibd_id, "peer_level": LEVEL_IBD, "pooling_reason": REASON_NONE,
            "member_provinces_merged": provinces[ibd_id], "member_ibds": ibd_id,
            "n_store_merged": row["n_store_raw"], "n_store_cat_merged": n_sc, "n_item_merged": len(items),
        }
        by_ibd = ibd.check_level == "ibd"
        need_pooling = row["low_store"] if by_ibd else (row["low_store_cat"] or row["low_item"])
        if not need_pooling:
            rows.append({**row, **same})
            continue

        best = None
        candidates = [] if by_ibd else [o for o in ibd_n_store if o != ibd_id and ibd_cell[o] == cell and (o, cat) in n_cat]
        for other in candidates:
            m_sc = n_sc + n_cat[(other, cat)]
            m_items = len(np.union1d(items, item_sets.get((other, cat), empty)))
            if not adequate(m_sc, m_items):
                continue
            het = heterogeneity(values[(other, cat)], values[(ibd_id, cat)], ibd.psi_bins, ibd.het_alpha, ibd.psi_min_bin_count)
            if not passes(het, ibd.vi_max, ibd.ks_max, ibd.psi_max, ibd.max_median_ratio):
                continue
            cand = {
                "other": other, "n_sc": m_sc, "n_item": m_items, "het": het,
                "distance": group_distance(define.distances[cell], ibd_provs[ibd_id], ibd_provs[other]),
            }
            if best is None or (cand["n_sc"], -het["vi"], -cand["distance"]) > (best["n_sc"], -best["het"]["vi"], -best["distance"]):
                best = cand

        if best is not None:
            members = sorted([ibd_id, best["other"]])
            merged_provs = "+".join(sorted(set(ibd_provs[ibd_id]) | set(ibd_provs[best["other"]])))
            rows.append(
                {
                    **row,
                    "ibd_id_merged": "+".join(members),
                    "peer_level": LEVEL_MERGED,
                    "pooling_reason": REASON_MERGE,
                    "member_provinces_merged": merged_provs,
                    "member_ibds": ";".join(members),
                    "n_store_merged": ibd_n_store[ibd_id] + ibd_n_store[best["other"]],
                    "n_store_cat_merged": best["n_sc"],
                    "n_item_merged": best["n_item"],
                    **best["het"],
                    "distance": best["distance"],
                }
            )
            continue

        cell_ibds = sorted(o for o in ibd_n_store if ibd_cell[o] == cell)
        cell_sc = cell_cat_store.get((cell, cat), 0)
        cell_items = len(np.unique(np.concatenate([item_sets.get((o, cat), empty) for o in cell_ibds])))
        cell_size = sum(ibd_n_store[o] for o in cell_ibds) if by_ibd else cell_sc
        if cell_size >= ibd.min_store:
            rows.append(
                {
                    **row,
                    "ibd_id_merged": f"base_cell:{cell_label(cell)}",
                    "peer_level": LEVEL_BASE,
                    "pooling_reason": REASON_BASE,
                    "member_provinces_merged": "+".join(sorted({p for o in cell_ibds for p in ibd_provs[o]})),
                    "member_ibds": ";".join(cell_ibds),
                    "n_store_merged": sum(ibd_n_store[o] for o in cell_ibds),
                    "n_store_cat_merged": cell_sc,
                    "n_item_merged": cell_items,
                }
            )
        else:
            rows.append(
                {
                    **row,
                    "ibd_id_merged": LEVEL_NA,
                    "peer_level": LEVEL_NA,
                    "pooling_reason": REASON_NA,
                    "member_provinces_merged": "",
                    "member_ibds": "",
                    "n_store_merged": 0,
                    "n_store_cat_merged": cell_sc,
                    "n_item_merged": cell_items,
                }
            )

    merge = pd.DataFrame(rows, columns=MERGE_COLUMNS)
    for col, name in zip(base_cell, range(len(base_cell))):
        merge.insert(2 + name, col, merge["cell"].str.split("|").str[name])
    summary = summarize_check(merge)
    if logger is not None:
        log_check(logger, period, merge, summary)
    return merge, summary


def summarize_check(merge: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"item": "rows", "value": len(merge)},
        {"item": "low_store", "value": int(merge["low_store"].sum())},
        {"item": "low_store_cat", "value": int(merge["low_store_cat"].sum())},
        {"item": "low_item", "value": int(merge["low_item"].sum())},
    ]
    for level in (LEVEL_IBD, LEVEL_MERGED, LEVEL_BASE, LEVEL_NA):
        rows.append({"item": f"peer_level={level}", "value": int((merge["peer_level"] == level).sum())})
    unchanged = int((merge["ibd_id_merged"] == merge["ibd_id_raw"]).sum())
    rows.append({"item": "unchanged", "value": unchanged})
    rows.append({"item": "unchanged_pct", "value": round(100 * unchanged / len(merge), 2) if len(merge) else 0.0})
    return pd.DataFrame(rows)


def log_check(logger, period, merge: pd.DataFrame, summary: pd.DataFrame) -> None:
    s = dict(zip(summary["item"], summary["value"]))
    flagged = int((merge["pooling_reason"] != REASON_NONE).sum())
    fallback = s[f"peer_level={LEVEL_BASE}"] + s[f"peer_level={LEVEL_NA}"]
    logger.info(
        "[IBD_CHECK] %s: low_store %d, low_store_cat %d, low_item %d; 不足 %d -> ibd_merged %d, base_cell %d, na %d; fallback 比例 %.1f%%",
        period, s["low_store"], s["low_store_cat"], s["low_item"], flagged,
        s[f"peer_level={LEVEL_MERGED}"], s[f"peer_level={LEVEL_BASE}"], s[f"peer_level={LEVEL_NA}"],
        100 * fallback / flagged if flagged else 0.0,
    )
    logger.info("[IBD_CHECK] %s: ibd_merge rows=%d, unchanged=%d (%.1f%%)", period, s["rows"], s["unchanged"], s["unchanged_pct"])
    merged = merge[merge["peer_level"] == LEVEL_MERGED]
    for r in merged.head(20).itertuples():
        logger.info(
            "[IBD_CHECK] merge %s x %s -> %s: n_store_cat %d->%d, n_item %d->%d, vi=%.3f ks=%.3f psi=%.3f",
            r.ibd_id_raw, r.category, r.ibd_id_merged, r.n_store_cat_raw, r.n_store_cat_merged,
            r.n_item_raw, r.n_item_merged, r.vi, r.ks, r.psi,
        )


def peer_assignment(sc_cur: pd.DataFrame, store_map: pd.DataFrame, merge: pd.DataFrame) -> pd.DataFrame:
    """Attach ibd_id / peer_group_id / peer_level / member_ibds to current store x category rows."""
    store_cols = ["store_id", "ibd_id"] + [c for c in ("soft_kept", "final_ok") if c in store_map.columns]
    out = sc_cur.merge(store_map[store_cols], on="store_id", how="left")
    cols = merge[["ibd_id_raw", "category", "ibd_id_merged", "peer_level", "member_ibds"]].rename(
        columns={"ibd_id_raw": "ibd_id", "ibd_id_merged": "peer_group_id"}
    )
    out = out.merge(cols, on=["ibd_id", "category"], how="left")
    missing = out["peer_group_id"].isna()
    out.loc[missing, "peer_group_id"] = out.loc[missing, "ibd_id"]
    out.loc[missing, "peer_level"] = LEVEL_IBD
    out.loc[missing, "member_ibds"] = out.loc[missing, "ibd_id"]
    return out
