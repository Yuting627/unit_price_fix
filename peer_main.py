"""Production entry: peer outlier detection for one period. Reads the params JSON, never writes it.

quality -> profile -> stratify (base_cell monitor) -> ibd_define -> ibd_check -> seasonal -> peer_check
-> evidence -> drilldown -> stability -> report. IBD, seasonal factors and base_cell checks are recomputed every period.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from core.config import flatten, load_params
from core.data_prep.data_prep import list_before_mp_files, list_store_target_files, load_panel, read_store_target, write_excel, zero_fill_categories
from core.report.report import adjustments, flag_raw, summary_by_dim, summary_overview, write_csv_gz
from core.outlier_rules.drilldown import run_drilldown
from core.outlier_rules.evidence import demote_zero_sales_from_fix, grade, log_evidence, summarize_levels
from core.ibd_define.ibd_check import check_ibd, peer_assignment
from core.ibd_define.ibd_define import define_ibd
from funcs.logger import close_logger, get_logger, log_section, production_log_path
from core.outlier_rules.peer_check import missing_sales, run_peer_check, store_drop
from core.data_prep.profile import profile
from core.data_prep.quality import check_quality
from core.outlier_rules.seasonal import compute_seasonal, group_members, log_seasonal, peer_panel, seasonal_records
from core.outlier_rules.stability import log_stability, rate_by_period, threshold_jaccard
from core.data_prep.stratify import check_base_cell
from core.unit_fix.unit_fix import run_unit_fix

DEFAULT_PARAMS = ROOT / "config" / "peer_params.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _with_ibd(items: pd.DataFrame, store_map: pd.DataFrame) -> pd.DataFrame:
    if items.empty:
        return items.assign(ibd_id=pd.Series(dtype=object))
    return items.merge(store_map[["store_id", "ibd_id"]], on="store_id", how="inner")


def detect(sc: pd.DataFrame, ppanel: pd.DataFrame, store_map, merge, period: int, params, logger=None):
    """Seasonal + peer_check + evidence for ``period`` using the current IBD / peer mapping.

    Peer sets are complete store x category grids: IBD stores not selling a category carry sales 0.
    """
    filled = zero_fill_categories(sc[sc["period_id"] == period], store_map.set_index("store_id")["ibd_id"])
    assigned = peer_assignment(filled, store_map, merge)
    assigned = assigned[assigned["ibd_id"].notna()]
    members = ppanel[["peer_group_id", "category", "ibd_id"]].drop_duplicates()
    sub = pd.concat([ppanel[ppanel["period_id"] < period], peer_panel(filled, store_map, members)], ignore_index=True)
    seas = compute_seasonal(sub, period, params.seasonal, base_cell=params.base_cell)
    if logger is not None:
        log_seasonal(logger, period, seas)
    detail = run_peer_check(assigned, sub, seas, period, params, logger=logger)
    drops = store_drop(sc, store_map, period, params.peer, logger=logger)
    all_missing = missing_sales(assigned, sub, period, params.peer)
    drops["n_missing"] = drops["store_id"].map(all_missing["store_id"].value_counts()).fillna(0).astype(int)
    missing = missing_sales(assigned, sub, period, params.peer, logger=logger, exclude_stores=drops["store_id"])
    graded = demote_zero_sales_from_fix(grade(detail, params.evidence.high_min))
    return seas, detail, graded, missing, drops


def run(params_path=DEFAULT_PARAMS, period: int | None = None) -> dict:
    params_path = Path(params_path)
    before_digest = _digest(params_path)
    params = load_params(params_path)
    item_key = params.data.item_key
    bmp_dir = params.path("before_mp_dir", ROOT)
    periods = [p for p in list_before_mp_files(bmp_dir) if period is None or p <= period]
    period = period or max(periods)
    if period not in periods:
        raise FileNotFoundError(f"no before_mp file for period {period}")
    prev = periods[-2] if len(periods) > 1 else None
    logger = get_logger("peer.main", production_log_path(params.path("log_dir", ROOT), period))
    try:
        logger.info("[MAIN] period=%s history=%s params=%s (只读)", period, periods, params_path)
        panel = load_panel(
            bmp_dir, params.path("store_target_dir", ROOT), item_key, periods=periods,
            item_periods={p for p in (period, prev) if p is not None},
            encoding=params.data.encoding, quality_fn=check_quality, logger=logger,
        )
        sc = panel.sc
        dq = panel.dq
        log_section(logger, "QUALITY", dq[(dq["period_id"] == period) & ((dq["n_rows"].fillna(0) > 0) | dq["n_rows"].isna())])

        prof = profile(sc, periods=[period])
        logger.info("[PROFILE] %s: %d stores, %d store x category rows", period, sc.loc[sc["period_id"] == period, "store_id"].nunique(), int((sc["period_id"] == period).sum()))

        sc_cur = sc[sc["period_id"] == period]
        anova, base_cell_suggested = check_base_cell(sc_cur, params.base_cell, params.stratify, logger=logger)
        define = define_ibd(sc, period, params.base_cell, params.ibd, logger=logger)
        items = panel.items
        items_cur = items[items["period_id"] == period]
        items_prev = items[items["period_id"] == prev] if prev is not None else items.iloc[0:0]
        merge, check_summary = check_ibd(sc_cur, items_cur, define, item_key, params.ibd, logger=logger)

        assigned_cur = peer_assignment(sc_cur, define.store_map, merge)
        ppanel = peer_panel(sc, define.store_map, group_members(assigned_cur))
        seas, detail, graded, missing, drops = detect(sc, ppanel, define.store_map, merge, period, params, logger=logger)
        levels = summarize_levels(graded)
        log_evidence(logger, period, levels)

        drill = run_drilldown(
            graded, _with_ibd(items_cur, define.store_map), _with_ibd(items_prev, define.store_map),
            seas, item_key, params.drilldown, logger=logger,
        )

        unit = run_unit_fix(
            graded, drill, _with_ibd(items_cur, define.store_map), _with_ibd(items_prev, define.store_map),
            define.store_map, params, logger=logger, drops=drops,
        )

        hist_levels = [levels]
        for p in panel.periods:
            if p < period and p != panel.periods[0]:
                _, _, g, _, _ = detect(sc, ppanel, define.store_map, merge, p, params)
                hist_levels.append(summarize_levels(g))
        all_levels = pd.concat(hist_levels, ignore_index=True).sort_values(["period_id", "level"])
        jac = threshold_jaccard(detail, params)
        rates = rate_by_period(all_levels, params.stability.rate_spike_ratio)
        log_stability(logger, jac, rates, params.stability.rate_spike_ratio)

        out_dir = params.path("output_dir", ROOT)
        min_level = params.report.adjust_min_level
        adj = adjustments(graded, min_level)
        raw_stats = {}
        raw_path = None
        if params.report.raw_output:
            raw = pd.read_csv(list_before_mp_files(bmp_dir)[period], encoding=params.data.encoding, low_memory=False)
            target = read_store_target(list_store_target_files(params.path("store_target_dir", ROOT))[period], encoding=params.data.encoding)
            flagged, raw_stats = flag_raw(raw, target[target["PERIODCODE"] == period], graded, adj, drops, drill, item_key, min_level, corrections=unit.corrections)
            raw_path = out_dir / f"peer_outlier_raw_{period}.csv.gz"
            write_csv_gz(flagged, raw_path)
            logger.info(
                "[REPORT] 原始数据 %d 行 -> %s; 有问题行 %d (%.2f%%), 销售额占比 %.2f%%",
                raw_stats["rows"], raw_path.name, raw_stats["issue_rows"], 100 * raw_stats["issue_rows"] / max(raw_stats["rows"], 1),
                100 * raw_stats["issue_sales"] / raw_stats["sales"] if raw_stats["sales"] else 0.0,
            )
            del raw, flagged
        overview = summary_overview(
            graded, adj, missing, drops, dq[dq["period_id"] == period], drill, raw_stats, min_level,
            unit=unit, ibd_map=define.ibd_map, high_min=params.evidence.high_min,
        )
        by_dim = summary_by_dim(graded, adj, missing, drops)
        log_section(logger, "REPORT 汇总", overview[["section", "item", "n", "pct_rows", "n_stores", "pct_sales", "adj_to_fence", "adj_to_median"]])
        seas_json = out_dir / f"seasonal_factor_{period}.json"
        seas_json.parent.mkdir(parents=True, exist_ok=True)
        seas_json.write_text(json.dumps({"period_id": period, "factors": seasonal_records(seas)}, ensure_ascii=False, indent=1), encoding="utf-8")

        graded_out = graded.drop(columns=["lv_med"], errors="ignore")
        xlsx = write_excel(
            {
                "summary": overview,
                "summary_by_dim": by_dim,
                "adjustment": adj,
                "dq": dq,
                "profile": prof,
                "anova": anova,
                "ibd_map": define.ibd_map,
                "ibd_merge": merge,
                "ibd_check": pd.concat([check_summary, define.summary.assign(item="ibd_define")], ignore_index=True),
                "seasonal_factor": seas,
                "peer_detail": graded_out,
                "drilldown": drill,
                "unit_fix": unit.corrections,
                "unit_recheck": unit.recheck,
                "store_shift": unit.store_shift,
                "unit_impact": unit.impact,
                "fix_funnel": unit.funnel,
                "missing_sales": missing,
                "store_drop": drops,
                "evidence_summary": all_levels,
                "stability": pd.concat([jac, rates], ignore_index=True),
                "params_used": pd.DataFrame(flatten(params.to_dict()), columns=["key", "value"]),
            },
            out_dir / f"peer_outlier_{period}.xlsx",
        )
        unchanged = _digest(params_path) == before_digest
        if not unchanged:
            raise RuntimeError(f"{params_path} was modified during production run")
        logger.info("[MAIN] 输出 %s, %s; %s 未修改", xlsx, seas_json, params_path.name)
        return {"xlsx": xlsx, "seasonal_json": seas_json, "graded": graded, "drilldown": drill, "missing_sales": missing, "store_drop": drops,
                "unit_fix": unit.corrections, "store_shift": unit.store_shift, "unit_impact": unit.impact,
                "summary": overview, "summary_by_dim": by_dim, "adjustment": adj, "raw_csv": raw_path, "base_cell_suggested": base_cell_suggested, "params_unchanged": unchanged}
    finally:
        close_logger(logger)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Peer outlier production run for one period")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS))
    parser.add_argument("--period", type=int, default=None, help="period to check (default: latest available)")
    args = parser.parse_args(argv)
    run(args.params, args.period)


if __name__ == "__main__":
    main()
