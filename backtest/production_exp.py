"""Production experiment: A0–A3 (T1/T3), B0/B3 (IBD soft=5, no forced), C3 (adjbox cell shrink).

Runs the real ``peer_main.run`` for 20261406/07/08 under isolated output dirs and writes a
comparison workbook with availability, alert structure, province lift, systematic share,
IBD quality, and slice metrics.

Usage:
    python backtest/production_exp.py                  # all arms (slow: ~7 x 3 full runs)
    python backtest/production_exp.py --phase 1        # A0–A3 only
    python backtest/production_exp.py --phase 2        # B0 B3 (needs phase-1 code)
    python backtest/production_exp.py --phase 3        # C3
    python backtest/production_exp.py --arms A0 A3 B3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "code", ROOT, ROOT / "backtest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import peer_main  # noqa: E402
import group_scheme_backtest as B  # noqa: E402

PERIODS = (20261406, 20261407, 20261408)
BASE_CONFIG = ROOT / "config" / "peer_params.json"
EXP_ROOT = ROOT / "data" / "backtest" / "exp"
OUT_XLSX = ROOT / "data" / "backtest" / "production_exp_compare.xlsx"
NA = "na"

# hist_start_filter always false so truncation does not confound T1/T3 attribution
COMMON = dict(hist_start_filter=False)

ARMS: dict[str, dict] = {
    "A0": {
        "fova": dict(COMMON, trend_fence_fallback=False, longitude_borrow_cell=False, adjbox_cell_shrink=False),
        "ibd": dict(min_store=20, min_store_soft=15, allow_forced_nearest=True),
    },
    "A1": {
        "fova": dict(COMMON, trend_fence_fallback=True, longitude_borrow_cell=False, adjbox_cell_shrink=False),
        "ibd": dict(min_store=20, min_store_soft=15, allow_forced_nearest=True),
    },
    "A2": {
        "fova": dict(COMMON, trend_fence_fallback=False, longitude_borrow_cell=True, adjbox_cell_shrink=False),
        "ibd": dict(min_store=20, min_store_soft=15, allow_forced_nearest=True),
    },
    "A3": {
        "fova": dict(COMMON, trend_fence_fallback=True, longitude_borrow_cell=True, adjbox_cell_shrink=False),
        "ibd": dict(min_store=20, min_store_soft=15, allow_forced_nearest=True),
    },
    "B0": {
        "fova": dict(COMMON, trend_fence_fallback=False, longitude_borrow_cell=False, adjbox_cell_shrink=False),
        "ibd": dict(min_store=10, min_store_soft=5, allow_forced_nearest=False),
    },
    "B3": {
        "fova": dict(COMMON, trend_fence_fallback=True, longitude_borrow_cell=True, adjbox_cell_shrink=False),
        "ibd": dict(min_store=10, min_store_soft=5, allow_forced_nearest=False),
    },
    "C3": {
        "fova": dict(COMMON, trend_fence_fallback=True, longitude_borrow_cell=True,
                     adjbox_cell_shrink=True, adjbox_shrink_min_n=10, adjbox_shrink_scale=3.0),
        "ibd": dict(min_store=10, min_store_soft=5, allow_forced_nearest=False),
    },
}

PHASE_ARMS = {
    1: ("A0", "A1", "A2", "A3"),
    2: ("B0", "B3"),
    3: ("C3",),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def config_for(arm: str) -> Path:
    raw = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    spec = ARMS[arm]
    raw["fova"].update(spec["fova"])
    raw["ibd"].update(spec["ibd"])
    out = EXP_ROOT / arm
    raw["paths"]["output_dir"] = str(out.relative_to(ROOT)).replace("\\", "/")
    raw["paths"]["log_dir"] = f"logs/exp/{arm}"
    # skip raw 2.6M-row dump in experiments (xlsx + seasonal still written)
    raw["report"]["raw_output"] = False
    dest = EXP_ROOT / f"_params_{arm}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    return dest


def dist(series: pd.Series) -> dict:
    c = series.value_counts()
    return {k: int(c.get(k, 0)) for k in ("high", "low", "ok", NA)}


def province_lift_max(detail: pd.DataFrame, flagged: pd.Series) -> float:
    if not flagged.any():
        return float("nan")
    share_all = detail["PROVINCE"].value_counts(normalize=True)
    share_flag = detail.loc[flagged, "PROVINCE"].value_counts(normalize=True)
    lift = (share_flag / share_all).dropna()
    return float(lift.max()) if len(lift) else float("nan")


def metrics_from_result(res: dict, period: int, arm: str) -> tuple[dict, list[dict], dict]:
    detail = res["graded"]
    n = len(detail)
    flagged = detail["fova_final"].isin(["high", "low"])
    n_flag = int(flagged.sum())
    flags = detail["fova_final"].to_numpy()
    sys_share, sys_n, sys_base = B.systematic_share(detail, flags)

    bad_mask = pd.Series(False, index=detail.index)
    if "final_ok" in detail.columns:
        bad_mask |= ~detail["final_ok"].fillna(True).astype(bool)
    if "soft_kept" in detail.columns:
        soft = detail["soft_kept"].fillna(False).astype(bool)
    else:
        soft = pd.Series(False, index=detail.index)
    if "pooling" in detail.columns:
        pass
    lift_bad = province_lift_max(detail[bad_mask | soft], flagged[bad_mask | soft]) if (bad_mask | soft).any() else float("nan")

    n_sell = pd.to_numeric(detail.get("n_sell", 0), errors="coerce").fillna(0)
    row = {
        "arm": arm, "period": period, "n_rows": n,
        "lon_available_pct": round(100 * float((detail["fova_longitude"] != NA).mean()), 2) if n else 0.0,
        "n_hist_eff_median": float(pd.to_numeric(detail.get("n_hist_eff", 0), errors="coerce").median()),
        "trend_available_pct": round(100 * float((detail["fova_trend"] != NA).mean()), 2) if n else 0.0,
        "final_available_pct": round(100 * float((detail["fova_final"] != NA).mean()), 2) if n else 0.0,
        "fence_src": detail["td_fence_src"].value_counts().to_dict() if "td_fence_src" in detail.columns else {},
        "lon_src": detail["lon_src"].value_counts().to_dict() if "lon_src" in detail.columns else {},
        "flag_rows": n_flag,
        "flag_rate_pct": round(100 * n_flag / n, 3) if n else 0.0,
        "province_lift_max": round(province_lift_max(detail, flagged), 2),
        "top10_group_share": round(float(
            detail.loc[flagged].groupby(["peer_group_id", "category"]).size().sort_values(ascending=False).head(10).sum() / n_flag
        ), 3) if n_flag else float("nan"),
        "systematic_share": round(float(sys_share), 4) if sys_share == sys_share else float("nan"),
        "systematic_n": int(sys_n), "systematic_base": int(sys_base),
        "lift_on_bad_ibd": round(lift_bad, 2) if lift_bad == lift_bad else float("nan"),
        "n_sell_lt5_pct": round(100 * float((n_sell < 5).mean()), 2),
        "n_sell_lt10_pct": round(100 * float((n_sell < 10).mean()), 2),
        "peer_level_na_pct": round(100 * float((detail["peer_level"] == "na").mean()), 2) if "peer_level" in detail.columns else 0.0,
        "soft_kept_pct": round(100 * float(soft.mean()), 2),
        "final_ok_false_pct": round(100 * float(bad_mask.mean()), 2),
        "adjbox_shrink_pct": round(100 * float((detail["adjbox_basis"] == "shrink").mean()), 2) if "adjbox_basis" in detail.columns else 0.0,
        "fova_longitude": dist(detail["fova_longitude"]),
        "fova_trend": dist(detail["fova_trend"]),
        "fova_final": dist(detail["fova_final"]),
        "adjbox_value": dist(detail["adjbox_value"]),
    }
    if "level" in detail.columns:
        row["level"] = detail["level"].value_counts().to_dict()

    # IBD quality from xlsx sheets if present
    ibd_q = {"arm": arm, "period": period}
    xlsx = res.get("xlsx")
    if xlsx and Path(xlsx).exists():
        try:
            ibd_map = pd.read_excel(xlsx, sheet_name="ibd_map")
            check = pd.read_excel(xlsx, sheet_name="ibd_check")
            ibd_q["n_ibd"] = int(ibd_map["ibd_id"].nunique()) if "ibd_id" in ibd_map.columns else 0
            if "pooling_reason" in ibd_map.columns:
                ibd_q["forced_nearest"] = int((ibd_map["pooling_reason"] == "forced_nearest").sum())
                ibd_q["soft_kept_prov"] = int(ibd_map["soft_kept"].astype(bool).sum()) if "soft_kept" in ibd_map.columns else 0
            if "final_ok" in ibd_map.columns:
                ibd_q["recheck_fail"] = int((~ibd_map["final_ok"].astype(bool)).sum())
            if "final_med_ratio" in ibd_map.columns:
                ratio = pd.to_numeric(ibd_map["final_med_ratio"], errors="coerce")
                ibd_q["med_ratio_gt2"] = int(((ratio > 2) | (ratio < 0.5)).sum())
            if "item" in check.columns and "n" in check.columns:
                pass
        except Exception as exc:  # noqa: BLE001
            ibd_q["ibd_error"] = str(exc)

    slices = []
    td_n = pd.to_numeric(detail.get("td_n", np.nan), errors="coerce").fillna(0)
    n_hist = pd.to_numeric(detail.get("n_hist", np.nan), errors="coerce").fillna(0)
    n_peer = pd.to_numeric(detail.get("n_peer", np.nan), errors="coerce").fillna(0)

    def add_slice(name: str, mask: pd.Series):
        if not mask.any():
            return
        sub = detail.loc[mask]
        fl = sub["fova_final"].isin(["high", "low"])
        slices.append({
            "arm": arm, "period": period, "slice": name, "n_rows": int(mask.sum()),
            "flag_rate_pct": round(100 * float(fl.mean()), 3),
            "province_lift_max": round(province_lift_max(sub, fl), 2),
            "lon_available_pct": round(100 * float((sub["fova_longitude"] != NA).mean()), 2),
            "trend_available_pct": round(100 * float((sub["fova_trend"] != NA).mean()), 2),
        })

    add_slice("td_n<10", td_n < 10)
    add_slice("td_n>=10", td_n >= 10)
    add_slice("n_hist<3", n_hist < 3)
    add_slice("n_hist>=3", n_hist >= 3)
    add_slice("soft_or_bad", soft | bad_mask)
    add_slice("good_ibd", ~(soft | bad_mask))
    add_slice("n_peer<5", n_peer < 5)
    add_slice("n_peer_5_9", (n_peer >= 5) & (n_peer < 10))
    add_slice("n_peer>=10", n_peer >= 10)
    return row, slices, ibd_q


def stability_jaccard(arm_rows: list[dict], details: dict[tuple[str, int], pd.DataFrame]) -> list[dict]:
    """Pairwise Jaccard of final flags for consecutive periods within an arm."""
    out = []
    for arm in {a for a, _ in details}:
        pers = sorted(p for a, p in details if a == arm)
        for a, b in zip(pers, pers[1:]):
            d0, d1 = details[(arm, a)], details[(arm, b)]
            k0 = d0.assign(_f=d0["fova_final"].isin(["high", "low"])).set_index(["store_id", "category"])["_f"]
            k1 = d1.assign(_f=d1["fova_final"].isin(["high", "low"])).set_index(["store_id", "category"])["_f"]
            both = k0.index.intersection(k1.index)
            if len(both) == 0:
                jac = float("nan")
            else:
                s0, s1 = k0.loc[both].to_numpy(), k1.loc[both].to_numpy()
                inter = int((s0 & s1).sum())
                union = int((s0 | s1).sum())
                jac = inter / union if union else float("nan")
            out.append({"arm": arm, "period_a": a, "period_b": b, "stability_jaccard": round(jac, 4) if jac == jac else float("nan"),
                        "n_common": int(len(both))})
    return out


def decide(compare: pd.DataFrame, stab: pd.DataFrame) -> pd.DataFrame:
    """Heuristic decision notes from the plan rules (for the decision sheet)."""
    notes = []

    def mean_of(arm, col):
        s = compare.loc[compare["arm"] == arm, col]
        return float(s.mean()) if len(s) else float("nan")

    def stab_mean(arm):
        s = stab.loc[stab["arm"] == arm, "stability_jaccard"]
        return float(s.mean()) if len(s) else float("nan")

    if {"A0", "A1"} <= set(compare["arm"]):
        thin0 = compare.loc[compare["arm"] == "A0", "fence_src"].apply(lambda d: d.get("thin", 0) if isinstance(d, dict) else 0).mean()
        thin1 = compare.loc[compare["arm"] == "A1", "fence_src"].apply(lambda d: d.get("thin", 0) if isinstance(d, dict) else 0).mean()
        notes.append({"rule": "T1", "pass": thin1 < thin0 and mean_of("A1", "province_lift_max") <= mean_of("A0", "province_lift_max") * 1.15,
                      "detail": f"thin rows A0={thin0:.0f} A1={thin1:.0f}; lift A0={mean_of('A0','province_lift_max'):.2f} A1={mean_of('A1','province_lift_max'):.2f}"})
    if {"A0", "A2"} <= set(compare["arm"]):
        notes.append({"rule": "T3", "pass": mean_of("A2", "lon_available_pct") > mean_of("A0", "lon_available_pct") + 20,
                      "detail": f"lon_avail A0={mean_of('A0','lon_available_pct'):.1f}% A2={mean_of('A2','lon_available_pct'):.1f}%"})
    if {"A0", "A3"} <= set(compare["arm"]):
        notes.append({"rule": "A3_default", "pass": mean_of("A3", "lon_available_pct") > 50,
                      "detail": f"A3 lon={mean_of('A3','lon_available_pct'):.1f}% lift={mean_of('A3','province_lift_max'):.2f} stab={stab_mean('A3'):.3f}"})
    if {"A0", "B0"} <= set(compare["arm"]):
        notes.append({"rule": "IBD_B0", "pass": True,
                      "detail": f"B0 forced check via ibd_quality; lift A0={mean_of('A0','province_lift_max'):.2f} B0={mean_of('B0','province_lift_max'):.2f}"})
    if {"A3", "B3"} <= set(compare["arm"]):
        ok = mean_of("B3", "systematic_share") <= mean_of("A3", "systematic_share") * 1.1
        notes.append({"rule": "IBD_B3", "pass": ok,
                      "detail": f"sys A3={mean_of('A3','systematic_share'):.4f} B3={mean_of('B3','systematic_share'):.4f}; "
                                f"lift A3={mean_of('A3','province_lift_max'):.2f} B3={mean_of('B3','province_lift_max'):.2f}"})
    if {"B3", "C3"} <= set(compare["arm"]):
        ok = mean_of("C3", "systematic_share") <= mean_of("B3", "systematic_share") * 1.05
        notes.append({"rule": "C3_adjbox_shrink", "pass": ok,
                      "detail": f"sys B3={mean_of('B3','systematic_share'):.4f} C3={mean_of('C3','systematic_share'):.4f}; "
                                f"shrink_pct={mean_of('C3','adjbox_shrink_pct'):.1f}"})
    return pd.DataFrame(notes)


def run_arms(arms: list[str]) -> None:
    compare_rows, slice_rows, ibd_rows = [], [], []
    details: dict[tuple[str, int], pd.DataFrame] = {}
    settings = []
    for arm in arms:
        cfg = config_for(arm)
        settings.append({"arm": arm, "params": json.dumps(ARMS[arm], ensure_ascii=False)})
        for period in PERIODS:
            log(f"run arm={arm} period={period}")
            t0 = time.time()
            res = peer_main.run(cfg, period)
            row, slices, ibd_q = metrics_from_result(res, period, arm)
            row["elapsed_s"] = round(time.time() - t0, 1)
            compare_rows.append(row)
            slice_rows.extend(slices)
            ibd_rows.append(ibd_q)
            details[(arm, period)] = res["graded"][["store_id", "category", "fova_final", "PROVINCE"]].copy()
            log(f"  -> rows={row['n_rows']} lon={row['lon_available_pct']}% flag={row['flag_rows']} "
                f"lift={row['province_lift_max']} ({row['elapsed_s']}s)")
        cfg.unlink(missing_ok=True)

    compare = pd.DataFrame(compare_rows)
    slices_df = pd.DataFrame(slice_rows)
    ibd_df = pd.DataFrame(ibd_rows)
    stab = pd.DataFrame(stability_jaccard(compare_rows, details))
    decision = decide(compare, stab)

    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        compare.to_excel(w, sheet_name="compare", index=False)
        slices_df.to_excel(w, sheet_name="by_slice", index=False)
        ibd_df.to_excel(w, sheet_name="ibd_quality", index=False)
        stab.to_excel(w, sheet_name="stability", index=False)
        pd.DataFrame(settings).to_excel(w, sheet_name="settings", index=False)
        decision.to_excel(w, sheet_name="decision", index=False)
    log(f"wrote {OUT_XLSX}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    show = ["arm", "period", "lon_available_pct", "flag_rows", "flag_rate_pct", "province_lift_max",
            "systematic_share", "n_sell_lt10_pct", "soft_kept_pct", "adjbox_shrink_pct"]
    print("\n===== compare =====")
    print(compare[[c for c in show if c in compare.columns]].to_string(index=False))
    print("\n===== decision =====")
    print(decision.to_string(index=False) if len(decision) else "(no rules matched)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Production experiment A/B/C arms")
    p.add_argument("--phase", type=int, choices=(1, 2, 3), default=None)
    p.add_argument("--arms", nargs="+", default=None, help="explicit arm names")
    args = p.parse_args(argv)
    if args.arms:
        arms = args.arms
    elif args.phase:
        arms = list(PHASE_ARMS[args.phase])
    else:
        arms = list(ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}; choose from {list(ARMS)}")
    log(f"arms={arms} periods={PERIODS}")
    run_arms(arms)


if __name__ == "__main__":
    main()
