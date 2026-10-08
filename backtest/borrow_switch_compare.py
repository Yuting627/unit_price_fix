"""Compare the FOVA reference-borrowing switches (T1 + T3) against current production settings.

Runs the real production entry point (``peer_main.run``) for 20261406 / 20261407 / 20261408 under two
configurations and reports the detector distributions side by side:

* ``baseline``  - ``fova.hist_start_filter=true``,  ``trend_fence_fallback=false``, ``longitude_borrow_cell=false``
                  (identical to the shipped config; reproduces the "longitude 全部 na" symptom)
* ``borrow``    - ``fova.hist_start_filter=false``, ``trend_fence_fallback=true``,  ``longitude_borrow_cell=true``

Only the *reference* is borrowed in both arms; the store's own observation is never shrunk. The
cross-sectional detectors (adjbox / share / item) must stay identical between arms - that is the control.

The ``borrow`` arm writes to ``data/backtest/borrow_on`` and ``logs/borrow`` so the shipped outputs in
``data/peer_outlier`` are preserved for inspection alongside the borrow arm's own artifacts.

Usage:
    python backtest/borrow_switch_compare.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "code", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import peer_main  # noqa: E402

PERIODS = (20261406, 20261407, 20261408)
BASE_CONFIG = ROOT / "config" / "peer_params.json"
TMP_CONFIG = ROOT / "data" / "backtest" / "_params_borrow_on.json"
ON_OUT = "data/backtest/borrow_on"
ON_LOG = "logs/borrow"
NA = "na"
ARMS = {
    "baseline": dict(hist_start_filter=True, trend_fence_fallback=False, longitude_borrow_cell=False),
    "borrow": dict(hist_start_filter=False, trend_fence_fallback=True, longitude_borrow_cell=True),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def config_for(arm: str) -> Path:
    if arm == "baseline":
        return BASE_CONFIG
    raw = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    raw["fova"].update(ARMS[arm])
    raw["paths"]["output_dir"] = ON_OUT
    raw["paths"]["log_dir"] = ON_LOG
    TMP_CONFIG.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    return TMP_CONFIG


def dist(detail: pd.DataFrame, col: str) -> dict:
    c = detail[col].value_counts()
    return {k: int(c.get(k, 0)) for k in ("high", "low", "ok", NA)}


def metrics(detail: pd.DataFrame, period: int, arm: str) -> dict:
    n = len(detail)
    flagged = detail["fova_final"].isin(["high", "low"])
    n_flag = int(flagged.sum())
    share_all = detail["PROVINCE"].value_counts(normalize=True)
    share_flag = detail.loc[flagged, "PROVINCE"].value_counts(normalize=True)
    lift = (share_flag / share_all).dropna()
    top_groups = detail.loc[flagged].groupby(["peer_group_id", "category"]).size().sort_values(ascending=False)
    row = {
        "arm": arm, "period": period, "n_rows": n,
        "lon_available": int((detail["fova_longitude"] != NA).sum()),
        "lon_available_pct": round(100 * float((detail["fova_longitude"] != NA).mean()), 2) if n else 0.0,
        "n_hist_eff_median": float(pd.to_numeric(detail["n_hist_eff"], errors="coerce").median()),
        "lon_src": detail["lon_src"].value_counts().to_dict(),
        "fence_src": detail["td_fence_src"].value_counts().to_dict(),
        "trend_available_pct": round(100 * float((detail["fova_trend"] != NA).mean()), 2) if n else 0.0,
        "final_available_pct": round(100 * float((detail["fova_final"] != NA).mean()), 2) if n else 0.0,
        "flag_rows": n_flag,
        "flag_rate_pct": round(100 * n_flag / n, 3) if n else 0.0,
        "province_lift_max": round(float(lift.max()), 2) if len(lift) else float("nan"),
        "top10_group_share": round(float(top_groups.head(10).sum() / n_flag), 3) if n_flag else float("nan"),
    }
    for col in ("fova_longitude", "fova_trend", "fova_final", "adjbox_value", "share_adjbox", "item_adjbox"):
        if col in detail.columns:
            row[f"{col}__{col}"] = dist(detail, col)
    if "level" in detail.columns:
        row["level__level"] = detail["level"].value_counts().to_dict()
    return row


def main() -> None:
    rows = []
    try:
        for arm in ARMS:
            cfg = config_for(arm)
            for period in PERIODS:
                log(f"run arm={arm} period={period} config={cfg.name}")
                res = peer_main.run(cfg, period)
                rows.append(metrics(res["graded"], period, arm))
                log(f"  -> rows={rows[-1]['n_rows']} lon_avail={rows[-1]['lon_available_pct']}% "
                    f"flag={rows[-1]['flag_rows']} ({rows[-1]['flag_rate_pct']}%)")
    finally:
        TMP_CONFIG.unlink(missing_ok=True)

    cmp = pd.DataFrame(rows)
    dest = ROOT / "data" / "backtest" / "borrow_switch_compare.xlsx"
    with pd.ExcelWriter(dest, engine="openpyxl") as w:
        cmp.to_excel(w, sheet_name="compare", index=False)
        settings = pd.DataFrame([
            ("periods", ",".join(str(p) for p in PERIODS)),
            ("baseline", json.dumps(ARMS["baseline"])),
            ("borrow", json.dumps(ARMS["borrow"])),
            ("borrow_output_dir", ON_OUT),
            ("note", "T1/T3 only replace the reference (fence / longitude limits); the store observation is untouched"),
        ], columns=["key", "value"])
        settings.to_excel(w, sheet_name="settings", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)
    show = ["arm", "period", "n_rows", "lon_available_pct", "n_hist_eff_median", "trend_available_pct",
            "final_available_pct", "flag_rows", "flag_rate_pct", "province_lift_max", "top10_group_share"]
    print("\n===== 三期对比：baseline vs borrow(T1+T3) =====")
    print(cmp[show].to_string(index=False))
    print("\n===== 参考来源与 longitude 可用率 =====")
    print(cmp[["arm", "period", "fence_src", "lon_src"]].to_string(index=False))
    print("\n===== 检测器分布（对照：横截面三项应完全一致）=====")
    for _, r in cmp.iterrows():
        print(f"{r['arm']:8s} {r['period']}  adjbox={r.get('adjbox_value__adjbox_value')}  "
              f"longitude={r.get('fova_longitude__fova_longitude')}  trend={r.get('fova_trend__fova_trend')}  "
              f"final={r.get('fova_final__fova_final')}")
    log(f"wrote {dest}")


if __name__ == "__main__":
    main()
