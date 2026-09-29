from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from core.classify import (
    HOLT_MIN_HIST,
    N17_WINDOW,
    cap_n0_by_province,
    identify_n0,
    identify_n1_5,
    identify_n6_11,
    identify_n8,
)
from core.composition import nearest_peer_shares, peer_field_for_names, share_wide
from core.share_change import short_term_share_change
from funcs.history import hist_share_series, split_current_hist
from funcs.io import read_province_cat, summarize_methods, write_outlier_xlsx
from funcs.merge import attach_july_marks
from funcs.share import add_province_weight, add_short_term_share_change, to_category_share

DEFAULT_INPUT = ROOT / "data" / "province_cat_salesvalue.xlsx"
DEFAULT_OUTPUT = ROOT / "data" / "province_cat_salesvalue_outlier.xlsx"
DEFAULT_PERIOD = 20261407


def _july_share(current: pd.DataFrame, province: str, category: str) -> tuple[float, float]:
    row = current[(current["province"] == province) & (current["category"] == category)]
    if row.empty:
        return 0.0, 0.0
    return float(row["share"].iloc[0]), float(row["sales_value"].iloc[0])


def score_all_keys(df: pd.DataFrame, current_period: int) -> pd.DataFrame:
    work = to_category_share(df)
    current, hist = split_current_hist(work, current_period)
    wide = share_wide(current)
    keys = sorted(set(zip(work["province"], work["category"])))
    window = N17_WINDOW

    prepared = []
    for province, category in keys:
        observed = hist_share_series(hist, province, category, fill_calendar=False)
        holt_hist = (
            hist_share_series(hist, province, category, fill_calendar=True)
            if len(observed) >= HOLT_MIN_HIST
            else observed
        )
        share_jul, sales_jul = _july_share(current, province, category)
        change_pp = float("nan")
        change_log = float("nan")
        if observed:
            chg = short_term_share_change(share_jul, observed, window=window)
            change_pp = chg["pp"]
            change_log = chg["log_g"]
        prepared.append(
            {
                "province": province,
                "category": category,
                "observed": observed,
                "holt_hist": holt_hist,
                "share_jul": share_jul,
                "sales_jul": sales_jul,
                "change_pp": change_pp,
                "change_log": change_log,
            }
        )

    rows = []
    for item in prepared:
        province = item["province"]
        category = item["category"]
        peer_names, peer_shares = nearest_peer_shares(wide, province, category)
        peer_changes = peer_field_for_names(prepared, peer_names, category, "change_log")
        peer_pps = peer_field_for_names(prepared, peer_names, category, "change_pp")
        prev_share = item["observed"][-1] if item["observed"] else 0.0
        pw_rows = current.loc[current["province"] == province, "province_weight"]
        province_weight = float(pw_rows.iloc[0]) if not pw_rows.empty else 0.0
        ns_rows = current.loc[
            (current["province"] == province) & (current["category"] == category),
            "national_share_loo",
        ]
        if ns_rows.empty:
            ns_rows = current.loc[current["category"] == category, "national_share_loo"]
        national_share = float(ns_rows.iloc[0]) if not ns_rows.empty else None
        n0 = identify_n0(
            item["share_jul"],
            peer_shares,
            national_share=national_share,
            province_weight=province_weight,
        )
        n15 = identify_n1_5(
            item["share_jul"],
            item["observed"],
            peer_changes,
            peer_pps=peer_pps,
            prev_share=prev_share,
            province_weight=province_weight,
        )
        n611 = identify_n6_11(
            item["share_jul"],
            item["observed"],
            prev_share=prev_share,
        )
        n8 = identify_n8(
            item["share_jul"],
            item["holt_hist"],
            prev_share=prev_share,
        )
        labels = [n0["label"], n15["label"], n611["label"], n8["label"]]
        comparable = [x for x in labels if x != "na"]
        note = ""
        if not item["observed"]:
            note = "zero_new" if item["sales_jul"] == 0.0 else "no_hist"
        if n0.get("note"):
            note = f"{note};{n0['note']}" if note else n0["note"]
        if n15.get("note"):
            note = f"{note};{n15['note']}" if note else n15["note"]
        rows.append(
            {
                "n_hist": len(item["observed"]),
                "share": item["share_jul"],
                "outlier_n0": n0["label"],
                "outlier_n1_5": n15["label"],
                "outlier_n6_11": n611["label"],
                "outlier_n8": n8["label"],
                "agree": len(set(comparable)) <= 1 if comparable else True,
                "n_methods": len(comparable),
                "note": note,
                "yhat_n0": n0["yhat_share"],
                "yhat_n1_5": n15["yhat_share"],
                "yhat_n6_11": n611["yhat_share"],
                "yhat_n8": n8["yhat_share"],
                "province": province,
                "category": category,
                "period_id": current_period,
                "national_share": national_share if national_share is not None else float("nan"),
            }
        )
    return cap_n0_by_province(pd.DataFrame(rows))


def run(
    input_path=DEFAULT_INPUT,
    output_path=DEFAULT_OUTPUT,
    current_period: int = DEFAULT_PERIOD,
) -> Path:
    original = read_province_cat(input_path)
    with_share = to_category_share(original)
    marks = score_all_keys(original, current_period)
    out = attach_july_marks(with_share, marks, current_period)
    out = add_province_weight(out)
    out = add_short_term_share_change(out)
    return write_outlier_xlsx(out, output_path)


if __name__ == "__main__":
    dest = run()
    print(dest)
    marked = pd.read_excel(dest, sheet_name="detail")
    counts, overlaps = summarize_methods(marked)
    print(counts.to_string(index=False))
    print(overlaps.to_string(index=False))
    july = marked[marked["period_id"] == DEFAULT_PERIOD]
    print(july[["outlier_n0", "outlier_n1_5", "outlier_n6_11", "outlier_n8"]].apply(pd.Series.value_counts).fillna(0).astype(int))
    comparable = july[july["n_methods"] >= 2]
    agree_flag = july["agree"].astype(bool)
    print("july rows", len(july))
    print("agree", int(agree_flag.sum()), "/", len(july))
    print("disagree", int((~agree_flag).sum()))
    print("compared at least 2 methods", len(comparable), "agree", int(comparable["agree"].astype(bool).sum()))
