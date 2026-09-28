from __future__ import annotations

import pandas as pd

from funcs.history import count_hist, hist_share_series, split_current_hist
from funcs.io import summarize_methods, write_outlier_xlsx
from funcs.merge import attach_july_marks
from funcs.share import (
    add_national_category_share,
    add_national_share_loo,
    add_short_term_share_change,
    to_category_share,
)
from funcs.compare_mix import category_share, compare_dwh_platform_mix
from funcs.store_mp import agg_province_shoptype, merge_before_mp_target


def _sample_df() -> pd.DataFrame:
    rows = []
    for period in (20261405, 20261406, 20261407):
        rows.append(
            {
                "period_id": period,
                "province": "广东省",
                "category": "BEER",
                "sales_value": 80.0 if period < 20261407 else 80.0,
            }
        )
        rows.append(
            {
                "period_id": period,
                "province": "广东省",
                "category": "BIS",
                "sales_value": 20.0,
            }
        )
        rows.append(
            {
                "period_id": period,
                "province": "贵州省",
                "category": "BEER",
                "sales_value": 20.0,
            }
        )
    rows.append(
        {
            "period_id": 20261406,
            "province": "上海市",
            "category": "BEER",
            "sales_value": 5.0,
        }
    )
    return pd.DataFrame(rows)


def test_agg_province_shoptype_sums_and_unique_stores():
    before = pd.DataFrame(
        {
            "period_id": [20261408, 20261408, 20261408],
            "store_id": ["A", "A", "B"],
            "category": ["BEER", "BEER", "BEER"],
            "sales_value": [10.0, 5.0, 20.0],
            "sales_unit": [2.0, 1.0, 4.0],
        }
    )
    target = pd.DataFrame(
        {
            "STOREID": ["A", "B", "C"],
            "PROVINCE": ["广东省", "广东省", "贵州省"],
            "SHOPTYPE": ["cvs", "cvs", "hmkt"],
            "PLATFORMNAME": ["ELEME", "ELEME", "MEITUAN"],
        }
    )
    merged = merge_before_mp_target(before, target)
    out = agg_province_shoptype(merged)
    row = out[
        (out["province"] == "广东省")
        & (out["shoptype"] == "cvs")
        & (out["category"] == "BEER")
        & (out["platform"] == "ELEME")
    ].iloc[0]
    assert float(row["sales_value"]) == 35.0
    assert float(row["sales_unit"]) == 7.0
    assert int(row["store_count"]) == 2
    assert "贵州省" not in set(out["province"])


def test_dwh_platform_mix_share_sums_and_pp():
    aug = pd.DataFrame(
        {
            "shoptype": ["dwh", "dwh", "dwh", "cvs"],
            "province": ["广东省", "广东省", "上海市", "广东省"],
            "category": ["BEER", "BIS", "BEER", "BEER"],
            "platform": ["ELEME", "ELEME", "MEITUAN", "ELEME"],
            "sales_value": [80.0, 20.0, 50.0, 999.0],
        }
    )
    hist = pd.DataFrame(
        {
            "period_id": [20261407, 20261407, 20261407, 20261407],
            "province": ["广东省", "广东省", "贵州省", "贵州省"],
            "category": ["BEER", "BIS", "BEER", "BIS"],
            "sales_value": [60.0, 40.0, 90.0, 10.0],
        }
    )
    grouped = category_share(aug[aug["shoptype"] == "dwh"], ["platform", "category"])
    eleme = grouped[grouped["platform"] == "ELEME"]
    assert abs(float(eleme["share"].sum()) - 1.0) < 1e-12

    by_platform, by_province = compare_dwh_platform_mix(aug, hist, july_period=20261407)
    eleme = by_platform[by_platform["platform"] == "ELEME"]
    meituan = by_platform[by_platform["platform"] == "MEITUAN"]
    assert abs(float(eleme["share_dwh"].sum()) - 1.0) < 1e-12
    assert abs(float(meituan["share_dwh"].sum()) - 1.0) < 1e-12
    beer_e = eleme[eleme["category"] == "BEER"].iloc[0]
    beer_m = meituan[meituan["category"] == "BEER"].iloc[0]
    assert abs(float(beer_e["share_dwh"]) - 0.8) < 1e-12
    assert abs(float(beer_e["share_july"]) - 0.75) < 1e-12
    assert abs(float(beer_e["pp"]) - 0.05) < 1e-12
    assert abs(float(beer_m["share_dwh"]) - 1.0) < 1e-12
    assert abs(float(beer_m["pp"]) - 0.25) < 1e-12
    gd = by_province[
        (by_province["province"] == "广东省")
        & (by_province["category"] == "BEER")
        & (by_province["platform"] == "ELEME")
    ].iloc[0]
    assert abs(float(gd["share_dwh"]) - 0.8) < 1e-12
    assert abs(float(gd["share_july"]) - 0.6) < 1e-12
    assert abs(float(gd["pp"]) - 0.2) < 1e-12


def test_share_is_category_over_province_total():
    df = to_category_share(_sample_df())
    grouped = df.groupby(["period_id", "province"])["share"].sum()
    assert ((grouped - 1.0).abs() < 1e-12).all()
    gd_beer = df[(df["province"] == "广东省") & (df["category"] == "BEER")].iloc[0]
    assert abs(float(gd_beer["share"]) - 0.8) < 1e-12
    period = int(gd_beer["period_id"])
    nat = float(df.loc[df["period_id"] == period, "sales_value"].sum())
    prov = float(df.loc[(df["period_id"] == period) & (df["province"] == "广东省"), "sales_value"].sum())
    assert abs(float(gd_beer["province_weight"]) - prov / nat) < 1e-12


def test_national_share_loo_excludes_this_province():
    raw = pd.DataFrame(
        {
            "period_id": [20261407] * 4,
            "province": ["广东省", "广东省", "贵州省", "贵州省"],
            "category": ["BEER", "BIS", "BEER", "BIS"],
            "sales_value": [80.0, 20.0, 20.0, 80.0],
        }
    )
    df = add_national_share_loo(to_category_share(raw))
    gd_beer = df[(df["province"] == "广东省") & (df["category"] == "BEER")].iloc[0]
    gz_beer = df[(df["province"] == "贵州省") & (df["category"] == "BEER")].iloc[0]
    assert abs(float(gd_beer["national_share"]) - 0.5) < 1e-12
    assert abs(float(gd_beer["national_share_loo"]) - 0.2) < 1e-12
    assert abs(float(gz_beer["national_share_loo"]) - 0.8) < 1e-12
    assert float(gd_beer["national_share_loo"]) != float(gd_beer["national_share"])


def test_national_share_is_category_over_national_total():
    df = add_national_category_share(to_category_share(_sample_df()))
    period = 20261407
    slice_ = df[df["period_id"] == period]
    beer_sales = float(slice_.loc[slice_["category"] == "BEER", "sales_value"].sum())
    nat = float(slice_["sales_value"].sum())
    beer = slice_[slice_["category"] == "BEER"]
    assert ((beer["national_share"] - beer_sales / nat).abs() < 1e-12).all()
    assert beer["national_share"].nunique() == 1


def test_short_term_share_change_on_all_periods():
    df = add_short_term_share_change(to_category_share(_sample_df()))
    gd = df[(df["province"] == "广东省") & (df["category"] == "BEER")].sort_values("period_id")
    first = gd.iloc[0]
    second = gd.iloc[1]
    assert pd.isna(first["share_chg_n1_7"])
    assert abs(float(second["share"]) - 0.8) < 1e-12
    assert abs(float(second["share_chg_n1_7"])) < 1e-12
    assert abs(float(second["share_pp_n1_7"])) < 1e-12


def test_n_hist_excludes_current():
    df = to_category_share(_sample_df())
    current, hist = split_current_hist(df, 20261407)
    assert (hist["period_id"] < 20261407).all()
    assert set(current["period_id"].unique()) == {20261407}
    assert count_hist(hist, "广东省", "BEER") == 2
    assert 20261407 not in hist["period_id"].tolist()
    series = hist_share_series(hist, "广东省", "BEER", fill_calendar=False)
    assert len(series) == 2


def test_attach_marks_keeps_history_values():
    original = to_category_share(_sample_df())
    marks = pd.DataFrame(
        [
            {
                "province": "广东省",
                "category": "BEER",
                "n_hist": 2,
                "share": 0.8,
                "outlier_n0": "ok",
                "outlier_n1_5": "ok",
                "outlier_n6_11": "na",
                "outlier_n8": "na",
                "agree": True,
                "n_methods": 2,
                "note": "",
                "yhat_n0": 0.8,
                "yhat_n1_5": 0.8,
                "yhat_n6_11": float("nan"),
                "yhat_n8": float("nan"),
            },
            {
                "province": "上海市",
                "category": "BEER",
                "n_hist": 1,
                "share": 0.0,
                "outlier_n0": "low",
                "outlier_n1_5": "low",
                "outlier_n6_11": "na",
                "outlier_n8": "na",
                "agree": True,
                "n_methods": 2,
                "note": "",
                "yhat_n0": 0.05,
                "yhat_n1_5": 0.05,
                "yhat_n6_11": float("nan"),
                "yhat_n8": float("nan"),
            },
        ]
    )
    out = attach_july_marks(original, marks, 20261407)
    hist = out[out["period_id"] < 20261407]
    raw_hist = original[original["period_id"] < 20261407]
    cols = ["period_id", "province", "category", "sales_value", "share", "province_weight"]
    assert hist[cols].reset_index(drop=True).equals(raw_hist[cols].reset_index(drop=True))
    assert hist["outlier_n0"].isna().all()
    assert hist["share"].notna().all()
    gd_hist_share = hist[(hist["province"] == "广东省") & (hist["category"] == "BEER")]["share"]
    assert (gd_hist_share - 0.8).abs().max() < 1e-12

    july_gd = out[(out["period_id"] == 20261407) & (out["province"] == "广东省")]
    assert july_gd["outlier_n0"].iloc[0] == "ok"
    assert bool(july_gd["agree"].iloc[0]) is True

    shanghai = out[(out["period_id"] == 20261407) & (out["province"] == "上海市")]
    assert len(shanghai) == 1
    assert float(shanghai["sales_value"].iloc[0]) == 0.0
    assert shanghai["outlier_n0"].iloc[0] == "low"


def test_summarize_methods_counts_and_n8_overlap():
    df = pd.DataFrame(
        {
            "period_id": [20261406, 20261407, 20261407, 20261407, 20261407],
            "outlier_n0": [None, "high", "high", "low", "ok"],
            "outlier_n1_5": [None, "ok", "ok", "low", "ok"],
            "outlier_n6_11": [None, "high", "ok", "ok", "high"],
            "outlier_n8": [None, "high", "high", "ok", "low"],
        }
    )
    counts, overlaps = summarize_methods(df)
    n0 = counts.set_index("method").loc["n0"]
    assert int(n0["high"]) == 2
    assert int(n0["low"]) == 1
    ov = overlaps.set_index("pair")
    assert int(ov.loc["n0_and_n8", "high_overlap"]) == 2
    assert int(ov.loc["n0_and_n8", "low_overlap"]) == 0
    assert int(ov.loc["n1_5_and_n8", "high_overlap"]) == 0
    assert int(ov.loc["n1_5_and_n8", "low_overlap"]) == 0
    assert int(ov.loc["n6_11_and_n8", "high_overlap"]) == 1
    assert int(ov.loc["n6_11_and_n8", "low_overlap"]) == 0


def test_write_outlier_xlsx_has_summary_sheet(tmp_path):
    df = pd.DataFrame(
        {
            "period_id": [20261407],
            "province": ["广东省"],
            "category": ["BEER"],
            "sales_value": [1.0],
            "share": [1.0],
            "outlier_n0": ["high"],
            "outlier_n1_5": ["ok"],
            "outlier_n6_11": ["ok"],
            "outlier_n8": ["high"],
        }
    )
    dest = write_outlier_xlsx(df, tmp_path / "out.xlsx")
    sheets = pd.read_excel(dest, sheet_name=None)
    assert "detail" in sheets and "summary" in sheets
    assert list(sheets["detail"]["outlier_n0"]) == ["high"]
    summary = sheets["summary"]
    assert "high" in summary.columns or summary.iloc[:, 0].notna().any()
