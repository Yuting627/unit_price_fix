from __future__ import annotations

import pandas as pd
import pytest

from core.data_prep.data_prep import (
    agg_province_shoptype,
    agg_store_category,
    list_before_mp_files,
    merge_by_period,
    read_before_mp,
    read_store_target,
    write_excel,
)


def _target(period=20261408):
    return pd.DataFrame(
        {
            "PERIODCODE": [period] * 3,
            "STOREID": ["A", "B", "C"],
            "PROVINCE": ["广东省", "广东省", "贵州省"],
            "SHOPTYPE": ["cvs", "cvs", "hmkt"],
            "PLATFORMNAME": ["ELEME", "ELEME", "MEITUAN"],
        }
    )


def test_merge_by_period_inner_join_and_unmatched():
    before = pd.DataFrame(
        {
            "period_id": [20261408, 20261408, 20261408, 20261408],
            "store_id": ["A", "A", "B", "Z"],
            "category": ["BEER"] * 4,
            "sales_value": [10.0, 5.0, 20.0, 1.0],
            "sales_unit": [2.0, 1.0, 4.0, 1.0],
        }
    )
    merged, unmatched = merge_by_period(before, _target())
    assert set(merged["store_id"]) == {"A", "B"}
    row = unmatched.iloc[0]
    assert (row["matched_stores"], row["bmp_only_stores"], row["target_only_stores"]) == (2, 1, 1)


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
    merged, _ = merge_by_period(before, _target())
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


def test_merge_by_period_uses_period_specific_shoptype():
    before = pd.DataFrame(
        {
            "period_id": [20261407, 20261408],
            "store_id": ["A", "A"],
            "category": ["BEER", "BEER"],
            "sales_value": [1.0, 2.0],
            "sales_unit": [1.0, 1.0],
        }
    )
    t7 = _target(20261407)
    t8 = _target(20261408)
    t8.loc[t8["STOREID"] == "A", "SHOPTYPE"] = "smkt"
    merged, _ = merge_by_period(before, pd.concat([t7, t8]))
    got = dict(zip(merged["period_id"], merged["SHOPTYPE"]))
    assert got == {20261407: "cvs", 20261408: "smkt"}


def test_read_before_mp_keeps_prod_id_and_desc(tmp_path):
    path = tmp_path / "O2O_itemcoding_output_20261408_newline.csv"
    pd.DataFrame(
        {
            "period_id": [20261408],
            "store_id": ["A"],
            "prod_id": [123],
            "prod_desc_raw": ["可乐 330ml"],
            "category": ["CSD"],
            "sales_value": [3.0],
            "sales_unit": [1.0],
            "nankey": [9],
        }
    ).to_csv(path, index=False, encoding="gb18030")
    df = read_before_mp(path)
    assert int(df.loc[0, "prod_id"]) == 123
    assert df.loc[0, "prod_desc_raw"] == "可乐 330ml"
    assert "prod_desc_raw" not in read_before_mp(path, keep_desc=False).columns


def test_list_before_mp_files_prefers_fixed(tmp_path):
    for name in (
        "O2O_itemcoding_output_20261408_newline.csv.gz",
        "O2O_itemcoding_output_20261408_newline_fixed.csv.gz",
        "O2O_itemcoding_output_20261407_newline.csv.gz",
    ):
        (tmp_path / name).write_bytes(b"")
    files = list_before_mp_files(tmp_path)
    assert list(files) == [20261407, 20261408]
    assert files[20261408].name.endswith("_fixed.csv.gz")


def test_read_store_target_dedups_by_period_and_store(tmp_path):
    path = tmp_path / "target_shop_qc_20261408.csv"
    df = pd.concat([_target(), _target().iloc[[0]]])
    df.to_csv(path, index=False, encoding="gb18030")
    out = read_store_target(path)
    assert len(out) == 3
    assert out["PERIODCODE"].dtype.kind == "i"


def test_agg_store_category_metrics():
    df = pd.DataFrame(
        {
            "period_id": [1, 1, 1, 1],
            "store_id": ["A", "A", "A", "A"],
            "category": ["BEER", "BEER", "BEER", "CSD"],
            "prod_id": [1, 2, 2, 3],
            "sales_value": [10.0, 20.0, 10.0, 60.0],
            "sales_unit": [1.0, 2.0, 1.0, 0.0],
            "PLATFORMNAME": ["ELEME"] * 4,
            "SHOPTYPE": ["cvs"] * 4,
            "PROVINCE": ["广东省"] * 4,
        }
    )
    out = agg_store_category(df, item_key="prod_id").set_index("category")
    assert out.loc["BEER", "n_item"] == 2
    assert out.loc["BEER", "avg_price"] == pytest.approx(10.0)
    assert out.loc["BEER", "cat_share"] == pytest.approx(0.4)
    assert pd.isna(out.loc["CSD", "avg_price"])


def test_write_excel_multi_sheet(tmp_path):
    dest = write_excel({"a": pd.DataFrame({"x": [1]}), "b": pd.DataFrame({"y": [2]})}, tmp_path / "o.xlsx")
    sheets = pd.read_excel(dest, sheet_name=None)
    assert set(sheets) == {"a", "b"}
