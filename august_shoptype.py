from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from funcs.store_mp import (
    agg_province_shoptype,
    merge_before_mp_target,
    read_before_mp,
    read_store_target,
    write_province_shoptype,
)

DEFAULT_BEFORE_MP = ROOT / "data" / "before_mp" / "O2O_itemcoding_output_20261408_newline_fixed.csv.gz"
DEFAULT_TARGET = ROOT / "data" / "store_target" / "target_shop_qc_20261408.csv"
DEFAULT_OUTPUT = ROOT / "data" / "province_shoptype_sales_20261408.xlsx"
DEFAULT_PERIOD = 20261408


def run(
    before_mp_path=DEFAULT_BEFORE_MP,
    target_path=DEFAULT_TARGET,
    output_path=DEFAULT_OUTPUT,
    period_id: int = DEFAULT_PERIOD,
) -> Path:
    before_mp = read_before_mp(before_mp_path)
    before_mp = before_mp[before_mp["period_id"] == period_id]
    store_target = read_store_target(target_path)
    merged = merge_before_mp_target(before_mp, store_target)
    summary = agg_province_shoptype(merged)
    return write_province_shoptype(summary, output_path)


if __name__ == "__main__":
    dest = run()
    print(dest)
    out = pd.read_excel(dest)
    print(out.head(10).to_string(index=False))
    print("rows", len(out))
    print("sales_value", float(out["sales_value"].sum()))
    print("sales_unit", float(out["sales_unit"].sum()))
    print("cells", out.groupby(["shoptype", "province", "category", "platform"]).ngroups)
