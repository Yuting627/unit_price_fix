from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from funcs.compare_mix import compare_dwh_platform_mix

DEFAULT_AUG = ROOT / "data" / "province_shoptype_sales_20261408.xlsx"
DEFAULT_JULY = ROOT / "data" / "province_cat_salesvalue.xlsx"
DEFAULT_OUTPUT = ROOT / "data" / "dwh_platform_vs_province_cat_mix.xlsx"
DEFAULT_JULY_PERIOD = 20261407


def run(
    aug_path=DEFAULT_AUG,
    july_path=DEFAULT_JULY,
    output_path=DEFAULT_OUTPUT,
    july_period: int = DEFAULT_JULY_PERIOD,
) -> Path:
    aug = pd.read_excel(aug_path)
    july = pd.read_excel(july_path)
    by_platform, by_province = compare_dwh_platform_mix(aug, july, july_period=july_period)
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(dest, engine="openpyxl") as writer:
        for platform, part in by_platform.groupby("platform", sort=True):
            part.to_excel(writer, sheet_name=str(platform)[:31], index=False)
        by_province.to_excel(writer, sheet_name="by_province", index=False)
    return dest


if __name__ == "__main__":
    dest = run()
    print(dest)
    book = pd.read_excel(dest, sheet_name=None)
    print("sheets", list(book))
    for name, df in book.items():
        if name == "by_province":
            continue
        print(name, "rows", len(df), "share_dwh", float(df["share_dwh"].sum()))
        print(df.head(5)[["category", "share_dwh", "share_july", "pp"]].to_string(index=False))
