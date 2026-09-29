from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLS = ("period_id", "province", "category", "sales_value")
METHOD_COLS = (
    ("n0", "outlier_n0"),
    ("n1_5", "outlier_n1_5"),
    ("n6_11", "outlier_n6_11"),
    ("n8", "outlier_n8"),
)


def read_province_cat(path) -> pd.DataFrame:
    df = pd.read_excel(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing} in {path}")
    out = df[list(REQUIRED_COLS)].copy()
    out["period_id"] = out["period_id"].astype(int)
    out["province"] = out["province"].astype(str)
    out["category"] = out["category"].astype(str)
    out["sales_value"] = out["sales_value"].astype(float)
    return out


def summarize_methods(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Counts and n0/n1-5/n6-11 vs n8 overlaps on rows that have method labels."""
    work = df.copy()
    if "outlier_n0" in work.columns:
        work = work[work["outlier_n0"].notna()]
    counts = []
    for name, col in METHOD_COLS:
        series = work[col] if col in work.columns else pd.Series(dtype=object)
        vc = series.value_counts()
        counts.append(
            {
                "method": name,
                "high": int(vc.get("high", 0)),
                "low": int(vc.get("low", 0)),
                "ok": int(vc.get("ok", 0)),
                "na": int(vc.get("na", 0)),
            }
        )
    n8 = work["outlier_n8"] if "outlier_n8" in work.columns else pd.Series(dtype=object)
    overlaps = []
    for name, col in METHOD_COLS[:-1]:
        other = work[col] if col in work.columns else pd.Series(dtype=object)
        overlaps.append(
            {
                "pair": f"{name}_and_n8",
                "high_overlap": int(((other == "high") & (n8 == "high")).sum()),
                "low_overlap": int(((other == "low") & (n8 == "low")).sum()),
            }
        )
    return pd.DataFrame(counts), pd.DataFrame(overlaps)


def write_outlier_xlsx(df: pd.DataFrame, path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    counts, overlaps = summarize_methods(df)
    with pd.ExcelWriter(dest, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="detail", index=False)
        counts.to_excel(writer, sheet_name="summary", index=False, startrow=1)
        overlaps.to_excel(writer, sheet_name="summary", index=False, startrow=len(counts) + 4)
        sheet = writer.sheets["summary"]
        sheet["A1"] = "high_low_count"
        sheet["A" + str(len(counts) + 4)] = "overlap_with_n8"
    return dest
