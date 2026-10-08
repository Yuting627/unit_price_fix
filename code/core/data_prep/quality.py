"""Data quality tags on item-level before_mp rows."""
from __future__ import annotations

import numpy as np
import pandas as pd

DUP_KEYS = ("period_id", "store_id", "prod_id")


def tag_quality(bmp: pd.DataFrame) -> pd.DataFrame:
    """Boolean tag columns per row (does not drop anything)."""
    keys = [k for k in DUP_KEYS if k in bmp.columns]
    tags = pd.DataFrame(index=bmp.index)
    tags["dup_period_store_prod"] = bmp.duplicated(keys, keep="first") if len(keys) == 3 else False
    tags["missing_value"] = bmp["sales_value"].isna() | bmp["sales_unit"].isna()
    tags["negative_value"] = (bmp["sales_value"] < 0) | (bmp["sales_unit"] < 0)
    tags["zero_unit_positive_value"] = (bmp["sales_unit"] == 0) & (bmp["sales_value"] > 0)
    return tags


def check_quality(
    bmp: pd.DataFrame,
    unmatched: pd.DataFrame | None = None,
    drop: tuple[str, ...] = ("dup_period_store_prod", "negative_value"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean rows, dq summary). Only duplicate and negative rows are dropped by default."""
    tags = tag_quality(bmp)
    rows = []
    for period, idx in bmp.groupby("period_id").groups.items():
        sub_tags = tags.loc[idx]
        stores = bmp.loc[idx, "store_id"]
        rows.append({"period_id": period, "check": "rows_total", "n_rows": len(idx), "n_stores": stores.nunique(), "action": ""})
        for col in tags.columns:
            mask = sub_tags[col]
            rows.append(
                {
                    "period_id": period,
                    "check": col,
                    "n_rows": int(mask.sum()),
                    "n_stores": int(stores[mask].nunique()),
                    "action": "dropped" if col in drop else "tagged",
                }
            )
    if unmatched is not None:
        for _, u in unmatched.iterrows():
            for col in ("bmp_only_stores", "target_only_stores", "matched_stores"):
                rows.append(
                    {
                        "period_id": int(u["period_id"]),
                        "check": col,
                        "n_rows": np.nan,
                        "n_stores": int(u[col]),
                        "action": "excluded" if col != "matched_stores" else "",
                    }
                )
    dq = pd.DataFrame(rows, columns=["period_id", "check", "n_rows", "n_stores", "action"])
    drop_mask = tags[list(drop)].any(axis=1) if drop else pd.Series(False, index=bmp.index)
    return bmp.loc[~drop_mask], dq
