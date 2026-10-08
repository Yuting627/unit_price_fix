"""Distribution profile of store x category x period data."""
from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE_DIMS = ("PLATFORMNAME", "SHOPTYPE", "PROVINCE", "category")


def _stats(g: pd.DataFrame, value: str) -> dict:
    x = g[value].to_numpy(dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {}
    q = np.quantile(x, [0.25, 0.5, 0.75, 0.9, 0.99])
    mean = x.mean()
    return {
        "n_store": g["store_id"].nunique(),
        "n_item_median": float(g["n_item"].median()) if "n_item" in g else np.nan,
        "n_obs": len(g),
        "zero_rate": float((x == 0).mean()),
        "median": q[1],
        "iqr": q[2] - q[0],
        "p90": q[3],
        "p99": q[4],
        "cv": float(x.std(ddof=1) / mean) if x.size > 1 and mean != 0 else np.nan,
    }


def profile(sc: pd.DataFrame, dims=PROFILE_DIMS, value: str = "sales_value", periods=None) -> pd.DataFrame:
    """One row per (period_id, dimension, level)."""
    data = sc if periods is None else sc[sc["period_id"].isin(periods)]
    rows = []
    for dim in dims:
        for (period, level), g in data.groupby(["period_id", dim], observed=True):
            stats = _stats(g, value)
            if stats:
                rows.append({"period_id": period, "dimension": dim, "level": level, **stats})
    return pd.DataFrame(rows)
