from __future__ import annotations

import numpy as np
import pandas as pd

CLR_EPS = 1e-6
DEFAULT_K = 8
EXPAND_K = 12
MIN_PEERS = 6


def _as_share_vector(mix, categories=None) -> np.ndarray:
    if isinstance(mix, dict):
        keys = list(categories) if categories is not None else list(mix.keys())
        return np.asarray([float(mix.get(k, 0.0)) for k in keys], dtype=float)
    return np.asarray(list(mix), dtype=float)


def clr_coords(shares, eps: float = CLR_EPS) -> np.ndarray:
    s = np.asarray(shares, dtype=float)
    s = np.clip(s, 0.0, None) + float(eps)
    total = float(s.sum())
    if total <= 0.0:
        s = np.full(s.shape, 1.0 / max(s.size, 1))
    else:
        s = s / total
    logs = np.log(s)
    return logs - float(logs.mean())


def aitchison_distance(left, right, drop_category: str | None = None, eps: float = CLR_EPS) -> float:
    """Euclidean distance of clr coordinates; optionally drop one category first."""
    if isinstance(left, dict) and isinstance(right, dict):
        cats = list(dict.fromkeys(list(left.keys()) + list(right.keys())))
        if drop_category is not None:
            cats = [c for c in cats if c != drop_category]
        a = _as_share_vector(left, cats)
        b = _as_share_vector(right, cats)
    else:
        a = _as_share_vector(left)
        b = _as_share_vector(right)
    return float(np.linalg.norm(clr_coords(a, eps=eps) - clr_coords(b, eps=eps)))


def share_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Province × category share matrix; missing cells are 0."""
    return (
        df.pivot_table(index="province", columns="category", values="share", aggfunc="sum", fill_value=0.0)
        .sort_index()
        .sort_index(axis=1)
    )


def _loo_row(row: pd.Series, category: str) -> np.ndarray:
    others = [c for c in row.index if c != category]
    return row.reindex(others, fill_value=0.0).to_numpy(dtype=float)


def nearest_peer_names(
    wide: pd.DataFrame,
    province: str,
    category: str,
    k: int = DEFAULT_K,
    min_peers: int = MIN_PEERS,
    expand_k: int = EXPAND_K,
) -> list[str]:
    """Nearest provinces by leave-one-category-out Aitchison distance."""
    others = [p for p in wide.index if p != province]
    if province not in wide.index or not others:
        return others
    own = _loo_row(wide.loc[province], category)
    dist = []
    for other in others:
        d = aitchison_distance(own, _loo_row(wide.loc[other], category))
        dist.append((d, str(other)))
    dist.sort(key=lambda x: (x[0], x[1]))
    take = min(int(k), len(dist))
    names = [name for _d, name in dist[:take]]
    if len(names) < int(min_peers):
        take = min(int(expand_k), len(dist))
        names = [name for _d, name in dist[:take]]
    if len(names) < int(min_peers):
        names = [name for _d, name in dist]
    return names


def nearest_peer_shares(
    wide: pd.DataFrame,
    province: str,
    category: str,
    k: int = DEFAULT_K,
    min_peers: int = MIN_PEERS,
    expand_k: int = EXPAND_K,
) -> tuple[list[str], list[float]]:
    """Peer category shares for Tukey: clr neighbors, then fallback to all others."""
    names = nearest_peer_names(
        wide, province, category, k=k, min_peers=min_peers, expand_k=expand_k
    )
    if category not in wide.columns:
        return names, [0.0] * len(names)
    shares = [float(wide.at[name, category]) for name in names]
    return names, shares


def peer_field_for_names(items, peer_names, category: str, field: str) -> list[float]:
    """Collect a numeric field from clr-neighbor rows of the same category."""
    allowed = set(peer_names)
    out = []
    for item in items:
        if item.get("category") != category or item.get("province") not in allowed:
            continue
        val = item.get(field)
        if val is None:
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(num):
            out.append(num)
    return out
