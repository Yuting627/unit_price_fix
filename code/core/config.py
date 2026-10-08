"""JSON parameter loader. The JSON file is the single source of business thresholds."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

NUM = (int, float)

SCHEMA: dict[str, Any] = {
    "paths": {"before_mp_dir": str, "store_target_dir": str, "output_dir": str, "log_dir": str},
    "data": {"encoding": str, "item_key": str},
    "base_cell": list,
    "ibd": {
        "method": str,
        "distance": str,
        "distance_mix_weight": NUM,
        "max_distance_quantile": NUM,
        "size_hist_periods": int,
        "size_max_splits": int,
        "min_store": int,
        "min_store_soft": int,
        "soft_stop": bool,
        "min_store_cat": int,
        "min_item": int,
        "min_overlap_periods": int,
        "trend_min_common_stores": int,
        "vi_max": NUM,
        "ks_max": NUM,
        "psi_max": NUM,
        "max_median_ratio": NUM,
        "psi_bins": int,
        "psi_min_bin_count": int,
        "het_alpha": NUM,
        "forced_nearest_warn_ratio": NUM,
        "allow_forced_nearest": bool,
        "check_level": str,
        "mix_lookback": int,
        "mix_weights": list,
        "mix_common_stores": bool,
        "mix_common_lookback": int,
        "mix_start_period": int,
    },
    "fova": {
        "adjbox_a": NUM,
        "adjbox_b": NUM,
        "adjbox_c": NUM,
        "longitude_iqr_coef": NUM,
        "trim_pct": NUM,
        "sign_crit_lift": NUM,
        "min_hist_periods": int,
        "recommended_hist_periods": int,
        "hist_start_period": int,
        "hist_start_filter": bool,
        "trend_fence_fallback": bool,
        "longitude_borrow_cell": bool,
        "longitude_borrow_hist_scale": NUM,
        "fence_fallback_min_n": int,
        "borrow_min_peer": int,
        "adjbox_cell_shrink": bool,
        "adjbox_shrink_min_n": int,
        "adjbox_shrink_scale": NUM,
    },
    "seasonal": {
        "baseline_periods": int,
        "sign_crit": NUM,
        "min_common_stores": int,
        "borrow_base_cell": bool,
    },
    "peer": {"evaluate_zero_rows": bool, "missing_sell_ratio": NUM, "missing_min_prev_ratio": NUM,
             "fallback_min_sell": int, "store_drop_ratio": NUM, "store_drop_min_prev_cat": int},
    "stratify": {"eta2_strong": NUM, "eta2_medium": NUM},
    "evidence": {"high_min": int},
    "drilldown": {"min_level": str, "top_n": int, "min_peer_stores": int},
    "stability": {"rate_spike_ratio": NUM, "grid": dict},
    "report": {"adjust_min_level": str, "raw_output": bool},
    "unit_fix": {
        "fuller_detect_limit": NUM,
        "fuller_correct_limit": NUM,
        "fuller_max_tail": int,
        "critical_ratio": NUM,
        "min_peer_stores": int,
        "packsize_tolerance": NUM,
        "score_min": NUM,
        "store_shift_ratio": NUM,
        "store_shift_min_cat": int,
        "acv_source": str,
    },
}

LEVELS = ("normal", "watch", "suspicious", "high_confidence")
CHECK_LEVELS = ("ibd", "category")
IBD_METHODS = ("province", "size")
IBD_DISTANCES = ("category_mix", "trend", "combined")


class Section:
    """Read-only attribute view over one JSON section."""

    def __init__(self, name: str, values: dict):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_values", MappingProxyType(copy.deepcopy(values)))

    def __getattr__(self, key):
        try:
            return self._values[key]
        except KeyError:
            raise AttributeError(f"{self._name}.{key} not in params") from None

    def __setattr__(self, key, value):
        raise AttributeError("params are read-only")

    def get(self, key, default=None):
        return self._values.get(key, default)

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self._values))


@dataclass(frozen=True)
class PeerParams:
    source: Path
    paths: Section
    data: Section
    base_cell: tuple[str, ...]
    ibd: Section
    fova: Section
    seasonal: Section
    peer: Section
    stratify: Section
    evidence: Section
    drilldown: Section
    stability: Section
    report: Section
    unit_fix: Section
    raw: MappingProxyType = field(repr=False)

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self.raw))

    def path(self, key: str, root: Path | None = None) -> Path:
        p = Path(self.paths.get(key))
        if p.is_absolute():
            return p
        base = root if root is not None else self.source.resolve().parent.parent
        return base / p


def _check_type(where: str, value, expected) -> None:
    if expected is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{where} expected bool, got {type(value).__name__}")
        return
    if isinstance(value, bool) or not isinstance(value, expected):
        raise TypeError(f"{where} expected {expected}, got {type(value).__name__}")


def validate(raw: dict) -> None:
    for section, spec in SCHEMA.items():
        if section not in raw:
            raise KeyError(f"missing section '{section}'")
        if not isinstance(spec, dict):
            _check_type(section, raw[section], spec)
            continue
        if not isinstance(raw[section], dict):
            raise TypeError(f"section '{section}' must be an object")
        for key, typ in spec.items():
            if key not in raw[section]:
                raise KeyError(f"missing key '{section}.{key}'")
            _check_type(f"{section}.{key}", raw[section][key], typ)
    if not raw["base_cell"] or not all(isinstance(c, str) for c in raw["base_cell"]):
        raise TypeError("base_cell must be a non-empty list of column names")
    if raw["ibd"]["check_level"] not in CHECK_LEVELS:
        raise ValueError(f"ibd.check_level must be one of {CHECK_LEVELS}")
    if raw["ibd"]["method"] not in IBD_METHODS:
        raise ValueError(f"ibd.method must be one of {IBD_METHODS}")
    if raw["ibd"]["distance"] not in IBD_DISTANCES:
        raise ValueError(f"ibd.distance must be one of {IBD_DISTANCES}")
    if not 0 <= raw["ibd"]["distance_mix_weight"] <= 1:
        raise ValueError("ibd.distance_mix_weight must be in [0, 1]")
    if not 0 < raw["ibd"]["max_distance_quantile"] <= 1:
        raise ValueError("ibd.max_distance_quantile must be in (0, 1]")
    if raw["ibd"]["max_median_ratio"] <= 1:
        raise ValueError("ibd.max_median_ratio must be > 1")
    if raw["ibd"]["size_hist_periods"] < 1 or raw["ibd"]["size_max_splits"] < 1:
        raise ValueError("ibd.size_hist_periods and ibd.size_max_splits must be >= 1")
    if not 0 < raw["ibd"]["min_store_soft"] <= raw["ibd"]["min_store"]:
        raise ValueError("ibd.min_store_soft must be in (0, ibd.min_store]")
    if raw["ibd"]["mix_lookback"] < 1:
        raise ValueError("ibd.mix_lookback must be >= 1")
    if raw["ibd"]["mix_common_lookback"] < 0:
        raise ValueError("ibd.mix_common_lookback must be >= 0 (0 = all mix window periods)")
    if raw["ibd"]["mix_start_period"] < 0:
        raise ValueError("ibd.mix_start_period must be >= 0 (0 disables the cut)")
    weights = raw["ibd"]["mix_weights"]
    if not isinstance(weights, list) or any(not isinstance(w, NUM) or w < 0 for w in weights):
        raise ValueError("ibd.mix_weights must be a list of non-negative numbers (empty = recency 2^t)")
    if weights and sum(float(w) for w in weights) <= 0:
        raise ValueError("ibd.mix_weights must sum to > 0 when non-empty")
    if not 0 < raw["ibd"]["het_alpha"] < 1:
        raise ValueError("ibd.het_alpha must be in (0, 1)")
    if not 0 < raw["peer"]["missing_sell_ratio"] <= 1:
        raise ValueError("peer.missing_sell_ratio must be in (0, 1]")
    if not 0 <= raw["peer"]["missing_min_prev_ratio"] <= 1:
        raise ValueError("peer.missing_min_prev_ratio must be in [0, 1]")
    for section in ("drilldown", "report"):
        key = "min_level" if section == "drilldown" else "adjust_min_level"
        if raw[section][key] not in LEVELS:
            raise ValueError(f"{section}.{key} must be one of {LEVELS}")
    for key in ("adjbox_c",):
        grid = raw["stability"]["grid"].get(key)
        if not isinstance(grid, list) or not grid:
            raise KeyError(f"missing list 'stability.grid.{key}'")
    if not 0 < raw["unit_fix"]["fuller_detect_limit"] < 1:
        raise ValueError("unit_fix.fuller_detect_limit must be in (0, 1)")
    if not 0 < raw["unit_fix"]["fuller_correct_limit"] < 1:
        raise ValueError("unit_fix.fuller_correct_limit must be in (0, 1)")
    if not 0 < raw["unit_fix"]["store_shift_ratio"] <= 1:
        raise ValueError("unit_fix.store_shift_ratio must be in (0, 1]")
    if raw["unit_fix"]["critical_ratio"] <= 1:
        raise ValueError("unit_fix.critical_ratio must be > 1")
    if raw["fova"]["fence_fallback_min_n"] < 1:
        raise ValueError("fova.fence_fallback_min_n must be >= 1")
    if raw["fova"]["borrow_min_peer"] < 1:
        raise ValueError("fova.borrow_min_peer must be >= 1")
    if raw["fova"]["longitude_borrow_hist_scale"] <= 0:
        raise ValueError("fova.longitude_borrow_hist_scale must be > 0")
    if raw["fova"]["adjbox_shrink_min_n"] < 1:
        raise ValueError("fova.adjbox_shrink_min_n must be >= 1")
    if raw["fova"]["adjbox_shrink_scale"] <= 0:
        raise ValueError("fova.adjbox_shrink_scale must be > 0")


def params_from_dict(raw: dict, source: Path | str = "config/peer_params.json") -> PeerParams:
    validate(raw)
    sections = {k: Section(k, raw[k]) for k in SCHEMA if k != "base_cell"}
    return PeerParams(
        source=Path(source),
        base_cell=tuple(raw["base_cell"]),
        raw=MappingProxyType(copy.deepcopy(raw)),
        **sections,
    )


def load_params(path) -> PeerParams:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return params_from_dict(raw, source=path)


def flatten(d: dict, prefix: str = "") -> list[dict]:
    """Dotted key / JSON value rows for Excel snapshots."""
    rows = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            rows.extend(flatten(v, f"{key}."))
        else:
            rows.append({"key": key, "value": json.dumps(v, ensure_ascii=False)})
    return rows
