from __future__ import annotations

import numpy as np
import pandas as pd

from core.holt import holt_damped_forecast, tolerance_bounds
from core.share_change import (
    current_period_share_check,
    share_change_combined_check,
    short_term_share_check,
    vs_national_share_check,
)

HOLT_MIN_HIST = 8
N0_CAP_TRIGGER = 5
N0_CAP_KEEP = 2
N17_WINDOW = 3
N15_MIN_HIST = 1
N611_MIN_HIST = 6
METHOD_LABEL_COLS = ("outlier_n0", "outlier_n1_5", "outlier_n6_11", "outlier_n8")


def _ok_type(prefix: str) -> str:
    return f"ok_{prefix}"


def _exc_type(prefix: str, direction: str) -> str:
    return f"{prefix}_{direction}"


def _demote(result: dict, prefix: str) -> dict:
    result["is_exception"] = False
    result["share_exception_dir"] = ""
    result["outlier_type"] = _ok_type(prefix)
    return result


def _apply_national_share(
    result: dict,
    share_jul: float,
    national_share: float | None,
    abs_threshold: float,
    prefix: str,
) -> dict:
    """n0 extra gate: keep only when share - national share matches the flag."""
    if national_share is None or not result.get("is_exception"):
        return result
    chk = vs_national_share_check(share_jul, national_share, abs_threshold=abs_threshold)
    if chk["is_exception"] and chk["direction"] == result.get("share_exception_dir"):
        return result
    return _demote(result, prefix)


def _apply_small_share(
    result: dict,
    share_jul: float,
    prev_share: float,
    small_share: float,
    prefix: str,
) -> dict:
    result["outlier_type_raw"] = result["outlier_type"]
    used = max(float(share_jul), float(prev_share or 0.0))
    if result.get("is_exception") and used < small_share:
        result["ignored_small"] = True
        result["is_exception"] = False
        result["outlier_type"] = _ok_type(prefix)
        result["share_exception_dir"] = ""
    else:
        result["ignored_small"] = False
    return result


def _from_current_share(
    share_jul: float,
    peer_shares,
    iqr_k: float,
    min_peers: int,
) -> dict:
    chk = current_period_share_check(
        share_jul, peer_shares, iqr_k=iqr_k, min_peers=min_peers
    )
    prefix = "current_share"
    direction = chk["direction"]
    is_exc = chk["is_exception"]
    return {
        "share": float(share_jul),
        "yhat_share": chk["baseline"],
        "lower_share": chk["lower"],
        "upper_share": chk["upper"],
        "is_exception": is_exc,
        "share_exception_dir": direction,
        "outlier_type": _exc_type(prefix, direction) if is_exc else _ok_type(prefix),
    }


def _from_short_term_peers(
    share_jul: float,
    hist_shares,
    peer_changes,
    window: int,
    iqr_k: float,
    min_peers: int,
    abs_threshold: float,
    peer_pps=None,
    peer_only: bool = False,
) -> dict:
    chk = share_change_combined_check(
        share_jul,
        hist_shares,
        peer_changes,
        window=window,
        iqr_k=iqr_k,
        min_peers=min_peers,
        abs_threshold=abs_threshold,
        peer_pps=peer_pps,
        peer_only=peer_only,
    )
    prefix = "short_term_share"
    direction = chk["direction"]
    is_exc = chk["is_exception"]
    return {
        "n_hist": len(list(hist_shares)),
        "share": float(share_jul),
        "yhat_share": chk["baseline"],
        "lower_share": chk["lower"],
        "upper_share": chk["upper"],
        "is_exception": is_exc,
        "share_exception_dir": direction,
        "outlier_type": _exc_type(prefix, direction) if is_exc else _ok_type(prefix),
    }


def _from_short_term(
    share_jul: float,
    hist_shares,
    threshold: float,
    window: int,
    abs_threshold: float = 0.03,
) -> dict:
    chk = short_term_share_check(
        share_jul,
        hist_shares,
        threshold=threshold,
        window=window,
        abs_threshold=abs_threshold,
    )
    prefix = "short_term_share"
    direction = chk["direction"]
    is_exc = chk["is_exception"]
    return {
        "n_hist": len(list(hist_shares)),
        "share": float(share_jul),
        "yhat_share": chk["baseline"],
        "lower_share": chk["lower"],
        "upper_share": chk["upper"],
        "is_exception": is_exc,
        "share_exception_dir": direction,
        "outlier_type": _exc_type(prefix, direction) if is_exc else _ok_type(prefix),
    }


def _from_holt(
    share_jul: float,
    hist_shares,
    alpha: float,
    beta: float,
    phi: float,
    pr: float,
) -> dict | None:
    yhat, _level, _trend, _errors, sigma, df = holt_damped_forecast(
        hist_shares, alpha=alpha, beta=beta, phi=phi
    )
    if df < 1:
        return None
    lower, upper, _t = tolerance_bounds(yhat, sigma, df, pr=pr)
    prefix = "holt_share"
    if sigma == 0.0:
        is_exc = not np.isclose(share_jul, yhat, atol=1e-12, rtol=0.0)
        if is_exc:
            direction = "high" if share_jul > yhat else "low"
        else:
            direction = ""
        lower = yhat
        upper = yhat
    else:
        if share_jul > upper:
            is_exc, direction = True, "high"
        elif share_jul < lower:
            is_exc, direction = True, "low"
        else:
            is_exc, direction = False, ""
    return {
        "n_hist": len(list(hist_shares)),
        "share": float(share_jul),
        "yhat_share": float(yhat),
        "lower_share": float(lower),
        "upper_share": float(upper),
        "is_exception": is_exc,
        "share_exception_dir": direction,
        "outlier_type": _exc_type(prefix, direction) if is_exc else _ok_type(prefix),
    }


def _label_from_result(result: dict) -> str:
    if result.get("is_exception"):
        return result.get("share_exception_dir") or "ok"
    return "ok"


def _na_method() -> dict:
    return {
        "label": "na",
        "is_exception": False,
        "share_exception_dir": "",
        "yhat_share": float("nan"),
        "lower_share": float("nan"),
        "upper_share": float("nan"),
        "ignored_small": False,
        "note": "",
    }


def _pack(raw: dict) -> dict:
    return {
        "label": _label_from_result(raw),
        "is_exception": raw["is_exception"],
        "share_exception_dir": raw["share_exception_dir"],
        "yhat_share": raw["yhat_share"],
        "lower_share": raw["lower_share"],
        "upper_share": raw["upper_share"],
        "ignored_small": raw["ignored_small"],
        "note": raw.get("note", ""),
    }


def identify_n0(
    share_jul: float,
    peer_shares=None,
    *,
    national_share: float | None = None,
    province_weight: float | None = None,
    iqr_k: float = 1.5,
    min_peers: int = 4,
    small_share: float = 0.005,
    abs_threshold: float = 0.01,
    min_province_weight: float = 0.01,
    peer_abs_threshold: float = 0.01,
) -> dict:
    """n0: current-period share vs other provinces, plus national mix."""
    peers = [] if peer_shares is None else [float(x) for x in peer_shares]
    if len(peers) < min_peers:
        return _na_method()
    raw = _from_current_share(share_jul, peers, iqr_k, min_peers)
    if raw.get("is_exception") and abs(float(share_jul) - float(raw["yhat_share"])) < peer_abs_threshold:
        raw = _demote(raw, "current_share")
    raw = _apply_small_share(raw, share_jul, share_jul, small_share, "current_share")
    note = ""
    if (
        raw.get("is_exception")
        and province_weight is not None
        and float(province_weight) < min_province_weight
    ):
        note = "n0_small_province"
        raw["is_exception"] = False
        raw["share_exception_dir"] = ""
        raw["outlier_type"] = _ok_type("current_share")
    raw = _apply_national_share(raw, share_jul, national_share, abs_threshold, "current_share")
    packed = _pack(raw)
    packed["note"] = note
    return packed


def _cap_label_by_province(
    marks,
    label_col: str,
    score: pd.Series,
    note_tag: str,
    trigger: int,
    keep: int,
):
    out = marks.copy()
    if out.empty or label_col not in out.columns:
        return out
    score_col = "_cap_score"
    out[score_col] = score.reindex(out.index)
    group_cols = ["province"]
    if "period_id" in out.columns:
        group_cols = ["period_id", "province"]
    flagged = out[label_col].isin(["high", "low"])
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        loc = list(idx)
        hit = [i for i in loc if bool(flagged.loc[i])]
        if len(hit) < int(trigger):
            continue
        ranked = out.loc[hit].sort_values(
            [score_col, "category"],
            ascending=[False, True],
            kind="mergesort",
        )
        drop = ranked.index[int(keep) :]
        out.loc[drop, label_col] = "ok"
        if "note" in out.columns:
            notes = out.loc[drop, "note"].fillna("").astype(str)
            out.loc[drop, "note"] = [
                f"{n};{note_tag}" if n else note_tag for n in notes
            ]
    out = out.drop(columns=[score_col])
    if {*METHOD_LABEL_COLS, "agree"}.issubset(out.columns):
        labels = out[list(METHOD_LABEL_COLS)]
        comparable = labels.replace("na", pd.NA)
        out["agree"] = comparable.nunique(axis=1, dropna=True).le(1)
    return out


def cap_n0_by_province(
    marks,
    *,
    trigger: int = N0_CAP_TRIGGER,
    keep: int = N0_CAP_KEEP,
):
    """If a province has too many n0 flags, keep the largest |share − national| only."""
    if marks.empty or "outlier_n0" not in marks.columns:
        return marks.copy()
    share = marks["share"].astype(float)
    national = marks["national_share"].astype(float) if "national_share" in marks.columns else 0.0
    return _cap_label_by_province(
        marks, "outlier_n0", (share - national).abs(), "n0_province_cap", trigger, keep
    )


def _identify_short_term(
    share_jul: float,
    hist_shares,
    peer_changes=None,
    *,
    peer_pps=None,
    prev_share: float | None = None,
    window: int = N17_WINDOW,
    iqr_k: float = 1.5,
    min_peers: int = 4,
    abs_threshold: float = 1e-4,
    small_share: float = 0.005,
    peer_only: bool = False,
) -> dict:
    hist = [float(x) for x in hist_shares]
    changes = [] if peer_changes is None else [float(x) for x in peer_changes]
    pps = None if peer_pps is None else [float(x) for x in peer_pps]
    if prev_share is None:
        prev_share = hist[-1]
    raw = _from_short_term_peers(
        share_jul,
        hist,
        changes,
        window,
        iqr_k,
        min_peers,
        abs_threshold,
        peer_pps=pps,
        peer_only=peer_only,
    )
    raw = _apply_small_share(raw, share_jul, prev_share, small_share, "short_term_share")
    return _pack(raw)


def identify_n1_5(
    share_jul: float,
    hist_shares,
    peer_changes=None,
    *,
    peer_pps=None,
    prev_share: float | None = None,
    window: int = N17_WINDOW,
    iqr_k: float = 1.5,
    min_peers: int = 4,
    abs_threshold: float = 1e-4,
    small_share: float = 0.005,
    province_weight: float | None = None,
    min_province_weight: float = 0.01,
) -> dict:
    """n1-5: clr-neighbor log-g AND Δpp Tukey when n_hist ≥ 1. No own IQR, no Holt."""
    hist = [float(x) for x in hist_shares]
    if len(hist) < N15_MIN_HIST:
        return _na_method()
    packed = _identify_short_term(
        share_jul,
        hist,
        peer_changes,
        peer_pps=peer_pps,
        prev_share=prev_share,
        window=window,
        iqr_k=iqr_k,
        min_peers=min_peers,
        abs_threshold=abs_threshold,
        small_share=small_share,
        peer_only=True,
    )
    if (
        packed.get("is_exception")
        and province_weight is not None
        and float(province_weight) < min_province_weight
    ):
        packed["is_exception"] = False
        packed["share_exception_dir"] = ""
        packed["label"] = "ok"
        packed["note"] = "n15_small_province"
    return packed


def identify_n6_11(
    share_jul: float,
    hist_shares,
    *,
    prev_share: float | None = None,
    window: int = N17_WINDOW,
    iqr_k: float = 1.5,
    min_peers: int = 4,
    abs_threshold: float = 1e-4,
    small_share: float = 0.005,
) -> dict:
    """n6-11: own wow Tukey AND 7-window Tukey when n_hist ≥ 6. No peers, no Holt."""
    hist = [float(x) for x in hist_shares]
    if len(hist) < N611_MIN_HIST:
        return _na_method()
    return _identify_short_term(
        share_jul,
        hist,
        None,
        peer_pps=None,
        prev_share=prev_share,
        window=window,
        iqr_k=iqr_k,
        min_peers=min_peers,
        abs_threshold=abs_threshold,
        small_share=small_share,
    )


def identify_n8(
    share_jul: float,
    hist_shares,
    *,
    prev_share: float | None = None,
    holt_min_hist: int = HOLT_MIN_HIST,
    alpha: float = 0.2,
    beta: float = 0.1,
    phi: float = 0.8,
    pr: float = 0.99,
    small_share: float = 0.005,
) -> dict:
    """n8: Holt damped trend on own history (current excluded)."""
    hist = [float(x) for x in hist_shares]
    if len(hist) < holt_min_hist:
        return _na_method()
    if prev_share is None:
        prev_share = hist[-1]
    raw = _from_holt(share_jul, hist, alpha, beta, phi, pr)
    if raw is None:
        return _na_method()
    raw = _apply_small_share(raw, share_jul, prev_share, small_share, "holt_share")
    return _pack(raw)


def score_four_methods(
    share_jul: float,
    hist_shares,
    sales_jul: float = 0.0,
    prev_share: float | None = None,
    holt_shares=None,
    peer_shares=None,
    peer_changes=None,
    peer_pps=None,
    province_weight: float | None = None,
    national_share: float | None = None,
    **kwargs,
) -> dict:
    """Run n0, n1-5, n6-11 and n8 side by side on the same July share."""
    hist = [float(x) for x in hist_shares]
    n_hist = len(hist)
    if prev_share is None:
        prev_share = hist[-1] if hist else 0.0

    holt_min = kwargs.get("holt_min_hist", HOLT_MIN_HIST)
    window = kwargs.get("short_term_window", N17_WINDOW)
    abs_threshold = kwargs.get("short_term_abs_threshold", 0.01)
    n17_abs_threshold = kwargs.get("n17_abs_threshold", 1e-4)
    n15_abs_threshold = kwargs.get("n15_abs_threshold", 1e-4)
    small_share = kwargs.get("small_share", 0.005)
    min_province_weight = kwargs.get("min_province_weight", 0.01)
    iqr_k = kwargs.get("current_iqr_k", 1.5)
    min_peers = kwargs.get("current_min_peers", 4)
    alpha = kwargs.get("alpha", 0.2)
    beta = kwargs.get("beta", 0.1)
    phi = kwargs.get("phi", 0.8)
    pr = kwargs.get("pr", 0.99)

    holt_hist = [float(x) for x in holt_shares] if holt_shares is not None else hist
    n0 = identify_n0(
        share_jul,
        peer_shares,
        national_share=national_share,
        province_weight=province_weight,
        iqr_k=iqr_k,
        min_peers=min_peers,
        small_share=small_share,
        abs_threshold=abs_threshold,
        min_province_weight=min_province_weight,
    )
    n15 = identify_n1_5(
        share_jul,
        hist,
        peer_changes,
        peer_pps=peer_pps,
        prev_share=prev_share,
        window=window,
        iqr_k=iqr_k,
        min_peers=min_peers,
        abs_threshold=n15_abs_threshold,
        small_share=small_share,
        province_weight=province_weight,
        min_province_weight=min_province_weight,
    )
    n611 = identify_n6_11(
        share_jul,
        hist,
        prev_share=prev_share,
        window=window,
        iqr_k=iqr_k,
        min_peers=min_peers,
        abs_threshold=n17_abs_threshold,
        small_share=small_share,
    )
    n8 = identify_n8(
        share_jul,
        holt_hist,
        prev_share=prev_share,
        holt_min_hist=holt_min,
        alpha=alpha,
        beta=beta,
        phi=phi,
        pr=pr,
        small_share=small_share,
    )

    labels = [n0["label"], n15["label"], n611["label"], n8["label"]]
    comparable = [x for x in labels if x != "na"]
    agree = len(set(comparable)) <= 1 if comparable else True

    note = ""
    if n_hist == 0:
        note = "zero_new" if float(sales_jul) == 0.0 else "no_hist"
    if n0.get("note"):
        note = f"{note};{n0['note']}" if note else n0["note"]
    if n15.get("note"):
        note = f"{note};{n15['note']}" if note else n15["note"]

    return {
        "n_hist": n_hist,
        "share": float(share_jul),
        "outlier_n0": n0["label"],
        "outlier_n1_5": n15["label"],
        "outlier_n6_11": n611["label"],
        "outlier_n8": n8["label"],
        "agree": agree,
        "n_methods": len(comparable),
        "note": note,
        "yhat_n0": n0["yhat_share"],
        "yhat_n1_5": n15["yhat_share"],
        "yhat_n6_11": n611["yhat_share"],
        "yhat_n8": n8["yhat_share"],
    }


def score_one_key(
    share_jul: float,
    hist_shares,
    sales_jul: float = 0.0,
    prev_share: float | None = None,
    *,
    holt_min_hist: int = HOLT_MIN_HIST,
    alpha: float = 0.2,
    beta: float = 0.1,
    phi: float = 0.8,
    pr: float = 0.99,
    short_term_threshold: float = 1.4,
    short_term_window: int = 3,
    short_term_abs_threshold: float = 0.005,
    small_share: float = 0.005,
) -> dict:
    """Score one province-category using only historical shares (current excluded)."""
    hist = [float(x) for x in hist_shares]
    n_hist = len(hist)
    if prev_share is None:
        prev_share = hist[-1] if hist else 0.0

    if n_hist == 0:
        kind = "zero_new" if float(sales_jul) == 0.0 else "no_hist"
        return {
            "n_hist": 0,
            "share": float(share_jul),
            "yhat_share": float("nan"),
            "lower_share": float("nan"),
            "upper_share": float("nan"),
            "is_exception": False,
            "share_exception_dir": "",
            "outlier_type": kind,
            "outlier_type_raw": kind,
            "ignored_small": False,
        }

    if n_hist >= holt_min_hist:
        holt_result = _from_holt(share_jul, hist, alpha, beta, phi, pr)
        if holt_result is not None:
            return _apply_small_share(
                holt_result, share_jul, prev_share, small_share, "holt_share"
            )

    result = _from_short_term(
        share_jul,
        hist,
        short_term_threshold,
        short_term_window,
        abs_threshold=short_term_abs_threshold,
    )
    return _apply_small_share(result, share_jul, prev_share, small_share, "short_term_share")
