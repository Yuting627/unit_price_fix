from __future__ import annotations

import numpy as np

N15_MAX_HIST = 5


def _finite_list(values) -> list[float]:
    if values is None:
        return []
    return [float(x) for x in values if np.isfinite(x)]


def log_growth(share_now: float, baseline: float) -> float:
    """log(1+g) = log(current / baseline). 0/0 -> 0; 0 baseline -> +inf."""
    current = float(share_now)
    base = float(baseline)
    if base <= 0.0 and current <= 0.0:
        return 0.0
    if base <= 0.0:
        return float("inf")
    if current <= 0.0:
        return float("-inf")
    return float(np.log(current / base))


def own_period_log_growths(hist_shares) -> list[float]:
    """log(1+g) of consecutive historical shares; g is arithmetic wow."""
    hist = [float(x) for x in hist_shares if np.isfinite(x)]
    out = []
    for prev, now in zip(hist, hist[1:]):
        g = log_growth(now, prev)
        if np.isfinite(g):
            out.append(float(g))
    return out


def own_period_pps(hist_shares) -> list[float]:
    """Consecutive historical share-point changes."""
    hist = [float(x) for x in hist_shares if np.isfinite(x)]
    return [float(now - prev) for prev, now in zip(hist, hist[1:])]


def own_wow_pp_tukey(
    share_now: float,
    hist_shares,
    iqr_k: float = 1.5,
    min_peers: int = 4,
) -> dict:
    """Tukey 1.5×IQR of current wow Δpp vs this province's own historical Δpp."""
    hist = [float(x) for x in hist_shares if np.isfinite(x)]
    peers = own_period_pps(hist)
    if len(hist) < 1 or len(peers) < int(min_peers):
        return {"is_exception": False, "direction": "", "skipped": True, "pp": float("nan")}
    current_pp = float(share_now) - hist[-1]
    chk = tukey_fence_check(current_pp, peers, iqr_k=iqr_k, min_peers=min_peers)
    chk["skipped"] = False
    chk["pp"] = current_pp
    return chk


def own_window_residuals(hist_shares, window: int, kind: str = "pp") -> list[float]:
    hist = [float(x) for x in hist_shares if np.isfinite(x)]
    w = int(window)
    out = []
    for i in range(w, len(hist)):
        base = float(np.mean(hist[i - w : i]))
        if kind == "log":
            g = log_growth(hist[i], base)
            if np.isfinite(g):
                out.append(float(g))
        else:
            out.append(float(hist[i] - base))
    return out


def own_window_tukey(
    share_now: float,
    hist_shares,
    window: int = 7,
    kind: str = "pp",
    iqr_k: float = 1.5,
    min_peers: int = 4,
    abs_threshold: float = 0.005,
) -> dict:
    """Tukey of current vs last ``window`` mean, plus |Δ| ≥ ``abs_threshold`` (default 0.5pp)."""
    hist = [float(x) for x in hist_shares if np.isfinite(x)]
    w = int(window)
    peers = own_window_residuals(hist, w, kind=kind)
    if len(hist) < w or len(peers) < int(min_peers):
        return {"is_exception": False, "direction": "", "skipped": True}
    base = float(np.mean(hist[-w:]))
    pp = float(share_now) - base
    value = log_growth(float(share_now), base) if kind == "log" else pp
    if abs(pp) < float(abs_threshold):
        return {"is_exception": False, "direction": "", "skipped": False, "pp": pp}
    if kind == "log" and not np.isfinite(value):
        direction = "high" if pp > 0 else "low"
        return {"is_exception": True, "direction": direction, "skipped": False, "pp": pp}
    chk = tukey_fence_check(value, peers, iqr_k=iqr_k, min_peers=min_peers)
    chk["skipped"] = False
    chk["pp"] = pp
    return chk


def own_wow_log_tukey(
    share_now: float,
    hist_shares,
    iqr_k: float = 1.5,
    min_peers: int = 4,
) -> dict:
    """Tukey 1.5×IQR of current log(1+wow) vs this province's own historical wow."""
    hist = [float(x) for x in hist_shares if np.isfinite(x)]
    peers = own_period_log_growths(hist)
    if len(hist) < 1 or len(peers) < int(min_peers):
        return {"is_exception": False, "direction": "", "skipped": True, "log_g": float("nan")}
    current_log = log_growth(float(share_now), hist[-1])
    if not np.isfinite(current_log):
        direction = "high" if float(share_now) > hist[-1] else "low"
        return {
            "is_exception": True,
            "direction": direction,
            "skipped": False,
            "log_g": current_log,
        }
    chk = tukey_fence_check(current_log, peers, iqr_k=iqr_k, min_peers=min_peers)
    chk["skipped"] = False
    chk["log_g"] = current_log
    return chk


def tukey_fence_check(
    value: float,
    peers,
    iqr_k: float = 1.5,
    min_peers: int = 4,
) -> dict:
    """Tukey fences against a peer vector."""
    peer_arr = np.asarray(list(peers), dtype=float)
    peer_arr = peer_arr[np.isfinite(peer_arr)]
    if peer_arr.size < int(min_peers):
        raise ValueError("Tukey check needs at least 4 peer values")

    q1 = float(np.percentile(peer_arr, 25))
    q3 = float(np.percentile(peer_arr, 75))
    iqr = q3 - q1
    baseline = float(np.median(peer_arr))
    lower = q1 - float(iqr_k) * iqr
    upper = q3 + float(iqr_k) * iqr
    pp = float(value - baseline)

    if iqr == 0.0:
        is_exception = not np.isclose(value, baseline, atol=1e-12, rtol=0.0)
    else:
        is_exception = bool(value > upper or value < lower)

    if is_exception:
        direction = "high" if pp > 0 else "low"
    else:
        direction = ""

    return {
        "baseline": baseline,
        "lower": lower,
        "upper": upper,
        "is_exception": is_exception,
        "direction": direction,
        "n_peers": int(peer_arr.size),
    }


def vs_national_share_check(
    share_now: float,
    national_share: float,
    abs_threshold: float = 0.01,
) -> dict:
    """Compare this province's category mix to the national mix of the same category."""
    baseline = float(national_share)
    pp = float(share_now) - baseline
    if pp > float(abs_threshold):
        direction = "high"
    elif pp < -float(abs_threshold):
        direction = "low"
    else:
        direction = ""
    return {
        "baseline": baseline,
        "pp": pp,
        "is_exception": direction != "",
        "direction": direction,
    }


def current_period_share_check(
    share_now: float,
    peer_shares,
    iqr_k: float = 1.5,
    min_peers: int = 4,
) -> dict:
    """Compare current share to the same-period peer distribution (Tukey fences)."""
    chk = tukey_fence_check(share_now, peer_shares, iqr_k=iqr_k, min_peers=min_peers)
    baseline = chk["baseline"]
    chk["rel"] = (
        float(share_now / baseline - 1.0)
        if baseline != 0
        else (0.0 if share_now == 0 else float("inf"))
    )
    return chk


def short_term_share_change(
    share_now: float,
    hist_shares,
    window: int = 3,
) -> dict:
    """Own short-term share change vs the last min(window, n_hist) mean."""
    hist = np.asarray(list(hist_shares), dtype=float)
    if hist.size < 1:
        raise ValueError("short-term change needs at least 1 historical share")
    n_use = min(int(window), int(hist.size))
    baseline = float(np.mean(hist[-n_use:]))
    pp = float(share_now - baseline)
    if baseline == 0.0:
        rel = 0.0 if share_now == 0.0 else float("inf")
    else:
        rel = float(share_now / baseline - 1.0)
    return {
        "baseline": baseline,
        "pp": pp,
        "rel": rel,
        "log_g": log_growth(share_now, baseline),
        "n_window": n_use,
    }


def share_change_peer_check(
    share_now: float,
    hist_shares,
    peer_changes,
    window: int = 3,
    iqr_k: float = 1.5,
    min_peers: int = 4,
) -> dict:
    """Compare this key's log-growth to same-period peer log-growths.

    Peer set is supplied by the caller (clr neighbors' log(1+g) in main).
    Implied share bands are baseline * exp(peer fence).
    """
    own = short_term_share_change(share_now, hist_shares, window=window)
    peers = _finite_list(peer_changes)
    empty = {
        "baseline": own["baseline"],
        "own_baseline": own["baseline"],
        "pp": own["pp"],
        "rel": own["rel"],
        "log_g": own["log_g"],
        "lower": own["baseline"],
        "upper": own["baseline"],
        "is_exception": False,
        "direction": "",
        "n_window": own["n_window"],
        "n_peers": len(peers),
        "skipped": True,
    }
    if len(peers) < int(min_peers):
        return empty
    if not np.isfinite(own["log_g"]):
        direction = "high" if own["pp"] > 0 else ("low" if own["pp"] < 0 else "")
        empty.update(
            {
                "is_exception": direction != "",
                "direction": direction,
                "skipped": False,
            }
        )
        return empty
    chk = tukey_fence_check(own["log_g"], peers, iqr_k=iqr_k, min_peers=min_peers)
    peer_med = chk["baseline"]
    base = own["baseline"]
    yhat = base * float(np.exp(peer_med)) if np.isfinite(peer_med) and base > 0 else base
    lower = base * float(np.exp(chk["lower"])) if np.isfinite(chk["lower"]) and base > 0 else 0.0
    upper = base * float(np.exp(chk["upper"])) if np.isfinite(chk["upper"]) and base > 0 else float("inf")
    return {
        "baseline": yhat,
        "own_baseline": own["baseline"],
        "pp": own["pp"],
        "rel": own["rel"],
        "log_g": own["log_g"],
        "lower": lower,
        "upper": upper,
        "is_exception": chk["is_exception"],
        "direction": chk["direction"],
        "n_window": own["n_window"],
        "n_peers": chk["n_peers"],
        "skipped": False,
    }


def share_change_pp_peer_check(
    share_now: float,
    hist_shares,
    peer_pps,
    window: int = 3,
    iqr_k: float = 1.5,
    min_peers: int = 4,
) -> dict:
    """Compare this key's Δpp to same-period peer share-point changes."""
    own = short_term_share_change(share_now, hist_shares, window=window)
    peers = _finite_list(peer_pps)
    if len(peers) < int(min_peers):
        return {
            "is_exception": False,
            "direction": "",
            "skipped": True,
            "pp": own["pp"],
            "n_peers": len(peers),
        }
    chk = tukey_fence_check(own["pp"], peers, iqr_k=iqr_k, min_peers=min_peers)
    chk["skipped"] = False
    chk["pp"] = own["pp"]
    return chk


def _hist_direction(pp: float, abs_threshold: float) -> tuple[str, bool]:
    if pp > abs_threshold:
        return "high", True
    if pp < -abs_threshold:
        return "low", True
    return "", False


def _n15_peer_intersect(
    share_now: float,
    hist_shares,
    peer_changes,
    peer_pps,
    window: int,
    iqr_k: float,
    min_peers: int,
    abs_threshold: float,
    own: dict,
) -> dict:
    hist_dir, hist_exc = _hist_direction(own["pp"], abs_threshold)
    gr = share_change_peer_check(
        share_now,
        hist_shares,
        peer_changes,
        window=window,
        iqr_k=iqr_k,
        min_peers=min_peers,
    )
    pp_peer = share_change_pp_peer_check(
        share_now,
        hist_shares,
        peer_pps,
        window=window,
        iqr_k=iqr_k,
        min_peers=min_peers,
    )
    gr_hit = (
        hist_exc
        and (not gr.get("skipped"))
        and gr["is_exception"]
        and gr["direction"] == hist_dir
    )
    pp_hit = (
        hist_exc
        and (not pp_peer.get("skipped"))
        and pp_peer["is_exception"]
        and pp_peer["direction"] == hist_dir
    )
    intersect_ok = gr_hit and pp_hit
    return {
        "baseline": own["baseline"],
        "own_baseline": own["baseline"],
        "pp": own["pp"],
        "rel": own["rel"],
        "log_g": own["log_g"],
        "lower": own["baseline"] - float(abs_threshold),
        "upper": own["baseline"] + float(abs_threshold),
        "is_exception": intersect_ok,
        "direction": hist_dir if intersect_ok else "",
        "hist_exception": hist_exc,
        "peer_exception": (not gr.get("skipped")) and bool(gr["is_exception"]),
        "pp_peer_exception": (not pp_peer.get("skipped")) and bool(pp_peer["is_exception"]),
        "vol_ok": intersect_ok,
        "n_window": own["n_window"],
        "n_peers": max(int(gr.get("n_peers") or 0), int(pp_peer.get("n_peers") or 0)),
    }


def share_change_combined_check(
    share_now: float,
    hist_shares,
    peer_changes=None,
    window: int = 3,
    iqr_k: float = 1.5,
    min_peers: int = 4,
    abs_threshold: float = 1e-4,
    peer_pps=None,
    long_window: int = 7,
    long_abs_threshold: float = 0.005,
    peer_only: bool = False,
) -> dict:
    """n1-5: peer Δlog-g AND Δpp. n6-11: own wow Tukey AND 7-period Tukey. No Holt."""
    own = short_term_share_change(share_now, hist_shares, window=window)
    n_hist = len([x for x in hist_shares if np.isfinite(x)])
    if peer_only or n_hist <= N15_MAX_HIST:
        return _n15_peer_intersect(
            share_now,
            hist_shares,
            peer_changes,
            peer_pps,
            window,
            iqr_k,
            min_peers,
            abs_threshold,
            own,
        )
    hist_dir, hist_exc = _hist_direction(own["pp"], abs_threshold)
    own_log = own_wow_log_tukey(share_now, hist_shares, iqr_k=iqr_k, min_peers=min_peers)
    own_pp = own_wow_pp_tukey(share_now, hist_shares, iqr_k=iqr_k, min_peers=min_peers)
    win_pp = own_window_tukey(
        share_now,
        hist_shares,
        window=long_window,
        kind="pp",
        iqr_k=iqr_k,
        min_peers=min_peers,
        abs_threshold=long_abs_threshold,
    )
    win_log = own_window_tukey(
        share_now,
        hist_shares,
        window=long_window,
        kind="log",
        iqr_k=iqr_k,
        min_peers=min_peers,
        abs_threshold=long_abs_threshold,
    )
    wow_dir = ""
    if hist_exc:
        if (not own_log.get("skipped")) and own_log["is_exception"] and own_log["direction"] == hist_dir:
            wow_dir = hist_dir
        if (not own_pp.get("skipped")) and own_pp["is_exception"] and own_pp["direction"] == hist_dir:
            wow_dir = hist_dir
    wow_hit = wow_dir != ""
    win_dirs = []
    if (not win_pp.get("skipped")) and win_pp["is_exception"]:
        win_dirs.append(win_pp["direction"])
    if (not win_log.get("skipped")) and win_log["is_exception"]:
        win_dirs.append(win_log["direction"])
    win_hit = bool(win_dirs)
    wow_skipped = bool(own_log.get("skipped") and own_pp.get("skipped"))
    long_skipped = bool(win_pp.get("skipped") and win_log.get("skipped"))
    if wow_skipped and long_skipped:
        union_ok = hist_exc
        union_dir = hist_dir
    elif wow_skipped:
        union_ok = win_hit
        union_dir = win_dirs[0] if win_hit and len(set(win_dirs)) == 1 else (hist_dir if win_hit else "")
    elif long_skipped:
        union_ok = wow_hit
        union_dir = wow_dir
    else:
        same = [d for d in win_dirs if d == wow_dir] if wow_hit else []
        union_ok = wow_hit and win_hit and bool(same)
        union_dir = wow_dir if union_ok else ""
    if union_ok and union_dir:
        direction = union_dir
        is_exception = True
    else:
        direction = ""
        is_exception = False
    return {
        "baseline": own["baseline"],
        "own_baseline": own["baseline"],
        "pp": own["pp"],
        "rel": own["rel"],
        "log_g": own["log_g"],
        "lower": own["baseline"] - float(abs_threshold),
        "upper": own["baseline"] + float(abs_threshold),
        "is_exception": is_exception,
        "direction": direction,
        "hist_exception": hist_exc,
        "peer_exception": False,
        "pp_peer_exception": False,
        "vol_ok": union_ok,
        "n_window": own["n_window"],
        "n_peers": 0,
    }


def short_term_share_check(
    share_now: float,
    hist_shares,
    threshold: float = 1.4,
    window: int = 3,
    abs_threshold: float = 0.005,
) -> dict:
    """Compare current share to the recent-window mean.

    Flag only if both are true:
    - relative change exceeds ``threshold`` (default 140%)
    - absolute share-point change exceeds ``abs_threshold`` (default 0.5pp)

    Defaults were chosen to maximize Jaccard overlap with Holt on n_hist>=8.
    """
    own = short_term_share_change(share_now, hist_shares, window=window)
    baseline = own["baseline"]
    rel = own["rel"]
    pp = own["pp"]
    rel_lower = baseline * (1.0 - threshold)
    rel_upper = baseline * (1.0 + threshold)
    abs_lower = baseline - abs_threshold
    abs_upper = baseline + abs_threshold
    lower = min(rel_lower, abs_lower)
    upper = max(rel_upper, abs_upper)
    outside_rel = abs(rel) > threshold if np.isfinite(rel) else share_now > 0.0
    outside_abs = abs(pp) > abs_threshold
    is_exception = bool(outside_rel and outside_abs)
    if is_exception:
        direction = "high" if pp > 0 else "low"
    else:
        direction = ""

    return {
        "baseline": baseline,
        "rel": rel,
        "lower": lower,
        "upper": upper,
        "is_exception": is_exception,
        "direction": direction,
        "n_window": own["n_window"],
    }
