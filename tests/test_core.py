from __future__ import annotations

import numpy as np
import pandas as pd

from core.classify import (
    cap_n0_by_province,
    identify_n0,
    identify_n1_5,
    identify_n6_11,
    identify_n8,
    score_one_key,
    score_four_methods,
)
from core.composition import (
    aitchison_distance,
    nearest_peer_shares,
    peer_field_for_names,
    share_wide,
)
from core.holt import holt_damped_forecast, tolerance_bounds
from core.share_change import (
    current_period_share_check,
    log_growth,
    own_period_log_growths,
    own_period_pps,
    share_change_combined_check,
    share_change_peer_check,
    share_change_pp_peer_check,
    short_term_share_change,
    short_term_share_check,
    vs_national_share_check,
)


def test_aitchison_distance_closer_for_similar_mix():
    gd = {"BEER": 0.50, "WATER": 0.30, "GRAIN": 0.20}
    js = {"BEER": 0.48, "WATER": 0.32, "GRAIN": 0.20}
    xz = {"BEER": 0.05, "WATER": 0.10, "GRAIN": 0.85}
    assert aitchison_distance(gd, js) < aitchison_distance(gd, xz)


def test_loo_category_distance_ignores_tested_share():
    a = {"BEER": 0.60, "WATER": 0.20, "GRAIN": 0.20}
    b = {"BEER": 0.10, "WATER": 0.20, "GRAIN": 0.20}
    assert aitchison_distance(a, b, drop_category="BEER") < 1e-9


def test_nearest_peer_shares_uses_clr_neighbors_then_fallback():
    rows = []
    for i, (prov, beer) in enumerate(
        [
            ("A", 0.50),
            ("B", 0.48),
            ("C", 0.47),
            ("D", 0.05),
        ]
    ):
        rows.append({"province": prov, "category": "BEER", "share": beer})
        rows.append({"province": prov, "category": "WATER", "share": 0.30 if beer > 0.4 else 0.10})
        rows.append({"province": prov, "category": "GRAIN", "share": 1.0 - beer - (0.30 if beer > 0.4 else 0.10)})
    wide = share_wide(pd.DataFrame(rows))
    names, shares = nearest_peer_shares(wide, "A", "BEER", k=2, min_peers=2, expand_k=2)
    assert set(names) == {"B", "C"}
    assert abs(shares[names.index("B")] - 0.48) < 1e-12
    few = share_wide(
        pd.DataFrame(
            [
                {"province": "A", "category": "BEER", "share": 0.5},
                {"province": "A", "category": "WATER", "share": 0.5},
                {"province": "B", "category": "BEER", "share": 0.4},
                {"province": "B", "category": "WATER", "share": 0.6},
                {"province": "C", "category": "BEER", "share": 0.3},
                {"province": "C", "category": "WATER", "share": 0.7},
            ]
        )
    )
    names_all, _ = nearest_peer_shares(few, "A", "BEER", k=8, min_peers=6, expand_k=12)
    assert set(names_all) == {"B", "C"}


def test_peer_field_for_names_keeps_clr_neighbors_only():
    items = [
        {"province": "A", "category": "BEER", "change_pp": 0.01, "change_log": 0.1},
        {"province": "B", "category": "BEER", "change_pp": 0.02, "change_log": 0.2},
        {"province": "C", "category": "BEER", "change_pp": 0.03, "change_log": 0.3},
        {"province": "D", "category": "BEER", "change_pp": 0.99, "change_log": 9.9},
        {"province": "B", "category": "WATER", "change_pp": 0.50, "change_log": 5.0},
    ]
    pps = peer_field_for_names(items, ["B", "C"], "BEER", "change_pp")
    logs = peer_field_for_names(items, ["B", "C"], "BEER", "change_log")
    assert pps == [0.02, 0.03]
    assert logs == [0.2, 0.3]


def test_holt_one_step_matches_hand_calc():
    y = [0.10, 0.11, 0.12, 0.12, 0.13, 0.13, 0.14, 0.14, 0.15, 0.15, 0.16]
    alpha, beta, phi = 0.2, 0.1, 0.8
    yhat_new, level, trend, errors, sigma, df = holt_damped_forecast(
        y, alpha=alpha, beta=beta, phi=phi
    )

    level0 = y[0]
    trend0 = y[1] - y[0]
    yhat1 = level0 + phi * trend0
    level1 = alpha * y[1] + (1 - alpha) * yhat1
    trend1 = beta * (level1 - level0) + (1 - beta) * phi * trend0
    assert abs(level[1] - level1) < 1e-12
    assert abs(trend[1] - trend1) < 1e-12
    assert errors.shape[0] == len(y) - 2
    assert df == len(y) - 3
    assert abs(yhat_new - (level[-1] + phi * trend[-1])) < 1e-12
    lower, upper, t_crit = tolerance_bounds(yhat_new, sigma, df, pr=0.99)
    assert t_crit > 0
    assert lower < yhat_new < upper


def test_short_term_n1_baseline_is_the_only_hist():
    chk = short_term_share_check(0.30, [0.10], threshold=1.4)
    assert chk["baseline"] == 0.10
    assert chk["n_window"] == 1
    assert chk["is_exception"] is True
    assert chk["direction"] == "high"


def test_short_term_n4_uses_last_three():
    chk = short_term_share_check(0.10, [0.40, 0.10, 0.10, 0.10], threshold=1.4)
    assert abs(chk["baseline"] - 0.10) < 1e-12
    assert chk["n_window"] == 3
    assert chk["is_exception"] is False


def test_n17_baseline_uses_last_three():
    hist = [0.40, 0.40, 0.40, 0.40, 0.04, 0.10, 0.16]
    chk = short_term_share_change(0.10, hist)
    assert chk["n_window"] == 3
    assert abs(chk["baseline"] - (0.04 + 0.10 + 0.16) / 3) < 1e-12
    out = identify_n6_11(0.10, hist)
    assert abs(out["yhat_share"] - (0.04 + 0.10 + 0.16) / 3) < 1e-12


def test_score_n11_uses_holt():
    hist = [0.10] * 11
    out = score_one_key(0.10, hist, sales_jul=1.0)
    assert out["n_hist"] == 11
    assert out["outlier_type"].startswith("ok_holt_share") or out["outlier_type"].startswith(
        "holt_share"
    )
    assert "short_term" not in out["outlier_type"]


def test_score_n4_and_n1_are_short_term():
    n4 = score_one_key(0.30, [0.10, 0.10, 0.10, 0.10], sales_jul=1.0)
    n1 = score_one_key(0.30, [0.10], sales_jul=1.0)
    assert n4["outlier_type"] == "short_term_share_high"
    assert n1["outlier_type"] == "short_term_share_high"
    assert n1["yhat_share"] == 0.10


def test_score_n0_no_hist_and_zero_new():
    fresh = score_one_key(0.02, [], sales_jul=10.0)
    zero = score_one_key(0.0, [], sales_jul=0.0)
    assert fresh["outlier_type"] == "no_hist"
    assert fresh["is_exception"] is False
    assert zero["outlier_type"] == "zero_new"


def test_small_share_demotes_exception():
    out = score_one_key(0.08, [0.02], sales_jul=1.0, small_share=0.10)
    assert out["is_exception"] is False
    assert out["ignored_small"] is True
    assert out["outlier_type"] == "ok_short_term_share"
    assert out["outlier_type_raw"] == "short_term_share_high"


def test_small_pp_change_not_short_term_outlier():
    # large relative but tiny absolute move: should stay ok
    chk = short_term_share_check(0.006, [0.002], threshold=1.4, abs_threshold=0.005)
    assert chk["is_exception"] is False


def test_constant_share_not_flagged_by_holt():
    hist = np.full(11, 0.08).tolist()
    out = score_one_key(0.08, hist, sales_jul=1.0)
    assert out["is_exception"] is False
    assert out["outlier_type"] == "ok_holt_share"


def test_current_share_flags_high_vs_peers():
    chk = current_period_share_check(0.40, [0.10] * 12)
    assert chk["is_exception"] is True
    assert chk["direction"] == "high"
    assert abs(chk["baseline"] - 0.10) < 1e-12


def test_current_share_stable_among_peers():
    chk = current_period_share_check(0.10, [0.09, 0.10, 0.11, 0.10, 0.08, 0.12])
    assert chk["is_exception"] is False


def test_log_growth_is_log_one_plus_g():
    assert abs(log_growth(0.30, 0.10) - np.log(3.0)) < 1e-12
    assert log_growth(0.10, 0.10) == 0.0
    assert log_growth(0.0, 0.0) == 0.0
    assert np.isposinf(log_growth(0.10, 0.0))
    assert np.isneginf(log_growth(0.0, 0.10))


def test_share_change_high_vs_peer_changes():
    chk = share_change_peer_check(0.30, [0.10, 0.10, 0.10], [0.0] * 12)
    assert chk["is_exception"] is True
    assert chk["direction"] == "high"
    assert abs(chk["pp"] - 0.20) < 1e-12
    assert abs(chk["log_g"] - np.log(3.0)) < 1e-12


def test_share_change_ok_when_peers_moved_the_same():
    same_log = [float(np.log(3.0))] * 12
    chk = share_change_peer_check(0.30, [0.10, 0.10, 0.10], same_log)
    assert chk["is_exception"] is False


def test_peer_fence_flags_same_pp_if_log_growth_is_larger():
    # +4pp from 2% is 3x; peers +4pp from 20% is only 1.2x
    chk = share_change_peer_check(0.06, [0.02, 0.02, 0.02], [float(np.log(1.2))] * 12)
    assert chk["is_exception"] is True
    assert chk["direction"] == "high"
    same_log = share_change_peer_check(0.06, [0.02, 0.02, 0.02], [float(np.log(3.0))] * 12)
    assert same_log["is_exception"] is False


def test_identify_n0_n17_n8_are_separate():
    n0 = identify_n0(0.40, [0.10] * 12, national_share=0.12, province_weight=0.05)
    n17 = identify_n1_5(0.30, [0.10] * 5, [0.0] * 12, peer_pps=[0.0] * 12)
    n8 = identify_n8(0.10, [0.10] * 11)
    assert n0["label"] == "high"
    assert n17["label"] == "high"
    assert n8["label"] == "ok"
    assert identify_n0(0.40, [])["label"] == "na"
    assert identify_n1_5(0.30, [0.10] * 4, [0.0] * 12, peer_pps=[0.0] * 12)["label"] == "high"
    assert identify_n6_11(0.30, [0.10] * 5)["label"] == "na"
    assert identify_n8(0.30, [0.10] * 4)["label"] == "na"


def test_n15_scores_long_hist_with_peers():
    assert identify_n1_5(0.30, [], [0.0] * 12)["label"] == "na"
    assert identify_n1_5(0.30, [0.10], [0.0] * 12, peer_pps=[0.0] * 12)["label"] == "high"
    assert identify_n1_5(0.30, [0.10] * 11, [0.0] * 12, peer_pps=[0.0] * 12)["label"] == "high"
    same = identify_n1_5(
        0.30,
        [0.10] * 11,
        [float(np.log(3.0))] * 12,
        peer_pps=[0.20] * 12,
    )
    assert same["label"] == "ok"
    assert identify_n6_11(0.30, [0.10] * 5)["label"] == "na"
    assert identify_n6_11(0.30, [0.10] * 6)["label"] == "high"


def test_three_methods_all_ok_when_share_stable():
    hist = [0.10] * 11
    out = score_four_methods(
        0.10, hist, sales_jul=1.0, peer_shares=[0.10] * 12, peer_changes=[0.0] * 12
    )
    assert out["outlier_n0"] == "ok"
    assert out["outlier_n1_5"] == "ok"
    assert out["outlier_n6_11"] == "ok"
    assert out["outlier_n8"] == "ok"
    assert out["agree"] is True
    assert out["n_methods"] == 4


def test_three_methods_n4_holt_is_na():
    out = score_four_methods(
        0.30,
        [0.10, 0.10, 0.10, 0.10],
        sales_jul=1.0,
        peer_shares=[0.10] * 12,
        peer_changes=[0.0] * 12,
        peer_pps=[0.0] * 12,
    )
    assert out["outlier_n0"] == "high"
    assert out["outlier_n1_5"] == "high"
    assert out["outlier_n6_11"] == "na"
    assert out["outlier_n8"] == "na"
    assert out["agree"] is True
    assert out["n_methods"] == 2


def test_three_methods_n0_all_na():
    out = score_four_methods(0.0, [], sales_jul=0.0)
    assert out["outlier_n0"] == "na"
    assert out["outlier_n1_5"] == "na"
    assert out["outlier_n6_11"] == "na"
    assert out["outlier_n8"] == "na"
    assert out["agree"] is True
    assert out["note"] == "zero_new"


def test_n0_small_province_weight_is_filtered():
    peers = [0.10] * 12
    raw = score_four_methods(0.40, [], sales_jul=1.0, peer_shares=peers, province_weight=0.004)
    kept = score_four_methods(0.40, [], sales_jul=1.0, peer_shares=peers, province_weight=0.05)
    assert raw["outlier_n0"] == "ok"
    assert kept["outlier_n0"] == "high"


def test_n15_small_province_weight_is_filtered():
    small = identify_n1_5(
        0.30,
        [0.10] * 5,
        [0.0] * 12,
        peer_pps=[0.0] * 12,
        province_weight=0.004,
    )
    kept = identify_n1_5(
        0.30,
        [0.10] * 5,
        [0.0] * 12,
        peer_pps=[0.0] * 12,
        province_weight=0.05,
    )
    assert small["label"] == "ok"
    assert small["note"] == "n15_small_province"
    assert kept["label"] == "high"


def test_n0_ignores_history_and_uses_current_peers():
    peers = [0.10] * 12
    no_hist = score_four_methods(0.40, [], sales_jul=1.0, peer_shares=peers)
    long_hist = score_four_methods(0.40, [0.40] * 11, sales_jul=1.0, peer_shares=peers)
    assert no_hist["outlier_n0"] == "high"
    assert long_hist["outlier_n0"] == "high"
    assert no_hist["yhat_n0"] == long_hist["yhat_n0"]
    assert no_hist["outlier_n1_5"] == "na"
    assert no_hist["outlier_n6_11"] == "na"
    assert long_hist["outlier_n8"] in {"ok", "high", "low"}


def test_three_methods_can_disagree():
    hist = [0.10] * 10 + [0.20]
    out = score_four_methods(
        0.20, hist, sales_jul=1.0, peer_shares=[0.20] * 12, peer_changes=[0.0] * 12
    )
    assert out["outlier_n0"] == "ok"
    assert out["outlier_n1_5"] in {"ok", "high"}
    assert out["outlier_n6_11"] in {"ok", "high"}
    assert out["n_methods"] == 4


def test_n17_ok_when_same_change_as_other_provinces():
    out = score_four_methods(
        0.30,
        [0.10, 0.25, 0.10, 0.25, 0.10],
        sales_jul=1.0,
        peer_shares=[0.10] * 12,
        peer_changes=[float(np.log(3.0))] * 12,
        peer_pps=[0.20] * 12,
    )
    assert out["outlier_n1_5"] == "ok"
    assert out["outlier_n6_11"] == "na"
    assert out["outlier_n0"] == "high"


def test_vs_national_share_same_direction_and_pp():
    high = vs_national_share_check(0.20, 0.10, abs_threshold=0.01)
    assert high["is_exception"] is True
    assert high["direction"] == "high"
    assert abs(high["pp"] - 0.10) < 1e-12
    low = vs_national_share_check(0.04, 0.10, abs_threshold=0.01)
    assert low["direction"] == "low"
    tiny = vs_national_share_check(0.108, 0.10, abs_threshold=0.01)
    assert tiny["is_exception"] is False


def test_n0_province_cap_keeps_top_two_when_five_or_more():
    rows = []
    for cat, share in [("A", 0.15), ("B", 0.14), ("C", 0.13), ("D", 0.12), ("E", 0.11)]:
        rows.append(
            {
                "province": "广西",
                "category": cat,
                "share": share,
                "national_share": 0.10,
                "outlier_n0": "high",
                "outlier_n1_5": "ok",
                "outlier_n6_11": "ok",
                "outlier_n8": "ok",
                "note": "",
            }
        )
    for cat, share in [("X", 0.20), ("Y", 0.18), ("Z", 0.16), ("W", 0.14)]:
        rows.append(
            {
                "province": "重庆",
                "category": cat,
                "share": share,
                "national_share": 0.10,
                "outlier_n0": "high",
                "outlier_n1_5": "ok",
                "outlier_n6_11": "ok",
                "outlier_n8": "ok",
                "note": "",
            }
        )
    out = cap_n0_by_province(pd.DataFrame(rows))
    gx = out[out["province"] == "广西"].set_index("category")
    assert gx.loc["A", "outlier_n0"] == "high"
    assert gx.loc["B", "outlier_n0"] == "high"
    assert gx.loc["C", "outlier_n0"] == "ok"
    assert gx.loc["D", "outlier_n0"] == "ok"
    assert gx.loc["E", "outlier_n0"] == "ok"
    assert "n0_province_cap" in str(gx.loc["C", "note"])
    cq = out[out["province"] == "重庆"]
    assert (cq["outlier_n0"] == "high").all()


def test_n17_needs_wow_and_seven_window_when_both_available():
    both = identify_n6_11(0.30, [0.10] * 11)
    wow_flat = identify_n6_11(0.20, [0.10] * 10 + [0.20])
    assert both["label"] == "high"
    assert wow_flat["label"] == "ok"


def test_n17_seven_window_needs_half_pp_even_if_tukey():
    tiny = identify_n6_11(0.104, [0.10] * 11)
    kept = identify_n6_11(0.106, [0.10] * 11)
    assert tiny["label"] == "ok"
    assert kept["label"] == "high"


def test_n0_needs_at_least_one_pp_vs_peers():
    peers = [0.10] * 12
    tiny = identify_n0(0.108, peers, national_share=0.05, province_weight=0.05)
    kept = identify_n0(0.112, peers, national_share=0.05, province_weight=0.05)
    assert tiny["label"] == "ok"
    assert kept["label"] == "high"


def test_n0_keeps_high_only_if_also_above_national():
    peers = [0.10] * 12
    kept = score_four_methods(
        0.40, [], sales_jul=1.0, peer_shares=peers, national_share=0.12
    )
    demoted = score_four_methods(
        0.40, [], sales_jul=1.0, peer_shares=peers, national_share=0.50
    )
    tiny_gap = score_four_methods(
        0.40, [], sales_jul=1.0, peer_shares=peers, national_share=0.398
    )
    assert kept["outlier_n0"] == "high"
    assert demoted["outlier_n0"] == "ok"
    assert tiny_gap["outlier_n0"] == "ok"


def test_n17_ignores_national_change():
    hist = [0.10] * 5
    peers = [0.10] * 12
    changes = [0.0] * 12
    same_as_national = score_four_methods(
        0.30,
        hist,
        sales_jul=1.0,
        peer_shares=peers,
        peer_changes=changes,
        peer_pps=[0.0] * 12,
        national_share=0.28,
    )
    assert same_as_national["outlier_n1_5"] == "high"
    assert same_as_national["outlier_n6_11"] == "na"
    assert same_as_national["outlier_n0"] == "high"


def test_n15_uses_peer_change_not_own_iqr():
    own_jump = [0.10] * 5
    flagged = identify_n1_5(0.30, own_jump, [0.0] * 12, peer_pps=[0.0] * 12)
    same_as_peers = identify_n1_5(
        0.30,
        own_jump,
        [float(np.log(3.0))] * 12,
        peer_pps=[0.20] * 12,
    )
    no_peers = identify_n1_5(0.30, own_jump)
    assert flagged["label"] == "high"
    assert same_as_peers["label"] == "ok"
    assert no_peers["label"] == "ok"


def test_n15_needs_gr_and_pp_peer():
    hist = [0.02] * 5
    gr_only = identify_n1_5(0.08, hist, [0.0] * 12, peer_pps=[0.06] * 12)
    pp_only = identify_n1_5(
        0.08,
        hist,
        [float(np.log(4.0))] * 12,
        peer_pps=[0.0] * 12,
    )
    both = identify_n1_5(0.08, hist, [0.0] * 12, peer_pps=[0.0] * 12)
    both_same = identify_n1_5(
        0.08,
        hist,
        [float(np.log(4.0))] * 12,
        peer_pps=[0.06] * 12,
    )
    assert gr_only["label"] == "ok"
    assert pp_only["label"] == "ok"
    assert both["label"] == "high"
    assert both_same["label"] == "ok"


def test_n15_has_no_half_pp_floor():
    tiny = identify_n1_5(0.104, [0.10] * 11, [0.0] * 12, peer_pps=[0.0] * 12)
    assert tiny["label"] == "high"


def test_n15_n_hist_5_still_uses_peers():
    hist = [0.10] * 5
    own_would_flag = identify_n1_5(
        0.30,
        hist,
        [float(np.log(3.0))] * 12,
        peer_pps=[0.20] * 12,
    )
    assert own_would_flag["label"] == "ok"


def test_share_change_pp_peer_check_flags_vs_peer_pps():
    high = share_change_pp_peer_check(0.30, [0.10, 0.10, 0.10], [0.0] * 12)
    same = share_change_pp_peer_check(0.30, [0.10, 0.10, 0.10], [0.20] * 12)
    skipped = share_change_pp_peer_check(0.30, [0.10, 0.10, 0.10], [0.0, 0.01])
    assert high["is_exception"] is True
    assert high["direction"] == "high"
    assert same["is_exception"] is False
    assert skipped["skipped"] is True


def test_n17_own_vol_demotes_when_history_already_swings():
    hist = [0.10, 0.25, 0.10, 0.25, 0.10, 0.25]
    noisy = identify_n6_11(0.30, hist)
    stable = identify_n6_11(0.30, [0.10] * 6)
    assert noisy["label"] == "ok"
    assert stable["label"] == "high"


def test_n17_own_log_or_own_pp_iqr():
    log_and_pp = identify_n6_11(0.30, [0.10] * 6)
    pp_only = identify_n6_11(0.155, [0.10, 0.11, 0.10, 0.11, 0.10, 0.11])
    assert log_and_pp["label"] == "high"
    assert pp_only["label"] == "high"


def test_own_period_log_growths_are_wow_log_one_plus_g():
    assert own_period_log_growths([0.10]) == []
    got = own_period_log_growths([0.10, 0.20, 0.10])
    assert abs(got[0] - np.log(2.0)) < 1e-12
    assert abs(got[1] - np.log(0.5)) < 1e-12
    assert own_period_pps([0.10, 0.20, 0.10]) == [0.10, -0.10]


def test_n17_own_iqr_uses_previous_period_wow():
    hist = [0.08, 0.08, 0.08, 0.08, 0.08, 0.20]
    out = identify_n6_11(0.20, hist)
    assert out["label"] == "ok"


def test_combined_needs_hist_mean_and_own_iqr_union():
    both = share_change_combined_check(0.30, [0.10] * 6)
    assert both["is_exception"] is True
    assert both["direction"] == "high"
    peers_ignored = share_change_combined_check(
        0.30, [0.10] * 6, peer_pps=[0.20] * 12
    )
    assert peers_ignored["is_exception"] is True
    tiny_vs_hist = share_change_combined_check(0.102, [0.10] * 6)
    assert tiny_vs_hist["is_exception"] is True
    below_min_pp = share_change_combined_check(
        0.10 + 5e-5, [0.10] * 6, abs_threshold=1e-4
    )
    assert below_min_pp["is_exception"] is False
    above_min_pp = share_change_combined_check(
        0.10 + 2e-4, [0.10] * 6, abs_threshold=1e-4
    )
    assert above_min_pp["is_exception"] is True
    noisy = share_change_combined_check(0.30, [0.10, 0.25, 0.10, 0.25, 0.10, 0.25])
    assert noisy["is_exception"] is False
