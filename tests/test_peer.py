from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from core.config import params_from_dict
from core.data_prep.quality import check_quality


# ---------------------------------------------------------------- config
def test_load_params_real_json():
    from conftest import PARAMS_PATH
    from core.config import load_params

    params = load_params(PARAMS_PATH)
    assert params.base_cell == ("PLATFORMNAME", "SHOPTYPE")
    assert params.ibd.min_store == 10 and params.ibd.min_store_soft == 5
    assert params.ibd.allow_forced_nearest is False
    assert params.fova.trend_fence_fallback is True and params.fova.longitude_borrow_cell is True
    assert params.fova.adjbox_cell_shrink is True
    assert params.fova.adjbox_shrink_min_n == 30
    assert params.fova.adjbox_c == 7
    assert params.seasonal.sign_crit == pytest.approx(1.281552)


def test_params_missing_key_raises(raw_params):
    bad = copy.deepcopy(raw_params)
    del bad["ibd"]["vi_max"]
    with pytest.raises(KeyError, match="ibd.vi_max"):
        params_from_dict(bad)


def test_params_wrong_type_raises(raw_params):
    bad = copy.deepcopy(raw_params)
    bad["ibd"]["min_store"] = "30"
    with pytest.raises(TypeError):
        params_from_dict(bad)


def test_params_read_only(params):
    with pytest.raises(AttributeError):
        params.ibd.min_store = 5
    with pytest.raises(Exception):
        params.base_cell = ("X",)


# ---------------------------------------------------------------- quality
def test_check_quality_tags_and_drops():
    bmp = pd.DataFrame(
        {
            "period_id": [1, 1, 1, 1, 1],
            "store_id": ["A", "A", "B", "B", "C"],
            "prod_id": [1, 1, 2, 3, 4],
            "category": ["X"] * 5,
            "sales_value": [5.0, 5.0, -1.0, 3.0, np.nan],
            "sales_unit": [1.0, 1.0, 1.0, 0.0, 1.0],
        }
    )
    unmatched = pd.DataFrame({"period_id": [1], "matched_stores": [3], "bmp_only_stores": [2], "target_only_stores": [4]})
    clean, dq = check_quality(bmp, unmatched)
    assert len(clean) == 3
    got = dq.set_index("check")
    assert got.loc["dup_period_store_prod", "n_rows"] == 1
    assert got.loc["negative_value", "n_rows"] == 1
    assert got.loc["missing_value", "n_rows"] == 1
    assert got.loc["zero_unit_positive_value", "n_rows"] == 1
    assert got.loc["target_only_stores", "n_stores"] == 4
    assert got.loc["dup_period_store_prod", "action"] == "dropped"
    assert got.loc["zero_unit_positive_value", "action"] == "tagged"


# ---------------------------------------------------------------- logger
def test_logger_files_separate(tmp_path):
    from funcs.logger import close_logger, get_logger, log_section, production_log_path

    a_path = production_log_path(tmp_path, 20261407)
    p_path = production_log_path(tmp_path, 20261408)
    assert p_path.name == "peer_outlier_20261408.log"

    lg = get_logger("peer.test", a_path)
    log_section(lg, "STRATIFY", {"eta2": 0.21, "base_cell": "PLATFORMNAME x SHOPTYPE"})
    close_logger(lg)
    lg = get_logger("peer.test", p_path)
    log_section(lg, "PEER", pd.DataFrame({"level": ["watch"], "n": [3]}))
    close_logger(lg)

    a_text = a_path.read_text(encoding="utf-8")
    p_text = p_path.read_text(encoding="utf-8")
    assert "[STRATIFY] eta2=0.21" in a_text and "[PEER]" not in a_text
    assert "[PEER]" in p_text and "watch" in p_text and "[STRATIFY]" not in p_text


# ---------------------------------------------------------------- profile / stratify
def _strat_df(seed=0, n=600):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "PLATFORMNAME": rng.choice(["ELEME", "MEITUAN"], n),
            "SHOPTYPE": rng.choice(["cvs", "hmkt", "smkt"], n),
            "PROVINCE": rng.choice(["P1", "P2", "P3", "P4"], n),
            "category": rng.choice(["BEER", "CSD"], n),
        }
    )
    eff = df["PLATFORMNAME"].map({"ELEME": 0.0, "MEITUAN": 1.0}) + df["SHOPTYPE"].map({"cvs": 0.0, "hmkt": 2.0, "smkt": 1.0})
    df["y"] = eff + rng.normal(0, 0.5, n)
    return df


def test_partial_eta2_matches_statsmodels_typ2():
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    from core.data_prep.stratify import partial_eta2

    df = _strat_df()
    table, r2 = partial_eta2(df, "y")
    model = ols("y ~ C(PLATFORMNAME) + C(SHOPTYPE) + C(PROVINCE) + C(category)", data=df).fit()
    ref = sm.stats.anova_lm(model, typ=2)
    for f in ("PLATFORMNAME", "SHOPTYPE", "PROVINCE", "category"):
        ss_ref = ref.loc[f"C({f})", "sum_sq"]
        got = table.set_index("factor").loc[f]
        assert got["ss"] == pytest.approx(ss_ref, rel=1e-6)
        assert got["partial_eta2"] == pytest.approx(ss_ref / (ss_ref + ref.loc["Residual", "sum_sq"]), rel=1e-6)
    assert r2 == pytest.approx(model.rsquared, rel=1e-6)


def test_run_stratify_strength_and_base_cell(params):
    from core.data_prep.stratify import run_stratify, suggest_base_cell

    df = _strat_df(n=2000)
    df["sales_value"] = np.expm1(df["y"] + 3)
    table, _ = run_stratify(df, params.stratify.eta2_strong, params.stratify.eta2_medium)
    strength = dict(zip(table["factor"], table["strength"]))
    assert strength["PLATFORMNAME"] == "strong" and strength["SHOPTYPE"] == "strong"
    assert strength["PROVINCE"] == "weak"
    assert suggest_base_cell(table, ("X",)) == ["PLATFORMNAME", "SHOPTYPE"]


def test_stratify_interaction_keeps_sign_flipping_factor(params):
    from core.data_prep.stratify import run_stratify, suggest_base_cell

    df = _strat_df(seed=3, n=3000)
    flip = np.where(df["SHOPTYPE"] == "cvs", 1.0, np.where(df["SHOPTYPE"] == "smkt", -1.0, 0.0))
    y = df["SHOPTYPE"].map({"cvs": 0.0, "hmkt": 2.0, "smkt": 1.0}) + flip * (df["PLATFORMNAME"] == "MEITUAN") - 0.5 * flip
    df["sales_value"] = np.expm1(y + np.random.default_rng(3).normal(0, 0.5, len(df)) + 3)
    table, _ = run_stratify(df, params.stratify.eta2_strong, params.stratify.eta2_medium)
    strength = dict(zip(table["factor"], table["strength"]))
    assert strength["PLATFORMNAME"] == "weak" and strength["PLATFORMNAME:SHOPTYPE"] == "strong"
    assert suggest_base_cell(table, ("SHOPTYPE",)) == ["PLATFORMNAME", "SHOPTYPE"]


def test_suggest_base_cell_hysteresis():
    from core.data_prep.stratify import suggest_base_cell

    def table(platform, inter):
        return pd.DataFrame({"factor": ["PLATFORMNAME", "SHOPTYPE", "PLATFORMNAME:SHOPTYPE"],
                             "strength": [platform, "strong", inter]})

    both = ("PLATFORMNAME", "SHOPTYPE")
    assert suggest_base_cell(table("weak", "medium"), both) == list(both)
    assert suggest_base_cell(table("weak", "medium"), ("SHOPTYPE",)) == ["SHOPTYPE"]
    assert suggest_base_cell(table("weak", "weak"), both) == ["SHOPTYPE"]
    assert suggest_base_cell(table("weak", "strong"), ("SHOPTYPE",)) == list(both)


def test_check_base_cell_warns_on_mismatch(params):
    from core.data_prep.stratify import check_base_cell

    df = _strat_df(n=2000)
    df["sales_value"] = np.expm1(df["y"] + 3)
    log = ListLogger()
    table, suggested = check_base_cell(df, ("SHOPTYPE",), params.stratify, logger=log)
    assert suggested == ["PLATFORMNAME", "SHOPTYPE"]
    assert (table["base_cell_current"] == "SHOPTYPE").all()
    assert any("建议 base_cell=PLATFORMNAME x SHOPTYPE" in w for w in log.warnings)

    log = ListLogger()
    _, suggested = check_base_cell(df, ("PLATFORMNAME", "SHOPTYPE"), params.stratify, logger=log)
    assert suggested == ["PLATFORMNAME", "SHOPTYPE"] and not log.warnings


def test_profile_stats():
    from core.data_prep.profile import profile

    sc = pd.DataFrame(
        {
            "period_id": [1] * 4,
            "store_id": ["A", "B", "C", "D"],
            "PLATFORMNAME": ["E"] * 4,
            "SHOPTYPE": ["cvs"] * 4,
            "PROVINCE": ["P"] * 4,
            "category": ["X"] * 4,
            "n_item": [1, 2, 3, 4],
            "sales_value": [0.0, 1.0, 2.0, 3.0],
        }
    )
    out = profile(sc)
    row = out[(out["dimension"] == "category")].iloc[0]
    assert row["n_store"] == 4 and row["zero_rate"] == 0.25 and row["median"] == 1.5
    assert set(out["dimension"]) == {"PLATFORMNAME", "SHOPTYPE", "PROVINCE", "category"}


# ---------------------------------------------------------------- robust
def test_medcouple_matches_statsmodels_and_sign():
    from statsmodels.stats.stattools import medcouple as sm_mc

    from funcs.robust import medcouple

    rng = np.random.default_rng(1)
    right = rng.lognormal(size=200)
    assert medcouple(right) == pytest.approx(float(sm_mc(right, use_fast=False)))
    assert medcouple(right) > 0
    assert medcouple(-right) < 0
    assert medcouple([1.0, 2.0]) == 0.0


def test_adjbox_fence_symmetric_equals_tukey_c():
    from funcs.robust import adjbox_fence, flag

    x = np.arange(1, 102, dtype=float)
    lo, hi = adjbox_fence(x, -4, 3, 1.5)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    assert lo == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert hi == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert list(flag([500.0, -500.0, 50.0, np.nan], lo, hi)) == ["high", "low", "ok", "na"]


def test_adjbox_skewed_widens_upper_side():
    from funcs.robust import adjbox_fence

    rng = np.random.default_rng(2)
    x = rng.lognormal(0, 1, 500)
    lo, hi = adjbox_fence(x, -4, 3, 1.5)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    assert hi - q3 > 1.5 * iqr
    assert q1 - lo < 1.5 * iqr


def test_sign_test():
    from funcs.robust import sign_test

    stat, sig, direction = sign_test(40, 10, 1.281552)
    assert sig and direction == "positive" and stat == pytest.approx(29 / np.sqrt(50))
    assert sign_test(12, 10, 1.281552)[1] is False
    assert sign_test(0, 0, 1.281552)[1] is False


def test_fova_longitude_limits_skew_adjusts_one_side():
    from funcs.robust import fova_longitude_limits

    rng = np.random.default_rng(3)
    hist = [rng.lognormal(1, 0.5, 300) for _ in range(5)]
    ll, ul, n = fova_longitude_limits(hist, 1.9992, 0.01)
    assert n == 5
    center = np.median([np.median(h) for h in hist])
    assert ul - center > center - ll
    sym = [np.linspace(1, 3, 101) for _ in range(3)]
    ll, ul, _ = fova_longitude_limits(sym, 1.9992, 0.0)
    assert ul - 2 == pytest.approx(2 - ll)


def test_heterogeneity_same_vs_shifted():
    from funcs.robust import heterogeneity, passes

    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 1, 400), rng.normal(0, 1, 400)
    het = heterogeneity(a, b, 10)
    assert passes(het, 1.2, 0.2, 0.25)
    shifted = heterogeneity(a, b + 3, 10)
    assert not passes(shifted, 1.2, 0.2, 0.25)
    assert shifted["vi"] > 1.2 and shifted["ks"] > 0.2 and shifted["psi"] > 0.25


def test_heterogeneity_small_samples_size_aware():
    from funcs.robust import heterogeneity, passes, psi_bins_for

    assert psi_bins_for(3, 40, 10, 5) == 0 and psi_bins_for(30, 40, 10, 5) == 6 and psi_bins_for(500, 900, 10, 5) == 10
    rng = np.random.default_rng(11)
    pop = rng.normal(0, 1, 5000)
    for n_from in (3, 10, 30):
        ok = [passes(heterogeneity(*np.split(rng.permutation(pop)[: 40 + n_from], [40]), 10), 1.2, 0.2, 0.25)
              for _ in range(200)]
        assert np.mean(ok) > 0.85
    tiny = heterogeneity(pop[:40], pop[40:43], 10)
    assert tiny["psi_bins"] == 0 and np.isnan(tiny["psi"])
    far = heterogeneity(pop[:40], pop[40:60] + 2.5, 10)
    assert not passes(far, 1.2, 0.2, 0.25)


# ---------------------------------------------------------------- ibd_define
class ListLogger:
    def __init__(self):
        self.infos, self.warnings = [], []

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)


PERIODS = (20261405, 20261406, 20261407, 20261408)
GROW = (1.0, 1.1, 1.21, 1.331)
ZIGZAG = (1.0, 0.7, 1.2, 0.8)


def _ibd_sc(categories=("BEER",)):
    spec = {
        "P1": (40, np.linspace(100, 1000, 40), GROW),
        "P2": (20, np.linspace(100, 1000, 20), GROW),
        "P3": (20, np.linspace(5000, 9000, 20), ZIGZAG),
        "P4": (40, np.linspace(100, 1000, 40), ZIGZAG),
    }
    rows = []
    for prov, (n, base, factors) in spec.items():
        for i in range(n):
            for t, period in enumerate(PERIODS):
                for cat in categories:
                    rows.append(
                        {
                            "period_id": period,
                            "store_id": f"{prov}_{i}",
                            "category": cat,
                            "PLATFORMNAME": "ELEME",
                            "SHOPTYPE": "cvs",
                            "PROVINCE": prov,
                            "sales_value": float(base[i] * factors[t]),
                            "sales_unit": 1.0,
                            "n_item": 20,
                        }
                    )
    df = pd.DataFrame(rows)
    df["cat_share"] = df["sales_value"] / df.groupby(["period_id", "store_id"])["sales_value"].transform("sum")
    return df


def test_province_distance_shop_similarity():
    from core.ibd_define.ibd_define import province_distance

    series = pd.DataFrame([[1.0, 2.0, 4.0, 8.0], [3.0, 6.0, 12.0, 24.0], [1.0, np.nan, np.nan, 2.0]], index=["a", "b", "c"])
    d = province_distance(series, 3)
    Da = np.diff(np.log1p(series.loc["a"].to_numpy()))
    Db = np.diff(np.log1p(series.loc["b"].to_numpy()))
    assert d.loc["a", "b"] == pytest.approx(np.linalg.norm(Da - Db) / 3)
    assert np.isnan(d.loc["a", "c"])
    series2 = pd.DataFrame([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]], index=["x", "y"])
    D = np.log(2.0)
    assert province_distance(series2, 3).loc["x", "y"] == pytest.approx(np.sqrt(3 * D**2) / 3)


def test_province_links_ignore_panel_replacement():
    from core.ibd_define.ibd_define import province_links, trend_distance

    rows = []
    for t, period in enumerate(PERIODS):
        for i in range(4):
            rows.append({"period_id": period, "store_id": f"a{i}", "SHOPTYPE": "cvs", "PROVINCE": "A", "sales_value": 100.0 * 1.1**t})
            rows.append({"period_id": period, "store_id": f"b{i}", "SHOPTYPE": "cvs", "PROVINCE": "B", "sales_value": 100.0 * 1.1**t})
        big = f"new{t // 2}"
        rows.append({"period_id": period, "store_id": big, "SHOPTYPE": "cvs", "PROVINCE": "B", "sales_value": 5000.0 * 1.1**t})
    tot = pd.DataFrame(rows)
    D = province_links(tot, ["SHOPTYPE"], 1)
    assert np.allclose(D.to_numpy(), np.log(1.1))
    dist = trend_distance(D.xs("cvs", level=0), 3)
    assert dist.loc["A", "B"] == pytest.approx(0.0)
    strict = province_links(tot, ["SHOPTYPE"], 5)
    assert strict.loc[("cvs", "A")].isna().all()
    assert strict.loc[("cvs", "B")].isna().tolist() == [False, True, False]


def test_merge_distance_cap_blocks_far_candidates(raw_params):
    from core.ibd_define.ibd_define import _merge_cell, distance_cap

    idx = ["S", "N", "F"]
    dist = pd.DataFrame([[0, 1, 5], [1, 0, 5], [5, 5, 0]], index=idx, columns=idx, dtype=float)
    assert distance_cap(dist, 0.5) == pytest.approx(5.0)
    assert distance_cap(dist, 0.3) < 5.0 and distance_cap(dist, 1.0) == np.inf
    rng = np.random.default_rng(3)
    values = {"S": rng.normal(5, 0.3, 10), "N": rng.normal(8, 0.3, 40), "F": rng.normal(5, 0.3, 40)}
    ibd = _override(raw_params, "ibd", max_distance_quantile=1.0).ibd
    groups, merges, _, _ = _merge_cell(idx, values, dist, ibd)
    assert merges[0]["into_members"] == "F" and merges[0]["pooling_reason"] == "accepted"
    ibd = _override(raw_params, "ibd", max_distance_quantile=0.3).ibd
    groups, merges, _, _ = _merge_cell(idx, values, dist, ibd)
    assert merges[0]["into_members"] == "N" and merges[0]["pooling_reason"] == "forced_nearest"


def test_allow_forced_nearest_false_soft_keeps_undersized(raw_params):
    from core.ibd_define.ibd_define import SOFT_KEPT, _merge_cell

    rng = np.random.default_rng(9)
    # Tiny group that fails size tests against a far larger neighbour -> would be forced when allowed.
    values = {"S": rng.normal(5, 0.3, 3), "N": rng.normal(8, 0.3, 40)}
    idx = ["S", "N"]
    dist = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], index=idx, columns=idx)
    forced = _merge_cell(idx, values, dist, _override(raw_params, "ibd", allow_forced_nearest=True, min_store=20, min_store_soft=15).ibd)
    assert any(m["pooling_reason"] == "forced_nearest" for m in forced[1])
    kept = _merge_cell(idx, values, dist, _override(raw_params, "ibd", allow_forced_nearest=False, min_store=20, min_store_soft=15).ibd)
    groups, merges, prov_info, soft = kept
    assert merges == [] and "S" in soft
    assert prov_info["S"]["pooling_reason"] == SOFT_KEPT


def test_passes_rejects_level_gap_in_tiny_samples():
    from funcs.robust import heterogeneity, passes

    rng = np.random.default_rng(8)
    big = rng.normal(8, 1.5, 100)
    tiny = np.array([5.0, 6.0, 7.0, 9.0])
    het = heterogeneity(big, tiny, 10)
    assert het["med_ratio"] < 0.3 and het["ks_p"] > 0.05
    assert passes(het, 1.2, 0.2, 0.25)
    assert not passes(het, 1.2, 0.2, 0.25, 2.0)


def test_merge_soft_stop_keeps_group_between_floor_and_target(raw_params):
    from core.ibd_define.ibd_define import _merge_cell

    rng = np.random.default_rng(4)
    values = {"A": rng.normal(6, 0.3, 22), "B": rng.normal(6, 0.3, 43), "C": rng.normal(6, 0.3, 5)}
    idx = list(values)
    dist = pd.DataFrame(1.0, index=idx, columns=idx)
    np.fill_diagonal(dist.values, 0.0)
    dist.loc["C", "A"] = dist.loc["A", "C"] = 0.5
    ibd = _override(raw_params, "ibd", soft_stop=False).ibd
    groups, _, _, soft = _merge_cell(idx, values, dist, ibd)
    assert len(groups) == 1 and not soft
    ibd = _override(raw_params, "ibd", soft_stop=True).ibd
    groups, _, info, soft = _merge_cell(idx, values, dist, ibd)
    assert sorted(map(sorted, groups.values())) == [["A", "C"], ["B"]]
    assert soft == {"A"} and info["A"]["pooling_reason"] == "soft_kept" and info["C"]["pooling_reason"] == "accepted"


def test_define_ibd_merges_similar_and_forced(raw_params):
    from core.ibd_define.ibd_define import define_ibd

    params = _override(raw_params, "ibd", min_store_soft=30, allow_forced_nearest=True, distance="trend")
    log = ListLogger()
    res = define_ibd(_ibd_sc(), 20261408, params.base_cell, params.ibd, logger=log)
    m = res.ibd_map.set_index("PROVINCE")
    assert m.loc["P1", "ibd_id"] == m.loc["P2", "ibd_id"]
    assert m.loc["P3", "ibd_id"] == m.loc["P4", "ibd_id"]
    assert m.loc["P1", "ibd_id"] != m.loc["P3", "ibd_id"]
    assert m.loc["P2", "pooling_reason"] == "accepted"
    assert m.loc["P3", "pooling_reason"] == "forced_nearest"
    assert m.loc["P1", "pooling_reason"] == "none"
    assert (res.ibd_map["n_store"] >= params.ibd.min_store).all()
    assert set(res.ibd_map.columns) >= {"ibd_id", "member_provinces", "n_store", "vi", "ks", "psi", "distance"}
    assert len(res.store_map) == 120 and res.store_map["ibd_id"].notna().all()
    assert res.forced_ratio == pytest.approx(0.5)
    assert any("forced_nearest_ratio" in w for w in log.warnings)
    assert ("ELEME", "cvs") in res.distances
    assert not m.loc["P3", "final_ok"] and not m.loc["P4", "final_ok"] and m.loc["P1", "final_ok"]
    assert any("合并后复查 2 省不通过" in w and "P3" in w for w in log.warnings)


def test_define_ibd_category_mix_pools_similar_taste(raw_params):
    from core.ibd_define.ibd_define import define_ibd, mix_distance, province_mix_profile

    rng = np.random.default_rng(5)
    mix = {"N1": (0.6, 0.4, 0.0), "N2": (0.6, 0.4, 0.0), "S1": (0.4, 0.2, 0.4), "S2": (0.4, 0.2, 0.4)}
    rows = []
    for prov, shares in mix.items():
        for i in range(18):
            size = float(rng.lognormal(7, 0.3))
            for cat, share in zip(("BEER", "CSD", "COFF"), shares):
                if share:
                    rows.append({"period_id": 20261408, "store_id": f"{prov}_{i}", "PLATFORMNAME": "ELEME", "SHOPTYPE": "cvs",
                                 "PROVINCE": prov, "category": cat, "sales_value": size * share, "cat_share": share})
    sc = pd.DataFrame(rows)
    prof = province_mix_profile(sc, ["PLATFORMNAME", "SHOPTYPE"])
    assert prof.loc[("ELEME", "cvs", "N1"), "COFF"] == 0.0
    d = mix_distance(prof.xs(("ELEME", "cvs")))
    assert d.loc["N1", "N2"] == pytest.approx(0.0) and d.loc["N1", "S1"] > 1.0
    params = _override(raw_params, "ibd", distance="category_mix")
    res = define_ibd(sc, 20261408, params.base_cell, params.ibd)
    ids = res.ibd_map.set_index("PROVINCE")["ibd_id"]
    assert ids["N1"] == ids["N2"] and ids["S1"] == ids["S2"] and ids["N1"] != ids["S1"]


def test_province_mix_lookback_weights_and_start_cut(raw_params):
    from core.ibd_define.ibd_define import mix_lookback_periods, mix_period_weights, province_mix_profile

    assert mix_lookback_periods([6, 7, 8], 8, 3, 0) == [6, 7, 8]
    assert mix_lookback_periods([6, 7, 8], 8, 3, 7) == [7, 8]
    assert mix_lookback_periods([6, 7, 8], 6, 3, 7) == [6]
    w = mix_period_weights(3, [0.2, 0.3, 0.5])
    assert w.tolist() == pytest.approx([0.2, 0.3, 0.5])
    rec = mix_period_weights(3, [])
    assert rec.tolist() == pytest.approx([1 / 7, 2 / 7, 4 / 7])

    rows = []
    for period, ice in ((20261406, 0.1), (20261407, 0.2), (20261408, 0.6)):
        for i in range(4):
            rows.append({"period_id": period, "store_id": f"s{i}", "PLATFORMNAME": "ELEME", "SHOPTYPE": "cvs",
                         "PROVINCE": "P", "category": "ICER", "sales_value": ice, "cat_share": ice})
            rows.append({"period_id": period, "store_id": f"s{i}", "PLATFORMNAME": "ELEME", "SHOPTYPE": "cvs",
                         "PROVINCE": "P", "category": "CSD", "sales_value": 1 - ice, "cat_share": 1 - ice})
    sc = pd.DataFrame(rows)
    one = province_mix_profile(sc, ["PLATFORMNAME", "SHOPTYPE"], period=20261408)
    params = _override(raw_params, "ibd", mix_lookback=3, mix_weights=[0.2, 0.3, 0.5])
    avg = province_mix_profile(sc, ["PLATFORMNAME", "SHOPTYPE"], ibd=params.ibd, period=20261408)
    expect = 0.2 * np.log1p(1000 * 0.1) + 0.3 * np.log1p(1000 * 0.2) + 0.5 * np.log1p(1000 * 0.6)
    assert avg.loc[("ELEME", "cvs", "P"), "ICER"] == pytest.approx(expect)
    assert one.loc[("ELEME", "cvs", "P"), "ICER"] == pytest.approx(np.log1p(1000 * 0.6))
    cut = _override(raw_params, "ibd", mix_lookback=3, mix_weights=[0.5, 0.5], mix_start_period=20261407)
    cut_p = province_mix_profile(sc, ["PLATFORMNAME", "SHOPTYPE"], ibd=cut.ibd, period=20261408)
    expect_cut = 0.5 * np.log1p(1000 * 0.2) + 0.5 * np.log1p(1000 * 0.6)
    assert cut_p.loc[("ELEME", "cvs", "P"), "ICER"] == pytest.approx(expect_cut)

    extra = sc[sc["period_id"] == 20261408].copy()
    extra["store_id"] = "only08"
    extra["cat_share"] = np.where(extra["category"] == "ICER", 0.9, 0.1)
    sc2 = pd.concat([sc, extra], ignore_index=True)
    both = _override(raw_params, "ibd", mix_lookback=3, mix_common_stores=True, mix_weights=[1, 1, 1])
    common = province_mix_profile(sc2, ["PLATFORMNAME", "SHOPTYPE"], ibd=both.ibd, period=20261408)
    noc = _override(raw_params, "ibd", mix_lookback=1)
    # common-store mix should ignore only08 (not in 06/07), so ICER below the 08-only spike
    cur = province_mix_profile(sc2, ["PLATFORMNAME", "SHOPTYPE"], ibd=noc.ibd, period=20261408)
    assert common.loc[("ELEME", "cvs", "P"), "ICER"] < cur.loc[("ELEME", "cvs", "P"), "ICER"]
    stay = sc[sc["period_id"].isin([20261407, 20261408])].copy()
    stay["store_id"] = "s07_08"
    stay["cat_share"] = np.where(stay["category"] == "ICER", 0.9, 0.1)
    sc3 = pd.concat([sc, stay], ignore_index=True)
    c3 = _override(raw_params, "ibd", mix_lookback=3, mix_common_stores=True, mix_common_lookback=0, mix_weights=[1, 1, 1])
    c2 = _override(raw_params, "ibd", mix_lookback=3, mix_common_stores=True, mix_common_lookback=2, mix_weights=[1, 1, 1])
    p3 = province_mix_profile(sc3, ["PLATFORMNAME", "SHOPTYPE"], ibd=c3.ibd, period=20261408)
    p2 = province_mix_profile(sc3, ["PLATFORMNAME", "SHOPTYPE"], ibd=c2.ibd, period=20261408)
    assert p2.loc[("ELEME", "cvs", "P"), "ICER"] > p3.loc[("ELEME", "cvs", "P"), "ICER"]


def test_params_mix_lookback_invalid(raw_params):
    bad = copy.deepcopy(raw_params)
    bad["ibd"]["mix_lookback"] = 0
    with pytest.raises(ValueError, match="mix_lookback"):
        params_from_dict(bad)


def test_combine_distances_scales_and_falls_back_to_mix():
    from core.ibd_define.ibd_define import combine_distances

    idx = ["A", "B", "C"]
    mix = pd.DataFrame([[0, 1, 3], [1, 0, 1], [3, 1, 0]], index=idx, columns=idx, dtype=float)
    trend = pd.DataFrame([[0, 10, np.inf], [10, 0, 30], [np.inf, 30, 0]], index=idx, columns=idx, dtype=float)
    d = combine_distances(mix, trend, 0.5)
    assert d.loc["A", "B"] == pytest.approx(0.5 * 1 + 0.5 * 0.5)
    assert d.loc["B", "C"] == pytest.approx(0.5 * 1 + 0.5 * 1.5)
    assert d.loc["A", "C"] == pytest.approx(3.0)


def test_define_ibd_by_size_bands_on_history(raw_params):
    from core.ibd_define.ibd_check import check_ibd
    from core.ibd_define.ibd_define import define_ibd

    params = _override(raw_params, "ibd", method="size", allow_forced_nearest=True)
    sc = _ibd_sc()
    spike = (sc["store_id"] == "P1_0") & (sc["period_id"] == 20261408)
    sc.loc[spike, "sales_value"] *= 100
    new = sc[(sc["store_id"] == "P1_1") & (sc["period_id"] == 20261408)].assign(store_id="NEW", sales_value=99999.0)
    sc = pd.concat([sc, new], ignore_index=True)
    log = ListLogger()
    res = define_ibd(sc, 20261408, params.base_cell, params.ibd, logger=log)
    m = res.ibd_map.set_index("band")
    assert list(m.index) == ["Small", "SmallMedium", "LargeMedium", "Large"]
    assert (m["n_store"] >= params.ibd.min_store).all() and m["n_store"].sum() == 121
    assert m.loc["Large", "n_size_from_current"] == 1
    smap = res.store_map.set_index("store_id")["ibd_id"]
    assert smap["P1_0"].endswith("|Small") and smap["NEW"].endswith("|Large")
    assert res.forced_ratio == 0.0 and any("4 档" in s for s in log.infos)
    s = res.summary.iloc[0]
    assert s["cv_value_bands"] < s["cv_value_cell"]

    cur = sc[sc["period_id"] == 20261408]
    items = cur.assign(nankey=1)[["period_id", "store_id", "category", "sales_value", "nankey"]]
    merge, _ = check_ibd(cur, items, res, "nankey", params.ibd)
    assert (merge["peer_level"] == "ibd").all()
    few = _override(raw_params, "ibd", method="size", check_level="category", min_store_cat=35)
    merge, _ = check_ibd(cur, items, res, "nankey", few.ibd)
    assert (merge["peer_level"] != "ibd").all()


def test_define_ibd_soft_kept_instead_of_forced(raw_params):
    from core.ibd_define.ibd_check import check_ibd
    from core.ibd_define.ibd_define import define_ibd

    # Fixture pins 30/20 for synthetic sizes; production JSON is 10/5 (see test_load_params_real_json).
    params = _override(raw_params, "ibd", min_store=30, min_store_soft=20, allow_forced_nearest=False)
    assert params.ibd.min_store_soft == 20
    sc = _ibd_sc()
    log = ListLogger()
    res = define_ibd(sc, 20261408, params.base_cell, params.ibd, logger=log)
    m = res.ibd_map.set_index("PROVINCE")
    assert m.loc["P3", "pooling_reason"] == "soft_kept" and m.loc["P3", "soft_kept"]
    assert m.loc["P3", "member_provinces"] == "P3" and m.loc["P3", "n_store"] == 20
    assert m.loc["P4", "pooling_reason"] == "none" and not m.loc["P4", "soft_kept"]
    assert res.forced_ratio == 0.0 and m["final_ok"].all()
    assert any("全部省份通过" in s for s in log.infos)
    cur = sc[sc["period_id"] == 20261408]
    items = cur.assign(nankey=1)[["period_id", "store_id", "category", "sales_value", "nankey"]]
    merge, _ = check_ibd(cur, items, res, "nankey", params.ibd)
    p3 = merge[merge["ibd_id_raw"] == m.loc["P3", "ibd_id"]].iloc[0]
    assert not p3["low_store"] and p3["peer_level"] == "ibd"


def test_params_min_store_soft_validated(raw_params):
    bad = copy.deepcopy(raw_params)
    bad["ibd"]["min_store_soft"] = 31
    with pytest.raises(ValueError, match="min_store_soft"):
        params_from_dict(bad)


# ---------------------------------------------------------------- ibd_check
def _check_inputs(params):
    from core.ibd_define.ibd_define import define_ibd

    sc = _ibd_sc()
    define = define_ibd(sc, 20261408, params.base_cell, params.ibd)
    ids = define.ibd_map.set_index("PROVINCE")["ibd_id"]
    g1, g2 = ids["P1"], ids["P4"]
    cur = sc[sc["period_id"] == 20261408].copy()
    extra = []

    def add(cat, prov, n, values):
        for i in range(n):
            extra.append({**cur[cur["store_id"] == f"{prov}_{i}"].iloc[0].to_dict(), "category": cat, "sales_value": float(values[i % len(values)])})

    add("CAT_M", "P1", 6, [10, 20, 30])
    add("CAT_M", "P4", 8, [10, 20, 30])
    add("CAT_NA", "P1", 5, [10, 20, 30])
    add("CAT_NA", "P4", 5, [9000, 9500, 9900])
    add("CAT_BASE", "P1", 25, np.linspace(10, 100, 25))
    add("CAT_BASE", "P4", 8, [9000, 9500, 9900])
    sc_cur = pd.concat([cur, pd.DataFrame(extra)], ignore_index=True)
    items = sc_cur[["store_id", "category"]].merge(pd.DataFrame({"nankey": range(12)}), how="cross")
    items["sales_value"] = 1.0
    return define, sc_cur, items, g1, g2


def _override(raw_params, section, **values):
    raw = copy.deepcopy(raw_params)
    raw[section].update(values)
    return params_from_dict(raw)


def test_ibd_check_merge_fallback_and_unchanged(raw_params):
    from core.ibd_define.ibd_check import check_ibd

    params = _override(raw_params, "ibd", check_level="category")
    define, sc_cur, items, g1, g2 = _check_inputs(params)
    log = ListLogger()
    merge, summary = check_ibd(sc_cur, items, define, "nankey", params.ibd, logger=log)
    m = merge.set_index(["ibd_id_raw", "category"])

    beer = m.loc[(g1, "BEER")]
    assert beer["peer_level"] == "ibd" and beer["pooling_reason"] == "none"
    assert beer["ibd_id_merged"] == g1
    for raw, merged in (("member_provinces_raw", "member_provinces_merged"), ("n_store_raw", "n_store_merged"),
                        ("n_store_cat_raw", "n_store_cat_merged"), ("n_item_raw", "n_item_merged")):
        assert beer[raw] == beer[merged]

    cm = m.loc[(g2, "CAT_M")]
    assert cm["low_store_cat"] and cm["peer_level"] == "ibd_merged" and cm["pooling_reason"] == "merge_nearest_ibd"
    assert cm["ibd_id_merged"] == "+".join(sorted([g1, g2])) and cm["n_store_cat_merged"] == 14
    assert np.isfinite(cm["vi"])

    base = m.loc[(g2, "CAT_BASE")]
    assert base["peer_level"] == "base_cell" and base["ibd_id_merged"] == "base_cell:ELEME|cvs"
    assert m.loc[(g1, "CAT_BASE")]["peer_level"] == "ibd"

    na = m.loc[(g2, "CAT_NA")]
    assert na["peer_level"] == "na" and na["ibd_id_merged"] == "na"

    s = dict(zip(summary["item"], summary["value"]))
    assert s["unchanged"] == int((merge["ibd_id_merged"] == merge["ibd_id_raw"]).sum())
    assert any("unchanged" in msg for msg in log.infos)


def test_ibd_check_level_ibd_ignores_category_counts(raw_params):
    from core.ibd_define.ibd_check import check_ibd

    params = _override(raw_params, "ibd", check_level="ibd")
    define, sc_cur, items, g1, g2 = _check_inputs(params)
    merge, _ = check_ibd(sc_cur, items, define, "nankey", params.ibd)
    assert (merge["peer_level"] == "ibd").all()
    assert (merge["ibd_id_merged"] == merge["ibd_id_raw"]).all()
    cm = merge.set_index(["ibd_id_raw", "category"]).loc[(g2, "CAT_NA")]
    assert cm["low_store_cat"] and cm["pooling_reason"] == "none"

    biggest = int(define.ibd_map["n_store"].max())
    base, _ = check_ibd(sc_cur, items, define, "nankey", _override(raw_params, "ibd", min_store=biggest + 1, min_store_soft=biggest + 1).ibd)
    assert (base["peer_level"] == "base_cell").all()
    na, _ = check_ibd(sc_cur, items, define, "nankey", _override(raw_params, "ibd", min_store=121, min_store_soft=121).ibd)
    assert (na["peer_level"] == "na").all()


def test_check_level_validated(raw_params):
    with pytest.raises(ValueError):
        _override(raw_params, "ibd", check_level="store")
    with pytest.raises(TypeError):
        _override(raw_params, "peer", evaluate_zero_rows=1)


def test_zero_fill_categories():
    from core.data_prep.data_prep import zero_fill_categories

    base = {"period_id": 20261408, "PLATFORMNAME": "ELEME", "SHOPTYPE": "cvs", "PROVINCE": "P1",
            "sales_unit": 1.0, "n_item": 2, "avg_price": 5.0, "cat_share": 0.5}
    sc = pd.DataFrame([
        {**base, "store_id": "A", "category": "X", "sales_value": 10.0},
        {**base, "store_id": "A", "category": "Y", "sales_value": 10.0},
        {**base, "store_id": "B", "category": "X", "sales_value": 20.0},
        {**base, "store_id": "C", "category": "Z", "sales_value": 30.0},
    ])
    out = zero_fill_categories(sc, pd.Series({"A": "I1", "B": "I1", "C": "I2"}))
    assert len(out) == 5
    filled = out[out["zero_filled"]]
    assert list(zip(filled["store_id"], filled["category"])) == [("B", "Y")]
    r = filled.iloc[0]
    assert r["sales_value"] == 0 and r["cat_share"] == 0 and r["n_item"] == 0 and np.isnan(r["avg_price"])
    assert r["PROVINCE"] == "P1"
    assert not ((out["store_id"] == "C") & (out["category"] != "Z")).any()


def test_ibd_merge_excel_roundtrip(raw_params, tmp_path):
    from core.data_prep.data_prep import write_excel
    from core.ibd_define.ibd_check import check_ibd, peer_assignment

    params = _override(raw_params, "ibd", check_level="category")
    define, sc_cur, items, g1, g2 = _check_inputs(params)
    merge, _ = check_ibd(sc_cur, items, define, "nankey", params.ibd)
    dest = write_excel({"ibd_map": define.ibd_map, "ibd_merge": merge}, tmp_path / "o.xlsx")
    back = pd.read_excel(dest, sheet_name="ibd_merge")
    assert {"ibd_id_raw", "ibd_id_merged", "peer_level", "pooling_reason"} <= set(back.columns)
    assigned = peer_assignment(sc_cur, define.store_map, merge)
    assert assigned["peer_group_id"].notna().all()
    row = assigned[(assigned["category"] == "CAT_M") & (assigned["ibd_id"] == g2)].iloc[0]
    assert row["peer_level"] == "ibd_merged"


# ---------------------------------------------------------------- seasonal
def _season_panel(group, prev_vals, cur_vals, base_extra=None):
    rows = []
    for i, (p, c) in enumerate(zip(prev_vals, cur_vals)):
        for period, v in ((20261407, p), (20261408, c)):
            rows.append({"period_id": period, "store_id": f"{group}_{i}", "peer_group_id": group, "category": "X",
                         "sales_value": float(v), "n_item": 10.0, "avg_price": 5.0})
    if base_extra is not None:
        for i, v in enumerate(base_extra):
            rows.append({"period_id": 20261406, "store_id": f"{group}_{i}", "peer_group_id": group, "category": "X",
                         "sales_value": float(v), "n_item": 10.0, "avg_price": 5.0})
    return rows


def test_compute_seasonal_reasons(params):
    from core.outlier_rules.seasonal import compute_seasonal

    rows = []
    rows += _season_panel("OK", [100] * 20, [130] * 18 + [90] * 2)
    rows += _season_panel("NS", [100] * 20, [110] * 11 + [90] * 9)
    rows += _season_panel("FEW", [100] * 5, [150] * 5)
    rows += _season_panel("CONF", [100] * 18 + [5000] * 2, [101] * 18 + [100] * 2)
    table = compute_seasonal(pd.DataFrame(rows), 20261408, params.seasonal)
    sv = table[table["metric"] == "sales_value"].set_index("peer_group_id")
    assert sv.loc["OK", "reason"] == "ok" and sv.loc["OK", "T_final"] == pytest.approx(sv.loc["OK", "T_raw"])
    assert sv.loc["OK", "T_raw"] == pytest.approx((18 * 130 + 2 * 90) / 20 / 100)
    assert (sv.loc["OK", "n_plus"], sv.loc["OK", "n_minus"]) == (18, 2)
    assert sv.loc["NS", "reason"] == "not_significant" and sv.loc["NS", "T_final"] == 1.0
    assert sv.loc["FEW", "reason"] == "few_common_stores" and sv.loc["FEW", "T_final"] == 1.0
    assert sv.loc["CONF", "reason"] == "direction_conflict" and sv.loc["CONF", "T_final"] == 1.0
    n_item = table[(table["metric"] == "n_item") & (table["peer_group_id"] == "OK")].iloc[0]
    assert n_item["T_final"] == 1.0


def test_compute_seasonal_chained_index_baseline(params):
    from core.outlier_rules.seasonal import compute_seasonal

    rows = _season_panel("OK", [100] * 20, [130] * 20, base_extra=[60] * 20)
    table = compute_seasonal(pd.DataFrame(rows), 20261408, params.seasonal)
    r = table[(table["metric"] == "sales_value")].iloc[0]
    assert r["idx_base"] == pytest.approx((1 + 100 / 60) / 2) and r["base_periods_used"] == 2
    assert r["link_t"] == pytest.approx(1.3) and r["source"] == "group"
    assert r["T_raw"] == pytest.approx(130 / 80)


def test_compute_seasonal_robust_to_store_panel_change(params):
    from core.outlier_rules.seasonal import compute_seasonal

    rows = []
    for i in range(20):
        rows.append({"period_id": 20261406, "store_id": f"old{i}", "peer_group_id": "G", "category": "X", "sales_value": 1000.0, "n_item": 10.0, "avg_price": 5.0})
    for i in range(20):
        rows.append({"period_id": 20261406, "store_id": f"new{i}", "peer_group_id": "G", "category": "X", "sales_value": 100.0, "n_item": 10.0, "avg_price": 5.0})
    for i in range(60):
        rows.append({"period_id": 20261407, "store_id": f"new{i}", "peer_group_id": "G", "category": "X", "sales_value": 110.0, "n_item": 10.0, "avg_price": 5.0})
        rows.append({"period_id": 20261408, "store_id": f"new{i}", "peer_group_id": "G", "category": "X", "sales_value": 121.0 if i < 50 else 110.0, "n_item": 10.0, "avg_price": 5.0})
    r = compute_seasonal(pd.DataFrame(rows), 20261408, params.seasonal).query("metric == 'sales_value'").iloc[0]
    link_08 = (50 * 121 + 10 * 110) / (60 * 110)
    assert r["idx_t"] == pytest.approx(1.1 * link_08)
    assert r["T_raw"] == pytest.approx(1.1 * link_08 / ((1 + 1.1) / 2))
    assert r["reason"] == "ok"


def test_compute_seasonal_borrows_base_cell(params):
    from core.outlier_rules.seasonal import compute_seasonal

    rows = []
    for grp, n in (("SMALL", 4), ("BIG", 30)):
        for i in range(n):
            for period, v in ((20261407, 100.0), (20261408, 150.0 if grp == "BIG" else 120.0)):
                rows.append({"period_id": period, "store_id": f"{grp}{i}", "peer_group_id": grp, "category": "X", "sales_value": v,
                             "n_item": 10.0, "avg_price": 5.0, "PLATFORMNAME": "ELEME", "SHOPTYPE": "cvs"})
    panel = pd.DataFrame(rows)
    t = compute_seasonal(panel, 20261408, params.seasonal, base_cell=("PLATFORMNAME", "SHOPTYPE"))
    s = t[t["metric"] == "sales_value"].set_index("peer_group_id")
    assert s.loc["SMALL", "source"] == "base_cell" and s.loc["SMALL", "n_links_borrowed"] == 1
    assert s.loc["SMALL", "link_t"] == pytest.approx((30 * 150 + 4 * 120) / 3400)
    assert s.loc["BIG", "source"] == "group" and s.loc["BIG", "T_raw"] == pytest.approx(1.5)
    no = compute_seasonal(panel, 20261408, params.seasonal)
    assert no.set_index(["metric", "peer_group_id"]).loc[("sales_value", "SMALL"), "reason"] == "few_common_stores"

# ---------------------------------------------------------------- peer_check
def _peer_inputs(n=40, spike=5000.0, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for t, period in enumerate(PERIODS):
        for i in range(n):
            v = float(rng.lognormal(5, 0.3))
            share = float(rng.uniform(0.25, 0.35))
            if i == 0 and period == PERIODS[-1]:
                v *= spike
                share = 0.95
            rows.append({"period_id": period, "store_id": f"S{i}", "category": "X", "PLATFORMNAME": "ELEME",
                         "SHOPTYPE": "cvs", "PROVINCE": "P1", "sales_value": v, "sales_unit": v / 10,
                         "n_item": 20, "avg_price": 10.0 * float(rng.uniform(0.95, 1.05)), "cat_share": share,
                         "ibd_id": "I", "peer_group_id": "G", "peer_level": "ibd", "member_ibds": "I"})
    panel = pd.DataFrame(rows)
    assigned = panel[panel["period_id"] == PERIODS[-1]].reset_index(drop=True)
    return assigned, panel


NO_BORROW = dict(hist_start_filter=True, trend_fence_fallback=False, longitude_borrow_cell=False)


def test_fova_combine_table():
    from core.outlier_rules.peer_check import fova_combine

    assert fova_combine("high", "high") == "high"
    assert fova_combine("high", "ok") == "high"
    assert fova_combine("ok", "low") == "low"
    assert fova_combine("ok", "ok") == "ok"
    assert fova_combine("high", "low") == "high"
    assert fova_combine("na", "low") == "low"
    assert fova_combine("low", "na") == "low"
    assert fova_combine("na", "na") == "na"


def test_run_peer_check_hist_start_period(raw_params):
    from core.outlier_rules.peer_check import run_peer_check

    assigned, panel = _peer_inputs()
    params = _override(raw_params, "fova", hist_start_period=PERIODS[1], **NO_BORROW)
    detail = run_peer_check(assigned, panel, pd.DataFrame(), PERIODS[-1], params)
    assert (detail["n_hist"] == 2).all() and (detail["fova_longitude"] == "na").all()
    s0 = detail[detail["store_id"] == "S0"].iloc[0]
    assert s0["fova_final"] == s0["fova_trend"] == "high"


def test_run_peer_check_flags_spike(raw_params):
    from core.outlier_rules.peer_check import run_peer_check
    from core.outlier_rules.seasonal import compute_seasonal

    params = _override(raw_params, "fova", hist_start_period=0)
    assigned, panel = _peer_inputs()
    seas = compute_seasonal(panel, PERIODS[-1], params.seasonal)
    log = ListLogger()
    detail = run_peer_check(assigned, panel, seas, PERIODS[-1], params, logger=log)
    s0 = detail[detail["store_id"] == "S0"].iloc[0]
    for col in ("adjbox_value", "share_adjbox", "fova_longitude", "fova_trend", "fova_final"):
        assert s0[col] == "high", col
    others = detail[detail["store_id"] != "S0"]
    assert (others["adjbox_value"] == "ok").mean() > 0.9
    assert s0["n_hist"] == 3
    assert any("hist_periods < 13" in w for w in log.warnings)


def test_price_adjbox_is_not_computed_or_used_as_evidence(params):
    from core.outlier_rules.evidence import EVIDENCE_COLS
    from core.outlier_rules.peer_check import run_peer_check

    assigned, panel = _peer_inputs()
    detail = run_peer_check(assigned, panel, pd.DataFrame(), PERIODS[-1], params)
    assert "price_adjbox" not in detail.columns
    assert "price_adjbox" not in EVIDENCE_COLS


def test_robust_z_is_not_computed_or_used_as_evidence(params):
    from core.outlier_rules.evidence import EVIDENCE_COLS
    from core.outlier_rules.peer_check import run_peer_check
    from core.outlier_rules.stability import threshold_jaccard

    assigned, panel = _peer_inputs()
    detail = run_peer_check(assigned, panel, pd.DataFrame(), PERIODS[-1], params)
    assert {"robust_z", "robust_z_value", "robust_basis"}.isdisjoint(detail.columns)
    assert "robust_z" not in EVIDENCE_COLS
    assert set(threshold_jaccard(detail, params)["detector"]) == {"adjbox_c"}


def test_run_peer_check_na_level_and_short_history(params):
    from core.outlier_rules.peer_check import run_peer_check

    assigned, panel = _peer_inputs()
    assigned.loc[assigned["store_id"] == "S1", "peer_level"] = "na"
    short = panel[panel["period_id"] >= PERIODS[-2]]
    detail = run_peer_check(assigned, short, pd.DataFrame(), PERIODS[-1], params)
    assert (detail.loc[detail["store_id"] == "S1", ["adjbox_value", "fova_final"]] == "na").all(axis=None)
    assert (detail["fova_longitude"] == "na").all()
    s0 = detail[detail["store_id"] == "S0"].iloc[0]
    assert s0["fova_final"] == s0["fova_trend"] == "high"


def _zero_rows(assigned, panel, ids):
    zero = {"sales_value": 0.0, "sales_unit": 0.0, "n_item": 0, "avg_price": np.nan, "cat_share": 0.0}
    mask = assigned["store_id"].isin(ids)
    assigned = assigned.assign(zero_filled=mask)
    assigned.loc[mask, list(zero)] = list(zero.values())
    cur = panel["period_id"] == PERIODS[-1]
    panel = pd.concat([panel[~cur], assigned.drop(columns="zero_filled")], ignore_index=True)
    return assigned, panel


def test_run_peer_check_zero_rows_in_reference(raw_params):
    from core.outlier_rules.peer_check import run_peer_check

    assigned, panel = _peer_inputs()
    assigned, panel = _zero_rows(assigned, panel, [f"S{i}" for i in range(30, 40)])
    detail = run_peer_check(assigned, panel, pd.DataFrame(), PERIODS[-1], _override(raw_params, "peer", evaluate_zero_rows=False))
    assert len(detail) == 30 and not detail["zero_filled"].any()
    assert (detail["n_peer"] == 40).all() and (detail["n_sell"] == 30).all()
    assert detail.loc[detail["store_id"] == "S0", "adjbox_value"].iloc[0] == "high"

    full = run_peer_check(assigned, panel, pd.DataFrame(), PERIODS[-1], _override(raw_params, "peer", evaluate_zero_rows=True))
    assert len(full) == 40 and full["zero_filled"].sum() == 10


def test_run_peer_check_degenerate_when_most_zero(params):
    from core.outlier_rules.peer_check import run_peer_check

    assigned, panel = _peer_inputs()
    half, half_panel = _zero_rows(assigned, panel, [f"S{i}" for i in range(15, 40)])
    detail = run_peer_check(half, half_panel, pd.DataFrame(), PERIODS[-1], params)
    assert len(detail) == 15
    assert (detail["adjbox_basis"] == "all").all()

    most, most_panel = _zero_rows(*_peer_inputs(n=60), [f"S{i}" for i in range(12, 60)])
    detail = run_peer_check(most, most_panel, pd.DataFrame(), PERIODS[-1], params)
    assert len(detail) == 12 and (detail["adjbox_basis"] == "sellers").all()
    s0 = detail[detail["store_id"] == "S0"].iloc[0]
    assert s0["adjbox_value"] == "high" and s0["share_adjbox"] == "high"
    assert (detail.loc[detail["store_id"] != "S0", "adjbox_value"] == "ok").mean() > 0.8

    few, few_panel = _zero_rows(assigned, panel, [f"S{i}" for i in range(8, 40)])
    detail = run_peer_check(few, few_panel, pd.DataFrame(), PERIODS[-1], params)
    assert len(detail) == 8 and (detail["adjbox_basis"] == "none").all()
    assert (detail["adjbox_value"] == "na").all() and (detail["share_adjbox"] == "na").all()


def test_run_peer_check_cell_sellers_fallback(params):
    from core.outlier_rules.peer_check import run_peer_check

    assigned, panel = _peer_inputs()
    rare, rare_panel = _zero_rows(assigned, panel, [f"S{i}" for i in range(3, 40)])
    other = assigned.assign(peer_group_id="G2", ibd_id="I2", member_ibds="I2", store_id="T" + assigned["store_id"].str[1:])
    other.loc[other["store_id"] == "T0", ["sales_value", "cat_share"]] = [150.0, 0.3]
    other = other.assign(zero_filled=False)
    cur = pd.concat([rare, other], ignore_index=True)
    full_panel = pd.concat([rare_panel, other.drop(columns="zero_filled")], ignore_index=True)
    detail = run_peer_check(cur, full_panel, pd.DataFrame(), PERIODS[-1], params)
    g = detail[detail["peer_group_id"] == "G"].set_index("store_id")
    assert len(g) == 3 and (g["adjbox_basis"] == "cell_sellers").all()
    assert g.loc["S0", "adjbox_value"] == "high"
    assert (g.loc[["S1", "S2"], "adjbox_value"] == "ok").all()
    assert (detail.loc[detail["peer_group_id"] == "G2", "adjbox_basis"] == "all").all()


def _thin_group_panel(n=40, seed=5, thin_n=3):
    """One large group G plus a thin group GSMALL that shares the same base_cell x category."""
    _, panel = _peer_inputs(n=n, seed=seed)
    thin_ids = [f"S{i}" for i in range(n - thin_n, n)]
    panel.loc[panel["store_id"].isin(thin_ids), "peer_group_id"] = "GSMALL"
    assigned = panel[panel["period_id"] == PERIODS[-1]].reset_index(drop=True)
    return assigned, panel


def test_longitude_parts_matches_fova_longitude_limits():
    from core.outlier_rules.peer_check import _longitude_parts
    from funcs.robust import fova_longitude_limits

    rng = np.random.default_rng(11)
    arrays = [rng.lognormal(4.0, 0.5, 30), rng.lognormal(4.3, 0.7, 22), rng.lognormal(3.9, 0.4, 17)]
    coef, trim_pct = 1.9992, 0.01
    ll, ul, n = fova_longitude_limits(arrays, coef, trim_pct)
    center, lo_w, hi_w, n_parts = _longitude_parts(arrays, coef, trim_pct)
    assert n_parts == n
    assert center - coef * lo_w == pytest.approx(ll)
    assert center + coef * hi_w == pytest.approx(ul)


def test_apply_fence_fallback_swaps_thin_group_for_cell():
    from core.outlier_rules.peer_check import _apply_fence_fallback

    ref = pd.DataFrame({
        "td_n": [2.0, 40.0], "c_td_n": [40.0, 40.0],
        "td_q1": [9.0, 1.0], "td_q3": [9.5, 2.0], "td_mc": [0.0, 0.1],
        "c_td_q1": [1.0, 1.0], "c_td_q3": [2.0, 2.0], "c_td_mc": [0.1, 0.1],
    })
    out = _apply_fence_fallback(ref, min_n=10, min_peer=5)
    assert list(out["td_fence_src"]) == ["cell", "group"]
    assert list(out["td_q1"]) == [1.0, 1.0] and list(out["td_q3"]) == [2.0, 2.0]
    assert list(out["td_mc"]) == [0.1, 0.1]
    assert list(ref["td_q1"]) == [9.0, 1.0]                # the caller's frame is left untouched

    thin_cell = ref.assign(c_td_n=[3.0, 40.0])             # cell itself too small to borrow from
    out2 = _apply_fence_fallback(thin_cell, min_n=10, min_peer=5)
    assert list(out2["td_fence_src"]) == ["thin", "group"]
    assert list(out2["td_q1"]) == [9.0, 1.0]               # group stats untouched

    plain = ref.drop(columns=["c_td_q1", "c_td_q3", "c_td_mc", "c_td_n"])
    out3 = _apply_fence_fallback(plain, min_n=10, min_peer=5)
    assert list(out3["td_fence_src"]) == ["thin", "group"]


def test_trend_fence_falls_back_to_cell_for_thin_group(raw_params):
    from core.outlier_rules.peer_check import run_peer_check

    assigned, panel = _thin_group_panel()
    off = run_peer_check(assigned, panel, pd.DataFrame(), PERIODS[-1], _override(raw_params, "fova", hist_start_period=0, **NO_BORROW))
    thin = off[off["peer_group_id"] == "GSMALL"]
    assert (thin["td_fence_src"] == "thin").all() and (thin["td_n"] == 3).all()
    assert (off.loc[off["peer_group_id"] == "G", "td_fence_src"] == "group").all()

    on = run_peer_check(
        assigned, panel, pd.DataFrame(), PERIODS[-1],
        _override(raw_params, "fova", **{**NO_BORROW, "hist_start_period": 0, "trend_fence_fallback": True}),
    )
    thin_on = on[on["peer_group_id"] == "GSMALL"]
    assert (thin_on["td_fence_src"] == "cell").all()
    assert (thin_on["c_td_n"] == 40).all()
    assert (on.loc[on["peer_group_id"] == "G", "td_fence_src"] == "group").all()
    # the thin group no longer determines its own fence: it now matches the cell distribution
    # (cell = all stores of ELEME x cvs x X, i.e. both groups, T = 1 since no seasonal table)
    from funcs.robust import adjbox_stats

    cur_v = panel.loc[panel["period_id"] == PERIODS[-1]].set_index("store_id")["sales_value"]
    prev_v = panel.loc[panel["period_id"] == PERIODS[-2]].set_index("store_id")["sales_value"]
    cell_q1 = adjbox_stats((np.log1p(cur_v) - np.log1p(prev_v)).to_numpy(dtype=float))[0]
    assert thin_on["td_q1"].iloc[0] == pytest.approx(cell_q1)
    own_q1 = off.loc[off["peer_group_id"] == "GSMALL", "td_q1"].iloc[0]
    assert thin_on["td_q1"].iloc[0] != pytest.approx(own_q1)


def test_adjbox_cell_shrink_blends_thin_group(raw_params):
    from core.outlier_rules.peer_check import run_peer_check

    assigned, panel = _thin_group_panel(n=40, thin_n=3)
    off = run_peer_check(
        assigned, panel, pd.DataFrame(), PERIODS[-1],
        _override(raw_params, "fova", hist_start_period=0, adjbox_cell_shrink=False, **NO_BORROW),
    )
    thin = off[off["peer_group_id"] == "GSMALL"]
    assert (thin["adjbox_shrink_w"] == 1.0).all()
    on = run_peer_check(
        assigned, panel, pd.DataFrame(), PERIODS[-1],
        _override(raw_params, "fova", hist_start_period=0, adjbox_cell_shrink=True, adjbox_shrink_min_n=10,
                  adjbox_shrink_scale=3.0, **NO_BORROW),
    )
    thin_on = on[on["peer_group_id"] == "GSMALL"]
    assert (thin_on["adjbox_basis"] == "shrink").all()
    assert (thin_on["adjbox_shrink_w"] < 1.0).all()
    # w = 3/(3+3) = 0.5 when n_peer=3
    assert thin_on["adjbox_shrink_w"].iloc[0] == pytest.approx(0.5)
    big = on[on["peer_group_id"] == "G"]
    assert (big["adjbox_shrink_w"] == 1.0).all()


def test_longitude_borrow_cell_restores_short_group_history(raw_params):
    from core.outlier_rules.peer_check import run_peer_check

    assigned, panel = _peer_inputs()
    late = dict(NO_BORROW, hist_start_period=PERIODS[1])
    off = run_peer_check(assigned, panel, pd.DataFrame(), PERIODS[-1], _override(raw_params, "fova", **late))
    assert (off["n_hist"] == 2).all() and (off["n_hist_cell"] == 0).all()
    assert (off["n_hist_eff"] == 2).all() and (off["fova_longitude"] == "na").all()
    assert (off["lon_src"] == "group").all()

    on = run_peer_check(
        assigned, panel, pd.DataFrame(), PERIODS[-1],
        _override(raw_params, "fova", **{**late, "longitude_borrow_cell": True}),
    )
    assert (on["n_hist"] == 2).all() and (on["n_hist_cell"] == 3).all()
    assert (on["n_hist_eff"] == 5).all()
    assert (on["lon_src"] == "blend").all()
    assert (on["fova_longitude"] != "na").any()
    assert on.loc[on["store_id"] == "S0", "fova_longitude"].iloc[0] == "high"


def test_hist_start_filter_off_uses_full_window(raw_params):
    from core.outlier_rules.peer_check import run_peer_check

    assigned, panel = _peer_inputs()
    late = dict(NO_BORROW, hist_start_period=PERIODS[1])
    cut = run_peer_check(assigned, panel, pd.DataFrame(), PERIODS[-1], _override(raw_params, "fova", **late))
    assert (cut["n_hist"] == 2).all()
    full = run_peer_check(
        assigned, panel, pd.DataFrame(), PERIODS[-1],
        _override(raw_params, "fova", **{**late, "hist_start_filter": False}),
    )
    assert (full["n_hist"] == 3).all()


def test_fova_borrow_params_validated(raw_params):
    for key, bad in (("borrow_min_peer", 0), ("fence_fallback_min_n", 0), ("longitude_borrow_hist_scale", 0.0)):
        raw = copy.deepcopy(raw_params)
        raw["fova"][key] = bad
        with pytest.raises(ValueError):
            params_from_dict(raw)


def test_missing_sales_levels(params):
    from core.outlier_rules.peer_check import missing_sales

    assigned, panel = _peer_inputs()
    assigned, panel = _zero_rows(assigned, panel, ["S1", "S2"])
    prev = panel["period_id"] == PERIODS[-2]
    panel.loc[prev & (panel["store_id"] == "S2"), "sales_value"] = 0.0  # long-term non-seller
    # Tiny previous sales vs peer median should also be ignored.
    panel.loc[prev & (panel["store_id"] == "S1"), "sales_value"] = 1.0
    out_tiny = missing_sales(assigned, panel, PERIODS[-1], params.peer)
    assert out_tiny.empty or "S1" not in set(out_tiny["store_id"])

    panel.loc[prev & (panel["store_id"] == "S1"), "sales_value"] = 100.0
    out = missing_sales(assigned, panel, PERIODS[-1], params.peer)
    m = out.set_index("store_id")
    assert set(m.index) == {"S1"}  # S2 never sold → ignored
    assert m.loc["S1", "level"] == "suspicious" and m.loc["S1", "prev_sales_value"] == pytest.approx(100.0)
    assert m.loc["S1", "sell_ratio"] == pytest.approx(38 / 40) and m.loc["S1", "n_peer"] == 40

    sparse, sparse_panel = _zero_rows(*_peer_inputs(), [f"S{i}" for i in range(10, 40)])
    assert missing_sales(sparse, sparse_panel, PERIODS[-1], params.peer).empty
    assert missing_sales(assigned, panel, PERIODS[-1], params.peer, exclude_stores=["S1"]).empty


def test_store_drop(params):
    from core.outlier_rules.peer_check import store_drop

    rows = []
    for p in PERIODS[-2:]:
        for s, n_cur in (("A", 2), ("B", 8), ("C", 1)):
            n = 10 if s != "C" else 3
            n = n if p == PERIODS[-2] else n_cur
            rows += [{"period_id": p, "store_id": s, "category": f"K{k}", "sales_value": 100.0,
                      "PLATFORMNAME": "ELEME", "SHOPTYPE": "cvs", "PROVINCE": "P1"} for k in range(n)]
    sc = pd.DataFrame(rows)
    store_map = pd.DataFrame({"store_id": ["A", "B", "C"], "ibd_id": "I"})
    out = store_drop(sc, store_map, PERIODS[-1], params.peer)
    assert list(out["store_id"]) == ["A"]
    r = out.iloc[0]
    assert (r["n_cat"], r["prev_n_cat"]) == (2, 10) and r["sales_ratio"] == pytest.approx(0.2)
    assert store_drop(sc[sc["period_id"] == PERIODS[-1]], store_map, PERIODS[-1], params.peer).empty


# ---------------------------------------------------------------- report
def _graded_for_report():
    base = {"period_id": 20261408, "PLATFORMNAME": "ELEME", "SHOPTYPE": "cvs", "PROVINCE": "P1", "ibd_id": "I",
            "peer_group_id": "G", "peer_level": "ibd", "category": "X", "n_evidence": 0, "evidence_list": "",
            "adjbox_value": "ok", "share_adjbox": "ok", "item_adjbox": "ok", "fova_final": "ok", "auto_action": "skip",
            "lv_lo": np.log1p(50.0), "lv_hi": np.log1p(200.0)}
    rows = [{**base, "store_id": f"S{i}", "sales_value": 100.0, "level": "normal", "direction": ""} for i in range(9)]
    rows.append({**base, "store_id": "HI", "sales_value": 1000.0, "level": "high_confidence", "direction": "high",
                 "n_evidence": 3, "auto_action": "fix"})
    rows.append({**base, "store_id": "LO", "sales_value": 10.0, "level": "suspicious", "direction": "low",
                 "n_evidence": 2, "auto_action": "fix"})
    rows.append({**base, "store_id": "W", "sales_value": 300.0, "level": "watch", "direction": "high",
                 "n_evidence": 1, "auto_action": "flag_only"})
    return pd.DataFrame(rows)


def test_adjustments_fence_and_median():
    from core.report.report import adjustments

    adj = adjustments(_graded_for_report(), "suspicious").set_index("store_id")
    assert set(adj.index) == {"HI", "LO"}
    assert adj.loc["HI", "expected_fence"] == pytest.approx(200.0) and adj.loc["HI", "adj_to_fence"] == pytest.approx(-800.0)
    assert adj.loc["HI", "peer_median_sales"] == pytest.approx(100.0) and adj.loc["HI", "adj_to_median"] == pytest.approx(-900.0)
    assert adj.loc["LO", "adj_to_fence"] == pytest.approx(40.0) and adj.loc["LO", "adj_to_median"] == pytest.approx(90.0)
    assert adj.loc["HI", "adj_pct_fence"] == pytest.approx(-0.8)


def test_summary_tables():
    from core.report.report import adjustments, summary_by_dim, summary_overview

    g = _graded_for_report()
    adj = adjustments(g, "suspicious")
    missing = pd.DataFrame({"store_id": ["S1"], "category": ["Y"], "PLATFORMNAME": ["ELEME"], "SHOPTYPE": ["cvs"], "PROVINCE": ["P1"],
                            "peer_median_sales": [50.0], "level": ["suspicious"]})
    drops = pd.DataFrame({"store_id": ["S2"], "sales_value": [10.0], "prev_sales_value": [110.0], "n_missing": [3]})
    dq = pd.DataFrame({"check": ["bmp_only_stores"], "n_stores": [4]})
    ov = summary_overview(g, adj, missing, drops, dq, pd.DataFrame(), {"rows": 100, "sales": 3000.0, "dropped_rows": 5}, "suspicious", high_min=3)
    r = ov.set_index(["section", "item"])
    total = g["sales_value"].sum()
    assert r.loc[("监测漏斗", "门店×品类 行数"), "n"] == len(g)
    assert r.loc[("分级规则", "计入证据的规则"), "n"] == 4
    assert "item_adjbox" in str(r.loc[("分级规则", "计入证据的规则"), "note"])
    assert "n_evidence=1" in str(r.loc[("异常分级", "watch"), "note"])
    assert "n_evidence≥3" in str(r.loc[("异常分级", "high_confidence"), "note"])
    assert r.loc[("自动动作", "fix"), "n"] == 2
    assert r.loc[("自动动作", "flag_only"), "n"] == 1
    assert r.loc[("异常分级", "suspicious+"), "n"] == 2
    assert r.loc[("异常分级", "suspicious+"), "pct_sales"] == pytest.approx(1010 / total)
    assert r.loc[("调整估算 (level≥suspicious)", "净调整"), "adj_to_fence"] == pytest.approx(-760.0)
    assert r.loc[("该卖没卖", "合计"), "adj_to_median"] == pytest.approx(50.0)
    assert r.loc[("整店下降", "门店"), "adj_to_median"] == pytest.approx(100.0)
    assert r.loc[("数据质量", "剔除行 (重复/负值)"), "pct_rows"] == pytest.approx(0.05)
    assert pd.isna(r.loc[("异常分级", "watch"), "adj_to_fence"])

    bd = summary_by_dim(g, adj, missing, drops).set_index(["dimension", "value"])
    assert bd.loc[("SHOPTYPE", "cvs"), "n_susp_plus"] == 2 and bd.loc[("SHOPTYPE", "cvs"), "n_store_drop"] == 1
    assert bd.loc[("category", "X"), "n_missing"] == 0


def test_flag_raw_keeps_all_rows_and_columns():
    from core.report.report import adjustments, flag_raw

    g = _graded_for_report()
    raw = pd.DataFrame({
        "Unnamed: 0": range(5), "period_id": 20261408, "store_id": ["HI", "HI", "S1", "S1", "NOPE"],
        "prod_id": [1.0, 2.0, 3.0, 3.0, 4.0], "category": "X", "brand": "b",
        "sales_value": [600.0, 400.0, 100.0, 5.0, 7.0], "sales_unit": [6.0, 4.0, 1.0, 1.0, 0.0], "nankey": [11.0, 12.0, 13.0, 13.0, 14.0],
    })
    target = pd.DataFrame({"PERIODCODE": 20261408, "STOREID": ["HI", "S1"], "PROVINCE": "P1", "SHOPTYPE": "cvs", "PLATFORMNAME": "ELEME"})
    drill = pd.DataFrame({"store_id": ["HI"], "category": ["X"], "nankey": [11], "rank": [1], "contribution": [0.7],
                          "item_flag": [""]})
    out, stats = flag_raw(raw, target, g, adjustments(g, "suspicious"), pd.DataFrame({"store_id": ["S1"]}), drill, "nankey", "suspicious")
    assert len(out) == 5 and list(out.columns[: len(raw.columns)]) == list(raw.columns)
    assert list(out["level"][:2]) == ["high_confidence"] * 2 and out["adj_to_fence"].iloc[0] == pytest.approx(-800.0)
    assert out["drill_rank"].iloc[0] == 1 and pd.isna(out["drill_rank"].iloc[1])
    assert out["dq_flags"].iloc[3] == "dup_period_store_prod" and out["dq_dropped"].iloc[3]
    assert out["dq_flags"].iloc[4] == "zero_unit_positive_value" and not out["store_matched"].iloc[4]
    assert out["issue"].iloc[0] == "peer异常"
    assert out["issue"].iloc[3] == "数据质量剔除;整店下降" and out["issue"].iloc[4] == "门店未匹配"
    assert out["issue"].iloc[2] == "整店下降" and stats["issue_rows"] == 5
    assert stats["dropped_rows"] == 1 and stats["unmatched_sales"] == pytest.approx(7.0)


# ---------------------------------------------------------------- drilldown
def _drill_inputs():
    rng = np.random.default_rng(9)
    cur, prev = [], []

    def item(rows, store, key, value, unit):
        rows.append({"store_id": store, "category": "X", "nankey": key, "sales_value": float(value), "sales_unit": float(unit),
                     "prod_desc_raw": f"item{key}", "ibd_id": "I", "period_id": 20261408 if rows is cur else 20261407})

    for i in range(10):
        for key in range(1, 6):
            price = rng.uniform(9.5, 10.5)
            item(cur, f"P{i}", key, 100, 100 / price)
        if i < 2:
            item(cur, f"P{i}", 6, 60, 6)
    item(cur, "T", 1, 500, 50)
    item(cur, "T", 2, 150, 5)
    item(cur, "T", 3, 100, 10)
    item(cur, "T", 6, 120, 12)
    item(cur, "T", 7, 40, 4)
    item(prev, "T", 6, 50, 5)
    for key in range(1, 5):
        item(cur, "L", key, 60, 6)
        item(prev, "L", key, 100, 10)
    item(prev, "L", 5, 300, 30)
    item(cur, "W", 1, 900, 90)
    base = {"period_id": 20261408, "category": "X", "peer_group_id": "G", "member_ibds": "I", "peer_level": "ibd"}
    graded = pd.DataFrame([
        {**base, "store_id": "T", "level": "high_confidence", "direction": "high"},
        {**base, "store_id": "L", "level": "suspicious", "direction": "low"},
        {**base, "store_id": "W", "level": "watch", "direction": "high"},
    ])
    return graded, pd.DataFrame(cur), pd.DataFrame(prev)


def test_drilldown_ranks_sources_and_flags(params):
    from core.outlier_rules.drilldown import run_drilldown

    graded, cur, prev = _drill_inputs()
    log = ListLogger()
    out = run_drilldown(graded, cur, prev, pd.DataFrame(), "nankey", params.drilldown, logger=log)
    assert set(out["store_id"]) == {"T", "L"}

    t = out[out["store_id"] == "T"].set_index("nankey")
    assert t["rank"].idxmin() == 1 and t.loc[1, "contribution"] == t["contribution"].max()
    assert t.loc[1, "expected_source"] == "peer" and t.loc[1, "delta"] == pytest.approx(400)
    assert t.loc[6, "expected_source"] == "own_prev" and t.loc[6, "expected_sales"] == pytest.approx(50)
    assert t.loc[7, "expected_source"] == "new_item" and t.loc[7, "item_flag"] == "new_item"
    assert 3 not in t.index
    assert t["contribution"].sum() == pytest.approx(1.0)

    low = out[out["store_id"] == "L"].set_index("nankey")
    assert (low["delta"] < 0).all()
    assert low.loc[5, "item_flag"] == "missing_item" and low.loc[5, "rank"] == 1
    assert any("[DRILLDOWN]" in m for m in log.infos)


def test_drilldown_top_n(raw_params):
    from core.config import params_from_dict
    from core.outlier_rules.drilldown import run_drilldown

    raw = copy.deepcopy(raw_params)
    raw["drilldown"]["top_n"] = 2
    p = params_from_dict(raw)
    graded, cur, prev = _drill_inputs()
    out = run_drilldown(graded, cur, prev, pd.DataFrame(), "nankey", p.drilldown)
    assert out.groupby("store_id").size().max() == 2


# ---------------------------------------------------------------- evidence / stability
def test_evidence_grade_levels_and_direction():
    from core.outlier_rules.evidence import EVIDENCE_COLS, grade

    assert "item_adjbox" in EVIDENCE_COLS
    base = {"period_id": 1, "category": "X", "peer_level": "ibd", "item_adjbox": "ok", "sales_value": 100.0, "zero_filled": False}
    detail = pd.DataFrame([
        {**base, "store_id": "A", "adjbox_value": "ok", "share_adjbox": "ok", "fova_final": "na"},
        {**base, "store_id": "B", "adjbox_value": "high", "share_adjbox": "ok", "fova_final": "ok"},
        {**base, "store_id": "C", "adjbox_value": "low", "share_adjbox": "ok", "fova_final": "ok"},
        {**base, "store_id": "D", "adjbox_value": "high", "share_adjbox": "high", "fova_final": "high", "item_adjbox": "high"},
        {**base, "store_id": "E", "adjbox_value": "ok", "share_adjbox": "high", "fova_final": "ok"},
        {**base, "store_id": "F", "adjbox_value": "high", "share_adjbox": "high", "fova_final": "ok"},  # suspicious
        {**base, "store_id": "Z", "adjbox_value": "high", "share_adjbox": "high", "fova_final": "ok",
         "sales_value": 0.0, "zero_filled": True},  # 该卖没卖：不得 fix
    ])
    g = grade(detail, 3).set_index("store_id")
    assert list(g.loc[["A", "B", "C", "D", "E", "F"], "level"]) == [
        "normal", "watch", "watch", "high_confidence", "watch", "suspicious",
    ]
    assert g.loc["C", "direction"] == "low" and g.loc["D", "direction"] == "high"
    assert g.loc["E", "direction"] == "high"
    assert g.loc["D", "n_evidence"] == 4
    assert g.loc["C", "evidence_list"] == "adjbox_value:low"
    assert g.loc["A", "auto_action"] == "skip"
    assert g.loc["B", "auto_action"] == "flag_only"
    assert g.loc["F", "auto_action"] == "fix"
    assert g.loc["D", "auto_action"] == "fix"
    assert g.loc["Z", "level"] == "suspicious" and g.loc["Z", "auto_action"] == "flag_only"


def test_summarize_levels_excludes_na():
    from core.outlier_rules.evidence import summarize_levels

    graded = pd.DataFrame({"period_id": [1, 1, 1, 1], "peer_level": ["ibd", "ibd", "ibd", "na"],
                           "level": ["normal", "watch", "high_confidence", "normal"], "direction": ["", "high", "low", ""]})
    s = summarize_levels(graded).set_index("level")
    assert s.loc["watch", "rate"] == pytest.approx(1 / 3)
    assert s.loc["high_confidence", "n_low"] == 1


def test_jaccard_and_threshold_grid(params):
    from core.outlier_rules.peer_check import run_peer_check
    from core.outlier_rules.stability import jaccard, threshold_jaccard

    assert jaccard(set(), set()) == 1.0
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assigned, panel = _peer_inputs()
    detail = run_peer_check(assigned, panel, pd.DataFrame(), PERIODS[-1], params)
    jac = threshold_jaccard(detail, params)
    assert set(jac["detector"]) == {"adjbox_c"}
    assert jac["jaccard"].between(0, 1).all()
    n_by_thr = jac.set_index("thr_a")["n_a"]
    assert n_by_thr.is_monotonic_decreasing


def test_rate_by_period_spike_warning():
    from core.outlier_rules.stability import log_stability, rate_by_period

    rows = []
    for period, rate in zip(range(1, 6), (0.01, 0.012, 0.011, 0.01, 0.08)):
        rows += [{"period_id": period, "level": "high_confidence", "rate": rate},
                 {"period_id": period, "level": "suspicious", "rate": 0.02},
                 {"period_id": period, "level": "normal", "rate": 0.9}]
    rates = rate_by_period(pd.DataFrame(rows), 3.0)
    spikes = rates[rates["spike"]]
    assert set(zip(spikes["period_id"], spikes["level"])) == {(5, "high_confidence"), (5, "suspicious+")}
    assert "normal" not in set(rates["level"]) and "suspicious+" in set(rates["level"])
    log = ListLogger()
    log_stability(log, pd.DataFrame(columns=["vs_config"]), rates, 3.0)
    assert any("WARNING 5" in w for w in log.warnings)
