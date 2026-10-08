"""unit_fix orchestrator tests: matching, triage, store shift, recheck, impact."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.unit_fix.unit_fix import (
    LV_BC_PK, LV_CAT, LV_NB_PK, LV_NK, LV_TAG,
    _impact_summary, _recheck, detect_store_shift, match_peers, run_unit_fix, triage_low,
)


def _target(**kw):
    base = {"nankey": 1, "brand": "A/A/A", "packsize": "1X48GM", "category": "BEER", "store_id": "S0"}
    base.update(kw)
    return pd.Series(base)


def _cands(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- matching
def test_match_key_fallback_chain(params):
    cands = _cands([
        {"store_id": "S1", "nankey": 1, "brand": "A/A/A", "packsize": "1X48GM", "category": "BEER"},
        {"store_id": "S2", "nankey": 1, "brand": "A/A/A", "packsize": "1X48GM", "category": "BEER"},
        {"store_id": "S3", "nankey": 2, "brand": "B/B/B", "packsize": "1X48GM", "category": "BEER"},
    ])
    peers, level = match_peers(_target(), cands, params)
    assert level == LV_NB_PK
    assert set(peers["store_id"]) == {"S1", "S2"}


def test_match_key_falls_back_to_nankey_only(params):
    cands = _cands([
        {"store_id": "S1", "nankey": 1, "brand": "X/X/X", "packsize": "9X99ZZ", "category": "BEER"},
    ])
    peers, level = match_peers(_target(), cands, params)
    assert level == LV_NK
    assert set(peers["store_id"]) == {"S1"}


def test_match_key_falls_back_to_brand_category_packsize(params):
    cands = _cands([
        {"store_id": "S1", "nankey": 99, "brand": "A/A/A", "packsize": "1X48GM", "category": "BEER"},
    ])
    peers, level = match_peers(_target(), cands, params)
    assert level == LV_BC_PK


def test_match_key_falls_back_to_category_and_tag(params):
    cands = _cands([
        {"store_id": "S1", "nankey": 99, "brand": "Z/Z/Z", "packsize": "1X48GM", "category": "BEER"},
    ])
    peers, level = match_peers(_target(), cands, params)
    assert level == LV_CAT
    peers2, level2 = match_peers(_target(), cands.assign(category="CSD"), params)
    assert level2 == LV_TAG


def test_match_packsize_tolerance_respected(params):
    # 2X24GM = 48GM volume: same bin as 1X48GM
    cands = _cands([
        {"store_id": "S1", "nankey": 1, "brand": "A/A/A", "packsize": "2X24GM", "category": "BEER"},
    ])
    peers, level = match_peers(_target(), cands, params)
    assert level == LV_NB_PK  # same volume -> matches packsize bin


# ---------------------------------------------------------------- low triage
def test_low_direction_three_way():
    drops = pd.DataFrame({"store_id": ["D1"]})
    assert triage_low(pd.Series({"store_id": "D1", "zero_filled": False, "sales_value": 5.0}), drops) == "store_drop"
    assert triage_low(pd.Series({"store_id": "S1", "zero_filled": True, "sales_value": 0.0}), drops) == "missing"
    assert triage_low(pd.Series({"store_id": "S2", "zero_filled": False, "sales_value": 3.0}), drops) == "undersell"


# ---------------------------------------------------------------- store shift
def test_store_shift_exemption(params):
    rows = []
    for i in range(5):
        rows.append({"period_id": 20261408, "store_id": "A", "category": f"C{i}", "peer_level": "ibd",
                     "direction": "high" if i < 4 else "ok", "sales_value": 100.0,
                     "PLATFORMNAME": "ELEME", "SHOPTYPE": "cvs", "PROVINCE": "P1"})
        rows.append({"period_id": 20261408, "store_id": "B", "category": f"C{i}", "peer_level": "ibd",
                     "direction": "high" if i < 2 else "ok", "sales_value": 100.0,
                     "PLATFORMNAME": "ELEME", "SHOPTYPE": "cvs", "PROVINCE": "P1"})
    graded = pd.DataFrame(rows)
    store_map = pd.DataFrame({"store_id": ["A", "B"], "ibd_id": ["ibdA", "ibdB"]})
    shift = detect_store_shift(graded, store_map, params)
    assert set(shift["store_id"]) == {"A"}
    assert shift.iloc[0]["high_ratio"] == pytest.approx(0.8)


def test_unit_fix_only_auto_action_fix(params):
    """Watch (flag_only) drill rows must not produce corrections."""
    graded = pd.DataFrame([
        {"period_id": 20261408, "store_id": "W", "category": "BEER", "peer_level": "ibd",
         "peer_group_id": "G", "member_ibds": "I1", "level": "watch", "direction": "high",
         "auto_action": "flag_only", "sales_value": 1000.0,
         "PLATFORMNAME": "ELEME", "SHOPTYPE": "cvs", "PROVINCE": "P1"},
        {"period_id": 20261408, "store_id": "P", "category": "BEER", "peer_level": "ibd",
         "peer_group_id": "G", "member_ibds": "I1", "level": "normal", "direction": "",
         "auto_action": "skip", "sales_value": 50.0,
         "PLATFORMNAME": "ELEME", "SHOPTYPE": "cvs", "PROVINCE": "P1"},
    ])
    drill = pd.DataFrame([{
        "period_id": 20261408, "store_id": "W", "category": "BEER", "direction": "high",
        "nankey": 1, "sales_value": 900.0, "sales_unit": 9.0, "rank": 1, "item_flag": "new_item",
    }])
    items = pd.DataFrame([
        {"period_id": 20261408, "store_id": "W", "category": "BEER", "nankey": 1, "brand": "A/A/A",
         "packsize": "1X48GM", "prod_desc_raw": "x", "sales_value": 900.0, "sales_unit": 9.0, "score": 1.0},
        {"period_id": 20261408, "store_id": "P", "category": "BEER", "nankey": 1, "brand": "A/A/A",
         "packsize": "1X48GM", "prod_desc_raw": "x", "sales_value": 50.0, "sales_unit": 1.0, "score": 1.0},
    ])
    store_map = pd.DataFrame({"store_id": ["W", "P"], "ibd_id": ["I1", "I1"]})
    out = run_unit_fix(graded, drill, items, items.iloc[0:0], store_map, params)
    assert out.corrections.empty
    funnel = out.funnel.set_index("reason")
    assert funnel.loc["skip_not_fix_action", "n"] == 1
    assert funnel.loc["corrected", "n"] == 0
    assert "drill_no_delta" in funnel.index


# ---------------------------------------------------------------- recheck
def _recheck_graded():
    rows = []
    for i, v in enumerate([8.0, 9.0, 10.0, 11.0, 12.0]):
        rows.append({"period_id": 20261408, "store_id": f"S{i}", "category": "BEER",
                     "peer_group_id": "G1", "sales_value": v})
    rows.append({"period_id": 20261408, "store_id": "T", "category": "BEER",
                 "peer_group_id": "G1", "sales_value": 1000.0})
    return pd.DataFrame(rows)


def test_recheck_resolved(params):
    graded = _recheck_graded()
    corrections = pd.DataFrame({"store_id": ["T"], "category": ["BEER"], "impact_value": [-990.0]})
    out = _recheck(graded, corrections, params, shift_stores=set())
    assert len(out) == 1
    row = out.iloc[0]
    assert row["corrected_value"] == pytest.approx(10.0)
    assert bool(row["resolved"]) is True
    assert bool(row["recheck_fail"]) is False


def test_recheck_still_outlier(params):
    graded = _recheck_graded()
    # only a tiny correction: stays above the clean-peer fence
    corrections = pd.DataFrame({"store_id": ["T"], "category": ["BEER"], "impact_value": [-10.0]})
    out = _recheck(graded, corrections, params, shift_stores=set())
    assert len(out) == 1
    assert bool(out.iloc[0]["resolved"]) is False


# ---------------------------------------------------------------- impact
def test_impact_summary_net_and_volume(params):
    corrections = pd.DataFrame({
        "store_id": ["S1", "S2"], "category": ["BEER", "CSD"], "brand": ["A/A/A", "B/B/B"],
        "impact_value": [10.0, -5.0],
    })
    graded = pd.DataFrame({"store_id": ["S1", "S2"], "PLATFORMNAME": ["ELEME", "MEITUAN"],
                           "SHOPTYPE": ["cvs", "cvs"], "PROVINCE": ["P1", "P2"]})
    out = _impact_summary(graded, corrections)
    overall = out[out["dimension"] == "overall"].iloc[0]
    assert overall["net_impact"] == pytest.approx(5.0)
    assert overall["total_volume"] == pytest.approx(15.0)
    assert overall["n_items"] == 2
    cats = out[out["dimension"] == "category"]
    assert set(cats["level"]) == {"BEER", "CSD"}
