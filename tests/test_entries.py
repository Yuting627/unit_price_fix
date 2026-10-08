"""Entry-point tests on a tiny synthetic data directory."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

PERIODS = (20261405, 20261406, 20261407, 20261408)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def workspace(tmp_path, raw_params):
    rng = np.random.default_rng(7)
    bmp_dir, tgt_dir = tmp_path / "before_mp", tmp_path / "store_target"
    bmp_dir.mkdir()
    tgt_dir.mkdir()
    provinces = {"P1": 40, "P2": 20, "P3": 35}
    stores = [(f"{p}_{i}", p) for p, n in provinces.items() for i in range(n)]
    for t, period in enumerate(PERIODS):
        rows, tgt = [], []
        for idx, (sid, prov) in enumerate(stores):
            platform = "ELEME" if idx % 2 == 0 else "MEITUAN"
            tgt.append({"PERIODCODE": period, "PLATFORMNAME": platform, "STOREID": sid, "SHOPID": 1,
                        "STORENAME": sid, "PROVINCE": prov, "CITY": "c", "SHOPTYPE": "cvs"})
            for cat in ("BEER", "CSD"):
                for k in range(12):
                    val = float(rng.lognormal(3, 0.3)) * (1 + 0.05 * t) * (5 if platform == "MEITUAN" else 1)
                    if sid == "P1_0" and cat == "BEER" and period == PERIODS[-1] and k == 0:
                        val *= 200
                    rows.append({"period_id": period, "store_id": sid, "prod_id": hash((sid, k, period)) % 10**9,
                                 "item_id": "x", "prod_desc_raw": f"{cat} item {k}", "category": cat,
                                 "sales_value": val, "sales_unit": max(1.0, round(val / 5)), "nankey": k + (100 if cat == "CSD" else 0)})
        pd.DataFrame(rows).to_csv(bmp_dir / f"O2O_itemcoding_output_{period}_newline.csv.gz", index=False, encoding="gb18030")
        pd.DataFrame(tgt).to_csv(tgt_dir / f"target_shop_qc_{period}.csv", index=False, encoding="gb18030")
    raw = copy.deepcopy(raw_params)
    raw["paths"] = {
        "before_mp_dir": str(bmp_dir), "store_target_dir": str(tgt_dir),
        "output_dir": str(tmp_path / "out"), "log_dir": str(tmp_path / "logs"),
    }
    raw["base_cell"] = ["SHOPTYPE"]
    params_path = tmp_path / "peer_params.json"
    params_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return tmp_path, params_path


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_main_production_outputs_and_params_untouched(workspace):
    tmp_path, params_path = workspace
    before = _digest(params_path)
    res = _load("peer_main").run(params_path, 20261408)
    assert _digest(params_path) == before and res["params_unchanged"]
    sheets = pd.read_excel(res["xlsx"], sheet_name=None)
    expected = {"summary", "summary_by_dim", "adjustment", "dq", "profile", "anova", "ibd_map", "ibd_merge", "ibd_check",
                "seasonal_factor", "peer_detail", "drilldown", "missing_sales", "store_drop", "fix_funnel",
                "evidence_summary",
                "stability", "params_used"}
    assert expected <= set(sheets)
    assert list(sheets)[0] == "summary"

    raw_in = pd.read_csv(tmp_path / "before_mp" / "O2O_itemcoding_output_20261408_newline.csv.gz", encoding="gb18030")
    flagged = pd.read_csv(res["raw_csv"], encoding="utf-8-sig")
    assert len(flagged) == len(raw_in)
    assert list(flagged.columns[: len(raw_in.columns)]) == list(raw_in.columns)
    assert {"dq_flags", "level", "direction", "evidence_list", "adj_to_fence", "store_drop", "issue"} <= set(flagged.columns)
    spike = flagged[(flagged["store_id"] == "P1_0") & (flagged["category"] == "BEER")]
    assert spike["level"].iloc[0] in ("suspicious", "high_confidence") and (spike["issue"].str.contains("peer异常")).all()
    s = sheets["summary"]
    assert {"监测漏斗", "分级规则", "总体", "异常分级", "该卖没卖", "整店下降", "数据质量"} <= set(s["section"])
    rules = s.set_index(["section", "item"])
    assert rules.loc[("分级规则", "计入证据的规则"), "n"] == 4
    assert "item_adjbox" in str(rules.loc[("分级规则", "计入证据的规则"), "note"])
    assert "n_evidence" in str(rules.loc[("异常分级", "watch"), "note"])
    assert "fix" in str(rules.loc[("分级规则", "auto_action"), "note"])
    assert "auto_action" in set(sheets["peer_detail"].columns)
    merge = sheets["ibd_merge"]
    same = merge["peer_level"] == "ibd"
    assert (merge.loc[same, "ibd_id_raw"] == merge.loc[same, "ibd_id_merged"]).all()
    assert set(sheets["evidence_summary"]["period_id"]) >= {20261406, 20261407, 20261408}
    data = json.loads(res["seasonal_json"].read_text(encoding="utf-8"))
    assert data["period_id"] == 20261408 and data["factors"]
    log_text = (tmp_path / "logs" / "peer_outlier_20261408.log").read_text(encoding="utf-8")
    assert res["base_cell_suggested"] == ["PLATFORMNAME"]
    assert "建议 base_cell=PLATFORMNAME, 配置为 SHOPTYPE" in log_text
    assert json.loads(params_path.read_text(encoding="utf-8"))["base_cell"] == ["SHOPTYPE"]
