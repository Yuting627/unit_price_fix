from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

PARAMS_PATH = ROOT / "config" / "peer_params.json"


@pytest.fixture
def raw_params() -> dict:
    """Real config with IBD thresholds pinned for synthetic fixtures (sized for 30/20).
    Production JSON defaults are min_store=10 / soft=5 / allow_forced_nearest=false (appendix G)."""
    with PARAMS_PATH.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    raw["ibd"].update(min_store=30, min_store_soft=20, allow_forced_nearest=True)
    return raw


@pytest.fixture
def params(raw_params):
    from core.config import params_from_dict

    return params_from_dict(raw_params, source=PARAMS_PATH)
