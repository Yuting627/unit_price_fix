"""Packsize / brand parsing tests."""
from __future__ import annotations

import math

from funcs.parse import packsize_volume, parse_packsize, split_brand


def _nan_close(a, b):
    if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b):
        return True
    return a == b


def test_packsize_qty_x_size():
    qty, size, unit = parse_packsize("1X48GM")
    assert qty == 1.0 and size == 48.0 and unit == "GM"


def test_packsize_unknown():
    qty, size, unit = parse_packsize("UNKNOWN")
    assert math.isnan(qty) and math.isnan(size) and unit == ""


def test_packsize_size_unit_only():
    qty, size, unit = parse_packsize("1PCK")
    assert math.isnan(qty) and size == 1.0 and unit == "PCK"


def test_packsize_none():
    qty, size, unit = parse_packsize(None)
    assert math.isnan(qty) and math.isnan(size) and unit == ""


def test_packsize_volume():
    assert packsize_volume(parse_packsize("1X48GM")) == 48.0
    assert packsize_volume(parse_packsize("2X24GM")) == 48.0
    assert math.isnan(packsize_volume(parse_packsize("UNKNOWN")))


def test_split_brand_hierarchical():
    assert split_brand("肌研/肌研/肌研") == ("肌研", "肌研", "肌研")


def test_split_brand_short_and_none():
    assert split_brand("肌研") == ("肌研", "", "")
    assert split_brand(None) == ("", "", "")
