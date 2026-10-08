"""Parsing helpers for ``packsize`` and hierarchical ``brand`` strings.

``packsize`` values look like ``1X48GM`` (qty x size unit), ``48G`` or ``UNKNOWN``;
``imdb_packsize`` holds an IMDb reference spec such as ``48G`` or ``1PCK``. ``brand`` is a
hierarchical string ``厂商/品牌/子品牌`` (manufacturer/brand/subbrand).
"""
from __future__ import annotations

import math
import re

_PACK_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[xX*]\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*$")
_SIZEUNIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$")
_UNKNOWN = {"", "UNKNOWN", "NA", "NAN", "NULL", "NONE", "-", "N/A"}


def parse_packsize(packsize) -> tuple[float, float, str]:
    """Parse a packsize string into ``(qty, size, unit)``.

    ``1X48GM`` -> ``(1.0, 48.0, "GM")``; ``48G`` -> ``(nan, 48.0, "G")``;
    ``UNKNOWN``/missing -> ``(nan, nan, "")``.
    """
    if packsize is None or (isinstance(packsize, float) and math.isnan(packsize)):
        return (math.nan, math.nan, "")
    s = str(packsize).strip().upper()
    if s in _UNKNOWN:
        return (math.nan, math.nan, "")
    m = _PACK_RE.match(s)
    if m:
        return (float(m.group(1)), float(m.group(2)), m.group(3).strip())
    m = _SIZEUNIT_RE.match(s)
    if m:
        return (math.nan, float(m.group(1)), m.group(2).strip())
    return (math.nan, math.nan, "")


def packsize_volume(parsed: tuple[float, float, str]) -> float:
    """Normalized volume ``qty * size`` (qty defaults to 1 when absent); nan when size unknown."""
    qty, size, _unit = parsed
    if not math.isfinite(size):
        return math.nan
    return (qty if math.isfinite(qty) else 1.0) * size


def split_brand(brand) -> tuple[str, str, str]:
    """Split a hierarchical brand string into ``(manufacturer, brand, subbrand)``.

    ``肌研/肌研/肌研`` -> ``("肌研", "肌研", "肌研")``; ``肌研`` -> ``("肌研", "", "")``.
    """
    if brand is None or (isinstance(brand, float) and math.isnan(brand)):
        return ("", "", "")
    parts = [p.strip() for p in str(brand).split("/")]
    parts = [p for p in parts if p]
    while len(parts) < 3:
        parts.append("")
    return (parts[0], parts[1], parts[2])
