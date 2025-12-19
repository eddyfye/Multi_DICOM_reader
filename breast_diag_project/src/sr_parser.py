"""Utilities for parsing DICOM Structured Reports (SR) into labels."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import pydicom


def _get_concept_name(item) -> Optional[str]:
    if "ConceptNameCodeSequence" not in item:
        return None
    code_item = item.ConceptNameCodeSequence[0]
    name = getattr(code_item, "CodeMeaning", None)
    if not name:
        name = getattr(code_item, "CodeValue", None)
    return name


def _find_assessment_category(
    items, keyword: str
) -> Optional[Tuple[Any, Any]]:
    for item in items:
        concept_name = _get_concept_name(item)
        if concept_name and keyword in concept_name.lower():
            raw_value = None
            meaning = None

            if "ConceptCodeSequence" in item:
                c = item.ConceptCodeSequence[0]
                raw_value = getattr(c, "CodeValue", None)
                meaning = getattr(c, "CodeMeaning", None)

            if raw_value is None and hasattr(item, "TextValue"):
                raw_value = item.TextValue

            # Found a matching SR node; bubble the values up the recursion stack.
            return raw_value, meaning

        if "ContentSequence" in item and item.ContentSequence:
            res = _find_assessment_category(item.ContentSequence, keyword)
            if res is not None:
                return res
    return None


def _extract_assessment_category_from_sr(
    sr_path: str,
    keyword: str,
) -> Tuple[Optional[int], Optional[Any], Optional[Any]]:
    try:
        ds = pydicom.dcmread(sr_path, stop_before_pixels=True, force=True)
    except Exception:
        return None, None, None

    if getattr(ds, "Modality", "").upper() != "SR":
        return None, None, None

    root_items: Sequence[Any] | None = getattr(ds, "ContentSequence", None)
    if not root_items:
        return None, None, None

    res = _find_assessment_category(root_items, keyword)
    if res is None:
        return None, None, None

    raw_value, meaning = res
    category_int = None

    if isinstance(raw_value, str):
        digits = [c for c in raw_value if c.isdigit()]
        if digits:
            category_int = int(digits[0])

    if category_int is None and isinstance(meaning, str):
        digits = [c for c in meaning if c.isdigit()]
        if digits:
            category_int = int(digits[0])

    return category_int, raw_value, meaning


def parse_sr_to_label(sr_path: str, keyword: str = "assessment category") -> Dict[str, Any]:
    """Parse a DICOM SR file and return training-ready labels using the keyword match."""

    category_int, raw_value, meaning = _extract_assessment_category_from_sr(
        sr_path, keyword.lower()
    )

    target = category_int if category_int is not None else 0
    diagnosis_text = meaning if meaning is not None else str(raw_value or "")

    return {"target": int(target), "diagnosis_text": diagnosis_text}


__all__ = ["parse_sr_to_label"]
