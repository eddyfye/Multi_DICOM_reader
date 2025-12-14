"""Utilities for parsing DICOM Structured Reports (SR) into labels."""
from __future__ import annotations

from typing import Any, Dict

import pydicom


def parse_sr_to_label(sr_path: str, label_config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a DICOM SR file and return training-ready labels.

    Notes:
        The extraction logic here is intentionally lightweight and contains
        TODO markers where project-specific codes should be inserted. The
        function ensures a consistent output structure downstream.
    """

    ds = pydicom.dcmread(sr_path, stop_before_pixels=True, force=True)

    # Placeholder logic: search through ContentSequence for malignancy flag.
    diagnosis_text = ""
    target = 0

    def _walk_content_sequence(sequence):
        nonlocal diagnosis_text, target
        for item in sequence:
            if hasattr(item, "TextValue") and not diagnosis_text:
                diagnosis_text = str(item.TextValue)
            if hasattr(item, "ConceptNameCodeSequence"):
                codes = [getattr(code, "CodeValue", "") for code in item.ConceptNameCodeSequence]
                # TODO: Replace with project-specific code detection using label_config["sr_field_codes"].
                if any(code in label_config.get("sr_field_codes", []) for code in codes):
                    if hasattr(item, "NumericValue"):
                        try:
                            target = 1 if float(item.NumericValue) > 0 else 0
                        except Exception:
                            target = 0
                    elif hasattr(item, "TextValue"):
                        target = 1 if str(item.TextValue).strip().lower() in {"yes", "malignant"} else 0
            if hasattr(item, "ContentSequence"):
                _walk_content_sequence(item.ContentSequence)

    if hasattr(ds, "ContentSequence"):
        _walk_content_sequence(ds.ContentSequence)

    return {"target": int(target), "diagnosis_text": diagnosis_text}


__all__ = ["parse_sr_to_label"]
