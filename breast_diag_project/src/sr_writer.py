"""Utilities for mapping predictions to BI-RADS labels and writing DICOM SR."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from breast_diag_project.src.config import ExperimentConfig


def _get_label_mapping(config: ExperimentConfig) -> Dict[str, Any]:
    return config.raw.get("inference", {}).get("label_mapping", {})


def _resolve_label_text(label: int, mapping: Dict[str, Any]) -> str:
    label_texts = mapping.get("label_texts", {})
    if isinstance(label_texts, dict):
        text = label_texts.get(str(label)) or label_texts.get(int(label))  # type: ignore[arg-type]
        if text:
            return str(text)
    return f"BI-RADS {label}"


def map_logits_to_label(logits: Iterable[float] | np.ndarray, config: ExperimentConfig) -> Dict[str, Any]:
    """Map raw logits/probabilities to the BI-RADS label schema."""

    mapping = _get_label_mapping(config)
    logits_arr = np.asarray(list(logits), dtype=np.float32)

    if logits_arr.ndim == 0 or logits_arr.shape[-1] == 1:
        logit = float(np.ravel(logits_arr)[0])
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        threshold = float(mapping.get("binary_threshold", 0.5))
        positive_label = int(mapping.get("positive_label", 4))
        negative_label = int(mapping.get("negative_label", 1))
        label = positive_label if prob >= threshold else negative_label
        diagnosis_text = _resolve_label_text(label, mapping)
        return {
            "target": label,
            "diagnosis_text": diagnosis_text,
            "probability": prob,
            "class_index": int(prob >= threshold),
        }

    exp_logits = np.exp(logits_arr - np.max(logits_arr))
    probs = exp_logits / np.sum(exp_logits)
    class_index = int(np.argmax(probs))
    class_to_label = mapping.get("class_to_label")
    label = class_index
    if isinstance(class_to_label, dict):
        label = int(class_to_label.get(str(class_index), class_to_label.get(class_index, class_index)))
    elif isinstance(class_to_label, (list, tuple)) and len(class_to_label) > class_index:
        label = int(class_to_label[class_index])

    diagnosis_text = _resolve_label_text(label, mapping)
    return {
        "target": int(label),
        "diagnosis_text": diagnosis_text,
        "probability": float(probs[class_index]),
        "class_index": class_index,
    }


def _build_file_meta() -> FileMetaDataset:
    meta = FileMetaDataset()
    meta.FileMetaInformationGroupLength = 0
    meta.FileMetaInformationVersion = b"\x00\x01"
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.88.11"  # Basic Text SR
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    return meta


def _copy_metadata(target: Dataset, source: Dataset, fields: Iterable[str]) -> None:
    for field in fields:
        if hasattr(source, field):
            setattr(target, field, getattr(source, field))


def build_sr_dataset(source: Dataset, label_fields: Dict[str, Any]) -> FileDataset:
    """Create a Basic Text SR dataset for the predicted assessment."""

    file_meta = _build_file_meta()
    now = dt.datetime.now()
    sr = FileDataset(
        None,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    sr.is_little_endian = True
    sr.is_implicit_VR = False

    sr.SOPClassUID = file_meta.MediaStorageSOPClassUID
    sr.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    sr.Modality = "SR"
    sr.SeriesInstanceUID = generate_uid()
    sr.StudyInstanceUID = getattr(source, "StudyInstanceUID", generate_uid())
    sr.StudyDate = getattr(source, "StudyDate", now.strftime("%Y%m%d"))
    sr.StudyTime = getattr(source, "StudyTime", now.strftime("%H%M%S"))
    sr.SeriesDate = now.strftime("%Y%m%d")
    sr.SeriesTime = now.strftime("%H%M%S")
    sr.ContentDate = now.strftime("%Y%m%d")
    sr.ContentTime = now.strftime("%H%M%S")
    sr.InstanceNumber = 1

    _copy_metadata(
        sr,
        source,
        [
            "PatientID",
            "PatientName",
            "PatientBirthDate",
            "PatientSex",
            "AccessionNumber",
            "StudyID",
            "ReferringPhysicianName",
        ],
    )

    sr.ValueType = "CONTAINER"
    sr.ContinuityOfContent = "SEPARATE"
    sr.ContentSequence = []

    assessment_item = Dataset()
    assessment_item.ValueType = "CODE"
    assessment_item.RelationshipType = "CONTAINS"
    assessment_item.ConceptNameCodeSequence = [
        Dataset(
            {
                "CodeValue": "ASMT",
                "CodingSchemeDesignator": "99LOCAL",
                "CodeMeaning": "Assessment Category",
            }
        )
    ]
    assessment_item.ConceptCodeSequence = [
        Dataset(
            {
                "CodeValue": str(label_fields.get("target", "")),
                "CodingSchemeDesignator": "99LOCAL",
                "CodeMeaning": label_fields.get("diagnosis_text", ""),
            }
        )
    ]
    sr.ContentSequence.append(assessment_item)

    probability = label_fields.get("probability")
    if probability is not None:
        prob_item = Dataset()
        prob_item.ValueType = "TEXT"
        prob_item.RelationshipType = "CONTAINS"
        prob_item.ConceptNameCodeSequence = [
            Dataset(
                {
                    "CodeValue": "PROB",
                    "CodingSchemeDesignator": "99LOCAL",
                    "CodeMeaning": "Probability",
                }
            )
        ]
        prob_item.TextValue = f"{float(probability):.4f}"
        sr.ContentSequence.append(prob_item)

    return sr


def write_sr(output_dir: Path, source: Dataset, label_fields: Dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sr = build_sr_dataset(source, label_fields)
    filename = f"{sr.StudyInstanceUID}_{sr.SeriesInstanceUID}_prediction.dcm"
    output_path = output_dir / filename
    sr.save_as(str(output_path), write_like_original=False)
    return output_path


__all__ = ["map_logits_to_label", "build_sr_dataset", "write_sr"]
