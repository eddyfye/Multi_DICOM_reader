"""Shared helpers for discovering DICOM series and SR files."""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

import pydicom
from tqdm import tqdm

logger = logging.getLogger(__name__)

ImageMeta = Tuple[str, str, str, str]


def collect_image_metadata(
    images_root: str,
    *,
    image_modality: str | None = "MR",
    series_description_keyword: str | None = "BLISS_AUTO",
) -> Dict[Tuple[str, str], List[ImageMeta]]:
    """Collect DICOM image series using metadata filters.

    Returns a mapping keyed by (study_uid, series_uid) to a list of tuples:
    (patient_id, study_uid, series_uid, file_path).
    """

    images_map: Dict[Tuple[str, str], List[ImageMeta]] = {}
    total_files = sum(len(files) for _, _, files in os.walk(images_root))
    modality_filter = (image_modality or "").upper()
    series_filter = (series_description_keyword or "").upper()

    with tqdm(total=total_files, desc="Scanning images", unit="file") as progress:
        for root, _, files in os.walk(images_root):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                except Exception as exc:  # pragma: no cover - logging only
                    logger.warning("Failed to read DICOM image %s: %s", fpath, exc)
                    progress.update()
                    continue

                modality = str(getattr(ds, "Modality", "") or "").upper()
                if modality_filter and modality != modality_filter:
                    progress.update()
                    continue

                series_description = str(getattr(ds, "SeriesDescription", "") or "").upper()
                if series_filter and series_filter not in series_description:
                    progress.update()
                    continue

                patient_id = getattr(ds, "PatientID", None)
                study_uid = getattr(ds, "StudyInstanceUID", None)
                series_uid = getattr(ds, "SeriesInstanceUID", None)
                if not study_uid or not series_uid:
                    logger.debug("Skipping file without Study/Series UID: %s", fpath)
                    progress.update()
                    continue

                key = (study_uid, series_uid)
                images_map.setdefault(key, []).append((patient_id, study_uid, series_uid, fpath))
                progress.update()
    return images_map


def collect_sr_metadata(
    sr_root: str, *, sr_modality: str | None = "SR"
) -> Dict[Tuple[str, str], List[str]]:
    """Collect metadata for DICOM-SR files using Modality to identify reports.

    Returns mapping from (patient_id, study_uid) to a list of SR file paths.
    """

    sr_map: Dict[Tuple[str, str], List[str]] = {}
    total_files = sum(len(files) for _, _, files in os.walk(sr_root))
    modality_filter = (sr_modality or "").upper()
    with tqdm(total=total_files, desc="Scanning SR", unit="file") as progress:
        for root, _, files in os.walk(sr_root):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                except Exception as exc:  # pragma: no cover - logging only
                    logger.warning("Failed to read SR file %s: %s", fpath, exc)
                    progress.update()
                    continue

                modality = str(getattr(ds, "Modality", "") or "").upper()
                if modality_filter and modality != modality_filter:
                    progress.update()
                    continue

                patient_id = getattr(ds, "PatientID", None) or ""
                study_uid = getattr(ds, "StudyInstanceUID", None)
                if not study_uid:
                    logger.debug("Skipping SR without StudyInstanceUID: %s", fpath)
                    progress.update()
                    continue
                sr_map.setdefault((patient_id, study_uid), []).append(fpath)
                progress.update()
    return sr_map


__all__ = ["collect_image_metadata", "collect_sr_metadata", "ImageMeta"]
