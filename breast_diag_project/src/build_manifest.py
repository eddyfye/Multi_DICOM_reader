"""Manifest builder for BREAST-DIAGNOSIS DICOM + DICOM-SR pairs.

This script walks DICOM image series and DICOM SR files, joins them on
StudyInstanceUID, and stores a manifest that downstream training jobs can
consume.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pydicom

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


ImageMeta = Tuple[str, str, str, str]


def _collect_image_metadata(images_root: str) -> Dict[Tuple[str, str], List[ImageMeta]]:
    """Collect metadata for all DICOM images.

    Returns a mapping keyed by (study_uid, series_uid) to a list of tuples:
    (patient_id, study_uid, series_uid, file_path).
    """

    images_map: Dict[Tuple[str, str], List[ImageMeta]] = {}
    for root, _, files in os.walk(images_root):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            except Exception as exc:  # pragma: no cover - logging only
                logger.warning("Failed to read DICOM image %s: %s", fpath, exc)
                continue

            patient_id = getattr(ds, "PatientID", None)
            study_uid = getattr(ds, "StudyInstanceUID", None)
            series_uid = getattr(ds, "SeriesInstanceUID", None)
            if not study_uid or not series_uid:
                logger.debug("Skipping file without Study/Series UID: %s", fpath)
                continue

            key = (study_uid, series_uid)
            images_map.setdefault(key, []).append((patient_id, study_uid, series_uid, fpath))
    return images_map


def _collect_sr_metadata(sr_root: str) -> Dict[str, Tuple[str, str]]:
    """Collect metadata for DICOM-SR files.

    Returns mapping from study_uid to (patient_id, sr_path).
    """

    sr_map: Dict[str, Tuple[str, str]] = {}
    for root, _, files in os.walk(sr_root):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            except Exception as exc:  # pragma: no cover - logging only
                logger.warning("Failed to read SR file %s: %s", fpath, exc)
                continue

            patient_id = getattr(ds, "PatientID", None)
            study_uid = getattr(ds, "StudyInstanceUID", None)
            if not study_uid:
                logger.debug("Skipping SR without StudyInstanceUID: %s", fpath)
                continue
            sr_map[study_uid] = (patient_id, fpath)
    return sr_map


def run(images_root: str, sr_root: str, output_manifest: str) -> None:
    """Build a manifest CSV aligning DICOM image series with SR files.

    Args:
        images_root: Root directory containing DICOM image files.
        sr_root: Root directory containing DICOM SR files.
        output_manifest: Path to write the manifest CSV.
    """

    images_root = os.path.abspath(images_root)
    sr_root = os.path.abspath(sr_root)
    output_manifest = os.path.abspath(output_manifest)
    logger.info("Scanning image root: %s", images_root)
    images_map = _collect_image_metadata(images_root)
    logger.info("Found %d image series", len(images_map))

    logger.info("Scanning SR root: %s", sr_root)
    sr_map = _collect_sr_metadata(sr_root)
    logger.info("Found %d SR files", len(sr_map))

    rows = []
    for (study_uid, series_uid), meta_list in images_map.items():
        patient_ids = {m[0] for m in meta_list if m[0] is not None}
        patient_id = next(iter(patient_ids), None)

        if study_uid not in sr_map:
            logger.warning("No SR found for study %s; skipping series %s", study_uid, series_uid)
            continue

        sr_patient_id, sr_path = sr_map[study_uid]
        if patient_id and sr_patient_id and patient_id != sr_patient_id:
            logger.warning(
                "Patient ID mismatch for study %s (images: %s, SR: %s); using SR value",
                study_uid,
                patient_id,
                sr_patient_id,
            )
            patient_id = sr_patient_id
        elif not patient_id:
            patient_id = sr_patient_id

        image_paths = [m[3] for m in sorted(meta_list, key=lambda x: x[3])]
        rows.append(
            {
                "patient_id": patient_id or "",
                "study_uid": study_uid,
                "series_uid": series_uid,
                "image_paths": json.dumps(image_paths),
                "sr_path": sr_path,
                "label_json": "{}",
            }
        )

    if not rows:
        logger.warning("No matched studies found; manifest will be empty")

    df = pd.DataFrame(rows)
    manifest_dir = Path(output_manifest).parent
    manifest_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_manifest, index=False)
    logger.info("Manifest saved to %s (%d rows)", output_manifest, len(df))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest aligning DICOM images with SR labels")
    parser.add_argument("--images-root", required=True, help="Root directory containing DICOM images")
    parser.add_argument("--sr-root", required=True, help="Root directory containing DICOM SR files")
    parser.add_argument("--output", required=True, help="Output CSV manifest path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(args.images_root, args.sr_root, args.output)


if __name__ == "__main__":
    main()
