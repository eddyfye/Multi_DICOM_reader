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
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
from pydicom import dcmread
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


ImageMeta = Tuple[str, str, str, str]


def _collect_image_metadata(
    images_root: str,
    *,
    image_modality: str = "MR",
    series_description_keyword: str = "BLISS_AUTO",
) -> Dict[Tuple[str, str], List[ImageMeta]]:
    """Collect MR BLISS_AUTO DICOM image series using DICOM metadata.

    Returns a mapping keyed by (study_uid, series_uid) to a list of tuples:
    (patient_id, study_uid, series_uid, file_path).
    """

    images_map: Dict[Tuple[str, str], List[ImageMeta]] = {}
    total_files = sum(len(files) for _, _, files in os.walk(images_root))

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
                if modality != image_modality.upper():
                    progress.update()
                    continue

                series_description = str(getattr(ds, "SeriesDescription", "") or "").upper()
                if series_description_keyword.upper() not in series_description:
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


def _collect_sr_metadata(
    sr_root: str, *, sr_modality: str = "SR"
) -> Dict[Tuple[str, str], List[str]]:
    """Collect metadata for DICOM-SR files using Modality to identify reports.

    Returns mapping from (patient_id, study_uid) to a list of SR file paths.
    """

    sr_map: Dict[Tuple[str, str], List[str]] = {}
    total_files = sum(len(files) for _, _, files in os.walk(sr_root))
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
                if modality != sr_modality.upper():
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


def run(
    images_root: str,
    sr_root: str,
    output_manifest: str,
    *,
    image_modality: str = "MR",
    series_description_keyword: str = "BLISS_AUTO",
    sr_modality: str = "SR",
) -> None:
    """Build a manifest CSV aligning DICOM image series with SR files.

    Args:
        images_root: Root directory containing DICOM image files.
        sr_root: Root directory containing DICOM SR files.
        output_manifest: Path to write the manifest CSV.
    """

    images_root = os.path.abspath(images_root)
    sr_root = os.path.abspath(sr_root)
    output_manifest = os.path.abspath(output_manifest)

    if Path(output_manifest).exists():
        logger.info("Manifest already exists at %s; skipping folder scan", output_manifest)
        return

    logger.info("Scanning image root: %s", images_root)
    images_map = _collect_image_metadata(
        images_root,
        image_modality=image_modality,
        series_description_keyword=series_description_keyword,
    )
    logger.info("Found %d image series", len(images_map))

    logger.info("Scanning SR root: %s", sr_root)
    sr_map = _collect_sr_metadata(sr_root, sr_modality=sr_modality)
    total_sr_files = sum(len(paths) for paths in sr_map.values())
    logger.info("Found %d SR files across %d studies", total_sr_files, len(sr_map))

    rows = []
    for (study_uid, series_uid), meta_list in images_map.items():
        patient_ids = {m[0] for m in meta_list if m[0] is not None}
        patient_id = next(iter(patient_ids), "")

        sr_key = (patient_id or "", study_uid)
        sr_candidates = sr_map.get(sr_key)
        if not sr_candidates and not patient_id:
            # Try matching by study when patient ID is missing in the image metadata.
            sr_candidates = sr_map.get(("", study_uid))

        if not sr_candidates:
            # Skip image series that lack a corresponding SR report.
            logger.warning(
                "No SR found for patient %s study %s; skipping series %s",
                patient_id or "<missing>",
                study_uid,
                series_uid,
            )
            continue

        sr_path = sorted(sr_candidates)[0]
        sr_patient_id = getattr(pydicom.dcmread(sr_path, stop_before_pixels=True, force=True), "PatientID", None)

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
            }
        )

    if not rows:
        logger.warning("No matched studies found; manifest will be empty")

    df = pd.DataFrame(rows)
    manifest_dir = Path(output_manifest).parent
    manifest_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_manifest, index=False)
    logger.info("Manifest saved to %s (%d rows)", output_manifest, len(df))


def choose_blissauto_dir(study_dir: Path) -> Path | None:
    """
    Given an IMAGE study directory:
        .../<patient>/<study>/
    find all BLISSAUTO series dirs and select ONLY the one with the MOST
    DICOM files.

    Steps:
      - Detect all subdirs where name contains 'BLISSAUTO'
      - Count *.dcm in each
      - Log counts
      - Return the dir with the largest count (ties broken by name)

    If no BLISSAUTO dirs exist, return None.
    """

    candidates = [
        d for d in study_dir.iterdir()
        if d.is_dir() and "BLISSAUTO" in d.name.upper()
    ]

    if not candidates:
        return None

    # Count dicom files in each candidate
    counts = []
    for d in candidates:
        num_dcm = sum(1 for _ in d.glob("*.dcm"))
        counts.append((d, num_dcm))

    # Log for visibility
    print(f"    [BLISSAUTO] Study: {study_dir.name}")
    for d, c in counts:
        print(f"        - {d.name}: {c} DICOM files")

    # Choose the one with the most DICOMs; break ties by directory name
    counts.sort(key=lambda x: (-x[1], x[0].name))
    best_dir, best_count = counts[0]

    if best_count == 0:
        print("    [WARN] All BLISSAUTO series have 0 DICOM files; using first anyway.")

    print(f"    [CHOSEN] {best_dir.name} (DICOM files: {best_count})")
    return best_dir


def choose_sr_dir(study_dir: Path) -> Path | None:
    """
    Given an SR study directory:
        .../<patient>/<study>/
    return the SR report series directory.

    Strategy:
      - Prefer directory whose name contains 'REPORT'
        (e.g. '1.000000-Standard Breast Imaging Report-549.1')
      - Otherwise, if exactly one subdir, use that.
      - Otherwise, return None (ambiguous).
    """

    series_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
    if not series_dirs:
        return None

    report_dirs = [d for d in series_dirs if "REPORT" in d.name.upper()]
    if report_dirs:
        # Prefer explicitly labeled report folders when present.
        return report_dirs[0]

    if len(series_dirs) == 1:
        # Fall back to the sole directory if it's unambiguous.
        return series_dirs[0]

    return None  # ambiguous


def find_blissauto(img_root: str) -> Dict[Tuple[str, str], Path]:
    """
    Scan IMG_ROOT tree and return:
        dict[(patient_id, study_name)] -> Path to chosen BLISSAUTO series dir
    """

    mapping = {}
    root = Path(img_root)

    patient_dirs = [d for d in root.iterdir() if d.is_dir()]

    for patient_dir in tqdm(patient_dirs, desc="Scanning IMG patients"):
        for study_dir in patient_dir.iterdir():
            if not study_dir.is_dir():
                continue

            # Pick the BLISSAUTO series with the highest slice count.
            bliss = choose_blissauto_dir(study_dir)
            if bliss is None:
                print(f"[IMG WARN] No BLISSAUTO in {patient_dir.name} / {study_dir.name}")
                continue

            mapping[(patient_dir.name, study_dir.name)] = bliss
            print(f"[IMG OK] {patient_dir.name} / {study_dir.name} -> {bliss.name}")

    return mapping


def find_sr(sr_root: str) -> Dict[Tuple[str, str], Path]:
    """
    Scan SR_ROOT tree and return:
        dict[(patient_id, study_name)] -> Path to SR series dir
    """

    mapping = {}
    root = Path(sr_root)

    patient_dirs = [d for d in root.iterdir() if d.is_dir()]

    for patient_dir in tqdm(patient_dirs, desc="Scanning SR patients"):
        for study_dir in patient_dir.iterdir():
            if not study_dir.is_dir():
                continue

            # Prefer clearly labeled report series; otherwise skip ambiguous cases.
            sr = choose_sr_dir(study_dir)
            if sr is None:
                print(f"[SR WARN] No unambiguous SR in {patient_dir.name} / {study_dir.name}")
                continue

            mapping[(patient_dir.name, study_dir.name)] = sr
            print(f"[SR OK] {patient_dir.name} / {study_dir.name} -> {sr.name}")

    return mapping


def load_image_volume(image_files: Iterable[Path]) -> np.ndarray | None:
    """Load DICOM slices into a volume sorted for consistent anatomical order."""

    files = list(image_files)
    if not files:
        return None

    datasets: List[pydicom.dataset.Dataset] = []
    for path in files:
        try:
            ds = dcmread(str(path))
            if not hasattr(ds, "pixel_array"):
                continue
            datasets.append(ds)
        except Exception as exc:  # pragma: no cover - logging only
            logger.warning("Could not read image DICOM %s: %s", path, exc)

    if not datasets:
        return None

    def sort_key(ds: pydicom.dataset.Dataset) -> Tuple[str, int, str]:
        series = getattr(ds, "SeriesInstanceUID", "")
        inst = getattr(ds, "InstanceNumber", 0)
        sop = getattr(ds, "SOPInstanceUID", "")
        try:
            inst_int = int(inst)
        except Exception:
            inst_int = 0
        return (series, inst_int, sop)

    datasets.sort(key=sort_key)
    slices = [ds.pixel_array.astype(np.float32) for ds in datasets]
    volume = np.stack(slices, axis=0)
    return volume


def _get_concept_name(item: pydicom.dataset.Dataset) -> str | None:
    if "ConceptNameCodeSequence" not in item:
        return None
    code_item = item.ConceptNameCodeSequence[0]
    name = getattr(code_item, "CodeMeaning", None) or getattr(code_item, "CodeValue", None)
    return name


def _find_assessment_category_in_items(items: Iterable[pydicom.dataset.Dataset]):
    """Recursively search ContentSequence for BI-RADS assessment category."""

    for item in items:
        concept_name = _get_concept_name(item)
        if concept_name and "assessment category" in concept_name.lower():
            raw_value = None
            meaning = None

            if "ConceptCodeSequence" in item:
                code = item.ConceptCodeSequence[0]
                raw_value = getattr(code, "CodeValue", None)
                meaning = getattr(code, "CodeMeaning", None)

            if raw_value is None and hasattr(item, "TextValue"):
                raw_value = item.TextValue

            return raw_value, meaning

        if "ContentSequence" in item and item.ContentSequence:
            nested = _find_assessment_category_in_items(item.ContentSequence)
            if nested is not None:
                return nested

    return None


def extract_assessment_category_from_sr(sr_path: Path):
    try:
        ds = dcmread(str(sr_path), stop_before_pixels=True)
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning("Could not read SR DICOM %s: %s", sr_path, exc)
        return None, None, None

    if getattr(ds, "Modality", "").upper() != "SR":
        return None, None, None

    root_items = getattr(ds, "ContentSequence", None)
    if not root_items:
        return None, None, None

    result = _find_assessment_category_in_items(root_items)
    if result is None:
        return None, None, None

    raw_value, meaning = result
    category_int = None

    if isinstance(raw_value, str):
        match = re.search(r"\d", raw_value)
        if match:
            category_int = int(match.group(0))

    if category_int is None and isinstance(meaning, str):
        match = re.search(r"\d", meaning)
        if match:
            category_int = int(match.group(0))

    return category_int, raw_value, meaning


def aggregate_exam_label(sr_files: Iterable[Path]) -> int | None:
    """Return worst-case BI-RADS assessment across SR files for an exam."""

    categories: List[int] = []

    for sr_path in sr_files:
        cat_int, _, _ = extract_assessment_category_from_sr(sr_path)
        if cat_int is not None:
            categories.append(cat_int)
        else:
            logger.warning("No assessment category found in %s", sr_path)

    if not categories:
        return None

    return max(categories)


def save_exam_pt(
    patient_id: str, study_uid: str, image_files: Iterable[Path], sr_files: Iterable[Path], out_root: Path
) -> bool:
    """Create interim PyTorch tensor for an exam paired with SR label."""

    image_files = list(image_files)
    sr_files = list(sr_files)

    logger.info("[PAIR] %s / %s | images=%d, sr=%d", patient_id, study_uid, len(image_files), len(sr_files))

    if not image_files:
        logger.warning("No image DICOMs found for %s / %s", patient_id, study_uid)
        return False
    if not sr_files:
        logger.warning("No SR DICOMs found for %s / %s", patient_id, study_uid)
        return False

    volume = load_image_volume(image_files)
    if volume is None:
        logger.warning("Could not build image volume for %s / %s", patient_id, study_uid)
        return False

    label_int = aggregate_exam_label(sr_files)
    if label_int is None:
        logger.warning("No valid assessment category found for %s / %s", patient_id, study_uid)
        return False

    image_tensor = torch.from_numpy(volume).unsqueeze(0).float()
    label_tensor = torch.tensor(label_int, dtype=torch.long)

    data = {
        "image": image_tensor,
        "label": label_tensor,
        "patient_id": patient_id,
        "study_name": study_uid,
        "resized_shape": tuple(int(dim) for dim in image_tensor.shape[-3:]),
    }

    safe_study = study_uid.replace(" ", "_").replace(".", "-")
    out_path = out_root / f"{patient_id}__{safe_study}.pt"

    if out_path.exists():
        logger.info("Found existing PT file %s; reusing without regeneration", out_path.name)
        try:
            existing = torch.load(out_path, map_location="cpu")
            if "resized_shape" not in existing and "image" in existing:
                existing["resized_shape"] = tuple(int(dim) for dim in existing["image"].shape[-3:])
                torch.save(existing, out_path)
            return True
        except Exception as err:
            logger.warning(
                "Existing PT file %s could not be loaded (%s); rebuilding", out_path, err
            )

    torch.save(data, out_path)

    logger.info(
        "Saved %s (image shape=%s, label=%d)",
        out_path.name,
        tuple(image_tensor.shape),
        label_int,
    )
    return True


def build_pt_dataset(
    img_root: str,
    sr_root: str,
    out_root: str,
    *,
    image_modality: str = "MR",
    series_description_keyword: str = "BLISS_AUTO",
    sr_modality: str = "SR",
) -> None:
    """Scan BLISS_AUTO MR images and SR reports to build interim PT tensors."""

    logger.info(
        "Scanning DICOM files under %s for %s series with %s in SeriesDescription",
        img_root,
        image_modality,
        series_description_keyword,
    )
    images_map = _collect_image_metadata(
        img_root,
        image_modality=image_modality,
        series_description_keyword=series_description_keyword,
    )

    logger.info(
        "Scanning DICOM files under %s for %s reports", sr_root, sr_modality
    )
    sr_map = _collect_sr_metadata(sr_root, sr_modality=sr_modality)

    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    num_saved = 0
    unmatched_images: List[Tuple[str, str]] = []
    for (study_uid, series_uid), meta_list in tqdm(
        sorted(images_map.items()), desc="Building PT dataset", unit="exam"
    ):
        patient_ids = {m[0] for m in meta_list if m[0]}
        patient_id = next(iter(patient_ids), "")

        sr_key = (patient_id or "", study_uid)
        sr_candidates = sr_map.get(sr_key)
        if not sr_candidates and not patient_id:
            sr_candidates = sr_map.get(("", study_uid))

        if not sr_candidates:
            unmatched_images.append((patient_id or "<missing>", study_uid))
            continue

        image_files = [Path(m[3]) for m in sorted(meta_list, key=lambda x: x[3])]
        sr_files = [Path(p) for p in sorted(sr_candidates)]

        if save_exam_pt(patient_id or "", study_uid, image_files, sr_files, out_root_path):
            num_saved += 1

    if unmatched_images:
        logger.warning(
            "Skipped studies without SR matches: %s",
            "; ".join([f"{p} / {s}" for p, s in unmatched_images]),
        )

    logger.info("Finished. Total exams saved: %d | Output: %s", num_saved, out_root_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tools for BREAST-DIAGNOSIS image/SR alignment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "manifest", help="Build CSV manifest aligning DICOM images with SR labels"
    )
    manifest_parser.add_argument("--images-root", required=True, help="Root directory containing DICOM images")
    manifest_parser.add_argument("--sr-root", required=True, help="Root directory containing DICOM SR files")
    manifest_parser.add_argument("--output", required=True, help="Output CSV manifest path")
    manifest_parser.add_argument(
        "--image-modality",
        default="MR",
        help="Modality tag value used to identify image series (default: MR)",
    )
    manifest_parser.add_argument(
        "--series-description-keyword",
        default="BLISS_AUTO",
        help="Keyword to search for in SeriesDescription when selecting image series",
    )
    manifest_parser.add_argument(
        "--sr-modality",
        default="SR",
        help="Modality tag value used to identify structured reports (default: SR)",
    )

    tensors_parser = subparsers.add_parser(
        "tensors",
        help="Create interim PyTorch tensors from BLISS_AUTO MR images and SR reports",
    )
    tensors_parser.add_argument("--images-root", required=True, help="Root directory containing BLISS_AUTO MR DICOM images")
    tensors_parser.add_argument("--sr-root", required=True, help="Root directory containing SR DICOM files")
    tensors_parser.add_argument(
        "--output-dir", required=True, help="Directory to store generated .pt files"
    )
    tensors_parser.add_argument(
        "--image-modality",
        default="MR",
        help="Modality tag value used to identify image series (default: MR)",
    )
    tensors_parser.add_argument(
        "--series-description-keyword",
        default="BLISS_AUTO",
        help="Keyword to search for in SeriesDescription when selecting image series",
    )
    tensors_parser.add_argument(
        "--sr-modality",
        default="SR",
        help="Modality tag value used to identify structured reports (default: SR)",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "manifest":
        run(
            args.images_root,
            args.sr_root,
            args.output,
            image_modality=args.image_modality,
            series_description_keyword=args.series_description_keyword,
            sr_modality=args.sr_modality,
        )
    else:
        build_pt_dataset(
            args.images_root,
            args.sr_root,
            args.output_dir,
            image_modality=args.image_modality,
            series_description_keyword=args.series_description_keyword,
            sr_modality=args.sr_modality,
        )


if __name__ == "__main__":
    main()
