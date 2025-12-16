"""
dicom_to_pt.py

Scan a root folder recursively for DICOMs, group them by
(PatientID, StudyInstanceUID), then for each exam:
  - stack all non-SR images into a 3D volume (numpy → torch tensor)
  - parse SR DICOMs to extract:
        Supplementary data -> Overall assessment -> Assessment category
    as a BI-RADS integer
  - save one PyTorch .pt file per exam:
        {
            'image': tensor [C=1, D, H, W],
            'label': tensor(),     # BI-RADS int
            'patient_id': str,
            'study_uid': str,
        }
"""

import os
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from pydicom import dcmread
from tqdm import tqdm


# ================== CONFIG ==================
# Root directory containing all DICOMs (any subdirectory structure)
# e.g. the output of combine_blissauto_sr.py
DICOM_ROOT = r"Y:\Dataset\BREAST-DIAGNOSIS_BLISSAUTO_SR"

# Output directory for .pt files (one per (patient, study))
OUT_ROOT = r"Y:\Dataset\BREAST-DIAGNOSIS_PT_EXAMS"

# Optional number of worker threads for header parsing (set >1 for speed).
# Default to the number of CPU cores to accelerate scanning on modern machines.
NUM_WORKERS = max(1, os.cpu_count() or 1)
# ===========================================


def _read_dicom_header(dcm_path: Path):
    """Read a DICOM header (without pixels) and extract routing info."""
    try:
        ds = dcmread(str(dcm_path), stop_before_pixels=True)
        return ds, None
    except Exception as e:
        return None, e


def build_exam_file_dict(root_dir: str, num_workers: int = 1):
    """
    Recursively walk root_dir, read each DICOM header, and build a dictionary:

        exam_dict = {
            (patient_id, study_uid): {
                "image_files": [Path(...), ...],  # Modality != 'SR'
                "sr_files":    [Path(...), ...],  # Modality == 'SR'
            },
            ...
        }

    Uses DICOM tags:
      - PatientID         (0010,0020)
      - StudyInstanceUID  (0020,000D)
      - Modality          (0008,0060)

    Args:
        root_dir: Path to the root directory containing DICOM files.
        num_workers: Optional number of worker threads for header parsing.
            Set >1 to speed up scanning on fast disks/CPUs.
    """
    root = Path(root_dir)
    exam_dict = {}

    files = list(root.rglob("*.dcm"))

    def handle_ds(ds, dcm_path):
        patient_id = getattr(ds, "PatientID", None)
        study_uid = getattr(ds, "StudyInstanceUID", None)
        if not patient_id or not study_uid:
            print(f"[WARN] Missing PatientID or StudyInstanceUID in {dcm_path}, skipping.")
            return

        modality = getattr(ds, "Modality", "").upper()

        key = (patient_id, study_uid)
        entry = exam_dict.setdefault(
            key,
            {"image_files": [], "sr_files": []}
        )

        if modality == "SR":
            entry["sr_files"].append(dcm_path)
        else:
            entry["image_files"].append(dcm_path)

    # Use a thread pool to speed up header parsing when desired.
    if num_workers and num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            def _parse_path(dcm_path):
                ds, err = _read_dicom_header(dcm_path)
                return dcm_path, ds, err

            for dcm_path, ds, err in tqdm(
                executor.map(_parse_path, files, chunksize=32),
                total=len(files),
                desc="Scanning DICOM files",
                unit="file",
            ):
                if err:
                    print(f"[WARN] Cannot read DICOM header {dcm_path}: {err}")
                    continue

                handle_ds(ds, dcm_path)
    else:
        for dcm_path in tqdm(files, desc="Scanning DICOM files", unit="file"):
            ds, err = _read_dicom_header(dcm_path)
            if err:
                print(f"[WARN] Cannot read DICOM header {dcm_path}: {err}")
                continue

            handle_ds(ds, dcm_path)

    return exam_dict


def load_image_volume(image_files):
    """
    Given a list of DICOM image files (non-SR) that belong to the same study,
    load them into a 3D numpy array [num_slices, H, W].

    Slices are sorted by:
        (SeriesInstanceUID, InstanceNumber, SOPInstanceUID)
    to ensure a stable and anatomically consistent order.
    """
    if not image_files:
        return None

    datasets = []
    for p in image_files:
        try:
            ds = dcmread(str(p))
            # Skip non-image instances just in case
            if not hasattr(ds, "pixel_array"):
                continue
            datasets.append(ds)
        except Exception as e:
            print(f"[WARN] Could not read image DICOM {p}: {e}")

    if not datasets:
        return None

    def sort_key(ds):
        series = getattr(ds, "SeriesInstanceUID", "")
        inst = getattr(ds, "InstanceNumber", 0)
        sop = getattr(ds, "SOPInstanceUID", "")
        try:
            inst_int = int(inst)
        except Exception:
            inst_int = 0
        return (series, inst_int, sop)

    datasets.sort(key=sort_key)

    slices = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        slices.append(arr)

    volume = np.stack(slices, axis=0)  # [num_slices, H, W]
    return volume


def get_concept_name(item):
    """
    Get the Concept Name Code Meaning (or Code Value) for an SR content item.

    Uses:
      - ConceptNameCodeSequence (0040,A043)
    """
    if "ConceptNameCodeSequence" not in item:
        return None
    code_item = item.ConceptNameCodeSequence[0]
    name = getattr(code_item, "CodeMeaning", None)
    if not name:
        name = getattr(code_item, "CodeValue", None)
    return name


def find_assessment_category_in_items(items):
    """
    Recursively search a list of SR content items (ContentSequence) for an item
    representing "Assessment Category" and return (raw_value, code_meaning).

    We look for ConceptName CodeMeaning containing "assessment category"
    (case-insensitive); in your reports this should correspond to:

        Supplementary data -> Overall assessment -> Assessment category
    """
    for item in items:
        concept_name = get_concept_name(item)
        if concept_name and "assessment category" in concept_name.lower():
            # Expect a CODE item with ConceptCodeSequence (0040,A168)
            raw_value = None
            meaning = None

            if "ConceptCodeSequence" in item:
                c = item.ConceptCodeSequence[0]
                raw_value = getattr(c, "CodeValue", None)
                meaning = getattr(c, "CodeMeaning", None)

            # Fallback: TEXT ValueType
            if raw_value is None and hasattr(item, "TextValue"):
                raw_value = item.TextValue

            return raw_value, meaning

        # Recurse into nested ContentSequence (0040,A730)
        if "ContentSequence" in item and item.ContentSequence:
            res = find_assessment_category_in_items(item.ContentSequence)
            if res is not None:
                return res

    return None


def extract_assessment_category_from_sr(sr_path: Path):
    """
    Read a DICOM SR and extract the BI-RADS assessment category.

    We use:
      - Modality (0008,0060) == 'SR'
      - ContentSequence (0040,A730)
      - ConceptNameCodeSequence (0040,A043)
      - ConceptCodeSequence (0040,A168)

    Returns:
        (category_int, raw_value, meaning) or (None, None, None)
    """
    try:
        ds = dcmread(str(sr_path), stop_before_pixels=True)
    except Exception as e:
        print(f"[WARN] Could not read SR DICOM {sr_path}: {e}")
        return None, None, None

    if getattr(ds, "Modality", "").upper() != "SR":
        return None, None, None

    root_items = getattr(ds, "ContentSequence", None)
    if not root_items:
        return None, None, None

    res = find_assessment_category_in_items(root_items)
    if res is None:
        return None, None, None

    raw_value, meaning = res
    category_int = None

    # Try to parse CodeValue or raw text as an integer (e.g. "4")
    if isinstance(raw_value, str):
        m = re.search(r"\d", raw_value)
        if m:
            category_int = int(m.group(0))

    # If that fails, try parsing from CodeMeaning (e.g. "BI-RADS 4: Suspicious")
    if category_int is None and isinstance(meaning, str):
        m = re.search(r"\d", meaning)
        if m:
            category_int = int(m.group(0))

    return category_int, raw_value, meaning


def aggregate_exam_label(sr_files):
    """
    For a list of SR files for one exam (same StudyInstanceUID),
    extract all assessment categories and aggregate them into
    a single label (BI-RADS category).

    Strategy:
      - Extract category for each SR
      - Use the maximum category (worst case) if multiple present.

    Returns:
        label_int or None
    """
    categories = []

    for sr_path in sr_files:
        cat_int, raw, meaning = extract_assessment_category_from_sr(sr_path)
        if cat_int is not None:
            categories.append(cat_int)
        else:
            print(f"[WARN] No assessment category found in {sr_path}")

    if not categories:
        return None

    return max(categories)  # worst-case BI-RADS


def main():
    dicom_root = Path(DICOM_ROOT)
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Build (patient, study) -> {image_files, sr_files} dictionary
    print(f"Scanning DICOM files under: {dicom_root}")
    exam_dict = build_exam_file_dict(dicom_root, num_workers=NUM_WORKERS)
    print(f"Found {len(exam_dict)} unique (PatientID, StudyInstanceUID) exams.\n")

    num_ok = 0

    # 2) For each exam, build volume + label and save .pt
    for (patient_id, study_uid) in sorted(exam_dict.keys()):
        entry = exam_dict[(patient_id, study_uid)]
        image_files = entry["image_files"]
        sr_files = entry["sr_files"]

        print(f"\n=== Exam: Patient={patient_id}, StudyUID={study_uid} ===")
        print(f"  Image files: {len(image_files)}")
        print(f"  SR files:    {len(sr_files)}")

        if not image_files:
            print("  [SKIP] No image DICOMs found.")
            continue
        if not sr_files:
            print("  [SKIP] No SR DICOMs found.")
            continue

        # Build image volume
        volume = load_image_volume(image_files)
        if volume is None:
            print("  [SKIP] Could not build image volume.")
            continue

        # Extract BI-RADS category from SR
        label_int = aggregate_exam_label(sr_files)
        if label_int is None:
            print("  [SKIP] No valid assessment category found in SR.")
            continue

        # Convert to torch tensors and resize to a standard 64x64x64 volume
        # Shape after resizing: [C=1, D=64, H=64, W=64]
        image_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
        image_tensor = F.interpolate(
            image_tensor,
            size=(64, 64, 64),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        label_tensor = torch.tensor(label_int, dtype=torch.long)

        data = {
            "image": image_tensor,
            "label": label_tensor,
            "patient_id": patient_id,
            "study_uid": study_uid,
        }

        # Use both patient_id and study_uid in filename to avoid collisions
        safe_study_uid = study_uid.replace(".", "").replace(" ", "")
        out_path = out_root / f"{patient_id}__{safe_study_uid}.pt"
        torch.save(data, out_path)
        num_ok += 1

        print(
            f"  [OK] Saved {out_path.name} "
            f"(image shape={tuple(image_tensor.shape)}, label={label_int})"
        )

    print(f"\nDone. Saved PyTorch objects for {num_ok} exams to {out_root}")


if __name__ == "__main__":
    main()
