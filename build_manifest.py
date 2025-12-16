"""build_manifest.py

Combine BLISSAUTO/SR directory discovery with direct conversion to
intermediate PyTorch tensors without copying DICOM files.

Steps performed:
1. Scan an image root for BLISSAUTO series per study (choose the largest).
2. Scan an SR root for report series (prefer names containing "REPORT").
3. For studies that have both, load BLISSAUTO image slices and SR reports,
   create a 3D tensor volume + BI-RADS label, and save to an interim folder.
"""

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import torch
from pydicom import dcmread


# ================== CONFIG ==================
# Root of IMAGE tree (stop at the BREAST-DIAGNOSIS folder)
IMG_ROOT = r"Y:\Dataset\TCIA_BREAST-DIAGNOSIS_06-22-2015\manifest-BbshIhaG7188578559074019493\BREAST-DIAGNOSIS"

# Root of SR tree (stop at the BREAST-DIAGNOSIS folder)
SR_ROOT = r"Y:\Dataset\BREAST-DIAGNOSIS_SR\manifest-1590525836082\BREAST-DIAGNOSIS"

# Interim output directory for .pt files
OUT_ROOT = r"Y:\Dataset\BREAST-DIAGNOSIS_PT_INTERIM"
# ===========================================


# ---- Discovery helpers (from combine_blissauto_sr.py, without copying) ----
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

    counts = []
    for d in candidates:
        num_dcm = sum(1 for _ in d.glob("*.dcm"))
        counts.append((d, num_dcm))

    print(f"    [BLISSAUTO] Study: {study_dir.name}")
    for d, c in counts:
        print(f"        - {d.name}: {c} DICOM files")

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
      - Otherwise, if exactly one subdir, use that.
      - Otherwise, return None (ambiguous).
    """
    series_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
    if not series_dirs:
        return None

    report_dirs = [d for d in series_dirs if "REPORT" in d.name.upper()]
    if report_dirs:
        return report_dirs[0]

    if len(series_dirs) == 1:
        return series_dirs[0]

    return None


def find_blissauto(img_root: str):
    """
    Scan IMG_ROOT tree and return:
        dict[(patient_id, study_name)] -> Path to chosen BLISSAUTO series dir
    """
    mapping = {}
    root = Path(img_root)

    for patient_dir in root.iterdir():
        if not patient_dir.is_dir():
            continue

        for study_dir in patient_dir.iterdir():
            if not study_dir.is_dir():
                continue

            bliss = choose_blissauto_dir(study_dir)
            if bliss is None:
                print(f"[IMG WARN] No BLISSAUTO in {patient_dir.name} / {study_dir.name}")
                continue

            mapping[(patient_dir.name, study_dir.name)] = bliss
            print(f"[IMG OK] {patient_dir.name} / {study_dir.name} -> {bliss.name}")

    return mapping


def find_sr(sr_root: str):
    """
    Scan SR_ROOT tree and return:
        dict[(patient_id, study_name)] -> Path to SR series dir
    """
    mapping = {}
    root = Path(sr_root)

    for patient_dir in root.iterdir():
        if not patient_dir.is_dir():
            continue

        for study_dir in patient_dir.iterdir():
            if not study_dir.is_dir():
                continue

            sr = choose_sr_dir(study_dir)
            if sr is None:
                print(f"[SR WARN] No unambiguous SR in {patient_dir.name} / {study_dir.name}")
                continue

            mapping[(patient_dir.name, study_dir.name)] = sr
            print(f"[SR OK] {patient_dir.name} / {study_dir.name} -> {sr.name}")

    return mapping


# ---- DICOM to PyTorch conversion helpers (from dicom_to_pt.py) ----
def load_image_volume(image_files: Iterable[Path]):
    """
    Given a list of DICOM image files (non-SR) that belong to the same study,
    load them into a 3D numpy array [num_slices, H, W].

    Slices are sorted by:
        (SeriesInstanceUID, InstanceNumber, SOPInstanceUID)
    to ensure a stable and anatomically consistent order.
    """
    files = list(image_files)
    if not files:
        return None

    datasets = []
    for p in files:
        try:
            ds = dcmread(str(p))
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
    """
    for item in items:
        concept_name = get_concept_name(item)
        if concept_name and "assessment category" in concept_name.lower():
            raw_value = None
            meaning = None

            if "ConceptCodeSequence" in item:
                c = item.ConceptCodeSequence[0]
                raw_value = getattr(c, "CodeValue", None)
                meaning = getattr(c, "CodeMeaning", None)

            if raw_value is None and hasattr(item, "TextValue"):
                raw_value = item.TextValue

            return raw_value, meaning

        if "ContentSequence" in item and item.ContentSequence:
            res = find_assessment_category_in_items(item.ContentSequence)
            if res is not None:
                return res

    return None


def extract_assessment_category_from_sr(sr_path: Path):
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

    if isinstance(raw_value, str):
        m = re.search(r"\d", raw_value)
        if m:
            category_int = int(m.group(0))

    if category_int is None and isinstance(meaning, str):
        m = re.search(r"\d", meaning)
        if m:
            category_int = int(m.group(0))

    return category_int, raw_value, meaning


def aggregate_exam_label(sr_files: Iterable[Path]):
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


# ---- Build manifest + save PT ----
def save_exam_pt(patient_id: str, study_name: str, bliss_dir: Path, sr_dir: Path, out_root: Path):
    image_files = list(bliss_dir.glob("*.dcm"))
    sr_files = list(sr_dir.glob("*.dcm"))

    print(f"[PAIR] {patient_id} / {study_name}")
    print(f"       Image files: {len(image_files)} | SR files: {len(sr_files)}")

    if not image_files:
        print("       [SKIP] No image DICOMs found.")
        return False
    if not sr_files:
        print("       [SKIP] No SR DICOMs found.")
        return False

    volume = load_image_volume(image_files)
    if volume is None:
        print("       [SKIP] Could not build image volume.")
        return False

    label_int = aggregate_exam_label(sr_files)
    if label_int is None:
        print("       [SKIP] No valid assessment category found in SR.")
        return False

    image_tensor = torch.from_numpy(volume).unsqueeze(0).float()
    label_tensor = torch.tensor(label_int, dtype=torch.long)

    data = {
        "image": image_tensor,
        "label": label_tensor,
        "patient_id": patient_id,
        "study_name": study_name,
    }

    safe_study = study_name.replace(" ", "_").replace(".", "-")
    out_path = out_root / f"{patient_id}__{safe_study}.pt"
    torch.save(data, out_path)

    print(
        f"       [OK] Saved {out_path.name} "
        f"(image shape={tuple(image_tensor.shape)}, label={label_int})"
    )
    return True


def main():
    print("Scanning for BLISSAUTO image series...")
    bliss_map = find_blissauto(IMG_ROOT)

    print("\nScanning for SR report series...")
    sr_map = find_sr(SR_ROOT)

    common_pairs = set(bliss_map.keys()) & set(sr_map.keys())

    print("\n============== SUMMARY ==============")
    print(f"BLISSAUTO (IMG) pairs found: {len(bliss_map)}")
    print(f"SR pairs found:              {len(sr_map)}")
    print(f"Matching IMG+SR pairs:       {len(common_pairs)}")
    print("=====================================\n")

    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    num_saved = 0
    for patient, study in sorted(common_pairs):
        bliss_dir = bliss_map[(patient, study)]
        sr_dir = sr_map[(patient, study)]

        ok = save_exam_pt(patient, study, bliss_dir, sr_dir, out_root)
        if ok:
            num_saved += 1

    incomplete_patients = (
        {p for (p, _) in bliss_map.keys()} | {p for (p, _) in sr_map.keys()}
    ) - {p for (p, _) in common_pairs}

    if incomplete_patients:
        print("\n############ WARNING ############")
        print("These patients were skipped because IMG or SR was missing:")
        for p in sorted(incomplete_patients):
            print(f"  - {p}")
        print("#################################")

    print("\nDone.")
    print(f"Patients with complete IMG+SR: {len(common_pairs)}")
    print(f"Total exams saved:             {num_saved}")
    print(f"Output root:                   {out_root}")


if __name__ == "__main__":
    main()
