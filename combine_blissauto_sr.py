from pathlib import Path
import shutil


# ================== CONFIG ==================
# Root of IMAGE tree (stop at the BREAST-DIAGNOSIS folder)
IMG_ROOT = r"Y:\Dataset\TCIA_BREAST-DIAGNOSIS_06-22-2015\manifest-BbshIhaG7188578559074019493\BREAST-DIAGNOSIS"

# Root of SR tree (stop at the BREAST-DIAGNOSIS folder)
SR_ROOT = r"Y:\Dataset\BREAST-DIAGNOSIS_SR\manifest-1590525836082\BREAST-DIAGNOSIS"

# Output directory — one folder per patient
OUT_ROOT = r"Y:\Dataset\BREAST-DIAGNOSIS_BLISSAUTO_SR"
# ===========================================


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
        return report_dirs[0]

    if len(series_dirs) == 1:
        return series_dirs[0]

    return None  # ambiguous


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


def copy_all_dicom_to_patient_folder(bliss_dir: Path, sr_dir: Path, patient_out: Path):
    """
    Copy BLISSAUTO + SR DICOM files into a single folder per patient.

    - Image files: ALWAYS named 'IMG_{original_name}'
      If 'IMG_{original_name}' already exists, append a numeric suffix:
          IMG_1_{original_name}, IMG_2_{original_name}, ...
    - SR files: ALWAYS named 'SR_{original_name}'
      If collision occurs, also suffix numerically.
    """
    patient_out.mkdir(parents=True, exist_ok=True)

    img_count = 0
    sr_count = 0

    # ---- Copy BLISSAUTO (image) DICOM files ----
    for dcm in bliss_dir.glob("*.dcm"):
        base_name = f"IMG_{dcm.name}"
        dest = patient_out / base_name

        if dest.exists():
            i = 1
            while True:
                candidate = patient_out / f"IMG_{i}_{dcm.name}"
                if not candidate.exists():
                    dest = candidate
                    break
                i += 1

        shutil.copy2(dcm, dest)
        img_count += 1

    # ---- Copy SR DICOM files ----
    for dcm in sr_dir.glob("*.dcm"):
        base_name = f"SR_{dcm.name}"
        dest = patient_out / base_name

        if dest.exists():
            i = 1
            while True:
                candidate = patient_out / f"SR_{i}_{dcm.name}"
                if not candidate.exists():
                    dest = candidate
                    break
                i += 1

        shutil.copy2(dcm, dest)
        sr_count += 1

    return img_count, sr_count


def main():
    print("Scanning for BLISSAUTO image series...")
    bliss_map = find_blissauto(IMG_ROOT)

    print("\nScanning for SR report series...")
    sr_map = find_sr(SR_ROOT)

    # Matching (patient, study) pairs that have both IMG + SR
    common_pairs = set(bliss_map.keys()) & set(sr_map.keys())

    print("\n============== SUMMARY ==============")
    print(f"BLISSAUTO (IMG) pairs found: {len(bliss_map)}")
    print(f"SR pairs found:              {len(sr_map)}")
    print(f"Matching IMG+SR pairs:       {len(common_pairs)}")
    print("=====================================\n")

    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    # Build per-patient list of valid (IMG+SR) studies
    patients_with_full_data = {}
    for patient, study in common_pairs:
        patients_with_full_data.setdefault(patient, []).append(study)

    total_img = 0
    total_sr = 0
    num_patients = 0

    # Copy data for patients that have at least one valid IMG+SR pair
    for patient, studies in sorted(patients_with_full_data.items()):
        num_patients += 1
        patient_out = out_root / patient
        patient_out.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Processing Patient: {patient} ===")

        for study in sorted(studies):
            bliss_dir = bliss_map[(patient, study)]
            sr_dir = sr_map[(patient, study)]

            print(f"[PAIR] {patient} / {study}")
            img_count, sr_count = copy_all_dicom_to_patient_folder(
                bliss_dir, sr_dir, patient_out
            )
            print(f"       Copied {img_count} IMG files, {sr_count} SR files")

            total_img += img_count
            total_sr += sr_count

    # Warn about incomplete patients (missing IMG or SR)
    all_patients = {p for (p, _) in bliss_map.keys()} | {p for (p, _) in sr_map.keys()}
    incomplete_patients = all_patients - set(patients_with_full_data.keys())

    if incomplete_patients:
        print("\n############ WARNING ############")
        print("These patients were skipped because IMG or SR was missing:")
        for p in sorted(incomplete_patients):
            print(f"  - {p}")
        print("#################################")

    print("\nDone.")
    print(f"Patients with complete IMG+SR: {num_patients}")
    print(f"Total IMG files copied:        {total_img}")
    print(f"Total SR files copied:         {total_sr}")
    print(f"Output root:                   {out_root}")


if __name__ == "__main__":
    main()
