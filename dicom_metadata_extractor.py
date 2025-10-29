"""Utility script for extracting metadata from DICOM files.

This module exposes a command-line interface that reads all DICOM files in a
given input directory, extracts a subset of metadata, and writes the results to
a CSV file in the specified output directory. The script is intentionally
standalone so it can easily be bundled into a PyInstaller executable.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pydicom
from pydicom.errors import InvalidDicomError


@dataclass
class DicomMetadata:
    """Subset of metadata extracted from a DICOM file."""

    source_path: Path
    patient_name: str
    sop_instance_uid: str
    study_instance_uid: str
    series_instance_uid: str

    def to_csv_row(self) -> List[str]:
        """Convert the metadata to a row suitable for ``csv.writer``."""

        return [
            str(self.source_path),
            self.patient_name,
            self.sop_instance_uid,
            self.study_instance_uid,
            self.series_instance_uid,
        ]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Extract key metadata from DICOM files and save as CSV."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Directory containing DICOM files to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory where the CSV file will be written.",
    )
    parser.add_argument(
        "--output-name",
        default="dicom_metadata.csv",
        help="Name of the CSV file to create inside the output directory.",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def find_dicom_files(root: Path) -> Iterable[Path]:
    """Yield file paths under ``root`` that look like DICOM files."""

    for path in root.rglob("*"):
        if path.is_file():
            yield path


def load_metadata(path: Path) -> DicomMetadata | None:
    """Read the DICOM file at ``path`` and return extracted metadata.

    Returns ``None`` when the file cannot be read as DICOM.
    """

    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
    except (InvalidDicomError, FileNotFoundError, PermissionError, OSError):
        return None

    def _get_value(tag: str) -> str:
        value = dataset.get(tag, "")
        if value is None:
            return ""
        return str(value)

    return DicomMetadata(
        source_path=path,
        patient_name=_get_value("PatientName"),
        sop_instance_uid=_get_value("SOPInstanceUID"),
        study_instance_uid=_get_value("StudyInstanceUID"),
        series_instance_uid=_get_value("SeriesInstanceUID"),
    )


def write_csv(output_path: Path, records: Iterable[DicomMetadata]) -> None:
    """Write ``records`` to ``output_path`` in CSV format."""

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "source_path",
                "patient_name",
                "sop_instance_uid",
                "study_instance_uid",
                "series_instance_uid",
            ]
        )
        for record in records:
            writer.writerow(record.to_csv_row())


def ensure_directory(path: Path) -> None:
    """Create the directory ``path`` if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def run(argv: Iterable[str] | None = None) -> int:
    """Entrypoint for command-line execution."""

    args = parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"Input directory does not exist or is not a directory: {args.input_dir}", file=sys.stderr)
        return 1

    ensure_directory(args.output_dir)
    output_path = args.output_dir / args.output_name

    metadata_records: List[DicomMetadata] = []
    for file_path in find_dicom_files(args.input_dir):
        metadata = load_metadata(file_path)
        if metadata is not None:
            metadata_records.append(metadata)

    write_csv(output_path, metadata_records)

    print(f"Wrote metadata for {len(metadata_records)} files to {output_path}")
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
