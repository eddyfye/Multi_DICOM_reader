"""Utility script for extracting metadata from DICOM files.

This module exposes a command-line interface that reads all DICOM files in a
given input directory, extracts a subset of metadata, and writes the results to
a CSV file in the specified output directory. The script is intentionally
standalone so it can easily be bundled into a PyInstaller executable.
"""

from __future__ import annotations

import argparse  # Parse CLI flags and options
import csv  # Write extracted rows to a CSV file
import sys  # Report errors and exit codes
from dataclasses import dataclass  # Provide lightweight record containers
from pathlib import Path  # Handle filesystem paths in a platform-agnostic way
from typing import Iterable, List  # Type hints for iterables and lists

import pydicom  # Core library for reading DICOM files
from pydicom.errors import InvalidDicomError  # Raised when a file is not valid DICOM


@dataclass  # Automatically generate init/eq/repr for the metadata holder
class DicomMetadata:
    """Subset of metadata extracted from a DICOM file."""

    source_path: Path  # Path to the DICOM file on disk
    patient_name: str  # Patient name tag
    sop_instance_uid: str  # Unique identifier for the image instance
    study_instance_uid: str  # Unique identifier for the study
    series_instance_uid: str  # Unique identifier for the series

    def to_csv_row(self) -> List[str]:
        """Convert the metadata to a row suitable for ``csv.writer``."""

        return [
            str(self.source_path),  # Convert Path to string for CSV compatibility
            self.patient_name,  # Already normalized to string
            self.sop_instance_uid,  # SOPInstanceUID column
            self.study_instance_uid,  # StudyInstanceUID column
            self.series_instance_uid,  # SeriesInstanceUID column
        ]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Extract key metadata from DICOM files and save as CSV."  # Help text
    )
    parser.add_argument(
        "--input_dir",  # Flag name used by callers
        type=Path,  # Coerce the string into a Path object
        help="Directory containing DICOM files to process.",  # Guidance for users
    )
    parser.add_argument(
        "--output_dir",  # Destination folder flag
        type=Path,  # Convert to Path for path-safe joins
        help="Directory where the CSV file will be written.",  # Describe expected value
    )
    parser.add_argument(
        "--output-name",  # Optional override of the CSV filename
        default="dicom_metadata.csv",  # Fallback filename when not provided
        help="Name of the CSV file to create inside the output directory.",  # CLI help
    )

    return parser.parse_args(list(argv) if argv is not None else None)  # Allow tests to inject argv


def find_dicom_files(root: Path) -> Iterable[Path]:
    """Yield file paths under ``root`` that look like DICOM files."""

    for path in root.rglob("*"):
        if path.is_file():  # Skip directories and other non-file entries
            # Only yield leaf files; directories are skipped implicitly.
            yield path  # Caller decides whether each file is valid DICOM


def load_metadata(path: Path) -> DicomMetadata | None:
    """Read the DICOM file at ``path`` and return extracted metadata.

    Returns ``None`` when the file cannot be read as DICOM.
    """

    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=True)  # Skip pixel data for speed
    except (InvalidDicomError, FileNotFoundError, PermissionError, OSError):
        return None  # Non-DICOM or unreadable files are ignored by the caller

    def _get_value(tag: str) -> str:
        value = dataset.get(tag, "")  # Fetch the element or empty default
        if value is None:  # Normalize missing values to an empty string
            return ""
        return str(value)  # Ensure consistent string output

    return DicomMetadata(
        source_path=path,  # Keep original file location for traceability
        patient_name=_get_value("PatientName"),  # Extract patient name
        sop_instance_uid=_get_value("SOPInstanceUID"),  # Instance UID
        study_instance_uid=_get_value("StudyInstanceUID"),  # Study UID
        series_instance_uid=_get_value("SeriesInstanceUID"),  # Series UID
    )


def write_csv(output_path: Path, records: Iterable[DicomMetadata]) -> None:
    """Write ``records`` to ``output_path`` in CSV format."""

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:  # Create/overwrite CSV
        writer = csv.writer(csvfile)  # Instantiate CSV writer
        # Emit a stable header row so downstream tools can rely on column names.
        writer.writerow(
            [
                "source_path",  # File path column
                "patient_name",  # Patient name column
                "sop_instance_uid",  # SOP UID column
                "study_instance_uid",  # Study UID column
                "series_instance_uid",  # Series UID column
            ]
        )
        for record in records:  # Iterate over extracted metadata objects
            # Each dataclass instance knows how to render its own row.
            writer.writerow(record.to_csv_row())  # Append a CSV row


def ensure_directory(path: Path) -> None:
    """Create the directory ``path`` if it does not already exist."""

    # ``exist_ok`` avoids errors when the folder already exists.
    path.mkdir(parents=True, exist_ok=True)  # Recursively create parent folders


def run(argv: Iterable[str] | None = None) -> int:
    """Entrypoint for command-line execution."""

    args = parse_args(argv)  # Capture validated CLI input

    if not args.input_dir.is_dir():  # Reject missing or non-directory inputs early
        print(f"Input directory does not exist or is not a directory: {args.input_dir}", file=sys.stderr)
        return 1  # Non-zero exit signals failure

    # Make sure the destination directory exists before writing the CSV file.
    ensure_directory(args.output_dir)  # Create output tree if needed
    output_path = args.output_dir / args.output_name  # Compose full CSV path

    # Collect metadata from every readable DICOM file in the input tree.
    metadata_records: List[DicomMetadata] = []  # Aggregated results container
    for file_path in find_dicom_files(args.input_dir):  # Walk all files
        metadata = load_metadata(file_path)  # Attempt to extract key tags
        if metadata is not None:  # Skip unreadable files
            metadata_records.append(metadata)  # Store successful extraction

    # Persist the aggregated rows and provide simple progress feedback.
    write_csv(output_path, metadata_records)  # Write header + rows to disk

    print(f"Wrote metadata for {len(metadata_records)} files to {output_path}")  # Friendly summary
    return 0  # Signal success


def main() -> None:
    sys.exit(run())  # Delegate to run() so tests can call run() directly


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
