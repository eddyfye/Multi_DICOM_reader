# Multi DICOM Reader

This repository provides a small command-line utility for reading DICOM files
and exporting selected metadata to a CSV file. The script is written to be
compatible with [PyInstaller](https://pyinstaller.org/en/stable/) so it can be
distributed as a standalone executable.

## Requirements

- Python 3.9 or later
- [pydicom](https://pydicom.github.io/pydicom/stable/)

## Usage

```
python dicom_metadata_extractor.py <input_dir> <output_dir> [--output-name OUTPUT_NAME]
```

- `input_dir`: Directory containing one or more DICOM files (searched
  recursively).
- `output_dir`: Directory where the resulting CSV file will be created. The
  directory is created if it does not already exist.
- `--output-name`: Optional name for the CSV file (defaults to
  `dicom_metadata.csv`).

The CSV file includes the following columns:

1. `source_path`
2. `patient_name`
3. `sop_instance_uid`
4. `study_instance_uid`
5. `series_instance_uid`

## Bundling with PyInstaller

To build a single-file executable using PyInstaller:

```
pyinstaller --onefile dicom_metadata_extractor.py
```

The resulting executable will accept the same arguments as the Python script.

## License

This project is provided without a specific license. Use at your own discretion.
