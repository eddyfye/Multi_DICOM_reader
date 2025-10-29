# Multi DICOM Reader

This repository provides a small command-line utility for reading DICOM files
and exporting selected metadata to a CSV file. The script is written to be
compatible with [PyInstaller](https://pyinstaller.org/en/stable/) so it can be
distributed as a standalone executable.

## Requirements

- Python 3.9 or later
- The Python packages listed in [`requirements.txt`](requirements.txt)

Install the dependencies into your active Python environment with:

```
./install_dependencies.sh
```

The script downloads the required wheels into `local_packages/` and installs
from that cache. Override the behaviour by setting `DOWNLOAD_DIR`,
`REQUIREMENTS_FILE`, or `PYTHON_BIN` environment variables as needed.

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

Use the provided helper script to create a standalone executable and a matching
PyInstaller spec file:

```
./build_executable.sh
```

This produces a one-file executable in the `dist/` directory and places the
generated spec file in `pyinstaller/`. Adjust the `APP_NAME`, `SPEC_DIR`,
`DIST_DIR`, or `BUILD_DIR` environment variables when invoking the script if you
need to customise where the build artefacts are written.

## License

This project is provided without a specific license. Use at your own discretion.
