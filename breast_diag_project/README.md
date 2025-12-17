# Breast diagnosis experiment helpers

This subpackage contains scripts and utilities used to prepare the TCIA
BREAST-DIAGNOSIS dataset and run training experiments. The tooling was written
for exploratory research rather than production deployment, but the steps below
describe how the pieces fit together.

## Project layout

- `configs/`: example configuration files for training and experiments.
- `src/build_manifest.py`: aligns DICOM image series with matching DICOM-SR
  reports and writes a manifest CSV that downstream steps can consume.
- `src/dataset.py`: dataset helpers that normalize volume shapes and pair image
  volumes with labels parsed from SR files.
- `src/sr_parser.py`: utilities for extracting BI-RADS assessment categories
  from structured report content sequences.
- `src/train.py`: training loop and evaluation helpers built on PyTorch.
- `src/run_experiment.py`: wraps the training logic with configuration parsing
  and experiment logging.
- `src/dicom_io.py`: shared routines for loading and transforming DICOM volumes.

## Typical workflow

1. **Assemble paired data**: run `build_manifest.py` to connect imaging series
   with their corresponding SR files. The script walks separate image and SR
   trees, matches them by `StudyInstanceUID`, and writes the manifest paths to a
   CSV for later reuse.
2. **Prepare tensors**: use `dicom_to_pt.py` from the repository root to convert
   the paired DICOMs into per-exam PyTorch tensors. Each tensor contains a
   stacked image volume and a BI-RADS label extracted from the SR.
3. **Train a model**: invoke `python -m breast_diag_project.src.run_experiment`
   with the appropriate configuration. The experiment runner loads the manifest,
   constructs datasets, and trains the model defined in `src/models.py`.

### Preprocessing guardrails

The dataset utilities can downsample overly large volumes before writing cached
`.pt` files. Configure `data.preprocessing.max_volume_bytes` or
`data.preprocessing.max_voxels` in your experiment config to cap the allowed
volume size. When a study exceeds these limits, the loader will log a message,
downsample the volume to fit within the cap, and skip the study entirely if it
still cannot meet the constraints and `downsample_on_overflow` is disabled.

## Example commands

Build a manifest CSV mapping image series to SR reports:

```bash
python -m breast_diag_project.src.build_manifest \
  --images_root /path/to/image/tree \
  --sr_root /path/to/sr/tree \
  --output_manifest /tmp/breast_manifest.csv
```

Train a model using a configuration file:

```bash
python -m breast_diag_project.src.run_experiment \
  --config breast_diag_project/configs/default.yaml \
  --manifest /tmp/breast_manifest.csv \
  --output-dir /tmp/breast_runs/exp1
```

The runner saves checkpoints, metrics, and logs to the specified `output-dir` so
you can track experiment results across runs.
