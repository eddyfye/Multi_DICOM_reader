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
  --images-root /path/to/image/tree \
  --sr-root /path/to/sr/tree \
  --output /tmp/breast_manifest.csv
```

By default the manifest builder looks for image series whose `Modality` equals
`MR` and whose `SeriesDescription` contains `BLISS_AUTO`, and pairs them with SR
files whose `Modality` equals `SR`. You can override these keywords either via
CLI flags or by setting `data.dicom_filters` in the experiment config.

Train a model using a configuration file:

```bash
python -m breast_diag_project.src.run_experiment \
  --configdir breast_diag_project/configs \
  --inputdir /path/to/data/raw \
  --preprocresultdir /tmp/breast_manifest_and_cache \
  --outputdir /tmp/breast_runs/exp1
```

The runner saves checkpoints, metrics, and logs to the specified `outputdir` so
you can track experiment results across runs.

## Hyperparameter tuning

Use the Optuna-backed tuner to launch multiple training trials and explore
hyperparameters:

```bash
python -m breast_diag_project.src.tune \
  --configdir breast_diag_project/configs \
  --inputdir /path/to/data/raw \
  --preprocresultdir /tmp/breast_manifest_and_cache \
  --outputdir /tmp/breast_tuning_runs
```

Add a `tuning` block to your experiment config to control the search. Example:

```json
{
  "tuning": {
    "n_trials": 20,
    "study_name": "breast_diag_study",
    "direction": "minimize",
    "storage": "sqlite:///tmp/breast_diag_study.db",
    "prune": true,
    "search_space": {
      "learning_rate": { "type": "float", "low": 0.0001, "high": 0.005, "log": true },
      "weight_decay": { "type": "float", "low": 0.000001, "high": 0.01, "log": true },
      "batch_size": { "type": "categorical", "choices": [1, 2, 4] },
      "dropout": { "type": "float", "low": 0.0, "high": 0.5 }
    }
  }
}
```

Study results are stored in the Optuna storage backend if you set `tuning.storage`
(for example, a SQLite URL). If `storage` is omitted, Optuna keeps the study in
memory for the duration of the run. Trial artifacts (checkpoints, logs, and
metrics) are written under the `outputdir` you pass to the CLI, organized into
`trial_<n>` subfolders.
