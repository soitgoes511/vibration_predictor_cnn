# Vibration Predictor (CNN)

This project trains a CNN to classify bearing condition from the FFT spectra you already write to InfluxDB2 from the ESP32 collector (`accelfreq` measurement with tags `operation`, `run_id`).

The baseline model is a 1D CNN over frequency bins with 3 channels (`x_freq`, `y_freq`, `z_freq`), built for predictive maintenance workflows.

## Prerequisites

- Python 3.12 (recommended for Windows + PyTorch compatibility)
- Access to your InfluxDB2 instance and token
- Labeled bearing runs (`run_id` + fault class)

## What It Includes

- InfluxDB2 data ingestion for your existing schema.
- Run-level tensor building (`run_id` -> fixed-size spectrum tensor).
- Supervised training pipeline with:
  - train/validation/test split
  - class weighting
  - early stopping
  - saved artifacts (model, metrics, confusion matrix, predictions)
- Evaluation CLI against fresh Influx data.
- Prediction CLI for unlabeled runs.

## Expected InfluxDB Schema

Frequency measurement:

```
accelfreq,operation=<operation>,run_id=<run_id> frequencies=<hz>,x_freq=<v>,y_freq=<v>,z_freq=<v> <timestamp_ns>
```

The loader groups rows by `(operation, run_id)` and sorts by `frequencies`.

## Setup (Verified)

1. Create and activate a Python 3.12 virtual environment.
2. Install this package.
3. Provide InfluxDB token.
4. Add labels CSV.
5. Run a quick CLI check.
6. Train.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .

$env:INFLUXDB_TOKEN = "your-token"
Copy-Item data\labels_template.csv data\labels.csv
# Edit data\labels.csv with real run_id/label values

vp-train --help
vp-evaluate --help
vp-predict --help
```

Train:

```powershell
vp-train --config configs/default.yaml
```

## Labels File

The training pipeline needs supervised labels in `data/labels.csv` (path configurable):

- Required columns: `run_id`, `label`
- Optional column: `operation` (recommended if run IDs can repeat across operations)

Example:

```csv
operation,run_id,label
L9OP600,12345678,healthy
L9OP600,12345679,inner_race_fault
```

## Evaluate and Predict

```powershell
vp-evaluate --config configs/default.yaml
vp-predict --config configs/default.yaml
```

By default these commands use `artifacts/default_run/model.pt`.

## Configuration

Main config: `configs/default.yaml`

Important knobs:

- `influx.start` / `influx.stop` for time window
- `influx.operation` to isolate one machine/line operation ID
- `dataset.target_bins` for frequency rebinning size
- `training.*` hyperparameters
- `paths.labels_csv` and `paths.output_dir`

## Output Artifacts

After training:

- `artifacts/default_run/model.pt`
- `artifacts/default_run/history.csv`
- `artifacts/default_run/metrics.json`
- `artifacts/default_run/confusion_matrix.csv`
- `artifacts/default_run/test_predictions.csv`
- `artifacts/default_run/dataset_index.csv`

## Notes

- This baseline assumes you have labeled fault states for run IDs.
- If labels are sparse, start with binary classes (`healthy` vs `fault`) and expand once performance stabilizes.
