# Vibration Predictor (CNN)

This project trains a CNN to classify bearing condition from the FFT spectra you already write to InfluxDB2 from the ESP32 collector (`accelfreq` measurement with tags `operation`, `run_id`).

The baseline model is a 1D CNN over frequency bins with 3 channels (`x_freq`, `y_freq`, `z_freq`), built for predictive maintenance workflows.

## Prerequisites

- Python 3.12 (recommended for Windows + PyTorch compatibility)
- Access to your InfluxDB2 instance and token
- Labeled bearing runs (`run_id` + fault class)

## What It Includes

- InfluxDB2 data ingestion for your existing schema.
- Run-level metadata filtering (`accelrunmeta`) for sample-rate / FFT / firmware consistency.
- Run-level tensor building (`run_id` -> fixed-size spectrum tensor).
- Supervised training pipeline with:
  - train/validation/test split
  - class weighting
  - early stopping
  - saved artifacts (model, metrics, confusion matrix, predictions)
- Optional bootstrap training data from external datasets (for example CWRU).
- Evaluation CLI against fresh Influx data.
- Prediction CLI for unlabeled runs.

## Expected InfluxDB Schema

Frequency measurement (`accelfreq`):

```
accelfreq,operation=<operation>,device_id=<device_id>,run_id=<run_id> frequencies=<hz>,x_freq=<v>,y_freq=<v>,z_freq=<v> <timestamp_ns>
```

The loader groups rows by `(operation, run_id)` and sorts by `frequencies`.

Run metadata measurement (`accelrunmeta`) is used to keep training data consistent:

```
accelrunmeta,operation=<operation>,device_id=<device_id>,run_id=<run_id> sample_rate_hz=3200i,fft_size=4096i,window="hann",fw="1.1.0" <timestamp_ns>
```

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
vp-bootstrap-cwru --help
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
L9OP600,6A4F-1739356800-1,healthy
L9OP600,6A4F-1739356900-2,inner_race_fault
```

## Evaluate and Predict

```powershell
vp-evaluate --config configs/default.yaml
vp-predict --config configs/default.yaml
```

By default these commands use `artifacts/default_run/model.pt`.

## Bootstrap with University Datasets

You can bootstrap the model with public bearing datasets before you have many local fault events.

Common sources:

- CWRU Bearing Data Center: https://engineering.case.edu/bearingdatacenter/download-data-file
- Paderborn University dataset page: https://mb.uni-paderborn.de/kat/main-research/datacenter/bearing-datacenter/open-source-rolling-bearing-datasets
- NASA mirror of IMS + FEMTO bearing sets: https://github.com/AlephNotation/bearing-data-set

Example bootstrap flow for CWRU:

```powershell
Copy-Item data\cwru_label_map_template.csv data\cwru_label_map.csv
# Edit patterns in data\cwru_label_map.csv for your file names

vp-bootstrap-cwru `
  --input-dir data\raw\cwru `
  --label-map-csv data\cwru_label_map.csv `
  --output data\bootstrap\cwru_bootstrap.npz `
  --segment-len 4096 `
  --segment-step 2048 `
  --target-bins 2048 `
  --log-scale `
  --normalize-per-axis
```

Then enable bootstrap in `configs/default.yaml`:

```yaml
bootstrap:
  enabled: true
  npz_path: "data/bootstrap/cwru_bootstrap.npz"
  use_for_training_only: true
```

If you want bootstrap-only training initially, leave `influx.token` empty and set:

```yaml
bootstrap:
  enabled: true
  use_for_training_only: false
```

## Configuration

Main config: `configs/default.yaml`

Important knobs:

- `influx.start` / `influx.stop` for time window
- `influx.operation` to isolate one machine/line operation ID
- `influx.device_ids` to restrict to specific ESP32 sensors
- `influx.require_run_metadata` + `influx.expected_*` to enforce consistent run settings
- `dataset.target_bins` for frequency rebinning size
- `training.*` hyperparameters
- `paths.labels_csv` and `paths.output_dir`
- `bootstrap.*` to include external labeled data

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
