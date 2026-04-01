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

## Why These Defaults Were Chosen

The values in `configs/default.yaml` are baseline settings chosen to make the first version of the pipeline stable on small-to-medium vibration datasets, not to claim they are universally optimal. They are meant to be easy to reason about, easy to retrain, and conservative enough that you can tune from a working starting point.

### Data and Preprocessing

- `expected_sample_rate_hz: 3200`, `expected_fft_size: 4096`, `expected_window: "hann"`, `expected_fw_prefix: "1."`
  These metadata filters are there to keep the training data physically comparable. A model can easily learn differences caused by a changed sample rate, FFT length, window, or firmware version instead of learning the bearing condition itself.
- `target_bins: 2048`
  Each run is interpolated onto a fixed frequency grid so the CNN always sees the same input width. `2048` keeps reasonably fine spectral detail while still being small enough for fast training and inference.
- `min_bins_per_run: 1800`
  Runs with too few frequency points are skipped because aggressive interpolation from a short spectrum would create noisy or misleading inputs. Keeping this threshold close to `target_bins` favors data quality over dataset size.
- `log_scale: true`
  FFT magnitudes often have a very wide dynamic range. `log1p` compresses large peaks so weaker but informative fault-related components are not drowned out.
- `normalize_per_axis: true`
  Each axis (`x_freq`, `y_freq`, `z_freq`) is standardized independently so the model focuses more on spectral shape than raw amplitude differences caused by mounting, gain, or operating variability.

### Model Architecture

- `conv_channels: [32, 64, 128]`
  The network widens gradually as it goes deeper: the early layers learn simple local spectral patterns and the later layers combine them into more class-specific features. This is a standard "small to medium" CNN size that is usually enough for 1D spectra without making the model heavy.
- `kernel_sizes: [7, 5, 3]`
  The first layer uses a wider receptive field to catch broader spectral structures, then later layers use smaller kernels to refine narrower local patterns. Using odd kernel sizes also makes it easy to use symmetric padding and preserve alignment before pooling.
- `MaxPool1d(kernel_size=2)` after each convolution block
  Pooling reduces the frequency resolution step by step, which lowers compute cost and encourages the network to learn more robust features instead of memorizing exact bin positions.
- `AdaptiveAvgPool1d(1)` before the classifier
  The classifier receives a compact summary of each learned feature map instead of a very large flattened vector. That keeps the number of trainable parameters down and reduces overfitting risk.
- `Linear(..., 128)` in the hidden classifier layer
  A single medium-sized dense layer gives the model some non-linear mixing capacity after convolution without making the classifier dominate the parameter count.
- `dropout: 0.3`
  `0.3` is a moderate regularization setting: enough to fight overfitting on limited labeled runs, but not so aggressive that training becomes unstable.

### Training Settings

- `batch_size: 32`
  This is a practical default that is usually stable on CPU or modest GPU hardware while still giving smooth enough gradient estimates.
- `epochs: 60`
  The upper bound is intentionally generous because early stopping is enabled. In practice, training often stops earlier.
- `learning_rate: 0.001`
  This is a common starting point for Adam-style optimizers and usually works well for a small CNN on standardized inputs.
- `weight_decay: 0.0001`
  A light amount of L2-style regularization helps generalization without overpowering the learning signal.
- `train_split: 0.7`, `val_split: 0.15`, test = remaining `0.15`
  The split keeps most labeled runs for training while still reserving separate validation and test sets.
- `early_stopping_patience: 10`
  Validation loss is allowed to plateau for a while before stopping, which helps avoid quitting too early because of a few noisy epochs.
- Class-weighted `CrossEntropyLoss`
  The training loop automatically upweights minority classes. That is important in predictive maintenance because healthy runs are often much more common than fault runs.
- `AdamW`
  AdamW is a strong default for this kind of tabular-to-tensor training workflow because it converges reliably with minimal tuning.

### Why PyTorch Instead of TensorFlow?

This project uses PyTorch mostly for pragmatic reasons:

- The model is a straightforward custom `nn.Module`, and PyTorch makes that style of iterative experimentation very direct.
- The training loop in `src/vibration_predictor/train.py` is explicit and easy to modify for class weighting, bootstrap data mixing, custom splits, and artifact export.
- For a small 1D CNN like this, PyTorch gives all the building blocks needed without extra framework ceremony.
- The README already recommends Python 3.12 specifically for Windows + PyTorch compatibility, so the current project setup is optimized around that stack.

It is not because TensorFlow would be incapable here. A TensorFlow/Keras version would be entirely feasible. PyTorch was simply the more convenient framework for the baseline implementation in this repository.

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
