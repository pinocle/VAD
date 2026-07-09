# VAD

Video anomaly detection preprocessing, feature extraction, and Z flow-matching pipeline.

## Structure

- `config/`: runtime configuration. `local.yaml` and `prod.yaml` share the same schema.
- `data/01_raw/`: raw datasets.
- `data/02_processed/`: standardized frames, labels, manifests, and window indices.
- `data/03_features/`: cached C_high, Z, and C_low artifacts.
- `data/04_predictions/`: prediction outputs.
- `entrypoints/`: command-line scripts.
- `src/`: reusable pipeline code.
- `tests/`: automated checks.
- `notebooks/`: exploration only.

## Preprocess ShanghaiTech

```bash
python entrypoints/preprocess.py --config config/local.yaml
```

Smoke run:

```bash
python entrypoints/preprocess.py \
  --config config/local.yaml \
  --limit-per-split 1 \
  --overwrite
```

Run tests:

```bash
python -m unittest discover tests
```

## Train / Inference

The Z DiT stage uses conditional flow matching: cached high-level condition
features `C_high` and optional low-level condition features `C_low` condition an
Euler sampler that predicts future VAE features `Z_hat` from cached target `Z`.

```bash
python entrypoints/train.py --config config/local.yaml --run-id flow_exp001
python entrypoints/inference.py --config config/local.yaml --run-id flow_exp001 --overwrite
```
