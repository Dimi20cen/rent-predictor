# rent-predictor

Swiss rental price prediction project with:
- an XGBoost model,
- reproducible train/evaluate/predict scripts,
- and a lightweight HTML/CSS/JS web app for interactive inference.

## Navigation

- [Results and Demo](#results-and-demo)
- [Dataset](#dataset)
- [Replicate Locally](#replicate-locally)
- [CLI Workflow](#cli-workflow)
- [Deployment](#deployment)
- [Quality Checks](#quality-checks)
- [Model Report](#model-report)
- [Productionization Decisions](#productionization-decisions)
- [Notebooks](#notebook-flow)
- [Project Structure](#project-structure)
- [Notes](#notes)

## Results and Demo

### Key Results (current baseline run)

- MAE: `347.02 CHF`
- RMSE: `748.05 CHF`
- R2: `0.7856`

These metrics come from the reproducible CLI pipeline (`train.py` / `evaluate.py`) on `data/processed/02_featured_data.pkl`.

### Demo

- Live demo: [https://dimy.dev/projects/rent-predictor](https://dimy.dev/projects/rent-predictor)

- Local interactive demo:

```bash
python app.py
```

## Dataset

### Data lineage

- Raw scrape (not committed): ~22,515 listings from ImmoScout24 (`notebooks/01_eda.ipynb`).
  - Scraped using [ImmoScraper](https://github.com/Dimi20cen/ImmoScraper), 12/2025.
- Cleaned residential set: ~16,399 rows after filtering/cleaning (`data/processed/01_cleaned_data.pkl`).
- Featured modeling set: engineered dataset used for training/inference (`data/processed/02_featured_data.pkl`).
- External enrichment: Swiss municipal/cantonal tax data from `data/external/tax_data_2025.csv`.

### Files in this repository

- `data/processed/01_cleaned_data.pkl` / `.csv`: output of EDA + cleaning.
- `data/processed/02_featured_data.pkl`: canonical training/evaluation dataset.
- `data/processed/rentals_ready_for_modeling.csv`: modeling-ready export.
- `data/external/tax_data_2025.csv`: tax feature source.

### Important preprocessing notes

- Tax feature (`tax_rate`) is merged by city/commune mapping.
- If an exact city match fails, notebooks apply canton-level median fallback.
- Training and app inference both expect raw columns such as `Zip`, `Canton`, and `SubType` before encoding.

## Replicate Locally

### 1) Create environment

```bash
conda env create -f environment.dev.yml
conda activate swiss-rental
```

For container/runtime deployment, `requirements.txt` is provided with runtime dependencies.

### 2) Train artifacts

```bash
python scripts/train.py \
  --data data/processed/02_featured_data.pkl \
  --model-out models/xgb_rent_model.pkl \
  --encoder-out models/zip_encoder.pkl \
  --metrics-out models/training_metrics.json \
  --feature-columns-out models/feature_columns.json \
  --manifest-out models/model_manifest.json \
  --split-strategy random
```

### 3) Evaluate saved artifacts

```bash
python scripts/evaluate.py \
  --data data/processed/02_featured_data.pkl \
  --model models/xgb_rent_model.pkl \
  --encoder models/zip_encoder.pkl \
  --manifest models/model_manifest.json \
  --metrics-out models/evaluation_metrics.json \
  --split-strategy random
```

### 4) (Optional) Batch prediction

```bash
python scripts/predict.py \
  --input-csv path/to/input_features.csv \
  --output-csv predictions.csv \
  --model models/xgb_rent_model.pkl \
  --encoder models/zip_encoder.pkl \
  --manifest models/model_manifest.json \
  --feature-columns models/feature_columns.json
```

## CLI Workflow

- `scripts/train.py`: trains model + encoder and writes training metrics/feature columns.
- `scripts/evaluate.py`: evaluates saved model/encoder on deterministic split (+ segmented metrics).
- `scripts/predict.py`: batch inference from input CSV with `predicted_rent_chf`, `model_version`, and validation warning fields.
- `scripts/healthcheck.py`: validates model artifact integrity from `models/model_manifest.json`.

Validation split strategies:
- `--split-strategy random`: standard random holdout.
- `--split-strategy group_zip`: group-based holdout by ZIP (harder, more realistic generalization check).

## Deployment

Containerized deployment is supported with startup artifact checks.

### One-command VPS deploy

From repo root on your VPS:

```bash
./scripts/deploy_vps.sh
```

This script will:
- pull latest `origin/master` (fast-forward only)
- rebuild the Docker image
- replace the running `rent-predictor` container
- print container status and recent logs

Optional: skip `git pull` if you already synced code:

```bash
SKIP_PULL=1 ./scripts/deploy_vps.sh
```

### Manual Docker run

```bash
docker build -t rent-predictor .
docker run --rm -p 8501:8501 rent-predictor
```

At container startup:
- `scripts/entrypoint.sh` runs `scripts/healthcheck.py`
- healthcheck verifies checksums in `models/model_manifest.json`
- the web app starts only if artifacts are valid

## Quality Checks

Run tests only:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Run full local gate:

```bash
./scripts/gate.sh
```

CI:
- GitHub Actions gate workflow: `.github/workflows/gate.yml`

Data quality:
- `scripts/data_quality_report.py` runs in `./scripts/gate.sh` and writes `models/data_quality_report.json`.
- Gate fails if critical schema checks fail or configured out-of-range ratios are exceeded.

## Productionization Decisions

- Script-first pipeline over notebook-only execution:
  - Decision: `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py` are canonical.
  - Why: reproducible runs, easier CI integration, lower handoff friction.

- Contract-first input validation:
  - Decision: strict shared schema checks in `src/ml_pipeline.py`.
  - Why: fail fast on bad inputs and avoid silent inference/training corruption.

- Explicit artifact integrity and versioning:
  - Decision: `models/model_manifest.json` with checksum verification at startup/inference.
  - Why: prevents stale/tampered artifact mixes and improves deploy safety.

- Feature alignment as a first-class artifact:
  - Decision: persist `models/feature_columns.json` and always align inference features.
  - Why: removes train/serve mismatch risk from one-hot columns.

- Two evaluation split modes:
  - Decision: support `random` and `group_zip` split strategies.
  - Why: report both optimistic baseline and harder generalization-to-unseen-ZIP behavior.

- Deployment startup guardrails:
  - Decision: healthcheck before app start (`scripts/entrypoint.sh` + `scripts/healthcheck.py`).
  - Why: fail early if deployment artifacts are inconsistent.

## Model Report

- Detailed before/after metrics and tradeoff case study:
  - `docs/model_report.md`

## Notebook Flow

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_features_and_baseline.ipynb`
3. `notebooks/03_ml_and_interpretability.ipynb`

## Project Structure

```text
rent-predictor/
  app.py
  scripts/
    train.py
    evaluate.py
    predict.py
  src/
    ml_pipeline.py
    mappings.py
  data/
    external/
    processed/
  models/
  notebooks/
  docs/
```

## Notes

- Batch prediction input must include raw preprocessing columns, including `Zip`, `Canton`, and `SubType`.
- `models/feature_columns.json` is required for robust feature alignment in app/CLI inference.
- `models/model_manifest.json` is used for artifact integrity validation and model version metadata.
