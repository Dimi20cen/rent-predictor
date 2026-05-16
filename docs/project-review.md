# Project Review

Date: 2026-02-08

## Scope Reviewed

- `app.py`
- `environment.dev.yml`
- `requirements.txt`
- `src/ml_pipeline.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/predict.py`
- `tests/test_ml_pipeline.py`
- `.github/workflows/gate.yml`

## Architecture Summary

- Canonical reproducible ML path is script-driven (`train` -> `evaluate` -> `predict`).
- Notebook flow remains available for exploration/analysis.
- Inference is served through the lightweight web app (`app.py` API plus `static/` frontend) with pre-trained artifacts.
- Feature generation includes:
  - geospatial hub distances,
  - tax enrichment (`tax_data_2025.csv` + city/commune mapping),
  - categorical encoding (ZIP target encoding + one-hot category alignment).

## Strengths

- End-to-end prototype is complete: data prep -> model -> interpretable notebook -> app.
- Model artifacts and required processed dataset are already versioned for local reproducibility.
- `app.py` aligns feature columns against model metadata, reducing inference mismatch risk.

## Risks and Gaps

- Pipeline quality checks are lightweight; no strict static typing tool is configured.
- App infers several binary fields via defaults/proxies, which may impact prediction realism.
- Large committed binary/data artifacts may slow collaboration and inflate repository size.
- Data refresh/versioning policy is still informal.

## Recommended Next Steps

1. Add stronger type checks (e.g., mypy) and schema validation for train/predict inputs.
2. Introduce artifact versioning convention (model version + training timestamp).
3. Add dataset/version documentation policy for refresh cadence and source traceability.
4. Add integration test that runs predict CLI on a tiny fixture file.
