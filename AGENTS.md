# AGENTS.md

Agent-oriented quick guide for `RentPredictor`.

Read `~/Projects/meta/agent-scripts/AGENTS.md` first.

## Start Here

1. Read `README.md` for project context and current workflow.
2. Run `./scripts/gate.sh` before and after non-trivial changes.
3. Prefer script-based workflow over notebook-only changes:
   - `scripts/train.py`
   - `scripts/evaluate.py`
   - `scripts/predict.py`

## Repository Map

- `app.py`: Streamlit app entry point.
- `src/ml_pipeline.py`: shared preprocessing/training/inference utilities.
- `scripts/`: reproducible CLI entry points.
- `tests/test_ml_pipeline.py`: pipeline smoke/unit tests.
- `models/`: runtime artifacts (`xgb_rent_model.pkl`, `zip_encoder.pkl`, `feature_columns.json`).
- `data/processed/02_featured_data.pkl`: canonical training/evaluation dataset.
- `docs/changes.md`: change log; update for non-trivial changes.

## Contracts and Assumptions

- App and CLI inference require `models/feature_columns.json` for robust feature alignment.
- Prediction input must include raw columns needed by preprocessing, including:
  - `Zip`
  - `Canton`
  - `SubType`
- Training target is log-transformed internally and converted back to CHF at prediction time.

## Validation Commands

- Full gate:
  - `./scripts/gate.sh`
- Tests only:
  - `python -m unittest discover -s tests -p "test_*.py" -v`
- App local:
  - `streamlit run app.py`

## Related Local Resources

- Agent scripts workspace:
  - `~/Projects/agent-scripts`
  - `/home/dim/Projects/agent-scripts`

Use these resources when you need reusable local automation/scripts.
