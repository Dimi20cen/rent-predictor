# Changes

## 2026-05-16 (Replace Streamlit with lightweight web app)

Summary of change:
- Replaced the Streamlit UI with a plain HTML/CSS/JS frontend served by `app.py`.
- Added JSON endpoints for property options and rent prediction.
- Kept the existing model artifacts, preprocessing, and prediction contract intact.
- Removed Streamlit from runtime dependencies and updated startup commands.

Affected files:
- `app.py`
- `static/index.html`
- `static/styles.css`
- `static/app.js`
- `requirements.txt`
- `environment.dev.yml`
- `scripts/entrypoint.sh`
- `scripts/gate.sh`
- `README.md`
- `docs/roadmap.md`
- `docs/project-review.md`
- `docs/changes.md`

Migration notes:
- No model, artifact, or data contract changes.
- Local app command is now `python app.py`.

Validation status:
- `conda run -n swiss-rental ./scripts/gate.sh` (pass).

## 2026-02-26 (Portfolio finishing docs: model report + case study)

Summary of change:
- Added `docs/model_report.md` with:
  - before/after model comparison table,
  - segment highlights,
  - explicit decision/tradeoff case study,
  - reproducible commands for rerunning comparisons.
- Linked the model report from README navigation/content.

Affected files:
- `docs/model_report.md`
- `README.md`
- `docs/changes.md`

Migration notes:
- Documentation-only update.
- No runtime behavior changed.

Validation status:
- `conda run -n swiss-rental ./scripts/gate.sh` (pass).

## 2026-02-26 (Portfolio polish: data-quality gate + demo UX + productionization rationale)

Summary of change:
- Added a dedicated data-quality reporting/validation script and integrated it into the local/CI gate flow.
- Improved Streamlit demo UX with:
  - sidebar model metadata (including model version),
  - manifest verification status,
  - sample scenario presets,
  - clearer prediction outputs (monthly + annual rent metrics).
- Added a `Productionization Decisions` section in README explaining key architecture/ops tradeoffs.

Affected files:
- `scripts/data_quality_report.py`
- `scripts/gate.sh`
- `app.py`
- `README.md`
- `docs/changes.md`

Migration notes:
- `./scripts/gate.sh` now includes a data-quality step that writes `models/data_quality_report.json`.
- Gate can fail on critical data-quality issues (schema/range checks), improving reliability signal.

Validation status:
- `conda run -n swiss-rental ./scripts/gate.sh` (pass).

## 2026-02-26 (Model quality upgrade: leakage-resistant encoding + derived features + split strategy)

Summary of change:
- Switched high-cardinality ZIP encoding from plain target encoding to leakage-resistant `CatBoostEncoder`.
- Added derived features in shared pipeline:
  - `area_per_room`
  - `log_area_m2`
  - `tax_distance_interaction`
  - `is_top_floor_proxy`
- Added configurable split strategy support across training/evaluation:
  - `random`
  - `group_zip` (group-based holdout by ZIP)
- Updated tests for derived features and ZIP-group split behavior.
- Updated README commands/docs to include split strategy usage.

Affected files:
- `src/ml_pipeline.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `tests/test_ml_pipeline.py`
- `README.md`
- `docs/changes.md`

Migration notes:
- New training runs now persist a `CatBoostEncoder` object in `models/zip_encoder.pkl`.
- `scripts/train.py` and `scripts/evaluate.py` now accept `--split-strategy`.

Validation status:
- `conda run -n swiss-rental python -m unittest discover -s tests -p "test_*.py" -v` (pass, 12 tests).
- `conda run -n swiss-rental ./scripts/gate.sh` (pass).
- Quick benchmark on `random` split after retraining in temp outputs:
  - previous artifact eval: MAE `347.02`, RMSE `748.05`, R2 `0.7856`
  - upgraded pipeline retrain: MAE `334.99`, RMSE `730.67`, R2 `0.7954`

## 2026-02-26 (Phase C+D completion: integrity, deployment, startup checks)

Summary of change:
- Completed Phase C metadata/interface alignment by standardizing model-version metadata in inference outputs.
- Added model artifact manifest support (`models/model_manifest.json`) with checksum generation and validation.
- Added startup integrity checks to Streamlit app and CLI paths using the manifest.
- Added deployment healthcheck and container runtime entrypoint:
  - `scripts/healthcheck.py`
  - `scripts/entrypoint.sh`
  - `Dockerfile`
  - `.dockerignore`
- Updated training flow to emit manifest output by default.
- Updated README with manifest-aware commands and deployment instructions.

Affected files:
- `src/ml_pipeline.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/predict.py`
- `scripts/healthcheck.py`
- `scripts/entrypoint.sh`
- `app.py`
- `models/model_manifest.json`
- `Dockerfile`
- `.dockerignore`
- `README.md`
- `tests/test_ml_pipeline.py`
- `docs/changes.md`

Migration notes:
- `scripts/train.py` now writes `models/model_manifest.json` by default (`--manifest-out`).
- `scripts/evaluate.py` and `scripts/predict.py` now accept `--manifest` and include model-version metadata in outputs.
- App startup now validates artifact integrity against `models/model_manifest.json` and fails fast on mismatch/missing artifacts.

Validation status:
- Executed in `swiss-rental` env:
  - `python -m unittest discover -s tests -p "test_*.py" -v` (pass, 10 tests)
  - `./scripts/gate.sh` (pass)

## 2026-02-26 (Phase C shared inference-prep refactor)

Summary of change:
- Added shared feature-column resolution utility to `src/ml_pipeline.py`.
- Refactored `scripts/predict.py` to use shared feature-name resolution instead of local duplicate logic.
- Refactored `scripts/evaluate.py` to use shared `transform_features_for_model` path for model-aligned inference features.
- Updated Streamlit app resource loading to use shared feature-name resolution helper.
- Added tests for feature-name resolver behavior (file-preferred and booster-fallback paths).

Affected files:
- `src/ml_pipeline.py`
- `scripts/predict.py`
- `scripts/evaluate.py`
- `app.py`
- `tests/test_ml_pipeline.py`
- `docs/changes.md`

Migration notes:
- `scripts/evaluate.py` now accepts `--feature-columns` (default `models/feature_columns.json`) for consistent feature alignment.
- If the feature-columns file is missing, evaluator/predictor can fall back to model booster feature names when available.

Validation status:
- Verified by local compile + tests + gate in `swiss-rental` environment.

## 2026-02-26 (Gate robustness for optional app dependency)

Summary of change:
- Updated `scripts/gate.sh` typecheck import smoke to treat Streamlit app import as optional when `streamlit` is not installed.
- Kept core module imports (`src.ml_pipeline`, `scripts.train`, `scripts.evaluate`, `scripts.predict`) strict.

Affected files:
- `scripts/gate.sh`
- `docs/changes.md`

Migration notes:
- In environments without `streamlit`, gate now skips importing `app.py` during typecheck smoke and continues with core checks.
- App runtime still requires `streamlit`.

Validation status:
- `./scripts/gate.sh`: typecheck now passes core imports and reports app import skip when `streamlit` is missing.
- Gate may still fail later on required dependencies missing for tests (for example `numpy`).

## 2026-02-26 (Phase B metadata + segmented evaluation)

Summary of change:
- Added deterministic dataframe fingerprint helper and feature-columns checksum support.
- Extended training metrics output with dataset fingerprint and feature columns SHA256.
- Extended evaluation output with segmented metrics by canton, subtype, and rent band.
- Added unit coverage for fingerprint stability and segmented metrics utilities.

Affected files:
- `src/ml_pipeline.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `tests/test_ml_pipeline.py`
- `docs/changes.md`

Migration notes:
- `models/training_metrics.json` now includes `data_fingerprint` and `feature_columns_sha256`.
- `scripts/evaluate.py --metrics-out ...` now writes `segment_metrics`.

Validation status:
- Pending in current shell due to missing dependencies (`numpy`, `streamlit`) in active environment.

## 2026-02-26 (Phase A contracts implementation)

Summary of change:
- Implemented strict shared input contracts in `src/ml_pipeline.py` for training and prediction paths.
- Added required-column, non-null, and numeric-type validations with actionable error messages.
- Wired contract enforcement into all script entry points (`train`, `evaluate`, `predict`) with fail-fast CLI errors.
- Updated Streamlit app inference path to use shared transform/predict utilities and display contract errors to users.
- Expanded unit tests to cover contract validation pass/fail scenarios.

Affected files:
- `src/ml_pipeline.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/predict.py`
- `app.py`
- `tests/test_ml_pipeline.py`
- `docs/changes.md`

Migration notes:
- Batch/app inference now expects full raw preprocessing columns and strictly validates input contract.
- Training/evaluation now fail early if featured data violates required schema/typing assumptions.

Validation status:
- `python -m unittest discover -s tests -p "test_*.py" -v`: blocked in active shell (`ModuleNotFoundError: numpy`).
- `./scripts/gate.sh`: blocked in active shell (`ModuleNotFoundError: streamlit` during import smoke).

## 2026-02-26 (Roadmap planning document)

Summary of change:
- Added a decision-complete roadmap document describing the full repository upgrade path from portfolio-grade to production-ready.
- Defined phase-based (non-time-bound) execution with explicit exit criteria, contract/interface additions, testing strategy, rollout/rollback, and risk mitigations.

Affected files:
- `docs/roadmap.md`
- `docs/changes.md`

Migration notes:
- No runtime behavior change in this commit.
- This is a planning/documentation addition to guide subsequent implementation work.

Validation status:
- Documentation-only update.
- Local gate execution currently blocked in active shell environment due to missing runtime dependency (`streamlit`) during import smoke.

## 2026-02-08

Summary of change:
- Added complete project README with setup, run instructions, architecture summary, and known limitations.
- Added a project technical review document.
- Added this change log.

Affected files:
- `README.md`
- `docs/project-review.md`
- `docs/changes.md`

Migration notes:
- No code/runtime behavior changes.
- No data/model migration required.

Validation status:
- Documentation-only update.
- Confirmed file paths and commands against current repository structure.

## 2026-02-08 (Notebook cleanup)

Summary of change:
- Compared `-Copy1` notebooks against canonical versions and confirmed they differ in content (not metadata-only).
- Moved notebook duplicates from `notebooks/` to `notebooks/archive/`.
- Updated README to explicitly define canonical notebook flow.

Affected files:
- `notebooks/archive/02_features_and_baseline-Copy1.ipynb`
- `notebooks/archive/03_ml_and_interpretability-Copy1.ipynb`
- `README.md`
- `docs/changes.md`

Migration notes:
- No app/runtime path changed.
- Canonical workflow remains `01_eda -> 02_features_and_baseline -> 03_ml_and_interpretability`.

Validation status:
- Verified duplicate files are no longer in root `notebooks/`.
- Verified archived files exist under `notebooks/archive/`.

## 2026-02-08 (Archive pruning)

Summary of change:
- Deleted old notebook copy snapshots (`*Copy*.ipynb`) from `notebooks/archive/`.

Affected files:
- `notebooks/archive/01_eda-Copy1.ipynb` (deleted)
- `notebooks/archive/01_eda-Copy2.ipynb` (deleted)
- `notebooks/archive/02_features_and_baseline-Copy1.ipynb` (deleted)
- `notebooks/archive/03_ml_and_interpretability-Copy1.ipynb` (deleted)
- `docs/changes.md`

Migration notes:
- No runtime/app impact.
- Canonical notebooks remain unchanged.

Validation status:
- Confirmed no `*Copy*.ipynb` files remain in `notebooks/archive/`.

## 2026-02-08 (Reproducible CLI pipeline)

Summary of change:
- Added reusable ML pipeline utilities in `src/ml_pipeline.py`.
- Added CLI scripts for `train`, `evaluate`, and `predict` workflows.
- Updated README with reproducible command-line workflow examples.

Affected files:
- `src/ml_pipeline.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/predict.py`
- `README.md`
- `docs/changes.md`

Migration notes:
- No Streamlit app behavior changed.
- Notebook workflow remains available; CLI is an additional reproducible path.

Validation status:
- Syntax validation performed for new Python modules/scripts.
- Runtime execution requires project environment dependencies (`pandas`, `xgboost`, `category_encoders`, etc.).

## 2026-02-08 (CLI robustness fixes)

Summary of change:
- Fixed script import behavior by adding project-root path bootstrap in all CLI scripts.
- Fixed `evaluate.py` for models where XGBoost booster feature names are not serialized.
- Added persisted feature column metadata in training output and wired `predict.py` to use it.

Affected files:
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/predict.py`
- `README.md`
- `docs/changes.md`

Migration notes:
- Training now additionally writes `models/feature_columns.json` by default.
- Batch prediction should pass `--feature-columns models/feature_columns.json` (or keep default path).

Validation status:
- Executed successfully in `swiss-rental` env:
  - `scripts/train.py`
  - `scripts/evaluate.py`
  - `scripts/predict.py`

## 2026-02-08 (README clarity + architecture diagram)

Summary of change:
- Rewrote README to be shorter and task-focused.
- Added a high-level architecture diagram (Mermaid) showing train/evaluate/predict/app flow and artifacts.
- Updated repository naming in README to `RentPredictor`.

Affected files:
- `README.md`
- `docs/changes.md`

Migration notes:
- No runtime changes.
- Documentation-only update.

Validation status:
- Confirmed commands/paths in README match current project layout.

## 2026-02-08 (Mermaid compatibility fix)

Summary of change:
- Updated architecture diagram node label to avoid Mermaid parse issues in GitHub renderer.

Affected files:
- `README.md`
- `docs/changes.md`

Migration notes:
- No runtime changes.
- Documentation-only fix.

Validation status:
- Mermaid block updated with GitHub-safe label format.

## 2026-02-08 (README restructure for results + replication)

Summary of change:
- Removed architecture diagram from README.
- Added dedicated `Results and Demo` section with current headline metrics and local app demo command.
- Reorganized README for easier replication with navigation + step-by-step local workflow.

Affected files:
- `README.md`
- `docs/changes.md`

Migration notes:
- No runtime changes.
- Documentation-only update.

Validation status:
- Verified section links and command paths in README.

## 2026-02-08 (README dataset section)

Summary of change:
- Added a concise `Dataset` section to README with data lineage, repository data files, and preprocessing caveats from notebooks.

Affected files:
- `README.md`
- `docs/changes.md`

Migration notes:
- No runtime changes.
- Documentation-only update.

Validation status:
- Verified referenced dataset files exist in `data/`.

## 2026-02-08 (Streamlit deployment dependencies)

Summary of change:
- Added `requirements.txt` with minimal runtime dependencies for Streamlit Cloud deployment.
- Updated README setup section to note `requirements.txt` is used for Streamlit Cloud runtime installs.

Affected files:
- `requirements.txt`
- `README.md`
- `docs/changes.md`

Migration notes:
- No app logic changes.
- This improves deployment startup reliability/speed compared to full Conda solve.

Validation status:
- Verified `requirements.txt` is present at repository root.

## 2026-02-08 (Force pip install on Streamlit Cloud)

Summary of change:
- Renamed `environment.yml` to `environment.dev.yml` so Streamlit Cloud does not select Conda environment solving.
- Kept local Conda workflow by updating README setup command to the new file name.

Affected files:
- `environment.dev.yml` (renamed from `environment.yml`)
- `README.md`
- `docs/changes.md`

Migration notes:
- Local setup command changed to `conda env create -f environment.dev.yml`.
- Streamlit Cloud should now install from `requirements.txt`.

Validation status:
- Confirmed `environment.yml` is absent and `environment.dev.yml` exists.

## 2026-02-08 (Streamlit prediction fix)

Summary of change:
- Fixed Streamlit runtime prediction crash caused by missing XGBoost booster feature names in pickle.
- Updated app inference path to load canonical feature columns from `models/feature_columns.json`.

Affected files:
- `app.py`
- `docs/changes.md`

Migration notes:
- `models/feature_columns.json` is now a required runtime artifact for app inference.

Validation status:
- Syntax check passed for `app.py` (`python -m compileall app.py`).

## 2026-02-08 (Repository polish: tests, gate, CI, hygiene)

Summary of change:
- Added lightweight unit tests for reusable ML pipeline utilities.
- Added local gate script with lint/typecheck/test/docs checks.
- Added GitHub Actions gate workflow for push/PR checks.
- Removed tracked Python bytecode artifact and updated `.gitignore` for Python cache files.
- Updated README and project review document to reflect current reproducible/quality-check workflow.

Affected files:
- `tests/test_ml_pipeline.py`
- `scripts/gate.sh`
- `.github/workflows/gate.yml`
- `.gitignore`
- `src/__pycache__/mappings.cpython-311.pyc` (deleted)
- `README.md`
- `docs/project-review.md`
- `docs/changes.md`

Migration notes:
- New recommended verification command: `./scripts/gate.sh`.
- CI now executes the same gate on GitHub Actions.

Validation status:
- Ran local gate successfully (`./scripts/gate.sh`).

## 2026-02-08 (README live demo link)

Summary of change:
- Added live project demo link (`https://dimy.dev/projects/rentpredictor`) in README demo section.

Affected files:
- `README.md`
- `docs/changes.md`

Migration notes:
- No runtime changes.
- Documentation-only update.

Validation status:
- Verified README demo section contains local and live demo links.
