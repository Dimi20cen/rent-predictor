# rent-predictor Roadmap

## Objective
Define a decision-complete, phase-based roadmap to evolve rent-predictor from portfolio-grade to production-ready ML deployment.

## Current State
- Reproducible CLI scripts exist: `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`.
- Lightweight web app exists: `app.py` serves the API and `static/` contains the HTML/CSS/JS frontend.
- Core artifacts exist: `models/xgb_rent_model.pkl`, `models/zip_encoder.pkl`, `models/feature_columns.json`.
- Quality gate exists locally and in CI: `./scripts/gate.sh`, `.github/workflows/gate.yml`.
- Main maturity gaps: artifact governance, strict contracts, deployment hardening, observability.

## Target State
- Deterministic training/evaluation with tracked run metadata.
- Strict schema validation for training and inference inputs.
- Versioned artifact package with manifest and integrity checks.
- Shared inference path between CLI and app.
- Expanded automated tests for contracts and regressions.
- Deployment path with health checks and basic telemetry.

## Guiding Principles
- Preserve existing user-facing interfaces where possible.
- Favor explicit contracts over implicit assumptions.
- Keep rollout incremental, reversible, and test-gated.
- Maintain notebook + script parity where meaningful, with scripts as canonical runtime path.

## Phase Plan

### Phase A: Contracts First
Scope:
- Define required train/predict schema contracts (columns, types, valid ranges).
- Enforce contract checks in `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`, and `app.py`.
- Introduce standardized validation errors with actionable messages.

Implementation:
- Add validation helpers in `src/ml_pipeline.py` (or a dedicated `src/contracts.py`).
- Validate early at each entry point before feature transforms.
- Add strict checks for required columns like `Zip`, `Canton`, `SubType`.

Exit Criteria:
- Invalid inputs fail fast with clear error messages.
- Contract tests cover success/failure cases.

### Phase B: Reproducibility and Evaluation Integrity
Scope:
- Persist training run metadata and reproducibility context.
- Expand evaluation output beyond single global metrics.

Implementation:
- Persist run metadata: seed, split config, feature checksum, training data fingerprint.
- Add segmented evaluation slices (by canton, price bands, subtype).
- Keep global metrics (MAE/RMSE/R2) plus segment table output.

Exit Criteria:
- Evaluation artifact includes global + segment metrics.
- Re-running with same seed/config yields equivalent outputs.

### Phase C: Inference Hardening
Scope:
- Unify preprocessing and inference preparation paths across CLI and app.
- Reduce drift risk from duplicated feature logic.

Implementation:
- Move shared inference prep into reusable functions in `src/ml_pipeline.py`.
- Refactor `app.py` and `scripts/predict.py` to call shared helpers.
- Add sanitizer/default policy for optional fields with explicit warnings.

Exit Criteria:
- App and CLI use the same core inference prep path.
- Integration tests confirm consistent output shape and metadata.

### Phase D: Deployment and Operations
Scope:
- Add a reproducible deploy artifact and operational readiness checks.

Implementation:
- Add containerized deployment path (`Dockerfile` and runtime entrypoint).
- Add startup checks for required model artifacts.
- Add structured prediction logs and basic runtime metrics hooks.

Exit Criteria:
- Deployment artifact builds reproducibly.
- Service starts only when required artifacts are valid.

### Phase E: Governance and Docs
Scope:
- Align project docs and operational runbooks with deployed behavior.

Implementation:
- Update `README.md` operational sections.
- Add `docs/runbook.md` for common failure modes and recovery steps.
- Keep `docs/changes.md` updated for all non-trivial changes.

Exit Criteria:
- A new collaborator can train, evaluate, predict, and troubleshoot via docs alone.

## Public Interfaces and Contract Additions

### New Artifact
- `models/model_manifest.json`

### Manifest Fields
- `model_file`
- `model_sha256`
- `encoder_file`
- `encoder_sha256`
- `feature_columns_file`
- `feature_columns_sha256`
- `training_data_fingerprint`
- `training_config`
- `training_metrics_summary`
- `model_version`

### Inference Response Contract (Internal Standard)
- `predicted_rent_chf`
- `model_version`
- `input_validation_warnings`

## Test Plan and Scenarios

### Unit Tests
- Schema validation pass/fail coverage.
- Feature alignment validation against `feature_columns.json`.
- Manifest generation and integrity check validation.

### Integration Tests
- End-to-end `train -> evaluate -> predict` on fixture data.
- App startup and prediction smoke tests with present/missing artifacts.

### Regression Guardrails
- Prevent catastrophic degradation vs baseline thresholds.
- Segment-level checks to detect localized failure (e.g., canton-specific collapse).

### Acceptance Scenarios
1. Missing `Zip` fails with actionable error.
2. Artifact hash mismatch blocks inference startup.
3. CLI and app produce consistent contract fields.
4. Gate fails when contract/docs checks do not pass.

## Rollout and Rollback
- Roll out phase-by-phase only when exit criteria are met.
- Keep compatibility loader for existing artifact format during transition.
- If contract checks or regression tests fail, revert to previous artifact manifest/version.

## Risks and Mitigations
- Risk: strict validation breaks ad-hoc usage.
  - Mitigation: support warning mode before strict enforcement.
- Risk: logic divergence between app and CLI.
  - Mitigation: single shared preprocessing/inference module.
- Risk: CI/runtime overhead increases.
  - Mitigation: split smoke checks from full checks with deterministic fixture data.

## Definition of Done
Roadmap execution is complete when:
- Contract validation is enforced across train/evaluate/predict/app.
- Artifact manifest + integrity verification are in place.
- Shared inference pipeline is used by all inference entry points.
- Tests cover contracts, integration flow, and regression guardrails.
- Deployment path includes startup checks and basic observability.
- Docs are sufficient for reproducible onboarding and operations.

## Assumptions and Defaults
- No date-based milestones are used; completion is phase/exit-criteria based.
- The lightweight web app remains the supported user interface.
- XGBoost remains the primary model family for this roadmap cycle.
- `./scripts/gate.sh` remains the canonical local validation command.
