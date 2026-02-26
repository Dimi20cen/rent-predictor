# Model Report

## Summary
This report compares the prior production artifact against the upgraded pipeline (leakage-resistant ZIP encoding + derived features) on the same featured dataset (`data/processed/02_featured_data.pkl`).

## Experiment Setup
- Dataset: `data/processed/02_featured_data.pkl` (`16005` rows, `34` columns)
- Seed: `42`
- Evaluation script: `scripts/evaluate.py`
- Metric focus: MAE (CHF), RMSE, R2

## Results

### 1) Random Holdout (same split style)

| Model | MAE (CHF) | RMSE (CHF) | R2 |
|---|---:|---:|---:|
| Previous production artifact | 347.02 | 748.05 | 0.7856 |
| Upgraded pipeline retrain | 334.99 | 730.67 | 0.7954 |
| Delta | **-12.02** | **-17.38** | **+0.0098** |

Observation:
- The upgraded pipeline improved all global metrics on the standard random holdout.

### 2) Group-by-ZIP Holdout (stress test)

| Model | MAE (CHF) | RMSE (CHF) | R2 |
|---|---:|---:|---:|
| Previous production artifact | 222.13 | 528.81 | 0.8659 |
| Upgraded pipeline retrain | 359.49 | 764.04 | 0.7200 |

Interpretation:
- This comparison is **not apples-to-apples** for production fairness because the previous artifact was trained under random split assumptions while the upgraded model here was trained and evaluated under stricter `group_zip` split.  
- The `group_zip` path should be treated as a harder generalization benchmark (unseen ZIP groups), not a direct replacement score.

## Segment Highlights (Random Split)
From `/tmp/existing_eval_random.json` vs `/tmp/rent_model_upgrade/eval_random.json`:

- `GE` canton MAE: **807.17 -> 717.75** (improved)
- `VD` canton MAE: **417.58 -> 389.90** (improved)
- `FLAT` subtype MAE: **298.91 -> 292.74** (improved)
- `SINGLE_HOUSE` subtype MAE: **1049.31 -> 848.06** (improved)
- `high` rent band MAE: **1443.25 -> 1344.14** (improved)

Known weak segment:
- `ZG` remains difficult due to limited sample size and extreme rent variance.

## Decision + Tradeoff Case Study

### Decision
Replace plain ZIP target encoding with leakage-resistant `CatBoostEncoder` and add compact, domain-grounded derived features.

### Why
- ZIP is high-cardinality (`2004` unique ZIP values).  
- Plain target encoding can overfit local priors and inflate apparent performance.
- Derived features (`area_per_room`, `log_area_m2`, `tax_distance_interaction`, `is_top_floor_proxy`) encode stable pricing structure with low complexity cost.

### Tradeoff
- Slightly more preprocessing complexity and tighter coupling to contract checks.
- In return, we get better random-holdout performance and a cleaner path to robust evaluation modes (`group_zip`).

### Outcome
- Measurable improvement on production-like random holdout.
- Stronger evaluation framework to avoid overclaiming performance from optimistic splits.

## Repro Commands

```bash
# Previous artifact random evaluation
python scripts/evaluate.py \
  --data data/processed/02_featured_data.pkl \
  --model models/xgb_rent_model.pkl \
  --encoder models/zip_encoder.pkl \
  --feature-columns models/feature_columns.json \
  --manifest models/model_manifest.json \
  --split-strategy random

# Upgraded retrain (random)
python scripts/train.py \
  --data data/processed/02_featured_data.pkl \
  --model-out /tmp/rent_model_upgrade/xgb_rent_model.pkl \
  --encoder-out /tmp/rent_model_upgrade/zip_encoder.pkl \
  --feature-columns-out /tmp/rent_model_upgrade/feature_columns.json \
  --metrics-out /tmp/rent_model_upgrade/training_metrics.json \
  --manifest-out /tmp/rent_model_upgrade/model_manifest.json \
  --split-strategy random

# Upgraded evaluation (random)
python scripts/evaluate.py \
  --data data/processed/02_featured_data.pkl \
  --model /tmp/rent_model_upgrade/xgb_rent_model.pkl \
  --encoder /tmp/rent_model_upgrade/zip_encoder.pkl \
  --feature-columns /tmp/rent_model_upgrade/feature_columns.json \
  --manifest /tmp/rent_model_upgrade/model_manifest.json \
  --split-strategy random
```
