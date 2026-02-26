#!/usr/bin/env python3
"""Batch prediction using saved model + encoder."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_pipeline import (
    ArtifactValidationError,
    ContractValidationError,
    get_model_version,
    predict_rent,
    resolve_feature_names,
    transform_features_for_model,
    validate_model_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference for rent prediction.")
    parser.add_argument("--input-csv", required=True, help="Input CSV with raw feature columns.")
    parser.add_argument("--output-csv", default="predictions.csv", help="Output CSV with predictions.")
    parser.add_argument("--model", default="models/xgb_rent_model.pkl", help="Path to trained model pickle.")
    parser.add_argument("--encoder", default="models/zip_encoder.pkl", help="Path to ZIP encoder pickle.")
    parser.add_argument(
        "--feature-columns",
        default="models/feature_columns.json",
        help="Path to JSON list of training feature column names.",
    )
    parser.add_argument("--manifest", default="models/model_manifest.json", help="Path to model manifest JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)
    with open(args.encoder, "rb") as f:
        encoder = pickle.load(f)
    input_validation_warnings: list[str] = []
    try:
        if Path(args.manifest).exists():
            validate_model_artifacts(args.manifest)
        else:
            input_validation_warnings.append(f"manifest_missing:{args.manifest}")
    except ArtifactValidationError as exc:
        raise SystemExit(f"Artifact integrity error: {exc}") from exc

    feature_names = resolve_feature_names(model, args.feature_columns)
    raw = pd.read_csv(args.input_csv)
    try:
        x = transform_features_for_model(raw, encoder, feature_names)
    except ContractValidationError as exc:
        raise SystemExit(f"Input contract error: {exc}") from exc

    pred = predict_rent(model, x)

    out = raw.copy()
    out["predicted_rent_chf"] = pred
    out["model_version"] = get_model_version(args.manifest)
    out["input_validation_warnings"] = ";".join(input_validation_warnings)
    out.to_csv(args.output_csv, index=False)

    print(f"Predictions complete: {len(out)} rows")
    print(f"Model version: {get_model_version(args.manifest)}")
    if input_validation_warnings:
        print(f"Input validation warnings: {', '.join(input_validation_warnings)}")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
