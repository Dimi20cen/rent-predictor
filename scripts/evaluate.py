#!/usr/bin/env python3
"""Evaluate saved model/encoder on deterministic holdout split."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_pipeline import (
    ArtifactValidationError,
    ContractValidationError,
    dataframe_fingerprint,
    get_model_version,
    load_featured_data,
    rent_bands,
    regression_metrics,
    resolve_feature_names,
    segmented_regression_metrics,
    save_json,
    split_train_test,
    split_features_target,
    predict_rent,
    transform_features_for_model,
    validate_model_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rent prediction model.")
    parser.add_argument("--data", default="data/processed/02_featured_data.pkl", help="Path to featured dataset pickle.")
    parser.add_argument("--model", default="models/xgb_rent_model.pkl", help="Path to trained model pickle.")
    parser.add_argument("--encoder", default="models/zip_encoder.pkl", help="Path to ZIP encoder pickle.")
    parser.add_argument("--manifest", default="models/model_manifest.json", help="Path to model manifest JSON.")
    parser.add_argument(
        "--feature-columns",
        default="models/feature_columns.json",
        help="Path to JSON list of training feature column names.",
    )
    parser.add_argument("--metrics-out", default="", help="Optional output JSON path for evaluation metrics.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--split-strategy",
        choices=["random", "group_zip"],
        default="random",
        help="How to split train/test; group_zip evaluates unseen ZIP groups.",
    )
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

    try:
        df = load_featured_data(args.data)
        x, y = split_features_target(df)
    except ContractValidationError as exc:
        raise SystemExit(f"Input contract error: {exc}") from exc

    _, x_test_raw, _, y_test = split_train_test(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        strategy=args.split_strategy,
    )

    feature_names = resolve_feature_names(model, args.feature_columns)
    x_test = transform_features_for_model(x_test_raw, encoder, feature_names)

    preds = predict_rent(model, x_test)
    metrics = regression_metrics(y_test, preds)
    segment_metrics = {
        "canton": segmented_regression_metrics(y_test, preds, x_test_raw["Canton"], "canton"),
        "subtype": segmented_regression_metrics(y_test, preds, x_test_raw["SubType"], "subtype"),
        "rent_band": segmented_regression_metrics(y_test, preds, rent_bands(y_test), "rent_band"),
    }

    if args.metrics_out:
        save_json(
            args.metrics_out,
            {
                "data_path": args.data,
                "data_fingerprint": dataframe_fingerprint(df),
                "model_path": args.model,
                "encoder_path": args.encoder,
                "model_version": get_model_version(args.manifest),
                "input_validation_warnings": input_validation_warnings,
                "random_state": args.random_state,
                "test_size": args.test_size,
                "split_strategy": args.split_strategy,
                "test_rows": int(len(x_test)),
                "metrics": metrics,
                "segment_metrics": segment_metrics,
            },
        )

    print("Evaluation complete")
    print(f"MAE:  {metrics['mae']:.2f} CHF")
    print(f"RMSE: {metrics['rmse']:.2f} CHF")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"Model version: {get_model_version(args.manifest)}")
    if input_validation_warnings:
        print(f"Input validation warnings: {', '.join(input_validation_warnings)}")
    print(
        "Segments: "
        f"canton={len(segment_metrics['canton'])}, "
        f"subtype={len(segment_metrics['subtype'])}, "
        f"rent_band={len(segment_metrics['rent_band'])}"
    )
    if args.metrics_out:
        print(f"Saved metrics to: {args.metrics_out}")


if __name__ == "__main__":
    main()
