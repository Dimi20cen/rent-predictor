#!/usr/bin/env python3
"""Train XGBoost rent model from featured dataset."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_pipeline import (
    build_model_manifest,
    ContractValidationError,
    dataframe_fingerprint,
    encode_train_test,
    load_featured_data,
    regression_metrics,
    save_json,
    sha256_of_text,
    split_train_test,
    split_features_target,
    train_xgb_log_target,
    predict_rent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rent prediction model.")
    parser.add_argument("--data", default="data/processed/02_featured_data.pkl", help="Path to featured dataset pickle.")
    parser.add_argument("--model-out", default="models/xgb_rent_model.pkl", help="Output path for trained model pickle.")
    parser.add_argument("--encoder-out", default="models/zip_encoder.pkl", help="Output path for ZIP encoder pickle.")
    parser.add_argument("--metrics-out", default="models/training_metrics.json", help="Output path for training metrics JSON.")
    parser.add_argument(
        "--feature-columns-out",
        default="models/feature_columns.json",
        help="Output path for model feature column names JSON.",
    )
    parser.add_argument(
        "--manifest-out",
        default="models/model_manifest.json",
        help="Output path for model manifest JSON.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--split-strategy",
        choices=["random", "group_zip"],
        default="random",
        help="How to split train/test; group_zip avoids ZIP overlap leakage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        df = load_featured_data(args.data)
        x, y = split_features_target(df)
    except ContractValidationError as exc:
        raise SystemExit(f"Input contract error: {exc}") from exc

    x_train_raw, x_test_raw, y_train, y_test = split_train_test(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        strategy=args.split_strategy,
    )

    x_train, x_test, encoder = encode_train_test(x_train_raw, x_test_raw, y_train)
    model = train_xgb_log_target(x_train, y_train, x_test, y_test)
    preds = predict_rent(model, x_test)
    metrics = regression_metrics(y_test, preds)

    model_out = Path(args.model_out)
    encoder_out = Path(args.encoder_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    encoder_out.parent.mkdir(parents=True, exist_ok=True)

    with model_out.open("wb") as f:
        pickle.dump(model, f)
    with encoder_out.open("wb") as f:
        pickle.dump(encoder, f)
    feature_columns = list(x_train.columns)
    feature_columns_json = json.dumps(feature_columns, indent=2)
    Path(args.feature_columns_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.feature_columns_out).write_text(feature_columns_json, encoding="utf-8")

    training_payload = {
        "data_path": args.data,
        "data_fingerprint": dataframe_fingerprint(df),
        "random_state": args.random_state,
        "test_size": args.test_size,
        "split_strategy": args.split_strategy,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": int(len(x_train.columns)),
        "feature_columns_sha256": sha256_of_text(feature_columns_json),
        "metrics": metrics,
    }
    save_json(args.metrics_out, training_payload)
    manifest_payload = build_model_manifest(
        model_path=args.model_out,
        encoder_path=args.encoder_out,
        feature_columns_path=args.feature_columns_out,
        metrics_path=args.metrics_out,
        training_payload=training_payload,
    )
    save_json(args.manifest_out, manifest_payload)

    print("Training complete")
    print(f"MAE:  {metrics['mae']:.2f} CHF")
    print(f"RMSE: {metrics['rmse']:.2f} CHF")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"Saved model to:   {args.model_out}")
    print(f"Saved encoder to: {args.encoder_out}")
    print(f"Saved columns to: {args.feature_columns_out}")
    print(f"Saved metrics to: {args.metrics_out}")
    print(f"Saved manifest to: {args.manifest_out}")


if __name__ == "__main__":
    main()
