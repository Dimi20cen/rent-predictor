#!/usr/bin/env python3
"""Generate and validate a lightweight data quality report for featured training data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_pipeline import ContractValidationError, save_json, validate_training_dataframe


RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "Rent": (400.0, 20000.0),
    "Area_m2": (10.0, 600.0),
    "Rooms": (0.5, 20.0),
    "Floor": (-2.0, 50.0),
    "Lat": (45.0, 48.5),
    "Lon": (5.0, 11.0),
    "tax_rate": (0.0, 1000.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate featured dataset quality and emit JSON report.")
    parser.add_argument("--data", default="data/processed/02_featured_data.pkl", help="Featured dataset path.")
    parser.add_argument(
        "--output",
        default="models/data_quality_report.json",
        help="Where to write the quality report JSON.",
    )
    parser.add_argument(
        "--max-out-of-range-ratio",
        type=float,
        default=0.02,
        help="Fail if any checked column exceeds this out-of-range ratio.",
    )
    return parser.parse_args()


def _range_stats(df: pd.DataFrame, column: str, lower: float, upper: float) -> dict[str, float | int]:
    series = df[column]
    out_of_range = ((series < lower) | (series > upper)).sum()
    ratio = float(out_of_range) / float(len(df)) if len(df) else 0.0
    return {
        "column": column,
        "min_allowed": lower,
        "max_allowed": upper,
        "out_of_range_count": int(out_of_range),
        "out_of_range_ratio": ratio,
    }


def main() -> None:
    args = parse_args()
    df = pd.read_pickle(args.data)

    try:
        validate_training_dataframe(df, context="data quality check")
    except ContractValidationError as exc:
        raise SystemExit(f"data-quality-failed: {exc}") from exc

    range_results: list[dict[str, float | int]] = []
    failures: list[str] = []
    for column, (lower, upper) in RANGE_CHECKS.items():
        if column not in df.columns:
            continue
        stats = _range_stats(df, column, lower, upper)
        range_results.append(stats)
        if stats["out_of_range_ratio"] > args.max_out_of_range_ratio:
            failures.append(
                f"{column}: ratio={stats['out_of_range_ratio']:.4f} > max={args.max_out_of_range_ratio:.4f}"
            )

    duplicate_rows = int(df.duplicated().sum())
    duplicate_ratio = float(duplicate_rows) / float(len(df)) if len(df) else 0.0
    if duplicate_ratio > 0.05:
        failures.append(f"duplicate_rows_ratio={duplicate_ratio:.4f} > max=0.0500")

    payload = {
        "data_path": args.data,
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "duplicate_rows": duplicate_rows,
        "duplicate_rows_ratio": duplicate_ratio,
        "range_checks": range_results,
        "status": "fail" if failures else "pass",
        "failures": failures,
    }
    save_json(args.output, payload)

    print(f"data-quality status={payload['status']} rows={payload['row_count']} cols={payload['column_count']}")
    print(f"report={args.output}")
    if failures:
        for item in failures:
            print(f"failure: {item}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
