#!/usr/bin/env python3
"""Runtime healthcheck for deployed model artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_pipeline import ArtifactValidationError, get_model_version, validate_model_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate deployment artifact integrity.")
    parser.add_argument("--manifest", default="models/model_manifest.json", help="Path to model manifest JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        validate_model_artifacts(args.manifest)
    except ArtifactValidationError as exc:
        raise SystemExit(f"healthcheck-failed: {exc}") from exc

    print(f"healthcheck-ok model_version={get_model_version(args.manifest)}")


if __name__ == "__main__":
    main()
