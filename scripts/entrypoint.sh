#!/usr/bin/env bash
set -euo pipefail

python scripts/healthcheck.py --manifest "${MODEL_MANIFEST_PATH:-models/model_manifest.json}"
exec python app.py
