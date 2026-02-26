#!/usr/bin/env bash
set -euo pipefail

python scripts/healthcheck.py --manifest "${MODEL_MANIFEST_PATH:-models/model_manifest.json}"
exec streamlit run app.py --server.address=0.0.0.0 --server.port="${PORT:-8501}"
