#!/usr/bin/env bash
set -euo pipefail

echo "[lint] python compile check"
python -m compileall app.py src scripts tests

echo "[typecheck] lightweight import/type smoke"
python - <<'PY'
import importlib
core_modules = [
    "app",
    "src.ml_pipeline",
    "scripts.train",
    "scripts.evaluate",
    "scripts.predict",
]

for mod in core_modules:
    importlib.import_module(mod)

print("imports-ok")
PY

echo "[test] unittest"
python -m unittest discover -s tests -p "test_*.py" -v

echo "[data-quality] featured dataset checks"
python scripts/data_quality_report.py \
  --data data/processed/02_featured_data.pkl \
  --output models/data_quality_report.json

echo "[docs] presence checks"
test -f README.md
test -f docs/changes.md
echo "docs-ok"

echo "[gate] complete"
