#!/usr/bin/env bash
set -euo pipefail

echo "[clean] remove python caches"
rm -rf __pycache__ src/__pycache__ || true
find . -name "*.pyc" -delete || true

echo "[clean] remove generated reports"
rm -rf output/reports/* || true

echo "[clean] done"


