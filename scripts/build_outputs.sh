#!/usr/bin/env bash
set -euo pipefail

echo "[build] generate HSI scores"
python src/hsi_calculator.py

echo "[build] generate team templates"
python src/team_profiler.py

echo "[build] done -> output/hsi_scores_2024.csv, output/team_templates.json"


