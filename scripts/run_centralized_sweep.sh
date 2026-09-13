#!/usr/bin/env bash
# Runs the centralized baseline for all 3 paper seeds.
# Takes ~2-3 minutes total on the Pi server (no network overhead).
#
# Usage (from repo root):
#   source activate.sh && bash scripts/run_centralized_sweep.sh
#
# Results land in logs/centralized_s{SEED}_{timestamp}/server_rounds.csv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate venv / PYTHONPATH if not already done (safe to source twice).
if [ -f "$REPO_ROOT/activate.sh" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/activate.sh"
fi

SEEDS=(2025 2026 2027)

echo "============================================================"
echo " Centralized baseline sweep: ${#SEEDS[@]} seeds"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- seed=$seed ---"
    FL_SEED="$seed" python3 "$SCRIPT_DIR/run_centralized.py"
done

echo ""
echo "============================================================"
echo " Centralized sweep complete."
echo " Results:"
ls -td logs/centralized_s* 2>/dev/null | head -6 | while read -r d; do
    mae=$(tail -1 "$d/server_rounds.csv" 2>/dev/null | cut -d',' -f3)
    echo "  $d  final_MAE=$mae"
done
echo "============================================================"
