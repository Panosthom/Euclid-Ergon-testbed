#!/usr/bin/env bash
# Run all 4 CBM experiments sequentially on the Pi testbed.
# Each run acquires/releases the lock on its own.
#
# Usage:
#   bash scripts/run_all_experiments.sh              # full run
#   bash scripts/run_all_experiments.sh --dry-run    # deploy only, no FL start
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN="${1:-}"
SLEEP_BETWEEN=30  # seconds between runs, lets nodes settle

EXPERIMENTS=(
    "studies.cbm_iid_fedavg:make_spec"
    "studies.cbm_iid_fedprox:make_spec"
    "studies.cbm_noniid_fedavg:make_spec"
    "studies.cbm_noniid_fedprox:make_spec"
)

total=${#EXPERIMENTS[@]}
failed=()

for i in "${!EXPERIMENTS[@]}"; do
    spec="${EXPERIMENTS[$i]}"
    n=$((i + 1))
    echo ""
    echo "========================================================"
    echo "[$n/$total] $spec"
    echo "========================================================"

    if bash "$SCRIPT_DIR/run_on_testbed.sh" "$spec" $DRY_RUN; then
        echo "[all] [$n/$total] done: $spec"
    else
        echo "[all] [$n/$total] FAILED: $spec" >&2
        failed+=("$spec")
    fi

    if [[ $n -lt $total ]]; then
        echo "[all] sleeping ${SLEEP_BETWEEN}s before next run..."
        sleep "$SLEEP_BETWEEN"
    fi
done

echo ""
echo "========================================================"
echo "All $total experiments attempted."
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All succeeded."
else
    echo "Failed (${#failed[@]}):"
    for f in "${failed[@]}"; do echo "  - $f"; done
    exit 1
fi
echo ""
echo "Latest Pi run logs:"
ls -td logs/pi_run_* 2>/dev/null | head -4 | while read -r d; do echo "  $d"; done
