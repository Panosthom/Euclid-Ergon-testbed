#!/usr/bin/env bash
# Full 36-run paper experiment sweep:
#   3 seeds × 3 network profiles × 4 experiments = 36 Pi runs
#
# Seeds control data sharding + model init for statistical validity.
# The FL_SEED env variable is read by each study spec automatically.
#
# Estimated runtime: ~90 minutes (each run ~2 min + 30 s cooldown)
# Results land in logs/pi_run_<timestamp>/ with experiment_id encoding
# the seed (e.g. cbm_iid_fedavg_s2026_<timestamp>).
#
# Usage:
#   bash scripts/run_paper_experiments.sh             # full 36 runs
#   bash scripts/run_paper_experiments.sh --dry-run   # deploy only, no FL
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=""
for a in "$@"; do [[ "$a" == "--dry-run" ]] && DRY_RUN="--dry-run"; done

SEEDS=(2025 2026 2027)
declare -a PROFILES=("" "--network-profile vsat" "--network-profile satellite")
declare -a PROFILE_NAMES=("baseline" "vsat" "satellite")

TOTAL_BATCHES=$(( ${#SEEDS[@]} * ${#PROFILES[@]} ))
TOTAL_RUNS=$(( TOTAL_BATCHES * 4 ))
BATCH=0
FAILED_BATCHES=()

START_TS=$(date +%s)

echo "============================================================"
echo " Paper experiment sweep: $TOTAL_RUNS runs"
echo " Seeds:    ${SEEDS[*]}"
echo " Profiles: ${PROFILE_NAMES[*]}"
[[ -n "$DRY_RUN" ]] && echo " Mode:     DRY-RUN"
echo "============================================================"
echo ""

for seed in "${SEEDS[@]}"; do
  export FL_SEED="$seed"

  for pi in "${!PROFILES[@]}"; do
    net_arg="${PROFILES[$pi]}"
    net_name="${PROFILE_NAMES[$pi]}"
    BATCH=$(( BATCH + 1 ))

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Batch $BATCH/$TOTAL_BATCHES — seed=$seed  network=$net_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if bash "$SCRIPT_DIR/run_all_experiments.sh" $net_arg $DRY_RUN; then
      echo "[paper] batch $BATCH OK (seed=$seed network=$net_name)"
    else
      echo "[paper] batch $BATCH FAILED (seed=$seed network=$net_name)" >&2
      FAILED_BATCHES+=("seed=$seed network=$net_name")
    fi

    # Estimate remaining time
    NOW=$(date +%s)
    ELAPSED=$(( NOW - START_TS ))
    if [[ $BATCH -gt 0 && $BATCH -lt $TOTAL_BATCHES ]]; then
      PER_BATCH=$(( ELAPSED / BATCH ))
      REMAINING=$(( PER_BATCH * (TOTAL_BATCHES - BATCH) ))
      echo "[paper] elapsed=${ELAPSED}s  ~${REMAINING}s remaining"
    fi
  done
done

echo ""
echo "============================================================"
ELAPSED=$(( $(date +%s) - START_TS ))
echo " Sweep complete in ${ELAPSED}s ($(( ELAPSED/60 ))m$(( ELAPSED%60 ))s)"
echo " Total batches: $TOTAL_BATCHES — Failed: ${#FAILED_BATCHES[@]}"

if [[ ${#FAILED_BATCHES[@]} -gt 0 ]]; then
  echo " Failed batches:"
  for f in "${FAILED_BATCHES[@]}"; do echo "   - $f"; done
  echo "============================================================"
  exit 1
fi
echo " All $TOTAL_RUNS runs succeeded."
echo "============================================================"
echo ""
echo "Latest Pi run logs:"
ls -td logs/pi_run_* 2>/dev/null | head -6 | while read -r d; do echo "  $d"; done
