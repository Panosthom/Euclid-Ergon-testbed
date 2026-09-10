#!/usr/bin/env bash
# Run all 4 CBM experiments sequentially on the Pi testbed.
# Each run acquires/releases the lock on its own.
#
# Usage:
#   bash scripts/run_all_experiments.sh                              # no network emulation
#   bash scripts/run_all_experiments.sh --network-profile vsat      # VSAT constraints
#   bash scripts/run_all_experiments.sh --network-profile satellite # satellite constraints
#   bash scripts/run_all_experiments.sh --dry-run                   # deploy only, no FL start
#   bash scripts/run_all_experiments.sh --network-profile vsat --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENTS=(
    "studies.cbm_iid_fedavg:make_spec"
    "studies.cbm_iid_fedprox:make_spec"
    "studies.cbm_noniid_fedavg:make_spec"
    "studies.cbm_noniid_fedprox:make_spec"
)

NET_PROFILE=""
DRY_RUN=""

for arg in "$@"; do
    case "$arg" in
        --network-profile) ;;
        --dry-run) DRY_RUN="--dry-run" ;;
        vsat|satellite|clear)
            # value following --network-profile
            [[ -z "$NET_PROFILE" ]] && NET_PROFILE="$arg" ;;
        *) ;;
    esac
done

# parse --network-profile <value> properly
args=("$@")
for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "--network-profile" ]]; then
        NET_PROFILE="${args[$((i+1))]:-}"
    fi
done

SLEEP_BETWEEN=30  # seconds between runs, lets nodes settle

# ── Apply network profile ──────────────────────────────────────────────────────
apply_profile() {
    local profile="$1"
    if [[ -n "$profile" && "$profile" != "clear" ]]; then
        echo "[all] setting network profile: $profile"
        bash "$SCRIPT_DIR/set_network_profile.sh" "$profile"
    fi
}

# ── Cleanup trap: always clear tc rules when done ─────────────────────────────
cleanup_network() {
    if [[ -n "$NET_PROFILE" && "$NET_PROFILE" != "clear" ]]; then
        echo ""
        echo "[all] clearing network emulation from all nodes ..."
        bash "$SCRIPT_DIR/set_network_profile.sh" clear || true
    fi
}
trap cleanup_network EXIT

# ── Main ──────────────────────────────────────────────────────────────────────
if [[ -n "$NET_PROFILE" ]]; then
    echo "[all] network profile: $NET_PROFILE"
else
    echo "[all] network profile: none (gigabit LAN baseline)"
fi
echo ""

apply_profile "$NET_PROFILE"

total=${#EXPERIMENTS[@]}
failed=()

for i in "${!EXPERIMENTS[@]}"; do
    spec="${EXPERIMENTS[$i]}"
    n=$((i + 1))
    echo ""
    echo "========================================================"
    echo "[$n/$total] $spec"
    [[ -n "$NET_PROFILE" ]] && echo "          network: $NET_PROFILE"
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
