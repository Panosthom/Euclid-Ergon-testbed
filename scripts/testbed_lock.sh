#!/usr/bin/env bash
# =============================================================================
# testbed_lock.sh -- advisory exclusive lock for the 4 physical client Pis.
#
# The clients are physical: only one federated run can use them at a time. Every
# student on the server Pi must coordinate through this shared lock.
#
#   bash scripts/testbed_lock.sh status         # who (if anyone) holds it
#   bash scripts/testbed_lock.sh acquire         # block until free, then hold (FD 9)
#   flock -n <file> <cmd>                         # see run_on_testbed.sh
#
# For real runs use scripts/run_on_testbed.sh, which wraps the whole run in a
# single `flock` so the lock auto-releases when the run exits (even on crash).
# This helper is for manual inspection / ad-hoc blocking.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$REPO_ROOT/testbed/testbed.env"
[[ -f "$ENV_FILE" ]] || { echo "[ERROR] $ENV_FILE not found (copy testbed.env.example)"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

command -v flock >/dev/null 2>&1 || { echo "[ERROR] flock not available"; exit 1; }
touch "$TESTBED_LOCK"
META="${TESTBED_LOCK}.holder"

cmd="${1:-status}"
case "$cmd" in
    status)
        if flock -n "$TESTBED_LOCK" true 2>/dev/null; then
            echo "FREE: $TESTBED_LOCK"
        else
            echo "HELD: $TESTBED_LOCK"
            [[ -f "$META" ]] && echo "  $(cat "$META")"
        fi
        ;;
    acquire)
        echo "Waiting for $TESTBED_LOCK ..."
        exec 9>"$TESTBED_LOCK"
        flock 9
        echo "$(whoami)@$(hostname) since $(date -Is) pid=$$" > "$META"
        echo "Acquired. Hold open in this shell; Ctrl-D or exit to release."
        # Keep FD 9 open by reading until EOF.
        cat
        ;;
    *)
        echo "Usage: $0 {status|acquire}"; exit 1 ;;
esac
