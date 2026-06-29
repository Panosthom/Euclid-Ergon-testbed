#!/usr/bin/env bash
# =============================================================================
# plot.sh -- turn a run's metrics into PNGs (train/val/local-test/global-test).
#
#   bash scripts/plot.sh                  # newest run in logs/
#   bash scripts/plot.sh logs/<run_dir>   # a specific run
#
# Sources activate.sh so the platform's plotting (utils.plot_paper_results) is
# importable, then writes PNGs to logs/<run>/plots/.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
# shellcheck disable=SC1090
source "$REPO_ROOT/activate.sh"

python -m testbed.plot "$@"
