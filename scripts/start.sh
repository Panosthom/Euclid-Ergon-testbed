#!/usr/bin/env bash
# =============================================================================
# start.sh -- the student "Start" button (LOCAL SIMULATION on the server).
#
# This is the educational default: no physical Pis, no lock, many students can
# run at once. It (1) shards your current data per your spec, then (2) runs the
# federated experiment as local processes on the server.
#
#   bash scripts/start.sh                                   # uses the example spec
#   bash scripts/start.sh studies.my_experiment:make_spec   # your spec
#
# When you are ready for a real run on the 4 client Pis, use:
#   bash scripts/run_on_testbed.sh <your-spec>
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Set up imports + venv from testbed/testbed.env so `bash scripts/start.sh` works
# on its own -- no need to `source activate.sh` first.
# shellcheck disable=SC1090
source "$REPO_ROOT/activate.sh"

SPEC_SOURCE="${1:-studies.example_experiment:make_spec}"

echo "[start] sharding data from spec ..."
python -m testbed.prepare --experiment-source "$SPEC_SOURCE"

echo "[start] running local simulation ..."
python -m testbed.run_sim --experiment-source "$SPEC_SOURCE"

echo "[start] done. See logs/<name>_<timestamp>/ for manifest.json + metrics."
