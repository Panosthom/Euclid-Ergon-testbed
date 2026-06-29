#!/usr/bin/env bash
# =============================================================================
# run_on_testbed.sh -- one command to run YOUR experiment on the 5 Pis.
#
# Wraps the whole run in a single exclusive `flock`, so the 4 physical clients
# are reserved for you and the lock auto-releases when the run exits (even on
# crash / Ctrl-C). Steps, in order:
#   1. acquire the shared testbed lock (fail fast if someone else holds it)
#   2. reap stale server/client processes on all nodes (frees 0.0.0.0:8000)
#   3. deploy your plugins/studies/data_processes + shards to all 5 Pis
#   4. launch the experiment via the platform orchestrator (controller.start_all),
#      derived from your ExperimentSpec, and block until all rounds finish
#   5. collect logs/model into this workspace, then (on exit) remove the deployed
#      student code from every node so the shared platform tree is left pristine
#
# Usage:
#   bash scripts/run_on_testbed.sh [studies.example_experiment:make_spec] [--dry-run]
#
# Prereqs: testbed/testbed.env configured; shards prepared (utils.prepare_data);
# platform + this template on PYTHONPATH; SSH key auth to all Pis.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# activate.sh sources testbed/testbed.env (giving TESTBED_LOCK, DATASET, ...) AND
# sets PYTHONPATH + venv, so the python -m calls below resolve the platform.
# shellcheck disable=SC1090
source "$REPO_ROOT/activate.sh"

SPEC_SOURCE="${1:-studies.example_experiment:make_spec}"
[[ "${1:-}" == --* ]] && SPEC_SOURCE="studies.example_experiment:make_spec"
DRY=""
for a in "$@"; do [[ "$a" == "--dry-run" ]] && DRY="--dry-run"; done

# Single source of truth: derive the dataset name from the spec (not testbed.env),
# so prepare/deploy/launch all agree. deploy_student.sh honours RUN_DATASET.
RUN_DATASET="$(python - "$SPEC_SOURCE" <<'PY'
import sys, register  # noqa: F401
from registry.manager import load_callable
print(load_callable(sys.argv[1])().dataset.name)
PY
)"
export RUN_DATASET
echo "[spec] dataset=$RUN_DATASET"

# Kill any leftover server.server / client.client on every node so a new run gets
# a clean 0.0.0.0:8000 (a Ctrl-C'd run leaves the remote nohup processes alive).
# The lock guarantees exclusive use of the 4 clients, so this only reaps our own
# stale processes. Skipped on --dry-run (no side effects on running state).
cleanup_stale_processes() {
    echo "[cleanup] stopping stale server/client processes on all nodes ..."
    while IFS=$'\t' read -r nip nuser nkey; do
        [[ -n "$nip" ]] || continue
        nkey="${nkey:-$SSH_KEY}"
        ssh -n -i "${nkey/#\~/$HOME}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$nuser@$nip" \
            "pkill -f server.server; pkill -f client.client; true" >/dev/null 2>&1 || true
    done < <(python3 - "$CONTROLLER_CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
def addr(n): return (n.get("ssh_ip") or n.get("ip") or "").strip()
for n in [cfg.get("server", {})] + cfg.get("clients", []):
    print("\t".join([addr(n), str(n.get("user", "")).strip(), str(n.get("key", "")).strip()]))
PY
)
}

# Leave the shared platform tree pristine after a run: remove the deployed student
# code from every node. Otherwise it lingers in the SERVER's tree and -- because the
# sim client runs with cwd=<platform root> -- shadows each student's live workspace
# plugins (a multi-tenant hazard: one student's Pi run breaks others' sim). Runs
# from the EXIT trap so it fires on success, error, or Ctrl-C. The `&& rm` only runs
# if `cd` into the repo succeeds, and only ever removes student artifacts.
cleanup_deployed_code() {
    echo "[cleanup] removing deployed student code from the platform tree on all nodes ..."
    local repo
    repo="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8')).get('remote_repo','~/federated-edge-testbed'))" "$CONTROLLER_CONFIG")"
    while IFS=$'\t' read -r nip nuser nkey; do
        [[ -n "$nip" ]] || continue
        nkey="${nkey:-$SSH_KEY}"
        ssh -n -i "${nkey/#\~/$HOME}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$nuser@$nip" \
            "cd $repo 2>/dev/null && rm -rf plugins studies data_processes register.py; true" >/dev/null 2>&1 || true
    done < <(python3 - "$CONTROLLER_CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
def addr(n): return (n.get("ssh_ip") or n.get("ip") or "").strip()
for n in [cfg.get("server", {})] + cfg.get("clients", []):
    print("\t".join([addr(n), str(n.get("user", "")).strip(), str(n.get("key", "")).strip()]))
PY
)
}

command -v flock >/dev/null 2>&1 || { echo "[ERROR] flock not available"; exit 1; }
touch "$TESTBED_LOCK"

# Acquire non-blocking; tell the user who holds it on failure.
exec 9>"$TESTBED_LOCK"
if ! flock -n 9; then
    echo "[busy] The 4 client Pis are already running another experiment — try again when free."
    [[ -f "${TESTBED_LOCK}.holder" ]] && echo "       in use by: $(cat "${TESTBED_LOCK}.holder")"
    echo "       check anytime: bash scripts/testbed_lock.sh status"
    echo "       (simulation always works now: bash scripts/start.sh <spec>)"
    exit 1
fi
echo "$(whoami)@$(hostname) since $(date -Is) pid=$$ spec=$SPEC_SOURCE" > "${TESTBED_LOCK}.holder"
trap 'rm -f "${TESTBED_LOCK}.holder"; cleanup_deployed_code' EXIT
echo "[lock] acquired -> ${TESTBED_LOCK}"

[[ -z "$DRY" ]] && cleanup_stale_processes

echo "[1/3] sharding data from spec ..."
python -m testbed.prepare --experiment-source "$SPEC_SOURCE"

echo "[2/3] deploying student code + shards to all nodes ..."
bash "$SCRIPT_DIR/deploy_student.sh"

echo "[3/3] launching experiment (blocking until complete) ..."
python -m testbed.launch --experiment-source "$SPEC_SOURCE" --wait $DRY

# Pull the run's logs/model from every node into THIS workspace (real runs only).
[[ -z "$DRY" ]] && bash "$SCRIPT_DIR/collect_results.sh"

echo "[done] run finished; lock released on exit."

