#!/usr/bin/env bash
# =============================================================================
# collect_results.sh -- pull a Pi run's outputs from every node into YOUR
# workspace (logs/pi_run_<timestamp>/), so results land where you work instead of
# in the read-only platform tree. run_on_testbed.sh calls this automatically after
# a real run; you can also run it by hand.
#
#   bash scripts/collect_results.sh
#
# Pulls (per node, via the shared deploy key):
#   server node  -> logs/server.log, global_model.pt
#   each client  -> logs/client_<id>.log
# (Extend the pulls below for artifacts/metrics or CodeCarbon emissions.)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
# shellcheck disable=SC1090
source "$REPO_ROOT/activate.sh"

# Honour the spec-derived dataset from run_on_testbed (fallback: testbed.env).
DATASET="${RUN_DATASET:-$DATASET}"

DEST="$STUDENT_WORKSPACE/logs/pi_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEST"

# remote_repo (e.g. ~/federated-edge-testbed); rsync expands the leading ~ on the
# remote side, so no per-node $HOME lookup is needed here.
REMOTE_REPO="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8')).get('remote_repo','~/federated-edge-testbed'))" "$CONTROLLER_CONFIG")"

# Topology as role<TAB>id<TAB>ip<TAB>user<TAB>key, captured up front (no ssh inside
# the read loop, so nothing can swallow it).
TOPO="$(python3 - "$CONTROLLER_CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
def addr(n): return (n.get("ssh_ip") or n.get("ip") or "").strip()
s = cfg.get("server", {})
print("\t".join(["server", "-", addr(s), str(s.get("user","")).strip(), str(s.get("key","")).strip()]))
for c in sorted(cfg.get("clients", []), key=lambda c: int(c.get("id", 0))):
    print("\t".join(["client", str(c.get("id","")), addr(c), str(c.get("user","")).strip(), str(c.get("key","")).strip()]))
PY
)"

pull() {  # pull <user> <ip> <key> <remote_path> <local_name>
    local user="$1" ip="$2" key="$3" remote="$4" name="$5"
    if rsync -az -e "ssh -i ${key/#\~/$HOME} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
        "$user@$ip:$remote" "$DEST/$name" 2>/dev/null; then
        echo "  + $name"
    else
        echo "  - $name (not found)"
    fi
}

# Per-client metric CSVs land here (metrics/<dataset>/client_*.csv) so the run dir
# matches a sim run's layout and scripts/plot.sh works on Pi runs too.
mkdir -p "$DEST/metrics/$DATASET"

echo "[collect] pulling run outputs -> $DEST"
while IFS=$'\t' read -r role cid ip user key; do
    [[ -n "$ip" ]] || continue
    key="${key:-$SSH_KEY}"
    if [[ "$role" == "server" ]]; then
        echo "[server] $user@$ip"
        pull "$user" "$ip" "$key" "$REMOTE_REPO/logs/server.log" "server.log"
        pull "$user" "$ip" "$key" "$REMOTE_REPO/global_model.pt" "global_model.pt"
        # round metrics (server.server default path) -> enables plot.sh
        pull "$user" "$ip" "$key" "$REMOTE_REPO/artifacts/metrics/$DATASET/server_rounds.csv" "server_rounds.csv"
    else
        echo "[client$cid] $user@$ip"
        pull "$user" "$ip" "$key" "$REMOTE_REPO/logs/client_${cid}.log" "client_${cid}.log"
        # per-client metrics (client.client default path) -> train/val/local-test plots
        pull "$user" "$ip" "$key" "$REMOTE_REPO/artifacts/metrics/$DATASET/client_${cid}.csv" "metrics/$DATASET/client_${cid}.csv"
    fi
done <<< "$TOPO"

echo "[collect] done -> $DEST"
ls -1 "$DEST"
