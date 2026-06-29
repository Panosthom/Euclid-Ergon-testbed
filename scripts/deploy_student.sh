#!/usr/bin/env bash
# =============================================================================
# deploy_student.sh -- push THIS student's custom code + data shards onto all
# five Pis, into the shared platform tree, before a testbed run.
#
# Why this exists: the platform's deploy_to_pi.sh syncs the *platform* repo only.
# Your plugins/ and studies/ are resolved INSIDE the remote server/client
# processes (e.g. --model-source plugins.models...:build_model), so they must be
# present on every node or the run fails. The exclusive-access lock
# (scripts/testbed_lock.sh) makes writing into the shared tree safe: only one
# student's code sits there during a run.
#
# Usage:
#   bash scripts/deploy_student.sh                 # code + shards to all nodes
#   bash scripts/deploy_student.sh --code-only
#   bash scripts/deploy_student.sh --data-only
# Reads config from testbed/testbed.env (copy from testbed.env.example first).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$REPO_ROOT/testbed/testbed.env"
[[ -f "$ENV_FILE" ]] || { echo "[ERROR] $ENV_FILE not found (copy testbed/testbed.env.example)"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

# Prefer the dataset derived from the spec by run_on_testbed (single source of
# truth); fall back to testbed.env DATASET for standalone use.
DATASET="${RUN_DATASET:-$DATASET}"

MODE="all"
case "${1:-}" in
    --code-only) MODE="code" ;;
    --data-only) MODE="data" ;;
    "" ) ;;
    * ) echo "Usage: $0 [--code-only|--data-only]"; exit 1 ;;
esac

command -v rsync >/dev/null 2>&1 || { echo "[ERROR] rsync required"; exit 1; }
[[ -f "$CONTROLLER_CONFIG" ]] || { echo "[ERROR] CONTROLLER_CONFIG not found: $CONTROLLER_CONFIG"; exit 1; }

# Emit TAB-separated rows: role<TAB>id<TAB>ssh_ip<TAB>user<TAB>key
read_topology() {
    python3 - "$CONTROLLER_CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
def ip(n): return (n.get("ssh_ip") or n.get("ip") or "").strip()
s = cfg.get("server", {})
print("\t".join(["server", "-", ip(s), str(s.get("user","")).strip(), str(s.get("key","")).strip()]))
for c in sorted(cfg.get("clients", []), key=lambda c: int(c.get("id", 0))):
    print("\t".join(["client", str(c.get("id","")), ip(c), str(c.get("user","")).strip(), str(c.get("key","")).strip()]))
PY
}

rsync_to() {  # rsync_to <user> <ip> <key> <dest_dir> <src...>  (no --delete: dest is the platform root)
    local user="$1" ip="$2" key="$3" dest="$4"; shift 4
    # -n: do NOT read stdin -- otherwise ssh swallows the while-read loop's topology rows.
    ssh -n -i "${key/#\~/$HOME}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$user@$ip" "mkdir -p '$dest'"
    rsync -az \
        --exclude='__pycache__' --exclude='*.pyc' \
        -e "ssh -i ${key/#\~/$HOME} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
        "$@" "$user@$ip:$dest/"
}

rsync_dir_clean() {  # rsync_dir_clean <user> <ip> <key> <src_dir> <dest_dir>  (mirrors with --delete)
    local user="$1" ip="$2" key="$3" src="$4" dest="$5"
    # --delete is scoped to <dest_dir> only (a student package dir), never the
    # platform root, so stale files from a previous student are removed safely.
    ssh -n -i "${key/#\~/$HOME}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$user@$ip" "mkdir -p '$dest'"
    rsync -az --delete \
        --exclude='__pycache__' --exclude='*.pyc' \
        -e "ssh -i ${key/#\~/$HOME} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
        "$src/" "$user@$ip:$dest/"
}

push_code() {  # push_code <user> <ip> <key> <label> <repo>
    local user="$1" ip="$2" key="$3" label="$4" repo="$5"
    echo "[$label] code -> $user@$ip:$repo"
    # Package dirs: mirror with --delete (zero leftovers between students).
    for p in plugins studies data_processes; do
        [[ -d "$STUDENT_WORKSPACE/$p" ]] && \
            rsync_dir_clean "$user" "$ip" "$key" "$STUDENT_WORKSPACE/$p" "$repo/$p"
    done
    # register.py: a single file at the platform root, plain copy.
    [[ -f "$STUDENT_WORKSPACE/register.py" ]] && \
        rsync_to "$user" "$ip" "$key" "$repo" "$STUDENT_WORKSPACE/register.py"
}

push_server_data() {  # push_server_data <user> <ip> <key> <repo>
    local user="$1" ip="$2" key="$3" repo="$4"
    local dd="$STUDENT_WORKSPACE/data/$DATASET"
    local files=()
    for f in train.pt test.pt meta.json; do [[ -f "$dd/$f" ]] && files+=("$dd/$f"); done
    [[ ${#files[@]} -gt 0 ]] || { echo "[server] no train/test in $dd (run utils.prepare_data first)"; return; }
    echo "[server] data -> $user@$ip"
    rsync_to "$user" "$ip" "$key" "$repo/data/$DATASET" "${files[@]}"
}

push_client_shard() {  # push_client_shard <user> <ip> <key> <id> <repo>
    local user="$1" ip="$2" key="$3" cid="$4" repo="$5"
    local shard="$STUDENT_WORKSPACE/data/$DATASET/shards/client_${cid}.pt"
    [[ -f "$shard" ]] || { echo "[ERROR] shard missing: $shard (run utils.prepare_data --num-clients ...)"; exit 1; }
    echo "[client$cid] shard -> $user@$ip"
    local extra=("$shard")
    [[ -f "$STUDENT_WORKSPACE/data/$DATASET/shards/manifest.json" ]] && extra+=("$STUDENT_WORKSPACE/data/$DATASET/shards/manifest.json")
    [[ -f "$STUDENT_WORKSPACE/data/$DATASET/meta.json" ]] && rsync_to "$user" "$ip" "$key" "$repo/data/$DATASET" "$STUDENT_WORKSPACE/data/$DATASET/meta.json"
    rsync_to "$user" "$ip" "$key" "$repo/data/$DATASET/shards" "${extra[@]}"
}

# Platform path on the remote nodes: from controller.json "remote_repo" (e.g.
# ~/federated-edge-testbed), expanded against EACH node's $HOME -- client Pis have
# different home dirs than the server, so a single absolute path would be wrong.
REMOTE_REPO="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8')).get('remote_repo','~/federated-edge-testbed'))" "$CONTROLLER_CONFIG")"

while IFS=$'\t' read -r role cid ip user key; do
    [[ -n "$ip" ]] || { echo "[WARN] node '$role $cid' has no ssh ip; skipping"; continue; }
    key="${key:-$SSH_KEY}"
    keyfile="${key/#\~/$HOME}"
    node_home="$(ssh -n -i "$keyfile" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$user@$ip" 'printf %s "$HOME"')"
    node_repo="${REMOTE_REPO/#\~/$node_home}"
    if [[ "$role" == "server" ]]; then
        [[ "$MODE" != "data" ]] && push_code "$user" "$ip" "$key" "server" "$node_repo"
        [[ "$MODE" != "code" ]] && push_server_data "$user" "$ip" "$key" "$node_repo"
    else
        [[ "$MODE" != "data" ]] && push_code "$user" "$ip" "$key" "client$cid" "$node_repo"
        [[ "$MODE" != "code" ]] && push_client_shard "$user" "$ip" "$key" "$cid" "$node_repo"
    fi
done < <(read_topology)

echo "[deploy_student] done (mode=$MODE)"
