#!/usr/bin/env bash
# Apply or clear network emulation profiles on all testbed nodes via tc (traffic control).
# Simulates realistic maritime satellite connectivity constraints.
#
# Requires: sudo tc access on each node for the SSH user (set up once by admin).
#
# Admin one-time setup on each Pi node:
#   echo "<user> ALL=(ALL) NOPASSWD: /sbin/tc" | sudo tee /etc/sudoers.d/tc-nopasswd
#
# Usage:
#   bash scripts/set_network_profile.sh clear       # remove all tc rules (gigabit LAN)
#   bash scripts/set_network_profile.sh vsat        # 512 kbps, 200ms latency
#   bash scripts/set_network_profile.sh satellite   # 128 kbps, 600ms latency
#
# Override network interface (default eth0):
#   NET_IFACE=enp1s0 bash scripts/set_network_profile.sh vsat
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Source testbed.env for CONTROLLER_CONFIG and SSH_KEY defaults
# shellcheck disable=SC1090
source "$REPO_ROOT/activate.sh"

PROFILE="${1:-}"
IFACE="${NET_IFACE:-eth0}"

# ── Profile definitions ────────────────────────────────────────────────────────
case "$PROFILE" in
    clear)
        LABEL="Gigabit LAN baseline (no emulation)"
        TC_REMOTE="tc qdisc del dev $IFACE root 2>/dev/null || true"
        ;;
    vsat)
        # VSAT Ku-band broadband: ~512 kbps symmetric, ~200ms RTT (typical merchant ship)
        LABEL="VSAT broadband: 512 kbps, 200ms ±20ms latency"
        TC_REMOTE="tc qdisc del dev $IFACE root 2>/dev/null || true && \
                   tc qdisc add dev $IFACE root netem delay 200ms 20ms rate 512kbit"
        ;;
    satellite)
        # Low-throughput satellite (Iridium / L-band): ~128 kbps, ~600ms RTT
        LABEL="Low-throughput satellite: 128 kbps, 600ms ±50ms latency"
        TC_REMOTE="tc qdisc del dev $IFACE root 2>/dev/null || true && \
                   tc qdisc add dev $IFACE root netem delay 600ms 50ms rate 128kbit"
        ;;
    ""|--help|-h)
        echo "Usage: $0 <clear|vsat|satellite>"
        echo ""
        echo "Profiles:"
        echo "  clear      Gigabit LAN baseline — remove all tc rules"
        echo "  vsat       VSAT broadband: 512 kbps, 200ms latency (merchant ship)"
        echo "  satellite  Low-throughput satellite: 128 kbps, 600ms latency (Iridium)"
        echo ""
        echo "Override interface: NET_IFACE=eth1 $0 vsat"
        exit 0
        ;;
    *)
        echo "Unknown profile '$PROFILE'. Choose: clear, vsat, satellite" >&2
        exit 1
        ;;
esac

echo "[net] profile: $PROFILE — $LABEL"
echo "[net] interface: $IFACE"
echo ""

failed=0

while IFS=$'\t' read -r nip nuser nkey; do
    [[ -n "$nip" ]] || continue
    nkey="${nkey:-$SSH_KEY}"
    printf "  %-20s %s@%s ... " "$PROFILE" "$nuser" "$nip"
    if ssh -n \
           -i "${nkey/#\~/$HOME}" \
           -o StrictHostKeyChecking=no \
           -o UserKnownHostsFile=/dev/null \
           -o ConnectTimeout=5 \
           "$nuser@$nip" \
           "sudo bash -c \"$TC_REMOTE\"" 2>/dev/null; then
        echo "ok"
    else
        echo "FAILED"
        failed=$((failed + 1))
    fi
done < <(python3 - "$CONTROLLER_CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
def addr(n): return (n.get("ssh_ip") or n.get("ip") or "").strip()
for n in [cfg.get("server", {})] + cfg.get("clients", []):
    print("\t".join([addr(n), str(n.get("user", "")).strip(), str(n.get("key", "")).strip()]))
PY
)

echo ""
if [[ $failed -eq 0 ]]; then
    echo "[net] all nodes updated to profile '$PROFILE'."
else
    echo "[net] WARNING: $failed node(s) failed to update." >&2
    exit 1
fi
