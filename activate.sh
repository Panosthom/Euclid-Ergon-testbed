# Source this (do NOT execute) to prepare your shell for a run:
#
#     source activate.sh && bash scripts/start.sh
#
# It puts the platform on PYTHONPATH and (optionally) activates the shared venv,
# reading the paths from testbed/testbed.env -- so you never export them by hand.
# Copy testbed/testbed.env.example to testbed/testbed.env and set PLATFORM_DIR
# (and VENV_ACTIVATE) first.

# Resolve this file's directory whether sourced from bash or zsh.
if [ -n "${BASH_SOURCE:-}" ]; then
    _src="${BASH_SOURCE[0]}"
else
    _src="$0"
fi
_ws="$(cd "$(dirname "$_src")" && pwd)"

_env="$_ws/testbed/testbed.env"
if [ ! -f "$_env" ]; then
    echo "activate.sh: $_env not found -- copy testbed/testbed.env.example and edit it." >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
. "$_env"

if [ -z "${PLATFORM_DIR:-}" ]; then
    echo "activate.sh: PLATFORM_DIR is empty in $_env -- set it to the shared platform path." >&2
    return 1 2>/dev/null || exit 1
fi

# Platform first, then this workspace (so plugins/studies resolve too).
export PYTHONPATH="${PLATFORM_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="${_ws}:${PYTHONPATH}"

# Activate the shared venv if testbed.env defines how (e.g.
# VENV_ACTIVATE="source /home/pi-server/.venv/fed/bin/activate").
if [ -n "${VENV_ACTIVATE:-}" ]; then
    eval "$VENV_ACTIVATE"
fi

echo "[activate] PLATFORM_DIR=$PLATFORM_DIR"
if python -c "import registry, experiments, extensions" 2>/dev/null; then
    echo "[activate] platform import OK"
else
    echo "[activate] WARNING: platform not importable -- check PLATFORM_DIR / read permissions" >&2
fi
