#!/usr/bin/env bash
# run-channelflow.sh — Run channelflow programs with correct Python environment
#
# This wrapper sets PYTHONPATH so channelflow executables can find
# the Dedalus Python scripts without requiring .bashrc modifications.
#
# Usage:
#   scripts/run-channelflow.sh findsoln -sys active_matter -T 10 ubest.nc
#   scripts/run-channelflow.sh python dedalus/LLE_example.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Examples:"
    echo "  $0 findsoln -sys active_matter -T 10 ubest.nc"
    echo "  $0 python dedalus/LLE_example.py"
    exit 1
fi

# Activate the right Python environment
if [ -n "${CONDA_PREFIX:-}" ]; then
    : # conda already active
elif [ -d "${PROJECT_DIR}/.venv" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

# Add channelflow binaries, libraries, and dedalus scripts to paths
export PATH="${PROJECT_DIR}/install/bin:${PATH}"
export LD_LIBRARY_PATH="${PROJECT_DIR}/install/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}/dedalus:${PYTHONPATH:-}"

exec "$@"
