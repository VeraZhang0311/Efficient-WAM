#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT=""
OUTPUT=""
NUM_WORKERS="${NUM_WORKERS:-8}"
SPLITS=("clean" "randomized")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            ROOT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --splits)
            shift
            SPLITS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SPLITS+=("$1")
                shift
            done
            ;;
        -h|--help)
            cat <<'USAGE'
Usage:
  bash data/robotwin2/robotwin_data_convert/compute_qpos_stats.sh \
    --root /path/to/converted_robotwin \
    --output /path/to/converted_robotwin/stats/qpos_mean_std.json \
    --num-workers 16 \
    --splits clean randomized
USAGE
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$ROOT" ]]; then
    echo "Error: --root is required" >&2
    exit 1
fi

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="${ROOT%/}/stats/qpos_mean_std.json"
fi

if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: No Python interpreter found" >&2
    exit 1
fi

"$PYTHON_CMD" "${SCRIPT_DIR}/compute_qpos_stats.py" \
    --root "$ROOT" \
    --output "$OUTPUT" \
    --num-workers "$NUM_WORKERS" \
    --splits "${SPLITS[@]}"
