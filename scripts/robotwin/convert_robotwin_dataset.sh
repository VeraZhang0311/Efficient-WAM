#!/usr/bin/env bash
# Auto conversion script for RoboTwin dataset

echo "Starting RoboTwin dataset conversion at $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONVERTER_DIR="${REPO_ROOT}/data/robotwin2/robotwin_data_convert"

# ============================================================================
# Load Configuration
# ============================================================================
CONFIG_FILE="${CONFIG_FILE:-"${CONVERTER_DIR}/config.yml"}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'USAGE'
Usage:
  bash scripts/robotwin/convert_robotwin_dataset.sh --config data/robotwin2/robotwin_data_convert/config.yml

Environment:
  CONFIG_FILE can also be used to provide the conversion config path.
USAGE
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please pass --config or set CONFIG_FILE."
    exit 1
fi

CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"

echo "Loading configuration from: $CONFIG_FILE"

format_bytes() {
    local bytes="$1"
    if command -v numfmt >/dev/null 2>&1; then
        numfmt --to=iec "$bytes"
    else
        awk -v b="$bytes" 'BEGIN {
            split("B KiB MiB GiB TiB", unit, " ");
            i = 1;
            while (b >= 1024 && i < 5) { b /= 1024; i++ }
            printf "%.1f%s", b, unit[i]
        }'
    fi
}

# Parse YAML configuration (improved - remove comments and extra whitespace)
SOURCE_ROOT=$(grep "^source_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
TARGET_ROOT=$(grep "^target_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
MAX_WORKERS=$(grep "^max_workers:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
LOG_LEVEL=$(grep "^log_level:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
ENABLE_T5=$(grep "^enable_t5_embeddings:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
ENABLE_QPOS_STATS=$(grep "^enable_qpos_stats:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
QPOS_STATS_OUTPUT=$(grep "^qpos_stats_output:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
QPOS_STATS_WORKERS=$(grep "^qpos_stats_num_workers:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Default values if not in config
MAX_WORKERS=${MAX_WORKERS:-"4"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}
ENABLE_T5=${ENABLE_T5:-"false"}
ENABLE_QPOS_STATS=${ENABLE_QPOS_STATS:-"true"}
QPOS_STATS_OUTPUT=${QPOS_STATS_OUTPUT:-"${TARGET_ROOT}/stats/qpos_mean_std.json"}
QPOS_STATS_WORKERS=${QPOS_STATS_WORKERS:-"$MAX_WORKERS"}

# ============================================================================
# Validation
# ============================================================================
if [ -z "$SOURCE_ROOT" ]; then
    echo "Error: source_root is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$TARGET_ROOT" ]; then
    echo "Error: target_root is not set in $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$SOURCE_ROOT" ]; then
    echo "Error: Source root not found: $SOURCE_ROOT"
    exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_ROOT"

echo "Configuration loaded successfully:"
echo "  Source Root: $SOURCE_ROOT"
echo "  Target Root: $TARGET_ROOT"
echo "  Max Workers: $MAX_WORKERS"
echo "  Log Level: $LOG_LEVEL"
echo "  T5 Embeddings: $ENABLE_T5"
echo "  QPos Stats: $ENABLE_QPOS_STATS"
if [ "$ENABLE_QPOS_STATS" = "true" ]; then
    echo "  QPos Stats Output: $QPOS_STATS_OUTPUT"
    echo "  QPos Stats Workers: $QPOS_STATS_WORKERS"
fi

# ============================================================================
# Environment Setup
# ============================================================================
# Check if required Python packages are available
echo "Checking Python environment..."

# Use the python from current environment (conda or system)
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: No Python interpreter found"
    exit 1
fi

# Check if we're in a conda environment
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
    echo "Python executable: $(which $PYTHON_CMD)"
else
    echo "Using system Python: $(which $PYTHON_CMD)"
fi

# Verify torch is available
if ! $PYTHON_CMD -c "import torch" &> /dev/null; then
    echo "Error: PyTorch not found in current Python environment"
    echo "Please ensure you're in the correct conda environment with PyTorch installed"
    echo "Current environment: ${CONDA_DEFAULT_ENV:-system}"
    echo "Python path: $(which $PYTHON_CMD)"
    exit 1
fi

echo "Python environment check passed"

# ============================================================================
# Pre-conversion Checks
# ============================================================================
echo "Performing pre-conversion checks..."

# Check disk space
SOURCE_SIZE=$(du -sk "$SOURCE_ROOT" 2>/dev/null | awk '{print $1 * 1024}' || echo "0")
TARGET_PARENT=$(dirname "$TARGET_ROOT")
AVAILABLE_SPACE=$(df -P "$TARGET_PARENT" | awk 'NR==2 {printf "%.0f", $4 * 1024}')

if [ "$SOURCE_SIZE" -gt 0 ] && [ "$AVAILABLE_SPACE" -gt 0 ]; then
    # Estimate required space (videos typically 50-80% of HDF5 size)
    REQUIRED_SPACE=$((SOURCE_SIZE * 70 / 100))
    
    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        echo "Warning: May not have enough disk space"
        echo "  Estimated required: $(format_bytes "$REQUIRED_SPACE")"
        echo "  Available: $(format_bytes "$AVAILABLE_SPACE")"
        
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "Disk space check passed"
    fi
fi

# Check write permissions
if [ ! -w "$TARGET_PARENT" ]; then
    echo "Error: No write permission for target directory: $TARGET_PARENT"
    exit 1
fi

echo "Pre-conversion checks completed"

# ============================================================================
# Main Conversion Process
# ============================================================================
echo "Starting dataset conversion..."

# Create log directory
LOG_DIR="${CONVERTER_DIR}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/conversion_${TIMESTAMP}.log"

echo "Logs will be saved to: $LOG_FILE"

# Set up verbose flag
VERBOSE_FLAG=""
if [ "$LOG_LEVEL" = "DEBUG" ]; then
    VERBOSE_FLAG="--verbose"
fi

# Run the conversion
echo "Executing conversion script..."
cd "$CONVERTER_DIR"

$PYTHON_CMD "${CONVERTER_DIR}/robotwin_converter.py" \
    --config "$CONFIG_FILE" \
    $VERBOSE_FLAG \
    2>&1 | tee "$LOG_FILE"

CONVERSION_STATUS=${PIPESTATUS[0]}

# ============================================================================
# Post-conversion Processing
# ============================================================================
if [ $CONVERSION_STATUS -eq 0 ]; then
    echo "Conversion completed successfully!"

    if [ "$ENABLE_QPOS_STATS" = "true" ]; then
        echo ""
        echo "Computing qpos normalization statistics..."
        bash "${CONVERTER_DIR}/compute_qpos_stats.sh" \
            --root "$TARGET_ROOT" \
            --output "$QPOS_STATS_OUTPUT" \
            --num-workers "$QPOS_STATS_WORKERS" \
            --splits clean randomized \
            2>&1 | tee -a "$LOG_FILE"
        QPOS_STATS_STATUS=${PIPESTATUS[0]}
        if [ $QPOS_STATS_STATUS -ne 0 ]; then
            echo "QPos statistics calculation failed with exit code: $QPOS_STATS_STATUS"
            exit $QPOS_STATS_STATUS
        fi
    fi
    
    # ========================================================================
    # Final Report
    # ========================================================================
    echo ""
    echo "=========================================="
    echo "CONVERSION SUMMARY"
    echo "=========================================="
    echo "Start time: $(head -n 1 "$LOG_FILE" | grep -o "[0-9][0-9]:[0-9][0-9]:[0-9][0-9]" || echo "Unknown")"
    echo "End time: $(date +%H:%M:%S)"
    echo "Source: $SOURCE_ROOT"
    echo "Target: $TARGET_ROOT"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Count converted files
    if [ -d "$TARGET_ROOT" ]; then
        VIDEO_COUNT=$(find "$TARGET_ROOT" -name "*.mp4" | wc -l)
        QPOS_COUNT=$(find "$TARGET_ROOT" -name "*.pt" | wc -l)
        META_COUNT=$(find "$TARGET_ROOT" -name "*.txt" | wc -l)
        
        echo "Converted files:"
        echo "  Videos: $VIDEO_COUNT"
        echo "  QPos files: $QPOS_COUNT" 
        echo "  Meta files: $META_COUNT"
        
        if [ "$ENABLE_T5" = "true" ]; then
            T5_COUNT=$(find "$TARGET_ROOT" -path "*/umt5_wan/*.pt" | wc -l)
            echo "  T5 embeddings: $T5_COUNT"
        fi
        
        # Calculate total size
        TARGET_SIZE=$(du -sh "$TARGET_ROOT" 2>/dev/null | cut -f1)
        echo "  Total size: ${TARGET_SIZE:-Unknown}"
    fi
    
    echo ""
    echo "Conversion completed successfully!"
    echo "=========================================="
    
else
    echo ""
    echo "=========================================="
    echo "CONVERSION FAILED"
    echo "=========================================="
    echo "Exit code: $CONVERSION_STATUS"
    echo "Check log file for details: $LOG_FILE"
    echo "=========================================="
    exit $CONVERSION_STATUS
fi

# ============================================================================
# Optional: Cleanup and Optimization
# ============================================================================
if grep -q "cleanup_temp_files.*true" "$CONFIG_FILE" 2>/dev/null; then
    echo "Cleaning up temporary files..."
    find /tmp -name "robotwin_*" -mtime +1 -delete 2>/dev/null || true
fi

echo "Script completed at $(date)"
