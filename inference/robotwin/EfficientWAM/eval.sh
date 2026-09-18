#!/bin/bash
set -euo pipefail

# Unified EfficientWAM evaluation entry for RoboTwin.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$SCRIPT_DIR"
LOG_ROOT="${EFFICIENT_WAM_LOG_ROOT:-$POLICY_DIR}"
CONFIG_FILE="${EFFICIENT_WAM_CONFIG:-${POLICY_DIR}/deploy_policy.yml}"

MODE=""
TASK_NAME=""
GPU_ID=""
GPU_IDS_OVERRIDE=""
TASKS_FILE_OVERRIDE=""
TASK_CONFIG_OVERRIDE=""
SEED_OVERRIDE=""
EPISODE_NUM_OVERRIDE=""
EXTRA_OVERRIDES=()

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_RESET=$'\033[0m'
    C_BOLD=$'\033[1m'
    C_DIM=$'\033[2m'
    C_RED=$'\033[31m'
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'
    C_MAGENTA=$'\033[35m'
    C_CYAN=$'\033[36m'
else
    C_RESET=""
    C_BOLD=""
    C_DIM=""
    C_RED=""
    C_GREEN=""
    C_YELLOW=""
    C_BLUE=""
    C_MAGENTA=""
    C_CYAN=""
fi

paint() {
    local color="$1"
    shift
    printf "%b%s%b\n" "$color" "$*" "$C_RESET"
}

die() {
    paint "$C_RED" "Error: $*"
    exit 1
}

print_rule() {
    paint "$C_BLUE" "================================================================"
}

print_section() {
    echo ""
    print_rule
    paint "${C_BOLD}${C_BLUE}" "$1"
    print_rule
}

print_kv() {
    local label="$1"
    shift
    printf "  %b%-15s%b %b%s%b\n" "$C_DIM" "${label}:" "$C_RESET" "$C_CYAN" "$*" "$C_RESET"
}

usage() {
    cat << EOF
Usage:
  bash eval.sh [TASK_NAME] [GPU_ID]
  bash eval.sh --all [--gpus 0,1] [--tasks tasks_all.txt]

Options:
  --config FILE          Use a deploy_policy.yml-compatible config file.
  --task TASK            Run a single task.
  --gpu GPU              GPU for single-task evaluation.
  --all                  Run every task listed by tasks_file.
  --tasks FILE           Task list for --all. Relative paths are under policy/EfficientWAM.
  --gpus IDS             Comma-separated GPUs for --all. Empty [] in config auto-detects GPUs.
  --task-config NAME     Override task_config.
  --seed SEED            Override seed.
  --episode-num NUM      Override RoboTwin episode_num.
  --test-num NUM         Alias for --episode-num.
  --inference.num_inference_steps NUM
                         Override denoising steps for EfficientWAM inference.
  --inference.num_video_inference_steps NUM
                         Legacy: evenly space NUM VGM cache-refresh steps.
  --inference.video_refresh_steps LIST
                         Explicit VGM refresh action steps, e.g. "[0,3,6,9]".
  --inference.video_stop_cosine_threshold FLOAT
                         Stop future VGM refreshes in a chunk once video cosine exceeds FLOAT.
  --inference.action_skip_cosine_threshold FLOAT
                         Reuse action velocity on cache-only steps once action cosine exceeds FLOAT.
  --inference.teacache.enabled BOOL
                         Enable TeaCache-style dynamic block residual caching.
  --inference.teacache.delta FLOAT
                         TeaCache accumulated relative-L1 threshold.
  --inference.teacache.force_last_step BOOL
                         Force the final VGM TeaCache timestep to run full blocks.
  --inference.teacache.action_enabled BOOL
                         Enable TeaCache on action-only video-cache steps.
  --inference.teacache.action_delta FLOAT
                         Action-only TeaCache threshold; defaults to VGM delta.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --task)
            MODE="${MODE:-single}"
            TASK_NAME="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --all)
            MODE="multi"
            shift
            ;;
        --tasks)
            TASKS_FILE_OVERRIDE="$2"
            shift 2
            ;;
        --gpus)
            GPU_IDS_OVERRIDE="$2"
            shift 2
            ;;
        --task-config)
            TASK_CONFIG_OVERRIDE="$2"
            shift 2
            ;;
        --seed)
            SEED_OVERRIDE="$2"
            shift 2
            ;;
        --episode-num|--test-num)
            EPISODE_NUM_OVERRIDE="$2"
            shift 2
            ;;
        --inference.num_inference_steps|--num-inference-steps|--num_inference_steps)
            EXTRA_OVERRIDES+=(--num_inference_steps "$2")
            shift 2
            ;;
        --inference.num_video_inference_steps|--num-video-inference-steps|--num_video_inference_steps|--vgm-num-inference-steps|--vgm_num_inference_steps)
            EXTRA_OVERRIDES+=(--num_video_inference_steps "$2")
            shift 2
            ;;
        --inference.video_refresh_steps|--video-refresh-steps|--video_refresh_steps|--vgm-refresh-steps|--vgm_refresh_steps)
            EXTRA_OVERRIDES+=(--video_refresh_steps "$2")
            shift 2
            ;;
        --inference.video_stop_cosine_threshold|--video-stop-cosine-threshold|--video_stop_cosine_threshold|--vgm-stop-cosine-threshold|--vgm_stop_cosine_threshold)
            EXTRA_OVERRIDES+=(--video_stop_cosine_threshold "$2")
            shift 2
            ;;
        --inference.action_skip_cosine_threshold|--action-skip-cosine-threshold|--action_skip_cosine_threshold|--action-stop-cosine-threshold|--action_stop_cosine_threshold)
            EXTRA_OVERRIDES+=(--action_skip_cosine_threshold "$2")
            shift 2
            ;;
        --inference.teacache.enabled|--teacache-enabled|--teacache_enabled|--enable-teacache|--enable_teacache)
            EXTRA_OVERRIDES+=(--teacache_enabled "$2")
            shift 2
            ;;
        --inference.teacache.delta|--teacache-delta|--teacache_delta)
            EXTRA_OVERRIDES+=(--teacache_delta "$2")
            shift 2
            ;;
        --inference.teacache.force_last_step|--teacache-force-last-step|--teacache_force_last_step)
            EXTRA_OVERRIDES+=(--teacache_force_last_step "$2")
            shift 2
            ;;
        --inference.teacache.action_enabled|--teacache-action-enabled|--teacache_action_enabled)
            EXTRA_OVERRIDES+=(--teacache_action_enabled "$2")
            shift 2
            ;;
        --inference.teacache.action_delta|--teacache-action-delta|--teacache_action_delta)
            EXTRA_OVERRIDES+=(--teacache_action_delta "$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            paint "$C_RED" "Error: unknown option: $1"
            usage
            exit 1
            ;;
        *)
            MODE="${MODE:-single}"
            if [ -z "$TASK_NAME" ]; then
                TASK_NAME="$1"
            elif [ -z "$GPU_ID" ]; then
                GPU_ID="$1"
            else
                paint "$C_RED" "Error: unexpected positional argument: $1"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$(pwd)/${CONFIG_FILE}"
fi

yaml_scalar() {
    local key="$1"
    local file="$2"

    awk -v key="$key" '
        $0 ~ "^[[:space:]]*" key "[[:space:]]*:" {
            value = $0
            sub(/^[[:space:]]*[^:]+:[[:space:]]*/, "", value)
            sub(/[[:space:]]+#.*$/, "", value)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            gsub(/^"|"$/, "", value)
            if (value == "null" || value == "~") {
                value = ""
            }
            print value
            exit
        }
    ' "$file"
}

yaml_inline_list() {
    local raw
    raw="$(yaml_scalar "$1" "$2")"
    raw="${raw#[}"
    raw="${raw%]}"
    raw="${raw// /}"
    raw="${raw//\"/}"
    raw="${raw//\'/}"
    printf "%s\n" "$raw"
}

resolve_policy_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf "%s\n" "$path"
    else
        printf "%s\n" "${POLICY_DIR}/${path}"
    fi
}

load_config() {
    local gpu_ids_str

    POLICY_NAME="$(yaml_scalar policy_name "$CONFIG_FILE")"
    ROBOTWIN_ROOT="$(yaml_scalar robotwin_root "$CONFIG_FILE")"
    CONDA_ENV="$(yaml_scalar conda_env "$CONFIG_FILE")"
    PYTHON_BIN="$(yaml_scalar python_executable "$CONFIG_FILE")"
    CHECKPOINT_PATH="$(yaml_scalar ckpt_setting "$CONFIG_FILE")"
    if [ -z "$CHECKPOINT_PATH" ]; then
        CHECKPOINT_PATH="$(yaml_scalar checkpoint_path "$CONFIG_FILE")"
    fi
    WAN_PATH="$(yaml_scalar wan_path "$CONFIG_FILE")"
    TASK_CONFIG="$(yaml_scalar task_config "$CONFIG_FILE")"
    CONFIG_TASK_NAME="$(yaml_scalar task_name "$CONFIG_FILE")"
    SEED="$(yaml_scalar seed "$CONFIG_FILE")"
    TASKS_FILE="$(yaml_scalar tasks_file "$CONFIG_FILE")"
    EPISODE_NUM="$(yaml_scalar episode_num "$CONFIG_FILE")"
    MULTI_EPISODE_NUM="$(yaml_scalar multi_episode_num "$CONFIG_FILE")"

    POLICY_NAME="${POLICY_NAME:-EfficientWAM}"
    TASK_CONFIG="${TASK_CONFIG_OVERRIDE:-${TASK_CONFIG:-demo_clean}}"
    TASK_NAME="${TASK_NAME:-${CONFIG_TASK_NAME:-adjust_bottle}}"
    SEED="${SEED_OVERRIDE:-${SEED:-42}}"
    TASKS_FILE="${TASKS_FILE_OVERRIDE:-${TASKS_FILE:-tasks_all.txt}}"
    EPISODE_NUM="${EPISODE_NUM_OVERRIDE:-${EPISODE_NUM:-2}}"
    MULTI_EPISODE_NUM="${EPISODE_NUM_OVERRIDE:-${MULTI_EPISODE_NUM:-20}}"

    if [ -n "$GPU_IDS_OVERRIDE" ]; then
        gpu_ids_str="$GPU_IDS_OVERRIDE"
    else
        gpu_ids_str="$(yaml_inline_list gpu_ids "$CONFIG_FILE")"
    fi

    if [ -n "$gpu_ids_str" ]; then
        IFS=',' read -ra GPU_IDS <<< "$gpu_ids_str"
    else
        GPU_IDS=()
    fi
}

validate_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        die "config file not found: $CONFIG_FILE"
    fi
    if [ -z "$ROBOTWIN_ROOT" ]; then
        die "robotwin_root is not set in $CONFIG_FILE"
    fi
    if [ -z "$CHECKPOINT_PATH" ]; then
        die "ckpt_setting is not set in $CONFIG_FILE"
    fi
    if [ -z "$WAN_PATH" ]; then
        die "wan_path is not set in $CONFIG_FILE"
    fi
    if [ ! -d "$ROBOTWIN_ROOT" ]; then
        die "RoboTwin root not found: $ROBOTWIN_ROOT"
    fi
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        die "EfficientWAM checkpoint file not found: $CHECKPOINT_PATH"
    fi
    if [ ! -d "$WAN_PATH" ]; then
        die "WAN path not found: $WAN_PATH"
    fi
}

activate_runtime() {
    cd "$ROBOTWIN_ROOT" || exit 1

    if [ -n "${CONDA_ENV:-}" ]; then
        if ! command -v conda &> /dev/null; then
            die "conda not found"
        fi
        paint "$C_DIM" "Activating conda environment: $CONDA_ENV"
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV"
    fi

    if [ -z "$PYTHON_BIN" ]; then
        PYTHON_BIN="$(command -v python)"
    fi
    if [ ! -x "$PYTHON_BIN" ]; then
        die "Python executable not found: $PYTHON_BIN"
    fi
    if ! "$PYTHON_BIN" -c 'import sapien' >/dev/null 2>&1; then
        die "Python $PYTHON_BIN cannot import sapien; set python_executable to the RoboTwin .venv Python in $CONFIG_FILE"
    fi

    export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
}

run_task() {
    local task="$1"
    local episode_num="$2"
    local log_dir="$3"

    "$PYTHON_BIN" script/eval_policy.py \
        --config "$CONFIG_FILE" \
        --overrides \
        --config_path "$CONFIG_FILE" \
        --task_name "$task" \
        --task_config "$TASK_CONFIG" \
        --ckpt_setting "$CHECKPOINT_PATH" \
        --seed "$SEED" \
        --episode_num "$episode_num" \
        --test_num "$episode_num" \
        --policy_name "$POLICY_NAME" \
        --log_dir "$log_dir" \
        --wan_path "$WAN_PATH" \
        "${EXTRA_OVERRIDES[@]}"
}

run_single() {
    local log_dir log_file
    GPU_ID="${GPU_ID:-${GPU_IDS[0]:-0}}"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"

    log_dir="${LOG_ROOT}/logs_single_$(date +%Y%m%d_%H%M%S)"
    log_file="${log_dir}/${TASK_NAME}.log"
    mkdir -p "$log_dir"

    print_section "EfficientWAM Single Task Evaluation"
    print_kv "Config" "$CONFIG_FILE"
    print_kv "Task Name" "$TASK_NAME"
    print_kv "GPU" "$GPU_ID"
    print_kv "RoboTwin Root" "$ROBOTWIN_ROOT"
    print_kv "Python" "$PYTHON_BIN"
    print_kv "Checkpoint" "$CHECKPOINT_PATH"
    print_kv "WAN Path" "$WAN_PATH"
    print_kv "Task Config" "$TASK_CONFIG"
    print_kv "Seed" "$SEED"
    print_kv "Episode Num" "$EPISODE_NUM"
    print_kv "Log File" "$log_file"
    print_rule

    set +e
    PYTHONWARNINGS=ignore::UserWarning run_task "$TASK_NAME" "$EPISODE_NUM" "$log_dir" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -e

    if [ "$exit_code" -eq 0 ]; then
        paint "$C_GREEN" "Task $TASK_NAME completed successfully"
    else
        paint "$C_RED" "Task $TASK_NAME failed with exit code $exit_code"
        print_kv "Log File" "$log_file"
    fi
    write_results_summary "$log_dir" "$EPISODE_NUM" "$TASK_NAME" || true
    return "$exit_code"
}

detect_gpus_if_needed() {
    if [ "${#GPU_IDS[@]}" -gt 0 ]; then
        return
    fi
    if command -v nvidia-smi &> /dev/null; then
        mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
        paint "$C_CYAN" "Auto-detected ${#GPU_IDS[@]} GPUs: ${GPU_IDS[*]}"
    else
        paint "$C_YELLOW" "Warning: nvidia-smi not found, using GPU 0"
        GPU_IDS=(0)
    fi
}

extract_success_rate() {
    local log_file="$1"
    local score

    if [ ! -f "$log_file" ]; then
        echo "N/A"
        return 0
    fi

    score="$(
        awk '
            BEGIN { IGNORECASE = 1 }
            /Success rate:/ { line = $0 }
            END {
                gsub(/\033\[[0-9;]*m/, "", line)
                if (line == "") {
                    exit
                }
                part_count = split(line, parts, "=>")
                target = parts[part_count]
                if (match(target, /[0-9]+(\.[0-9]+)?%/)) {
                    score = substr(target, RSTART, RLENGTH)
                    gsub(/%/, "", score)
                    printf "%.1f", score
                }
            }
        ' "$log_file"
    )"
    if [ -n "$score" ]; then
        echo "$score"
        return 0
    fi

    score="$(
        awk -F ':' '
            BEGIN { IGNORECASE = 1 }
            /success_rate:/ { value = $2 }
            END {
                gsub(/[^0-9.]/, "", value)
                if (value != "") {
                    printf "%.1f", value * 100.0
                }
            }
        ' "$log_file"
    )"
    if [ -n "$score" ]; then
        echo "$score"
        return 0
    fi

    echo "N/A"
}

write_results_summary() {
    local log_dir="$1"
    local episode_num="$2"
    shift 2
    local -a result_tasks=("$@")
    local summary results_file velocity_task_file velocity_global_file velocity_tmp success failed scored_count total_score average_score average_label total
    local task log_file score

    summary="${log_dir}/evaluation_summary.txt"
    results_file="${log_dir}/task_success_rates.csv"
    velocity_task_file="${log_dir}/velocity_cosine_by_task_step.csv"
    velocity_global_file="${log_dir}/velocity_cosine_all_tasks_by_step.csv"
    velocity_tmp="${log_dir}/.velocity_cosine_task_steps.tsv"

    success=0
    failed=0
    scored_count=0
    total_score=0
    total=0

    {
        echo "EfficientWAM Evaluation Summary"
        echo "=========================="
        echo "Date: $(date)"
        echo "Host: $(hostname)"
        echo "RoboTwin: $ROBOTWIN_ROOT"
        echo "Checkpoint: $CHECKPOINT_PATH"
        echo "WAN Path: $WAN_PATH"
        echo "Policy: $POLICY_NAME"
        echo "Task Config: $TASK_CONFIG"
        echo "Seed: $SEED"
        echo "Episode Num: $episode_num"
        echo "GPUs: ${GPU_IDS[*]}"
        echo ""
        echo "Task Results:"
        echo "-------------"
    } > "$summary"
    echo "task,success_rate_percent,status,log_file" > "$results_file"
    echo "task,metric,step,count,avg_cos" > "$velocity_task_file"
    : > "$velocity_tmp"

    for task in "${result_tasks[@]}"; do
        [ -z "$task" ] && continue
        ((total+=1))
        log_file="${log_dir}/${task}.log"
        score="$(extract_success_rate "$log_file")"
        if [ ! -f "$log_file" ]; then
            echo "  $task: LOG NOT FOUND" >> "$summary"
            echo "$task,N/A,LOG_NOT_FOUND,$log_file" >> "$results_file"
            ((failed+=1))
        elif grep -q "failed with exit code\|Traceback" "$log_file" 2>/dev/null; then
            echo "  $task: ERROR (see $log_file)" >> "$summary"
            echo "$task,N/A,ERROR,$log_file" >> "$results_file"
            ((failed+=1))
        elif [ "$score" != "N/A" ]; then
            echo "  $task: ${score}%" >> "$summary"
            echo "$task,$score,OK,$log_file" >> "$results_file"
            total_score="$(awk "BEGIN {printf \"%.4f\", $total_score + $score}")"
            ((scored_count+=1))
            if grep -q "completed successfully\|Success rate:" "$log_file" 2>/dev/null; then
                ((success+=1))
            else
                ((failed+=1))
            fi
        else
            echo "  $task: N/A" >> "$summary"
            echo "$task,N/A,NO_SCORE,$log_file" >> "$results_file"
            ((failed+=1))
        fi
        if [ -f "$log_file" ]; then
            awk -v task="$task" -v csv="$velocity_task_file" -v raw="$velocity_tmp" '
                /EfficientWAM task velocity cosine step:/ {
                    metric = ""
                    step = ""
                    count = ""
                    avg = ""
                    for (i = 1; i <= NF; i++) {
                        split($i, kv, "=")
                        if (kv[1] == "metric") metric = kv[2]
                        else if (kv[1] == "step") step = kv[2]
                        else if (kv[1] == "count") count = kv[2]
                        else if (kv[1] == "avg_cos") avg = kv[2]
                    }
                    if (metric != "" && step != "" && count != "" && avg != "") {
                        print task "," metric "," step "," count "," avg >> csv
                        print metric "\t" step "\t" count "\t" avg >> raw
                    }
                }
            ' "$log_file"
        fi
    done

    echo "metric,step,count,avg_cos" > "$velocity_global_file"
    if [ -s "$velocity_tmp" ]; then
        awk -F '\t' '
            {
                key = $1 SUBSEP $2
                count[key] += $3
                weighted_sum[key] += $3 * $4
            }
            END {
                for (key in count) {
                    split(key, parts, SUBSEP)
                    if (count[key] > 0) {
                        printf "%s,%s,%d,%.6f\n", parts[1], parts[2], count[key], weighted_sum[key] / count[key]
                    }
                }
            }
        ' "$velocity_tmp" | sort -t, -k1,1 -k2,2n >> "$velocity_global_file"
    fi

    if [ "$scored_count" -gt 0 ]; then
        average_score="$(awk "BEGIN {printf \"%.1f\", $total_score / $scored_count}")"
        average_label="${average_score}%"
    else
        average_score="N/A"
        average_label="N/A"
    fi
    echo "AVERAGE,$average_score,,$results_file" >> "$results_file"

    {
        echo ""
        echo "Summary Statistics:"
        echo "-------------------"
        echo "Successful: $success"
        echo "Failed: $failed"
        echo "Total: $total"
        echo "Average Success Rate: $average_label"
        echo ""
        echo "Results CSV: $results_file"
        echo "Velocity Cosine By Task/Step CSV: $velocity_task_file"
        echo "Velocity Cosine All Tasks/Step CSV: $velocity_global_file"
        echo "Logs: $log_dir"
    } >> "$summary"

    paint "$C_GREEN" "Successful: $success"
    if [ "$failed" -eq 0 ]; then
        paint "$C_GREEN" "Failed: $failed"
    else
        paint "$C_RED" "Failed: $failed"
    fi
    paint "$C_MAGENTA" "Average Success Rate: $average_label"
    print_kv "Results CSV" "$results_file"
    print_kv "Velocity CSV" "$velocity_task_file"
    print_kv "Global Velocity CSV" "$velocity_global_file"
    print_kv "Summary" "$summary"

    [ "$failed" -eq 0 ]
}

run_multi() {
    local tasks_path log_dir total completed failed launch_idx
    local -a tasks pids
    declare -A gpu_pid

    EPISODE_NUM="$MULTI_EPISODE_NUM"
    tasks_path="$(resolve_policy_path "$TASKS_FILE")"
    if [ ! -f "$tasks_path" ]; then
        die "tasks file not found: $tasks_path"
    fi
    mapfile -t tasks < "$tasks_path"
    if [ "${#tasks[@]}" -eq 0 ]; then
        die "no tasks found in $tasks_path"
    fi

    detect_gpus_if_needed
    for gpu_id in "${GPU_IDS[@]}"; do
        gpu_pid[$gpu_id]=""
    done

    log_dir="${LOG_ROOT}/logs_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$log_dir"

    print_section "EfficientWAM Task List Evaluation"
    print_kv "Config" "$CONFIG_FILE"
    print_kv "Tasks File" "$tasks_path"
    print_kv "Tasks" "${#tasks[@]}"
    print_kv "GPUs" "${GPU_IDS[*]}"
    print_kv "RoboTwin Root" "$ROBOTWIN_ROOT"
    print_kv "Python" "$PYTHON_BIN"
    print_kv "Checkpoint" "$CHECKPOINT_PATH"
    print_kv "WAN Path" "$WAN_PATH"
    print_kv "Task Config" "$TASK_CONFIG"
    print_kv "Seed" "$SEED"
    print_kv "Episode Num" "$EPISODE_NUM"
    print_kv "Log Dir" "$log_dir"
    print_rule

    is_running() {
        [ -n "$1" ] && kill -0 "$1" 2>/dev/null
    }

    get_free_gpu() {
        while true; do
            for gpu_id in "${GPU_IDS[@]}"; do
                if ! is_running "${gpu_pid[$gpu_id]}"; then
                    echo "$gpu_id"
                    return 0
                fi
            done
            sleep 2
        done
    }

    pids=()
    launch_idx=0
    for task in "${tasks[@]}"; do
        [ -z "$task" ] && continue
        ((launch_idx+=1))
        gpu_id="$(get_free_gpu)"
        task_log_dir="${log_dir}/${task}"
        log_file="${log_dir}/${task}.log"

        printf "%b[%02d/%02d]%b %bTASK%b %b%-32s%b %bGPU%b %b%-2s%b %bLOG%b %s\n" \
            "$C_DIM" "$launch_idx" "${#tasks[@]}" "$C_RESET" \
            "$C_BLUE" "$C_RESET" \
            "$C_CYAN" "$task" "$C_RESET" \
            "$C_DIM" "$C_RESET" \
            "$C_MAGENTA" "$gpu_id" "$C_RESET" \
            "$C_DIM" "$C_RESET" \
            "$log_file"
        (
            export CUDA_VISIBLE_DEVICES="$gpu_id"
            mkdir -p "$task_log_dir"
            if PYTHONWARNINGS=ignore::UserWarning run_task "$task" "$EPISODE_NUM" "$task_log_dir" > "$log_file" 2>&1; then
                echo "Task $task completed successfully" >> "$log_file"
            else
                exit_code=$?
                echo "Task $task failed with exit code $exit_code" >> "$log_file"
                exit "$exit_code"
            fi
        ) &

        pid=$!
        gpu_pid[$gpu_id]=$pid
        pids+=("$pid")
        sleep 1
    done

    completed=0
    total=${#pids[@]}
    if [ "$total" -eq 0 ]; then
        die "no runnable tasks found in $tasks_path"
    fi
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            ((failed+=1))
        fi
        ((completed+=1))
    done
    paint "$C_GREEN" "Completed $completed/$total tasks"

    write_results_summary "$log_dir" "$EPISODE_NUM" "${tasks[@]}"
}

paint "${C_BOLD}${C_BLUE}" "Starting EfficientWAM evaluation on RoboTwin at $(date)"
print_kv "Config" "$CONFIG_FILE"

load_config
MODE="${MODE:-single}"
validate_config
activate_runtime

if [ "$MODE" = "multi" ]; then
    run_multi
else
    run_single
fi
