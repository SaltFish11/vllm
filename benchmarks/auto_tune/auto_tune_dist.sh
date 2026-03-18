#!/bin/bash
# set -x
# This script aims to tune the best server parameter combinations to maximize throughput for given requirement.
# See details in README (benchmarks/auto_tune/README.md).
TAG=${TAG:-""}
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
BASE=${BASE:-"$SCRIPT_DIR/../../.."}
MODEL=${MODEL:-""}
SYSTEM=${SYSTEM:-"GLM4.7"}
TP=${TP:-8}
DP=${DP:-4}
EP=${EP:-32}
DP_SIZE_LOCAL=${DP_SIZE_LOCAL:-1}
TOTAL_NODES=${TOTAL_NODES:-${ARNOLD_WORKER_NUM:-$DP}}
TOOL_PARSER=${TOOL_PARSER:-"glm47"}
DOWNLOAD_DIR=${DOWNLOAD_DIR:-""}
INPUT_LEN=${INPUT_LEN:-9000}
OUTPUT_LEN=${OUTPUT_LEN:-3000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-150000}
MIN_CACHE_HIT_PCT=${MIN_CACHE_HIT_PCT:-0}
MAX_LATENCY_ALLOWED_MS=${MAX_LATENCY_ALLOWED_MS:-100000000000}
NUM_SEQS_LIST=${NUM_SEQS_LIST:-"256 512"}
NUM_BATCHED_TOKENS_LIST=${NUM_BATCHED_TOKENS_LIST:-"1024 2048 4096 8192"}
ARNOLD_ID=${ARNOLD_ID:-0}
AUTO_BENCH_ROOT="${BASE}/auto-benchmark"
mkdir -p "$AUTO_BENCH_ROOT"
if [[ -z "$TAG" ]]; then
    if [[ "$ARNOLD_ID" -eq 0 ]]; then
        TAG=$(date +"%Y_%m_%d_%H_%M")
        echo "$TAG" > "$AUTO_BENCH_ROOT/current_tag"
    else
        for i in {1..300}; do
            if [[ -f "$AUTO_BENCH_ROOT/current_tag" ]]; then
                TAG=$(cat "$AUTO_BENCH_ROOT/current_tag")
                break
            fi
            sleep 1
        done
    fi

    
fi
if [[ -z "$TAG" ]]; then
    echo "Error: Failed to determine TAG." >&2
    exit 1
fi

MASTER_HOST=$ARNOLD_WORKER_0_HOST
PORTS=$ARNOLD_WORKER_0_PORT
# 提取第一个端口
PORT=$(echo "$PORTS" | cut -d',' -f1)
# 提取第二个端口
DP_RPC_PORT=$(echo "$PORTS" | cut -d',' -f2)
MASTER_PORT=$(echo "$PORTS" | cut -d',' -f3)
VLLM_PORT_USE_LIST=$PORTS
export VLLM_PORT_USE_LIST=$(echo "$VLLM_PORT_USE_LIST" | cut -d',' -f4-)
if [[ -z "$MASTER_HOST" ]]; then
    echo "Error: Failed to determine hostname." >&2
    exit 1
fi

LOG_FOLDER="$AUTO_BENCH_ROOT/$TAG"
barrier_comm_dir="${LOG_FOLDER}/tmp"

RESULT="$LOG_FOLDER/result.txt"
PROFILE_PATH="$LOG_FOLDER/profile"

echo "====================== AUTO TUNE PARAMETERS ===================="
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "BASE=$BASE"
echo "MODEL=$MODEL"
echo "SYSTEM=$SYSTEM"
echo "TP=$TP"
echo "TOTAL_NODES=$TOTAL_NODES"
echo "DOWNLOAD_DIR=$DOWNLOAD_DIR"
echo "INPUT_LEN=$INPUT_LEN"
echo "OUTPUT_LEN=$OUTPUT_LEN"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "MIN_CACHE_HIT_PCT=$MIN_CACHE_HIT_PCT"
echo "MAX_LATENCY_ALLOWED_MS=$MAX_LATENCY_ALLOWED_MS"
echo "NUM_SEQS_LIST=$NUM_SEQS_LIST"
echo "NUM_BATCHED_TOKENS_LIST=$NUM_BATCHED_TOKENS_LIST"
echo "VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL"
echo "RESULT_FILE=$RESULT"
echo "====================== AUTO TUNEPARAMETERS ===================="

# rm -rf $LOG_FOLDER
# rm -rf $PROFILE_PATH
mkdir -p $LOG_FOLDER
mkdir -p $PROFILE_PATH

cd "$BASE/vllm"

pip install -q datasets

current_hash=$(git rev-parse HEAD)
echo "hash:$current_hash" >> "$RESULT"
echo "current_hash: $current_hash"

TOTAL_LEN=$((INPUT_LEN + OUTPUT_LEN))
RED='\033[0;31m'
if (( TOTAL_LEN > MAX_MODEL_LEN )); then
    echo -e "${RED}FAILED: INPUT_LEN($INPUT_LEN) + OUTPUT_LEN($OUTPUT_LEN) = $TOTAL_LEN, which is > MAX_MODEL_LEN = $MAX_MODEL_LEN.\033[0m" >&2
    exit 1
fi

best_throughput=0
best_max_num_seqs=0
best_num_batched_tokens=0
best_goodput=0
best_request_rate=0

build_role_config_args() {
    START_RANK=$((DP_SIZE_LOCAL * ARNOLD_ID))
    local args=""
    if [[ $ARNOLD_ID -ne 0 ]]; then
        args+="  --data-parallel-start-rank $START_RANK"
        args+="  --headless"
    fi
    echo "$args"
}
mkdir -p "$barrier_comm_dir"

barrier_func() {
    # 每个节点执行完任务后
    local suffix_str=$1
    local total_nodes=$2
    local failed_suffix_str=$3
    # 等待所有节点完成
    while true; do
        finished=$(ls "$barrier_comm_dir"/*.${suffix_str} 2>/dev/null | wc -l)
        if [ $finished -ge $total_nodes ]; then
            # sudo rm -rf "$barrier_comm_dir"/*.${suffix_str}
            # if [[ -n "$failed_suffix_str" ]]; then
            #     # sudo rm -rf "$barrier_comm_dir"/*.${failed_suffix_str}
            # fi
            break
        fi
        if [[ -n "$failed_suffix_str" ]]; then
            failed=$(ls "$barrier_comm_dir"/*.${failed_suffix_str} 2>/dev/null | wc -l)
            if [ $failed -ge 1 ]; then
                # sudo rm -rf "$barrier_comm_dir"/*.${failed_suffix_str}
                # sudo rm -rf "$barrier_comm_dir"/*.${suffix_str}
                echo "有其他 node 发生错误"
                return 1
            fi
        fi
        sleep 1
    done
    echo "所有节点完成！开始下一阶段..."
    return 0
}

start_server() {
    local gpu_memory_utilization=$1
    local max_num_seqs=$2
    local max_num_batched_tokens=$3
    local vllm_log=$4
    local profile_dir=$5
    local failed_suffix_str=$6

    pkill -if "vllm serve" || true
    pkill -9 vllm || true
    pkill -9 VLLM::Worker_DP || true
    pkill -9 VLLM::DPCoordin || true
    pkill -9 VLLM::EngineCor || true
    pkill -9 VLLM::APIServer || true
    # Define the common arguments as a bash array.
    # Each argument and its value are separate elements.
    local common_args_array=(
        "$MODEL"
        "--disable-log-requests"
        "--port" "$PORT"
        "--host" "$MASTER_HOST"
        "--gpu-memory-utilization" "$gpu_memory_utilization"
        "--max-num-seqs" "$max_num_seqs"
        "--max-num-batched-tokens" "$max_num_batched_tokens"
        "--tensor-parallel-size" "$TP"
        "--enable-prefix-caching"
        "--load-format" "dummy"
        "--download-dir" "$DOWNLOAD_DIR"
        "--max-model-len" "$MAX_MODEL_LEN"
        "--enable-auto-tool-choice"
        "--tool-call-parser" "$TOOL_PARSER"
        "--enable-expert-parallel"
        "--async-scheduling"
        "--data-parallel-size" "$DP"
        "--data-parallel-size-local" "$DP_SIZE_LOCAL"
        "--data-parallel-address" "$MASTER_HOST"
        "--data-parallel-rpc-port" "$DP_RPC_PORT"
        "--master-addr" "$MASTER_HOST"
        "--master-port" "$MASTER_PORT"
    )
    common_args_array+=($(build_role_config_args))

    # Use the array expansion "${common_args_array[@]}"
    # This correctly passes each element as a separate argument.
    echo "common_args_array: ${common_args_array[@]}"
    if [[ -n "$profile_dir" ]]; then
        # Start server with profiling enabled
        local profile_config_json="{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$profile_dir\"}"
        VLLM_SERVER_DEV_MODE=1 \
            vllm serve --profiler-config "$profile_config_json" "${common_args_array[@]}" > "$vllm_log" 2>&1 &
    else
        # Start server without profiling
        VLLM_SERVER_DEV_MODE=1 \
            vllm serve "${common_args_array[@]}" > "$vllm_log" 2>&1 &
    fi
    local server_pid=$!

    # wait for 10 minutes...
    server_started=0
    for i in {1..60}; do
        # This line checks whether the server is still alive or not,
        # since that we should always have permission to send signal to the server process.
        kill -0 $server_pid 2> /dev/null || break

        if [[ "$MASTER_HOST" == *:* ]]; then
            health_url="http://[${MASTER_HOST}]:${PORT}/health"
        else
            health_url="http://${MASTER_HOST}:${PORT}/health"
        fi
        RESPONSE=$(curl -s -X GET "$health_url" -w "%{http_code}" -o /dev/stdout)
        STATUS_CODE=$(echo "$RESPONSE" | tail -n 1)
        if [[ "$STATUS_CODE" -eq 200 ]]; then
            server_started=1
            break
        else
            sleep 18
        fi
        if [[ -n "$failed_suffix_str" ]]; then
            failed=$(ls "$barrier_comm_dir"/*.${failed_suffix_str} 2>/dev/null | wc -l)
            if [ $failed -ge 1 ]; then
                echo "server failed to start. gpu_memory_utilization:$gpu_memory_utilization, max_num_seqs:$max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens"
                break
                # return 1
            fi
        fi
    done

    if (( server_started == 0 )); then
        touch "$barrier_comm_dir/${MASTER_HOST}_${ARNOLD_ID}.${failed_suffix_str}"
        pkill -9 VLLM::Worker_DP || true
        pkill -9 VLLM::DPCoordin || true
        pkill -9 vllm || true
        pkill -9 VLLM::EngineCor || true
        pkill -9 VLLM::APIServer || true
        echo "server did not start within 25 minutes or crashed. Please check server log at $vllm_log".
        return 1
    else
        return 0
    fi
}



run_benchmark() {
    pkill -if "vllm serve" || true
    sleep 10
    
    local max_num_seqs=$1
    local max_num_batched_tokens=$2
    local gpu_memory_utilization=$3
    echo "max_num_seq: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens"
    local vllm_log="$LOG_FOLDER/vllm_log_WORKER${ARNOLD_ID}_${max_num_seqs}_${max_num_batched_tokens}.txt"
    echo "vllm_log: $vllm_log"
    echo
    rm -f $vllm_log
    pkill -if "vllm serve" || true

    local done_suffix="done_${max_num_seqs}_${max_num_batched_tokens}"
    local failed_suffix="failed_${max_num_seqs}_${max_num_batched_tokens}"
    local bench_done_suffix="bench_done_${max_num_seqs}_${max_num_batched_tokens}"

    echo "starting server..."
    # Call start_server without a profile_dir to avoid profiling overhead
    start_server $gpu_memory_utilization $max_num_seqs $max_num_batched_tokens $vllm_log "" $failed_suffix
    result=$?
    if [[ "$result" -eq 1 ]]; then
        touch "$barrier_comm_dir/${MASTER_HOST}_${ARNOLD_ID}.${failed_suffix}"
        echo "server failed to start. gpu_memory_utilization:$gpu_memory_utilization, max_num_seqs:$max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens"
    else
        touch "$barrier_comm_dir/${MASTER_HOST}_${ARNOLD_ID}.${done_suffix}"
        echo "server started."
    fi

    barrier_func "$done_suffix" "$TOTAL_NODES" "$failed_suffix"
    result=$?
    if [[ "$result" -ne 0 ]]; then
        echo "some server failed. gpu_memory_utilization:$gpu_memory_utilization, max_num_seqs:$max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens"
        return 1
    fi

    if [[ "$ARNOLD_ID" -eq 0 ]]; then

        echo "run benchmark test..."
        
        meet_latency_requirement=0
        # get a basic qps by using request-rate inf
        bm_log="$LOG_FOLDER/bm_log_${max_num_seqs}_${max_num_batched_tokens}_requestrate_inf.txt"
        prefix_len=$(( INPUT_LEN * MIN_CACHE_HIT_PCT / 100 ))
        adjusted_input_len=$(( INPUT_LEN - prefix_len ))
        # --profile flag is removed from this call
        vllm bench serve --backend vllm \
            --model $MODEL  \
            --dataset-name random \
            --random-input-len $adjusted_input_len \
            --random-output-len $OUTPUT_LEN \
            --ignore-eos \
            --disable-tqdm \
            --request-rate inf \
            --percentile-metrics ttft,tpot,itl,e2el \
            --goodput e2el:$MAX_LATENCY_ALLOWED_MS \
            --num-prompts 200 \
            --random-prefix-len $prefix_len \
            --host "$MASTER_HOST" \
            --port "$PORT" &> "$bm_log"
        throughput=$(grep "Request throughput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')
        e2el=$(grep "P99 E2EL (ms):" "$bm_log" | awk '{print $NF}')
        goodput=$(grep "Request goodput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')
        meet_latency_requirement=0
        if (( $(echo "$e2el <= $MAX_LATENCY_ALLOWED_MS" | bc -l) )); then
            meet_latency_requirement=1
            request_rate=inf
        fi

        if (( ! meet_latency_requirement )); then
        # start from request-rate as int(throughput) + 1
            request_rate=$((${throughput%.*} + 1))
            while ((request_rate > 0)); do
                # clear prefix cache
                curl -X POST http://${MASTER_HOST}:$PORT/reset_prefix_cache
                sleep 5
                bm_log="$LOG_FOLDER/bm_log_${max_num_seqs}_${max_num_batched_tokens}_requestrate_${request_rate}.txt"
                vllm bench serve --backend vllm \
                    --model $MODEL  \
                    --dataset-name random \
                    --random-input-len $adjusted_input_len \
                    --random-output-len $OUTPUT_LEN \
                    --ignore-eos \
                    --disable-tqdm \
                    --request-rate $request_rate \
                    --percentile-metrics ttft,tpot,itl,e2el \
                    --goodput e2el:$MAX_LATENCY_ALLOWED_MS \
                    --num-prompts 100 \
                    --random-prefix-len $prefix_len \
                    --host "$MASTER_HOST" \
                    --port $PORT &> "$bm_log"
                throughput=$(grep "Request throughput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')
                e2el=$(grep "P99 E2EL (ms):" "$bm_log" | awk '{print $NF}')
                goodput=$(grep "Request goodput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')
                if (( $(echo "$e2el <= $MAX_LATENCY_ALLOWED_MS" | bc -l) )); then
                    meet_latency_requirement=1
                    break
                fi
                request_rate=$((request_rate-1))
            done
        fi
        # write the results and update the best result.
        if ((meet_latency_requirement)); then
            echo "max_num_seqs: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens, request_rate: $request_rate, e2el: $e2el, throughput: $throughput, goodput: $goodput"
            echo "max_num_seqs: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens, request_rate: $request_rate, e2el: $e2el, throughput: $throughput, goodput: $goodput" >> "$RESULT"
            if (( $(echo "$throughput > $best_throughput" | bc -l) )); then
                best_throughput=$throughput
                best_max_num_seqs=$max_num_seqs
                best_num_batched_tokens=$max_num_batched_tokens
                best_goodput=$goodput
                best_request_rate=$request_rate
            fi
        else
            echo "max_num_seqs: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens does not meet latency requirement ${MAX_LATENCY_ALLOWED_MS}"
            echo "max_num_seqs: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens does not meet latency requirement ${MAX_LATENCY_ALLOWED_MS}" >> "$RESULT"
        fi
        printf "best_throughput=%s\nbest_max_num_seqs=%s\nbest_num_batched_tokens=%s\nbest_goodput=%s\nbest_request_rate=%s\n" \
            "$best_throughput" \
            "$best_max_num_seqs" \
            "$best_num_batched_tokens" \
            "$best_goodput" \
            "$best_request_rate" > "${barrier_comm_dir}/best_save.txt"
        echo "best_max_num_seqs: $best_max_num_seqs, best_num_batched_tokens: $best_num_batched_tokens, best_throughput: $best_throughput"
        touch "$barrier_comm_dir/${MASTER_HOST}_${ARNOLD_ID}.${bench_done_suffix}"
    fi
    barrier_func "$bench_done_suffix" 1
    result=$?
    if [[ "$result" -ne 0 ]]; then
        echo "bench_done failed."
        return 1
    fi
    echo "===================="
    return 0
}

read -r -a num_seqs_list <<< "$NUM_SEQS_LIST"
read -r -a num_batched_tokens_list <<< "$NUM_BATCHED_TOKENS_LIST"

# first find out the max gpu-memory-utilization without HBM OOM.
gpu_memory_utilization=0.80
find_gpu_memory_utilization=0

while (( $(echo "$gpu_memory_utilization >= 0.75" | bc -l) )); do
    # Pass empty string for profile_dir argument
    util_tag=$(printf "%s" "$gpu_memory_utilization" | tr '.' '_')
    done_suffix="gm_done_${util_tag}"
    failed_suffix="gm_failed_${util_tag}"

    start_server $gpu_memory_utilization "${num_seqs_list[-1]}" "${num_batched_tokens_list[-1]}" "$LOG_FOLDER/vllm_log_gpu_memory_utilization_${gpu_memory_utilization}_worker${ARNOLD_ID}.log" "" "$failed_suffix"
    result=$?


    if [[ "$result" -eq 0 ]]; then
        touch "$barrier_comm_dir/${MASTER_HOST}_${ARNOLD_ID}.${done_suffix}"
    else
        touch "$barrier_comm_dir/${MASTER_HOST}_${ARNOLD_ID}.${failed_suffix}"
    fi

    barrier_func "$done_suffix" "$TOTAL_NODES" "$failed_suffix"
    result=$?
    if [[ "$result" -eq 0 ]]; then
        find_gpu_memory_utilization=1
        break
    else
        gpu_memory_utilization=$(echo "$gpu_memory_utilization - 0.01" | bc)
    fi
done

if [[ "$find_gpu_memory_utilization" -eq 1 ]]; then
    echo "Using gpu_memory_utilization=$gpu_memory_utilization to serve model."
else
    echo "Cannot find a proper gpu_memory_utilization over 0.9 to serve the model, please check logs in $LOG_FOLDER."
    exit 1
fi
# NUM_SEQS_LIST=${NUM_SEQS_LIST:-"128 256 512"}
# NUM_BATCHED_TOKENS_LIST=${NUM_BATCHED_TOKENS_LIST:-"512 1024 2048 4096 8192"}
for num_seqs in "${num_seqs_list[@]}"; do
    for num_batched_tokens in "${num_batched_tokens_list[@]}"; do
        run_benchmark $num_seqs $num_batched_tokens $gpu_memory_utilization
    done
done
echo "finish permutations"

# =================================================================================
# FINAL PROFILING RUN FOR THE BEST CONFIGURATION
# =================================================================================
profile_done_suffix="profile_done"
profile_server_done_suffix="profile_server_done"
profile_server_failed_suffix="profile_server_failed"

if [[ -f "${barrier_comm_dir}/best_save.txt" ]]; then
    source "${barrier_comm_dir}/best_save.txt"
fi

# source /mnt/bn/tiktok-mm-4/aiic/users/zhangbiao.168/vllm_deploy_test/bench_config/auto-benchmark/2026_03_18_13_52/tmp/best_save.txt
if (( $(echo "$best_throughput > 0" | bc -l) )); then
    echo
    echo "Benchmark tuning finished. Now running profiling on the best configuration found..."
    echo "Best config: max_num_seqs: $best_max_num_seqs, max_num_batched_tokens: $best_num_batched_tokens, throughput: $best_throughput"
    echo

    vllm_log="$LOG_FOLDER/vllm_log_BEST_PROFILE_${ARNOLD_ID}.txt"
    bm_log="$LOG_FOLDER/bm_log_BEST_PROFILE.txt"

    # Start server with the best params and profiling ENABLED
    echo "Starting server for profiling..."
    # start_server $gpu_memory_utilization $best_max_num_seqs $best_num_batched_tokens "$vllm_log" "$PROFILE_PATH" "$profile_server_failed_suffix"
    start_server $gpu_memory_utilization $best_max_num_seqs $best_num_batched_tokens "$vllm_log" "$PROFILE_PATH" "$profile_server_failed_suffix"

    result=$?
    if [[ "$result" -eq 0 ]]; then
        touch "$barrier_comm_dir/${MASTER_HOST}_${ARNOLD_ID}.${profile_server_done_suffix}"
    else
        touch "$barrier_comm_dir/${MASTER_HOST}_${ARNOLD_ID}.${profile_server_failed_suffix}"
    fi
    barrier_func "$profile_server_done_suffix" "$TOTAL_NODES" "$profile_server_failed_suffix"
    result=$?
    if [[ "$result" -ne 0 ]]; then
        echo "profile_server_failed."
        return 1
    fi
    if [[ "$ARNOLD_ID" -eq 0 ]]; then

        # Run benchmark with the best params and the --profile flag
        echo "Running benchmark with profiling..."
        prefix_len=$(( INPUT_LEN * MIN_CACHE_HIT_PCT / 100 ))
        adjusted_input_len=$(( INPUT_LEN - prefix_len ))
        vllm bench serve --backend vllm \
            --model $MODEL \
            --dataset-name random \
            --random-input-len $adjusted_input_len \
            --random-output-len $OUTPUT_LEN \
            --ignore-eos \
            --disable-tqdm \
            --request-rate $best_request_rate \
            --percentile-metrics ttft,tpot,itl,e2el \
            --goodput e2el:$MAX_LATENCY_ALLOWED_MS \
            --num-prompts 100 \
            --random-prefix-len $prefix_len \
            --host "$MASTER_HOST" \
            --port $PORT &> "$bm_log"
            # --profile 
        echo "best_max_num_seqs: $best_max_num_seqs, best_num_batched_tokens: $best_num_batched_tokens, best_throughput: $best_throughput, profile saved in: $PROFILE_PATH"
        echo "best_max_num_seqs: $best_max_num_seqs, best_num_batched_tokens: $best_num_batched_tokens, best_throughput: $best_throughput, profile saved in: $PROFILE_PATH" >> "$RESULT"
        touch "$barrier_comm_dir/${MASTER_HOST}_${ARNOLD_ID}.${profile_done_suffix}"
    fi
    barrier_func "$profile_done_suffix" 1 "$profile_server_failed_suffix"
    result=$?
    if [[ "$result" -ne 0 ]]; then
        echo "profile_server_failed."
        return 1
    fi
else
    echo "No configuration met the latency requirements. Skipping final profiling run."
fi
echo "all task finished!!!"
pkill -if "vllm serve" || true
