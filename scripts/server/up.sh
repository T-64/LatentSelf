#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
    reader9|reader27-offload|reader27-int8)
        exec bash "$(dirname "$0")/start-reader.sh" "$1" ;;
esac
case "${1:-}" in
    embed|query|text|rerank|llm|llm-private) ;;
    *) echo "usage: $0 embed|query|text|rerank|llm|llm-private" >&2; exit 2 ;;
esac
default_fraction=0.80
default_kv=auto
if [[ $1 == query ]]; then
    default_fraction=0.36
    default_kv=1610612736
fi
fraction=${LATENTSELF_GPU_MEMORY_FRACTION:-$default_fraction}
[[ $fraction =~ ^0\.[0-9]+$ ]] || { echo "invalid GPU memory fraction" >&2; exit 2; }
kv_bytes=${LATENTSELF_KV_CACHE_BYTES:-$default_kv}
[[ $kv_bytes == auto || $kv_bytes =~ ^[1-9][0-9]*$ ]] || { echo "invalid KV cache byte count" >&2; exit 2; }
slots=${LATENTSELF_JUDGE_SLOTS:-1}
[[ $slots =~ ^[1-4]$ ]] || { echo "invalid number of judgment slots" >&2; exit 2; }
ssh -o BatchMode=yes server bash -s -- "$1" "$fraction" "$kv_bytes" "$slots" <<'REMOTE'
set -euo pipefail
role=$1
if [[ $role == llm ]]; then
    # Starting an existing Docker container reuses its immutable port mappings.
    published=$(docker inspect -f '{{range (index .HostConfig.PortBindings "8000/tcp")}}{{.HostIp}}:{{.HostPort}}{{end}}' qwen-backend)
    if [[ $published != 127.0.0.1:8000 ]]; then
        echo "Existing Qwen stack publishes outside loopback. Resolve the binding before starting it." >&2
        exit 1
    fi
    bash -ic 'qwen-start'
    exit
fi
case "$role" in
    embed|query) model=Qwen/Qwen3-VL-Embedding-8B; port=8001; revision=2c4565515e0f265c6511776e7193b22c0968ddc7 ;;
    text) model=Qwen/Qwen3-Embedding-8B; port=8003; revision=1d8ad4ca9b3dd8059ad90a75d4983776a23d44af ;;
    rerank) model=Qwen/Qwen3-VL-Reranker-8B; port=8002; revision=b212dc8c91a8164aef1ea2de9c1a867611e75c04 ;;
    llm-private) port=8010 ;;
esac
name="latentself-$role"
if [[ $role == embed || $role == query ]]; then
    other=embed
    [[ $role == embed ]] && other=query
    if [[ $(docker inspect -f '{{.State.Running}}' "latentself-$other" 2>/dev/null || :) == true ]]; then
        echo "$other already serves port 8001. Stop it before starting $role." >&2
        exit 1
    fi
fi
wait_ready() {
    for ((attempt=0; attempt<150; attempt++)); do
        if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            echo "$role ready on port $port"
            return
        fi
        if [[ $(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || :) != true ]]; then
            echo "$name exited during startup" >&2
            return 1
        fi
        sleep 2
    done
    echo "$name is still starting after the readiness timeout" >&2
    return 1
}
if docker inspect "$name" >/dev/null 2>&1; then
    test "$(docker inspect -f '{{index .Config.Labels "latentself.role"}}' "$name")" = "$role"
    docker start "$name" >/dev/null
    wait_ready
    exit
fi
if [[ $role == llm-private ]]; then
    # Reuse the installed weights and exact image without publishing the owner's proxy.
    image=$(docker inspect -f '{{.Image}}' qwen-backend)
    models=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/models"}}{{.Source}}{{end}}{{end}}' qwen-backend)
    test -s "$models/Qwen3.8-27B-UD-Q6_K_XL.gguf"
    test -s "$models/mmproj-Qwen3.8-27B-F16.gguf"
    docker run -d --rm --name "$name" --label "latentself.role=$role" --gpus all \
        --health-cmd='curl -fsS http://127.0.0.1:8000/health' --health-start-period=180s \
        --log-driver none --tmpfs /tmp --read-only --mount "type=bind,src=$models,dst=/models,readonly" \
        -p 127.0.0.1:8010:8000 "$image" -m /models/Qwen3.8-27B-UD-Q6_K_XL.gguf \
        --mmproj /models/mmproj-Qwen3.8-27B-F16.gguf --alias qwen3.8-27b \
        -c "$((24576 * $4))" -np "$4" -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
        --cache-ram 0 --jinja --reasoning-budget 0 --host 0.0.0.0 --port 8000
    wait_ready
    exit
fi
cache="$HOME/.cache/huggingface"
modeldir="hub/models--${model/\//--}"
snapshot="$modeldir/snapshots/$revision"
test -s "$cache/$snapshot/config.json" || { echo "Model snapshot missing: $model" >&2; exit 1; }
extra=()
if [[ $3 != auto ]]; then
    extra+=(--kv-cache-memory-bytes "$3")
fi
if [[ $role == query ]]; then
    extra+=(--language-model-only)
elif [[ $role != text ]]; then
    extra+=(--limit-mm-per-prompt '{"image":3,"video":0}')
fi
if [[ $role == rerank ]]; then
    template=/vllm-workspace/examples/pooling/score/template/qwen3_vl_reranker.jinja
    extra+=(--hf-overrides '{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' --chat-template "$template")
fi
docker run -d --rm --name "$name" --label "latentself.role=$role" --gpus all \
    --log-driver none --shm-size 2g --tmpfs /tmp:exec \
    -p "127.0.0.1:$port:8000" -v "$cache:/hf:ro" \
    -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
    vllm/vllm-openai:v0.34.0 --model "/hf/$snapshot" --served-model-name "$model" \
    --runner pooling --dtype bfloat16 --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --enforce-eager --gpu-memory-utilization "$2" "${extra[@]}"
wait_ready
REMOTE
