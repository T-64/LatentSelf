#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
    embed|text|rerank|llm) ;;
    *) echo "usage: $0 embed|text|rerank|llm" >&2; exit 2 ;;
esac
ssh -o BatchMode=yes server bash -s -- "$1" <<'REMOTE'
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
    embed) model=Qwen/Qwen3-VL-Embedding-8B; port=8001; revision=2c4565515e0f265c6511776e7193b22c0968ddc7 ;;
    text) model=Qwen/Qwen3-Embedding-8B; port=8003; revision= ;;
    rerank) model=Qwen/Qwen3-VL-Reranker-8B; port=8002; revision=b212dc8c91a8164aef1ea2de9c1a867611e75c04 ;;
esac
name="latentself-$role"
if docker inspect "$name" >/dev/null 2>&1; then
    test "$(docker inspect -f '{{index .Config.Labels "latentself.role"}}' "$name")" = "$role"
    docker start "$name" >/dev/null
    exit
fi
cache="$HOME/.cache/huggingface"
modeldir="hub/models--${model/\//--}"
if [[ -z $revision ]]; then revision=$(cat "$cache/$modeldir/refs/main"); fi
snapshot="$modeldir/snapshots/$revision"
test -s "$cache/$snapshot/config.json" || { echo "Model snapshot missing: $model" >&2; exit 1; }
extra=()
if [[ $role == rerank ]]; then
    template=$(curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/v0.34.0/examples/pooling/score/template/qwen3_vl_reranker.jinja)
    extra+=(--hf-overrides '{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' --chat-template "$template")
fi
docker run -d --rm --name "$name" --label "latentself.role=$role" --gpus all \
    --log-driver none --shm-size 2g --tmpfs /tmp:exec \
    -p "127.0.0.1:$port:8000" -v "$cache:/hf:ro" \
    -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
    vllm/vllm-openai:v0.34.0 --model "/hf/$snapshot" --served-model-name "$model" \
    --runner pooling --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.43 "${extra[@]}"
REMOTE
