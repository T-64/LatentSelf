#!/usr/bin/env bash
# Publish only Python source to the user's GPU host. A host-network container listens on loopback.
set -euo pipefail
case "${1:-}" in
    reader9|reader27-offload|reader27-int8) role=$1 ;;
    *) exit 2 ;;
esac
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
revision=$(git -C "$root" rev-parse HEAD)
remote_dir=$(ssh -o BatchMode=yes server 'mkdir -p "$HOME/.cache/latentself"; mktemp -d "$HOME/.cache/latentself/reader-src-XXXXXXXX"')
[[ $remote_dir =~ ^/[a-zA-Z0-9_./-]+$ ]]
git -C "$root" archive "$revision" latentself | ssh -o BatchMode=yes server "tar -xf - -C '$remote_dir'"
ssh -o BatchMode=yes server bash -s -- "$role" "$remote_dir" "$revision" <<'REMOTE'
set -euo pipefail
role=$1 src=$2 revision=$3
name=latentself-$role
if docker inspect "$name" >/dev/null 2>&1; then
    test "$(docker inspect -f '{{index .Config.Labels "latentself.revision"}}' "$name")" = "$revision"
    docker start "$name" >/dev/null
    exit
fi
if ss -H -ltn | awk '$4 ~ /:8003$/ {found=1} END {exit !found}'; then
    echo "Reader requires the existing listener on port 8003 to be stopped first." >&2
    exit 1
fi
case "$role" in
    reader9) model=Qwen3.5-9B-Base; sae=SAE-Res-Qwen3.5-9B-Base-W64K-L0_50
        model_rev=68c46c4b3498877f3ef123c856ecfde50c39f404; sae_rev=7bb2370d360e566b2d8acb8b3d15a0b2b88f52a8; precision=bf16 ;;
    reader27-*) model=Qwen3.5-27B; sae=SAE-Res-Qwen3.5-27B-W80K-L0_50
        model_rev=81a287ca484a5e71148f5beb7f367bb5286d70f9; sae_rev=13d4221569f7ca5d3c1e605e3e3dc95117e4807c
        if [[ $role == reader27-int8 ]]; then precision=int8; else precision=bf16-offload; fi ;;
esac
cache="$HOME/.cache/huggingface/hub"
model_path="models--Qwen--$model/snapshots/$model_rev"
sae_path="models--Qwen--$sae/snapshots/$sae_rev"
test -s "$cache/$model_path/config.json"
test -s "$cache/$sae_path/config.json"
extra=()
baseline=()
public_file="$HOME/.cache/latentself/wildchat-zh-unique.jsonl"
if [[ ! -f $public_file ]]; then public_file="$HOME/.cache/latentself/wildchat-zh.jsonl"; fi
if [[ -f $public_file ]]; then
    extra+=(-v "$public_file:/public.jsonl:ro")
    baseline+=(--baseline /public.jsonl)
fi
if [[ $precision == int8 ]]; then
    quant="$HOME/.cache/latentself/reader-python"
    mkdir -p "$quant"
    if [[ ! -d $quant/bitsandbytes ]]; then
        docker run --rm --log-driver none -v "$quant:/extras" --entrypoint python3 vllm/vllm-openai:v0.34.0 \
            -m pip install --target /extras --no-deps bitsandbytes
    fi
    extra+=(-v "$quant:/extras:ro")
fi
docker run -d --rm --name "$name" --label "latentself.role=$role" --label "latentself.revision=$revision" \
    --gpus all --network host --read-only --tmpfs /tmp:exec --log-driver none \
    -v "$cache:/hub:ro" -v "$src:/app:ro" "${extra[@]}" -e HOME=/tmp -e PYTHONPATH=/app:/extras \
    -e HF_HUB_OFFLINE=1 --entrypoint python3 vllm/vllm-openai:v0.34.0 -B -m latentself.interp.reader \
    --model-path "/hub/$model_path" --sae-path "/hub/$sae_path" --reader "Qwen/$model" \
    --sae-repo "Qwen/$sae" --precision "$precision" "${baseline[@]}" >/dev/null
for ((i=0;i<240;i++)); do
    if curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:8003/health >/dev/null 2>&1; then
        echo "$role ready on loopback"; exit
    fi
    test "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || :)" = true
    sleep 2
done
echo "Reader has not reached readiness" >&2
exit 1
REMOTE
