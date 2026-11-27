#!/usr/bin/env bash
# Run on the GPU host. Download only the reader and the five experiment layers per model.
set -euo pipefail
if [[ $(uname -s) != Linux ]]; then
    echo "Run this script on the Linux inference server." >&2
    exit 2
fi
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
fetch() {
    local repo=$1 revision=$2
    shift 2
    local options=()
    for pattern in "$@"; do options+=(--include "$pattern"); done
    for attempt in 1 2 3; do
        if hf download "$repo" --revision "$revision" --max-workers 3 "${options[@]}"; then return; fi
        sleep "$((10 * attempt))"
    done
    return 1
}
configs=('*.json' '*.txt' '*.jinja' 'tokenizer.model' '*.safetensors' 'LICENSE*')
fetch Qwen/Qwen3.5-9B-Base 68c46c4b3498877f3ef123c856ecfde50c39f404 "${configs[@]}"
fetch Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50 7bb2370d360e566b2d8acb8b3d15a0b2b88f52a8 'config.json' 'LICENSE*' layer{8,12,16,20,24}.sae.pt
fetch Qwen/Qwen3.5-27B 81a287ca484a5e71148f5beb7f367bb5286d70f9 "${configs[@]}"
fetch Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 13d4221569f7ca5d3c1e605e3e3dc95117e4807c 'config.json' 'LICENSE*' layer{16,24,32,40,48}.sae.pt
