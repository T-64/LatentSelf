#!/usr/bin/env bash
# Downloads run on the GPU host. HF snapshots resume partial transfers.
set -euo pipefail
case "${1:-hub}" in
    hub|mirror) endpoint="${1:-hub}" ;;
    *) echo "usage: $0 [hub|mirror]" >&2; exit 2 ;;
esac
ssh -o BatchMode=yes server bash -s -- "$endpoint" <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if [[ $1 == mirror ]]; then export HF_ENDPOINT=https://hf-mirror.com; fi
fetch() {
    hf download "$1" --revision "$2" ||
        HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=90 hf download "$1" --revision "$2"
}
fetch Qwen/Qwen3-VL-Embedding-8B 2c4565515e0f265c6511776e7193b22c0968ddc7
fetch Qwen/Qwen3-VL-Reranker-8B b212dc8c91a8164aef1ea2de9c1a867611e75c04
fetch Qwen/Qwen3-Embedding-8B 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af
REMOTE
