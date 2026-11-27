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
hf download Qwen/Qwen3-VL-Embedding-8B --revision 2c4565515e0f265c6511776e7193b22c0968ddc7
hf download Qwen/Qwen3-VL-Reranker-8B --revision b212dc8c91a8164aef1ea2de9c1a867611e75c04
REMOTE
