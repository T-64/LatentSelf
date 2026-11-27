#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
    llm) exec ssh -o BatchMode=yes server "bash -ic 'qwen-off'" ;;
    embed|query|text|rerank|llm-private|reader9|reader27-offload|reader27-int8) ;;
    *) echo "usage: $0 embed|query|text|rerank|llm|llm-private" >&2; exit 2 ;;
esac
ssh -o BatchMode=yes server bash -s -- "$1" <<'REMOTE'
set -euo pipefail
name="latentself-$1"
if docker inspect "$name" >/dev/null 2>&1; then
    test "$(docker inspect -f '{{index .Config.Labels "latentself.role"}}' "$name")" = "$1"
    docker stop "$name"
fi
REMOTE
