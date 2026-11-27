#!/usr/bin/env bash
set -euo pipefail
ssh -o BatchMode=yes server 'bash -s' <<'REMOTE'
set -eu
docker ps -a --format '{{.Names}} {{.State}} {{.Ports}}'
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
systemctl --user is-active llm-panel llm-proxy || :
ss -H -ltn | awk '$4 ~ /:(8000|8001|8002|8003|8010|4100|4101)$/ {print $4}'
REMOTE
