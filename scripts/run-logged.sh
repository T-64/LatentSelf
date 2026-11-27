#!/usr/bin/env bash
# Run one resumable command on the Mac and keep its private output with the data.
set -euo pipefail
case "${1:-}" in
    enrich|embed|eval) task=$1; cmd=("$(command -v lself)" "$@") ;;
    fetch-models) task=$1; shift; cmd=("$(cd "$(dirname "$0")" && pwd)/server/fetch-models.sh" "$@") ;;
    *) echo "usage: $0 enrich|embed|eval|fetch-models [arguments ...]" >&2; exit 2 ;;
esac
umask 077
dir="${LATENTSELF_DATA:-$HOME/data}/processed/latentself/logs"
mkdir -p "$dir"
log="$dir/$task-$(date +%Y%m%dT%H%M%S)-$$.log"
PYTHONUNBUFFERED=1 python3 - "$log" "${cmd[@]}" <<'PY'
import subprocess
import sys

with open(sys.argv[1], "x") as log:
    worker = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True)
print(f"pid={worker.pid} log={log.name}")
PY
