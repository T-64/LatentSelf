#!/usr/bin/env bash
# Run one resumable command on the Mac and keep its private output with the data.
set -euo pipefail
case "${1:-}" in
    enrich|embed|eval) task=$1; cmd=(lself "$@") ;;
    fetch-models) task=$1; shift; cmd=("$(dirname "$0")/server/fetch-models.sh" "$@") ;;
    *) echo "usage: $0 enrich|embed|eval|fetch-models [arguments ...]" >&2; exit 2 ;;
esac
umask 077
dir="${LATENTSELF_DATA:-$HOME/data}/processed/latentself/logs"
mkdir -p "$dir"
log="$dir/$task-$(date +%Y%m%dT%H%M%S)-$$.log"
set -C
PYTHONUNBUFFERED=1 nohup "${cmd[@]}" >"$log" 2>&1 </dev/null &
echo "pid=$! log=$log"
