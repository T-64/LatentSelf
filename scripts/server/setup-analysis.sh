#!/usr/bin/env bash
# Install analysis dependencies on the Linux server in a project-scoped environment.
set -euo pipefail
ssh -o BatchMode=yes server bash -s <<'REMOTE'
set -euo pipefail
env_dir="$HOME/.cache/latentself/analysis-venv"
if [[ ! -x $env_dir/bin/python ]]; then
    python3 -m venv "$env_dir"
fi
"$env_dir/bin/python" -m pip install numpy==2.5.5 igraph==1.0.0 leidenalg==0.12.0 \
    pyarrow==26.0.1 huggingface_hub==2.5.0 xlrd==2.0.2
REMOTE
