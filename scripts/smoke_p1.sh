#!/usr/bin/env bash
set -euo pipefail
queries="${LATENTSELF_DATA:-$HOME/data}/processed/latentself/eval/queries.yaml"
first=$(/usr/bin/ruby -ryaml -e 'puts YAML.safe_load(File.read(ARGV[0])).fetch("queries").first.fetch("q")' "$queries")
photo=$(/usr/bin/ruby -ryaml -e 'puts YAML.safe_load(File.read(ARGV[0])).fetch("queries").find { |q| q["photo"] }.fetch("q")' "$queries")
lself search "$first" --strict
lself search "$photo" --images --sort score
