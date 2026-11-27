#!/usr/bin/env bash
set -euo pipefail
queries="${LATENTSELF_DATA:-$HOME/data}/processed/latentself/eval/queries.yaml"
exec /usr/bin/ruby -ryaml -rjson -ropen3 - "$queries" <<'RUBY'
def check(query, mode, *flags)
  output, status = Open3.capture2("lself", "search", query, *flags, "--json")
  abort "search exited #{status.exitstatus}" unless status.success?
  result = JSON.parse(output)
  rows = result.fetch("results")
  abort "#{mode} degraded: #{result.fetch("notices").join(" ")}" unless result["notices"].empty? && result["mode"] == mode
  abort "#{mode} returned no results" if rows.empty?
  field = mode == "strict" ? "reason" : "images"
  abort "missing #{field}" unless rows.all? { |row| row[field] && !row[field].empty? }
  puts "#{mode}: #{rows.size} conversations"
end

queries = YAML.safe_load(File.read(ARGV.fetch(0))).fetch("queries")
check(queries.first.fetch("q"), "strict", "--strict")
check(queries.find { |q| q["photo"] }.fetch("q"), "fast", "--images", "--sort", "score")
RUBY
