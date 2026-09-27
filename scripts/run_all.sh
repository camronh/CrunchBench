#!/usr/bin/env bash
# Run every model profile in ezvals.json, one EZVals run per model, then build the report.
#
#   scripts/run_all.sh v3            # session name
#   scripts/run_all.sh v3 --limit 5  # extra args go to `ezvals run`
#
# Each profile's "concurrency" (default 1) sets how many cases run at once.
# Expensive models use 1 so questions at the same context size hit the prompt
# cache. Parallel calls all miss it and pay the cache-write premium.
set -euo pipefail
cd "$(dirname "$0")/.."

session="${1:?usage: scripts/run_all.sh <session> [ezvals run args...]}"
shift

profiles=$(uv run python -c '
import json
for name, p in json.load(open("ezvals.json"))["configs"].items():
    print(name, p.get("concurrency", 1))
')
while read -r profile concurrency; do
  echo "=== $profile (concurrency $concurrency)"
  uv run ezvals run evals/ --config "$profile" --session "$session" -c "$concurrency" "$@" < /dev/null
done <<< "$profiles"
uv run scripts/report.py --session "$session"
