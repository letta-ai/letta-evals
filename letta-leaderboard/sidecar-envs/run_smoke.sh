#!/usr/bin/env bash
set -uo pipefail

if [[ -z "${LETTA_API_KEY:-}" ]]; then
  echo "LETTA_API_KEY is not set"
  exit 1
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
results_dir="$root_dir/results"
mkdir -p "$results_dir"

failed=0
for suite in "$root_dir"/*/suite_smoke.yaml; do
  env="$(basename "$(dirname "$suite")")"
  name="$(basename "$suite" .yaml)"
  out="$results_dir/${env}-${name}"
  echo "Running $suite -> $out"
  if ! uv run letta-evals run "$suite" --output "$out"; then
    failed=1
  fi
done

exit $failed
