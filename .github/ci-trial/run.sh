#!/usr/bin/env bash
# One measured pytest run of a unit shard. Appends a JSON line to $RESULTS.
# Usage: run.sh NAME WORKERS CACHE(off|on) ISA_CAP(none|AVX2)
# Reads SHARD_JSON, CACHE_DIR, RESULTS, CPU_MODEL, CPU_HASH, PHASE from the environment.
set -uo pipefail
name=$1 workers=$2 cache=$3 cap=$4

python=$(jq -r .python <<<"$SHARD_JSON")
package=$(jq -r .package <<<"$SHARD_JSON")
extras=$(jq -r .extras <<<"$SHARD_JSON")
label=$(jq -r .label <<<"$SHARD_JSON")
paths=$(jq -r .test_paths <<<"$SHARD_JSON")
args=$(jq -r .pytest_args <<<"$SHARD_JSON")
args="${args/-n auto/-n $workers}"

if [ "$cache" = "on" ]; then
  mkdir -p "$CACHE_DIR"
  export JAX_COMPILATION_CACHE_DIR="$CACHE_DIR"
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
fi
if [ "$cap" != "none" ]; then
  export XLA_FLAGS="--xla_cpu_max_isa=$cap"
fi

entries_before=$(find "$CACHE_DIR" -type f 2>/dev/null | wc -l)
marker=$(mktemp)
log="$RUNNER_TEMP/pytest-$name.log"
( while sleep 1; do free -m | awk '/^Mem:/{m=$3; a=$7} /^Swap:/{s=$3} END{print m, a, s}'; done ) > "$RUNNER_TEMP/mem-$name.log" &
sampler=$!
start=$(date +%s%N)
# shellcheck disable=SC2086
uv run --python "$python" --package "$package" $extras --group test \
  pytest $args --junitxml="$RUNNER_TEMP/junit-$name.xml" $paths > "$log" 2>&1
status=$?
wall_ms=$(( ($(date +%s%N) - start) / 1000000 ))
kill "$sampler" 2>/dev/null
tail -n 25 "$log"

summary=$(grep -E '^=+ .*(passed|failed|error).* in [0-9.]+s' "$log" | tail -1)
writes=$(find "$CACHE_DIR" -type f -newer "$marker" 2>/dev/null | wc -l)
entries_after=$(find "$CACHE_DIR" -type f 2>/dev/null | wc -l)
cache_mb=$(du -sm "$CACHE_DIR" 2>/dev/null | cut -f1)
read_errors=$(grep -c 'Error reading persistent compilation cache' "$log")
oom=$(sudo dmesg | grep -ciE 'out of memory|killed process')
mem="$RUNNER_TEMP/mem-$name.log"
jq -nc \
  --arg phase "$PHASE" --arg run "$name" --arg shard "$label" \
  --arg workers "$workers" --arg cache "$cache" --arg cap "$cap" \
  --arg cpu "$CPU_MODEL" --arg cpu_hash "$CPU_HASH" --arg summary "$summary" \
  --arg cache_hit "${CACHE_HIT:-}" \
  --argjson status "$status" --argjson wall "$wall_ms" \
  --argjson entries_before "$entries_before" --argjson entries_after "$entries_after" \
  --argjson writes "$writes" --argjson cache_mb "${cache_mb:-0}" \
  --argjson read_errors "$read_errors" --argjson oom "$oom" \
  --argjson peak_used_mb "$(awk 'BEGIN{m=0} {if ($1>m) m=$1} END{print m}' "$mem")" \
  --argjson min_avail_mb "$(awk 'BEGIN{m=1e9} {if ($2<m) m=$2} END{print m}' "$mem")" \
  --argjson peak_swap_mb "$(awk 'BEGIN{m=0} {if ($3>m) m=$3} END{print m}' "$mem")" \
  '$ARGS.named | .wall = (.wall / 1000)' | tee -a "$RESULTS"
