#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
MODEL=${QWEN4_MODEL:-$($SCRIPT_DIR/get-qwen4-model.sh)}
jobs=${QWEN4_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '4')}
port=${QWEN4_BENCH_PORT:-18080}
(( jobs > 11 )) && jobs=11
export OMP_NUM_THREADS=$jobs
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=ACTIVE
make -s -C "$ROOT" -j"$jobs" bin/qwen4
"$ROOT/bin/qwen4" --model "$MODEL" --server "$port" --context 2048 \
  --max-tokens 16 --no-thinking &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ready=0
attempt=0
while (( attempt < 120 )); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "qwen4 benchmark: server exited before becoming ready" >&2
    exit 1
  fi
  sleep 1
  attempt=$((attempt + 1))
done
(( ready )) || { echo "qwen4 benchmark: server did not become ready" >&2; exit 1; }

echo "qwen4 benchmark: warming model pages" >&2
curl -fsS "http://127.0.0.1:$port/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-in-c","messages":[{"role":"user","content":"List the numbers from one through thirty, separated by spaces."}],"max_tokens":16,"temperature":0}' \
  >/dev/null

echo "qwen4 benchmark: measured resident request" >&2
curl -fsS "http://127.0.0.1:$port/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-in-c","messages":[{"role":"user","content":"Write the numbers one through thirty, separated by spaces."}],"max_tokens":16,"temperature":0}'
: # keep the JSON response and shell prompt on separate lines
printf '\n'
