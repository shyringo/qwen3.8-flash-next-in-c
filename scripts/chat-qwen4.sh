#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

if jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null); then :; else jobs=4; fi
case "$jobs" in ''|*[!0-9]*|0) jobs=4 ;; esac
(( jobs > 11 )) && jobs=11
export OMP_NUM_THREADS=${QWEN4_THREADS:-$jobs}
export OMP_DYNAMIC=${OMP_DYNAMIC:-false}
if [[ -z ${OMP_WAIT_POLICY:-} ]]; then
  one_shot=0
  for argument in "$@"; do
    [[ $argument == --prompt ]] && one_shot=1
  done
  if (( one_shot )); then export OMP_WAIT_POLICY=ACTIVE; else export OMP_WAIT_POLICY=PASSIVE; fi
fi
make -s -C "$ROOT" -j"$jobs" bin/qwen4
if [[ $# -eq 1 && $1 == --help ]]; then
  exec "$ROOT/bin/qwen4" --help
fi
if [[ -n ${QWEN4_MODEL:-} ]]; then
  MODEL=$QWEN4_MODEL
else
  MODEL=$("$SCRIPT_DIR/get-qwen4-model.sh")
fi
[[ -r "$MODEL" ]] || { echo "qwen4: model is not readable: $MODEL" >&2; exit 1; }
exec "$ROOT/bin/qwen4" --model "$MODEL" --context "${QWEN4_CONTEXT:-8192}" "$@"
