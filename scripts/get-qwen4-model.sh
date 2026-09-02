#!/usr/bin/env bash
set -euo pipefail

REVISION=a25519c666383714aad56d7916060aa9b315dc0e
MODEL_DIR=${QWEN4_MODEL_DIR:-"${XDG_CACHE_HOME:-$HOME/.cache}/qwen3.8-flash-next-in-c/model/UD-IQ1_S"}
BASE=Qwen3.8-Flash-Next-UD-IQ1_S
FILES=(
  "$BASE-00001-of-00003.gguf"
  "$BASE-00002-of-00003.gguf"
  "$BASE-00003-of-00003.gguf"
)
SIZES=(10946624 49990818368 22544696352)
HASHES=(
  88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd
  3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6
  0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a
)

command -v curl >/dev/null 2>&1 || { echo "qwen4: curl is required" >&2; exit 1; }
if command -v sha256sum >/dev/null 2>&1; then
  hash_file() { sha256sum "$1" | awk '{ print $1 }'; }
elif command -v shasum >/dev/null 2>&1; then
  hash_file() { shasum -a 256 "$1" | awk '{ print $1 }'; }
else
  echo "qwen4: sha256sum or shasum is required" >&2
  exit 1
fi
mkdir -p "$MODEL_DIR"

download_resume() {
  local output=$1 expected=$2 primary=$3 fallback=$4
  local failures=0
  while :; do
    local current=0
    [[ -f $output ]] && current=$(wc -c < "$output")
    (( current == expected )) && return 0
    (( current > expected )) && return 1
    local url=$primary
    (( failures >= 4 )) && url=$fallback
    if curl -L --fail --connect-timeout 30 --speed-limit 1024 --speed-time 30 \
         -C - -o "$output" "$url"; then
      failures=0
    else
      failures=$((failures + 1))
      sleep 2
    fi
  done
}

remaining=0
for i in 0 1 2; do
  path="$MODEL_DIR/${FILES[$i]}"
  current=0
  [[ -f "$path" ]] && current=$(wc -c < "$path")
  if (( current > SIZES[i] )); then
    echo "qwen4: ${FILES[$i]} is larger than expected; move it aside and retry" >&2
    exit 1
  fi
  remaining=$((remaining + SIZES[i] - current))
done
available_kib=$(df -Pk "$MODEL_DIR" | awk 'END { print $4 }')
if (( available_kib * 1024 < remaining + 1073741824 )); then
  echo "qwen4: not enough disk space; need the remaining model plus 1 GiB reserve" >&2
  exit 1
fi

for i in 0 1 2; do
  file=${FILES[$i]}
  path="$MODEL_DIR/$file"
  marker="$path.${HASHES[$i]}.ok"
  if [[ -f "$marker" && $(wc -c < "$path") -eq ${SIZES[$i]} ]]; then
    continue
  fi
  if [[ ! -f "$path" || $(wc -c < "$path") -lt ${SIZES[$i]} ]]; then
    ms="https://www.modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/$REVISION/UD-IQ1_S/$file"
    hf="https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-IQ1_S/$file"
    echo "qwen4: downloading $file" >&2
    download_resume "$path" "${SIZES[$i]}" "$ms" "$hf" || {
      echo "qwen4: unable to resume $file" >&2; exit 1;
    }
  fi
  actual_size=$(wc -c < "$path")
  [[ $actual_size -eq ${SIZES[$i]} ]] || { echo "qwen4: size mismatch for $file" >&2; exit 1; }
  actual_hash=$(hash_file "$path")
  [[ $actual_hash == ${HASHES[$i]} ]] || { echo "qwen4: SHA-256 mismatch for $file" >&2; exit 1; }
  : > "$marker"
done

printf '%s\n' "$MODEL_DIR/${FILES[0]}"
