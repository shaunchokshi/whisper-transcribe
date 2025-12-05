#!/usr/bin/env bash
set -euo pipefail

SEARCH_PATH="$(pwd)"
read -e -i "$SEARCH_PATH" -p "Enter path to search/replace in:" SEARCH_PATH

mkdir -p ./data
if [ -f "$SEARCH_PATH" ]; then
  HOST_DIR="$(cd "$(dirname "$SEARCH_PATH")" && pwd)"
  INPUT_BASENAME="$(basename "$SEARCH_PATH")"
  INPUT_IN_CONTAINER="/data/$INPUT_BASENAME"
else
  HOST_DIR="$(cd "$SEARCH_PATH" && pwd)"
  INPUT_BASENAME=""
  INPUT_IN_CONTAINER="/data"
fi

echo "[info] host_dir=$HOST_DIR"
echo "[info] input_in_container=$INPUT_IN_CONTAINER"

MD_DATE_DEFAULT=""
read -e -i "$MD_DATE_DEFAULT" -p "Enter ISO date for markdown timestamps (YYYY-MM-DD, leave blank for time-only): " MD_DATE

GLOB_DEFAULT=""
read -e -i "$GLOB_DEFAULT" -p "Enter glob pattern for files (e.g. *.mkv, leave blank for all supported): " GLOB_PATTERN

docker compose build whisper

EXTRA_ARGS=()
if [ -n "$MD_DATE" ]; then
  EXTRA_ARGS+=(--md_date "$MD_DATE")
fi
if [ -n "$GLOB_PATTERN" ]; then
  EXTRA_ARGS+=(--glob "$GLOB_PATTERN")
fi

docker run --rm -it   --device=/dev/kfd --device=/dev/dri   --group-add video   -e HIP_VISIBLE_DEVICES=0   -e HSA_OVERRIDE_GFX_VERSION=10.3.0   -v "$HOST_DIR":/data   rocm-whisper:latest   python /app/transcribe.py --gpu --model medium --input "$INPUT_IN_CONTAINER" --out_dir /data/out --verbose_json "${EXTRA_ARGS[@]}"
