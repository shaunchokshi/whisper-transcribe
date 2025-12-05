#!/usr/bin/env bash
set -euo pipefail

#############################
# Color definitions
#############################

REDTEXT="\033[1;31m"
GREENTEXT="\033[1;32m"
NOCOLOR="\033[0m"
RED="31"
GREEN="32"
YELLOW="33"
BOLDGREEN="\e[1;${GREEN}m"
ITALICRED="\e[3;${RED}m"
BOLDRED="\e[1;${RED}m"
ENDCOLOR="\e[0m"
BOLDCYAN="\e[1;36m"
CYAN="\e[0;36m"
YELLOWTEXT="\e[0;33m"
BOLDYELLOW="\e[1;33m"
PURPLETEXT="\e[0;35m"
BOLDPURPLE="\e[1;35m"

info()  { printf "${BOLDCYAN}[INFO]${ENDCOLOR} %s\n" "$*"; }
warn()  { printf "${BOLDYELLOW}[WARN]${ENDCOLOR} %s\n" "$*"; }
error() { printf "${BOLDRED}[ERROR]${ENDCOLOR} %s\n" "$*"; }
ok()    { printf "${BOLDGREEN}[OK]${ENDCOLOR} %s\n" "$*"; }

#############################
# Intro
#############################

printf "${BOLDCYAN}ROCm Whisper Transcription Launcher${ENDCOLOR}\n"
printf "${PURPLETEXT}GPU-accelerated transcription with Markdown + ISO timestamps${ENDCOLOR}\n\n"

#############################
# Path selection
#############################

SEARCH_PATH="$(pwd)"
read -e -i "$SEARCH_PATH" -p "$(printf "${BOLDCYAN}Enter path to file or directory to transcribe:${ENDCOLOR} ")" SEARCH_PATH

if [ ! -e "$SEARCH_PATH" ]; then
  error "Path does not exist: $SEARCH_PATH"
  exit 1
fi

mkdir -p ./data  # not strictly required, but harmless

if [ -f "$SEARCH_PATH" ]; then
  HOST_DIR="$(cd "$(dirname "$SEARCH_PATH")" && pwd)"
  INPUT_BASENAME="$(basename "$SEARCH_PATH")"
  INPUT_IN_CONTAINER="/data/$INPUT_BASENAME"
  info "Input is a single file."
else
  HOST_DIR="$(cd "$SEARCH_PATH" && pwd)"
  INPUT_BASENAME=""
  INPUT_IN_CONTAINER="/data"
  info "Input is a directory; will scan for media files."
fi

info "Host directory: $HOST_DIR"
info "Container input path: $INPUT_IN_CONTAINER"

#############################
# Model selection
#############################

MODEL_DEFAULT="medium"
printf "\n"
read -e -i "$MODEL_DEFAULT" -p "$(printf "${BOLDCYAN}Whisper model (tiny, base, small, small.en, medium, large, etc.) [default: medium]: ${ENDCOLOR}")" MODEL
MODEL="${MODEL:-$MODEL_DEFAULT}"

#############################
# ISO date for Markdown
#############################

MD_DATE_DEFAULT=""
printf "\n"
read -e -i "$MD_DATE_DEFAULT" -p "$(printf "${BOLDCYAN}ISO date for Markdown timestamps (YYYY-MM-DD, leave blank for time-only): ${ENDCOLOR}")" MD_DATE

#############################
# Glob pattern
#############################

# Default to *.mkv because that’s your common case; blank = all supported types
GLOB_DEFAULT="*.mkv"
printf "\n"
read -e -i "$GLOB_DEFAULT" -p "$(printf "${BOLDCYAN}Glob pattern for files (e.g. *.mkv, leave blank for all supported): ${ENDCOLOR}")" GLOB_PATTERN

#############################
# Language selection
#############################

# Default to en; if user clears the field, we skip --language and let Whisper auto-detect.
LANG_DEFAULT="en"
printf "\n"
read -e -i "$LANG_DEFAULT" -p "$(printf "${BOLDCYAN}Language of content (e.g. de; default is en; clear for auto-detect): ${ENDCOLOR}")" LANGUAGE

#############################
# Build extra args
#############################

EXTRA_ARGS=()

if [ -n "$MD_DATE" ]; then
  EXTRA_ARGS+=(--md_date "$MD_DATE")
fi
if [ -n "$GLOB_PATTERN" ]; then
  EXTRA_ARGS+=(--glob "$GLOB_PATTERN")
fi
if [ -n "$LANGUAGE" ]; then
  EXTRA_ARGS+=(--language "$LANGUAGE")
fi

#############################
# Persistent Whisper cache
#############################

CACHE_DIR="${HOME}/.cache/whisper"
mkdir -p "$CACHE_DIR"
ok "Using persistent Whisper cache at: $CACHE_DIR"

#############################
# Summary
#############################

printf "\n${BOLDPURPLE}===== RUN SUMMARY =====${ENDCOLOR}\n"
printf "${BOLDYELLOW}Host dir:        ${ENDCOLOR}%s\n" "$HOST_DIR"
printf "${BOLDYELLOW}Input (container):${ENDCOLOR} %s\n" "$INPUT_IN_CONTAINER"
printf "${BOLDYELLOW}Model:           ${ENDCOLOR}%s\n" "$MODEL"
if [ -n "$MD_DATE" ]; then
  printf "${BOLDYELLOW}Markdown date:   ${ENDCOLOR}%s (timestamps like %sT00:00:00Z)\n" "$MD_DATE" "$MD_DATE"
else
  printf "${BOLDYELLOW}Markdown date:   ${ENDCOLOR}time-only (HH:MM:SS)\n"
fi
if [ -n "$GLOB_PATTERN" ]; then
  printf "${BOLDYELLOW}Glob pattern:    ${ENDCOLOR}%s\n" "$GLOB_PATTERN"
else
  printf "${BOLDYELLOW}Glob pattern:    ${ENDCOLOR}ALL supported media types\n"
fi
if [ -n "$LANGUAGE" ]; then
  printf "${BOLDYELLOW}Language:        ${ENDCOLOR}%s (forced)\n" "$LANGUAGE"
else
  printf "${BOLDYELLOW}Language:        ${ENDCOLOR}auto-detect\n"
fi
printf "${BOLDYELLOW}GPU:             ${ENDCOLOR}enabled (ROCm)\n"
printf "${BOLDYELLOW}Image:           ${ENDCOLOR}rocm-whisper:latest\n"
printf "${BOLDYELLOW}Cache:           ${ENDCOLOR}%s\n" "$CACHE_DIR"
printf "${BOLDPURPLE}=========================${ENDCOLOR}\n\n"

read -p "$(printf "${BOLDCYAN}Proceed with transcription? [Y/n]: ${ENDCOLOR}")" CONFIRM
CONFIRM="${CONFIRM:-Y}"
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
  warn "Aborting per user request."
  exit 0
fi

#############################
# Run container
#############################

info "Starting ROCm Whisper container..."

docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  -e HIP_VISIBLE_DEVICES=0 \
  -v "$HOST_DIR":/data \
  -v "$CACHE_DIR":/root/.cache/whisper \
  rocm-whisper:latest \
  python /app/transcribe.py \
    --gpu \
    --model "$MODEL" \
    --input "$INPUT_IN_CONTAINER" \
    --out_dir /data/out \
    --verbose_json \
    "${EXTRA_ARGS[@]}"

RUN_STATUS=$?

if [ $RUN_STATUS -eq 0 ]; then
  ok "Transcription run completed."
  info "Check the 'out' subdirectory under: $HOST_DIR"
else
  error "Transcription run exited with status $RUN_STATUS"
fi
