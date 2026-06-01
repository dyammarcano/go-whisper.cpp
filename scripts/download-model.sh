#!/usr/bin/env bash
# Download a pinned ggml whisper model with SHA256 verification into ./models/.
set -euo pipefail
NAME="${1:-tiny.en}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p models
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${NAME}.bin"
OUT="models/ggml-${NAME}.bin"
# Pinned SHA256 for ggml-tiny.en.bin (update if NAME changes).
declare -A SHA=( ["tiny.en"]="921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f" )
if [ -f "$OUT" ]; then echo "exists: $OUT"; else
  echo "downloading $URL"; curl -fL "$URL" -o "$OUT"
fi
if [ -n "${SHA[$NAME]:-}" ]; then
  echo "${SHA[$NAME]}  $OUT" | sha256sum -c - || { echo "checksum FAILED"; rm -f "$OUT"; exit 1; }
fi
echo "model ready: $OUT"
echo "export TEST_MODEL=$ROOT/$OUT"
