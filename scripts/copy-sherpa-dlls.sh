#!/usr/bin/env bash
# Copy sherpa-onnx + onnxruntime runtime DLLs from the Go module cache to a target dir
# (default: repo root, so `go test`/examples find them on PATH/CWD).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
DEST="${1:-$ROOT}"
VER="$(go list -m -f '{{.Version}}' github.com/k2-fsa/sherpa-onnx-go-windows 2>/dev/null || echo v1.13.2)"
GOMODCACHE="$(go env GOMODCACHE)"
LIBDIR="$GOMODCACHE/github.com/k2-fsa/sherpa-onnx-go-windows@$VER/lib/x86_64-pc-windows-gnu"
if [ ! -d "$LIBDIR" ]; then echo "lib dir not found: $LIBDIR (run 'go mod download' first)"; exit 1; fi
mkdir -p "$DEST"
for dll in onnxruntime.dll sherpa-onnx-c-api.dll sherpa-onnx-cxx-api.dll; do
  cp -f "$LIBDIR/$dll" "$DEST/" && echo "copied $dll -> $DEST"
done
