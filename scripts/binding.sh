#!/usr/bin/env bash
# Compile binding.cpp -> libbinding.a (binding.o only; whisper libs linked by cgo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CXX="${CXX:-g++}"
WINVER=""
case "$(uname -s)" in
  MINGW*|MSYS*) WINVER="-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00 -DNTDDI_VERSION=0x0A000007 -DGGML_NO_THREAD_POWER_THROTTLING";;
esac
"$CXX" -std=c++17 -O3 $WINVER \
  -I whisper.cpp/include -I whisper.cpp/ggml/include \
  -c binding.cpp -o binding.o
rm -f libbinding.a
ar rcs libbinding.a binding.o
echo "built libbinding.a"
