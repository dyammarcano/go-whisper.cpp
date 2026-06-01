#!/usr/bin/env bash
# Build whisper.cpp static libs (CPU or Vulkan). Output: whisper.cpp/build-<backend>/
set -euo pipefail
BACKEND="${1:-cpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
SRC="whisper.cpp"; BUILD="$SRC/build-$BACKEND"

EXTRA=""
[ "$BACKEND" = "vulkan" ] && EXTRA="-DGGML_VULKAN=ON"
# Prefer Ninja (fast; installed via scoop on the dev box, and the toolchain has no
# mingw32-make). Fall back to a platform make generator only if ninja is absent.
if command -v ninja >/dev/null 2>&1; then
  GEN="Ninja"
else
  case "$(uname -s)" in MINGW*|MSYS*) GEN="MinGW Makefiles";; *) GEN="Unix Makefiles";; esac
fi

# MinGW workaround (same as go-llama): gate THREAD_POWER_THROTTLING_STATE.
WINVER=""
case "$(uname -s)" in
  MINGW*|MSYS*)
    WINVER="-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00 -DNTDDI_VERSION=0x0A000007 -DGGML_NO_THREAD_POWER_THROTTLING"
    CPUC="$SRC/ggml/src/ggml-cpu/ggml-cpu.c"
    if [ -f "$CPUC" ] && ! grep -q "GGML_NO_THREAD_POWER_THROTTLING" "$CPUC"; then
      sed -i 's/#if _WIN32_WINNT >= 0x0602$/#if _WIN32_WINNT >= 0x0602 \&\& !defined(GGML_NO_THREAD_POWER_THROTTLING)/' "$CPUC"
    fi
  ;;
esac

rm -rf "$BUILD"
cmake -S "$SRC" -B "$BUILD" -G "$GEN" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF \
  -DWHISPER_SDL2=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF $EXTRA \
  ${WINVER:+-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_FLAGS="$WINVER" -DCMAKE_CXX_FLAGS="$WINVER"}
cmake --build "$BUILD" -j
echo "=== $BACKEND static libs ==="; find "$BUILD" -name '*.a' | sort
