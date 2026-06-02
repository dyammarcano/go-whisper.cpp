# ADR-0001: cgo binding over whisper.cpp via a thin C shim

- **Status:** Accepted
- **Date:** 2026-06-01

## Context

We need Go speech-to-text backed by [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
whisper.cpp is C/C++ with a C API and GPU backends (CUDA, Vulkan, Metal) through ggml.
Options considered: (a) shell out to the `whisper` CLI, (b) reimplement inference in Go,
(c) cgo bind the C API directly. (a) is fragile and slow per-call; (b) is infeasible
(would not track upstream or reuse ggml's optimized/GPU kernels).

whisper's by-value structs (`whisper_full_params`) and callback function pointers don't map
cleanly across cgo, and the C++ pieces must be compiled outside cgo's C compiler.

## Decision

Bind the C API with cgo through a **thin C shim** (`binding.cpp` / `binding.h`):

- The shim owns construction of `whisper_full_params`/`whisper_context_params` (by-value
  structs never cross cgo) and hosts the callback trampolines that call exported Go funcs.
- `binding.cpp` is compiled **outside cgo** by `scripts/binding.sh` into `libbinding.a`.
- whisper.cpp is a **pinned git submodule** (v1.7.4), built into `build-<backend>/` static
  libs (or MSVC DLLs for CUDA).
- Concurrency uses a **Model + Session split**: one read-only `whisper_context` shared across
  N per-goroutine `whisper_state`s via `whisper_full_with_state`.
- GPU backends are **opt-in build tags**, each with its own `link_*.go` LDFLAGS file.
- Callbacks/cancellation use `runtime/cgo.Handle` trampolines; the abort flag is atomic.

## Consequences

- **+** Reuses whisper.cpp's optimized CPU + GPU kernels; tracks upstream by bumping the submodule.
- **+** True parallel inference across goroutines (Session per goroutine).
- **+** Clean separation: only the shim touches cgo internals.
- **−** Requires a C/C++ toolchain to build (MinGW on Windows); not `go get`-and-go.
- **−** Platform/toolchain-specific gotchas, e.g. **`GGML_OPENMP=OFF` on MinGW** — aborting an
  in-flight ggml compute via the cgo `abort_callback` tore down a libgomp parallel region and
  faulted (`0xC0000005`). ggml's native Win32 threadpool aborts cleanly. See [ISSUES](../ISSUES.md).
- **−** The submodule shows dirty after building (a MinGW `ggml-cpu.c` sed patch); the parent
  repo must never stage it.
