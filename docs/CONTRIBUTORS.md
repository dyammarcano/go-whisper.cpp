# Contributors

## Maintainer

- **Dyam Marcano** ([@dyammarcano](https://github.com/dyammarcano)) — dyam.marcano@gmail.com

## Contributing

This is a cgo binding for [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (a pinned
git submodule) plus a sherpa-onnx-based diarization package. Building requires a C/C++
toolchain; the dev platform is **Windows + MinGW gcc**.

### Setup

```bash
task deps          # init the whisper.cpp submodule
task build:cpu     # build whisper.cpp CPU static libs + the cgo shim (libbinding.a)
task models:tiny   # download the pinned ggml-tiny.en test model
```

### Build backends

| Task | Backend | Requires |
|------|---------|----------|
| `task build:cpu` | CPU static (default) | MinGW gcc, Ninja |
| `task build:cuda` | CUDA (MSVC DLLs, arch 75) | VS2022 + CUDA toolkit (nvcc) |
| `task build:vulkan` | Vulkan (MinGW static) | Vulkan SDK |

> The MinGW builds set `GGML_OPENMP=OFF` (see [ISSUES](ISSUES.md) / [ADR-0001](adr/0001-cgo-binding-over-whisper-cpp.md)).
> The whisper.cpp submodule will show as dirty after building (a MinGW `ggml-cpu.c` sed patch) — **do not stage it.**

### Test

```bash
task test                                   # go test ./... (pure-Go + gated integration skips)
```

Integration tests are env-gated. Native paths required on Windows (not MSYS `/d/...`):

```bash
export TEST_MODEL="$(cygpath -m "$PWD")/models/ggml-tiny.en.bin"   # whisper integration
go test .
```

**Diarization tests need the compiled-exe-beside-DLLs pattern.** A stale
`System32\onnxruntime.dll` shadows the bundled ONNX Runtime and crashes any temp-dir test
binary, so compile the test into the repo root (next to the DLLs copied by `task diarize:dlls`):

```bash
task models:diarize && task diarize:dlls
go test -c -o diarize.test.exe ./diarize/ && ./diarize.test.exe -test.v && rm -f diarize.test.exe
```

For the full whisper suite without the test-reporter hook collapsing output:
`go test -c -o whisper.test.exe . && ./whisper.test.exe -test.v`.

### Lint & format

```bash
task lint   # golangci-lint run --fix ./...  (default: all linters, see .golangci.yml)
task fmt    # go fmt ./...
```

`golangci-lint` runs with `default: all`; CLI/example programs use the `run() error` +
`fmt.Fprintf(os.Stdout, …)` pattern to satisfy `forbidigo`/`gocritic`/`errcheck`.

### Conventions

- **Conventional commits** (`feat`, `fix`, `docs`, `test`, `build`, `style`, `chore`),
  scoped where useful (`feat(stream): …`, `build(diarize): …`). No AI attribution lines.
- **BSD-3-Clause** license (`LICENSE`).
- cgo preamble rule: prose comments must be separated from `#cgo`/`#include` directives by a
  true blank line. C++ shim (`binding.cpp`) is compiled outside cgo by `scripts/binding.sh`.
- Update [CHANGELOG.md](../CHANGELOG.md) under `[Unreleased]` with user-facing changes.
- Architecture decisions go in [docs/adr/](adr/).

CI (GitHub Actions) currently **disabled** at the repo level; it build-verifies the backend
build tags when enabled but does not run the audio-gated integration tests.
