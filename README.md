# go-whisper.cpp

Go bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (speech-to-text), mirroring the design of go-llama.cpp.

Module: `github.com/dyammarcano/go-whisper.cpp`

## Introduction

`go-whisper.cpp` is a thin cgo binding over whisper.cpp's C API via a small C shim (`binding.c`/`binding.h`). It exposes two types for concurrent use:

- **Model** — wraps a `whisper_context` (the loaded model weights). Safe to share across goroutines.
- **Session** — wraps a `whisper_state` (per-goroutine inference state). Create one per goroutine via `Model.NewSession`.

The package also includes a **pure-Go WAV reader** (`wav/`) with no cgo dependency.

License: BSD-3-Clause.

## Clone

```sh
git clone --recurse-submodules https://github.com/dyammarcano/go-whisper.cpp
```

If you cloned without `--recurse-submodules`:

```sh
git submodule update --init --recursive
# or
task deps
```

## Build

Install prerequisites first:

- **Windows**: MinGW-w64 (gcc/g++), cmake, ninja — e.g. via [Scoop](https://scoop.sh): `scoop install mingw cmake ninja`
- **macOS/Linux**: system compiler (clang/gcc), cmake, ninja

Then:

```sh
task deps        # initialises the whisper.cpp submodule
task build:cpu   # builds whisper.cpp static libs via cmake+Ninja and compiles libbinding.a
```

This produces `libbinding.a` in the repo root, which cgo links automatically.

CUDA, Vulkan, and Metal backends are supported — see [GPU backends](#gpu-backends) below.

## Get a model

```sh
task models:tiny   # downloads ggml-tiny.en.bin (SHA256-checked) into ./models/
```

For other model sizes and languages, see the [ggerganov/whisper.cpp HuggingFace collection](https://huggingface.co/ggerganov/whisper.cpp).

## Quickstart

```go
package main

import (
	"context"
	"fmt"
	"log"

	whisper "github.com/dyammarcano/go-whisper.cpp"
	"github.com/dyammarcano/go-whisper.cpp/wav"
)

func main() {
	// Load the model (shared; safe across goroutines).
	m, err := whisper.New("models/ggml-tiny.en.bin")
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()

	// Decode a 16 kHz mono WAV file.
	samples, err := wav.ReadFile("audio.wav")
	if err != nil {
		log.Fatal(err)
	}

	// Transcribe on the model directly (mutex-serialised; fine for CLI use).
	res, err := m.Transcribe(context.Background(), samples,
		whisper.WithLanguage("en"),
		whisper.WithBeamSearch(5),
		whisper.WithSegmentCallback(func(seg whisper.Segment) {
			fmt.Printf("[%s --> %s] %s\n", seg.Start, seg.End, seg.Text)
		}),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Detected language:", res.Language)
	for _, seg := range res.Segments {
		fmt.Printf("%s --> %s : %s\n", seg.Start, seg.End, seg.Text)
	}
}
```

### Concurrent transcription with Session

For parallel workloads, create one `Session` per goroutine:

```go
s, err := m.NewSession()
if err != nil {
    log.Fatal(err)
}
defer s.Close()

res, err := s.Transcribe(ctx, samples, whisper.WithLanguage("en"))
```

## Audio input (`wav` package)

`wav.ReadFile(path)` and `wav.ReadWAV(r)` decode a RIFF/WAVE file to **16 kHz mono float32** samples in `[-1, 1]` — exactly what whisper.cpp expects.

If the file is not 16 kHz, `wav.ErrNot16kHz` is returned. Resample first:

```sh
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

The `wav` package is pure Go with no cgo dependency.

## Speaker diarization (`diarize` package)

The `diarize` package answers **"who spoke when"** — natively, with **no Python**. It wraps
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) running the **pyannote segmentation-3.0**
(MIT-licensed) + **WeSpeaker** embedding models through ONNX Runtime. It is **CPU-only** (GPU
diarization is deferred) and depends only on `sherpa-onnx-go` — never on the whisper.cpp binding,
so the two cgo deps stay decoupled. Input is the same **16 kHz mono float32** as whisper.

### Provision the models + runtime DLLs

```sh
task models:diarize   # downloads pyannote seg-3.0 (MIT) + WeSpeaker embedding into ./models/
task diarize:dlls     # copies onnxruntime.dll + sherpa-onnx-c-api.dll + sherpa-onnx-cxx-api.dll
```

`task diarize:dlls` copies the three runtime DLLs from the Go module cache to the repo root. The
**DLLs must sit beside the executable** you run (Go searches the exe's directory before `System32`).

### Standalone diarization

```go
d, err := diarize.New(
    "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    "models/wespeaker_en_voxceleb_resnet34_LM.onnx",
    diarize.WithThreshold(0.5), // unknown speaker count
)
if err != nil {
    log.Fatal(err)
}
defer d.Close()

// samples is 16 kHz mono float32 (use wav.ReadFile or sherpa.ReadWave).
turns, err := d.Diarize(samples)
if err != nil {
    log.Fatal(err)
}
for _, t := range turns {
    fmt.Printf("[%s -> %s] speaker %d\n", t.Start, t.End, t.Speaker)
}
```

Each `Turn` is `{Start, End time.Duration; Speaker int}`. Choose the clustering strategy:

- **`diarize.WithNumSpeakers(n)`** — force exactly `n` speakers (use when the count is known).
- **`diarize.WithThreshold(t)`** — auto-detect the count (larger `t` => fewer speakers; default 0.5).

These are mutually exclusive — whichever is applied last wins. Other options:
`WithMinDuration(on, off)`, `WithThreads(n)`, `WithDebug`.

> **Speaker IDs are per-file** — a 0-based id is stable within one recording but is **not**
> comparable across different recordings.

### Labeling a transcript (whisper + diarize)

`diarize.Label` assigns each transcript segment the speaker whose turn has the greatest temporal
overlap (`-1` if none). It operates on the package-local `diarize.Segment` type so `diarize` never
imports the whisper binding — callers map `whisper.Segment` -> `diarize.Segment` in two lines:

```go
segs := make([]diarize.Segment, len(res.Segments))
for i, s := range res.Segments {
    segs[i] = diarize.Segment{Start: s.Start, End: s.End, Text: s.Text}
}
for _, ls := range diarize.Label(segs, turns) {
    fmt.Printf("[Speaker %d] %s\n", ls.Speaker, ls.Text)
}
```

See [`examples/diarize`](examples/diarize) (standalone) and
[`examples/transcribe-diarize`](examples/transcribe-diarize) (whisper transcript + speaker labels).
The combo example links **both** whisper.cpp and ONNX Runtime, so it needs `task build:cpu` (whisper
static libs) **and** `task diarize:dlls` (the sherpa DLLs) plus both model sets.

### Running the diarize tests (Windows)

A stale `C:\Windows\System32\onnxruntime.dll` (an older ONNX Runtime) can **shadow** the bundled
1.24.x and crash any `diarize` test binary with *"The requested API version is not available"*.
`go test ./diarize/` builds the test exe in a temp dir with no DLLs beside it, so `System32` wins.
Instead, compile the test exe **into the repo root next to the DLLs** and run it there:

```sh
task diarize:dlls
go test -c -o diarize.test.exe ./diarize/
./diarize.test.exe -test.run TestLabel -test.v
rm -f diarize.test.exe
```

The integration test (`TestDiarize_Integration`) is **env-gated** — set `DIARIZE_SEG_MODEL`,
`DIARIZE_EMB_MODEL`, and `DIARIZE_WAV` to run it; otherwise it skips.

## Running tests

```sh
task models:tiny   # download the test model once
```

Then set the `TEST_MODEL` environment variable and run:

```sh
# Linux / macOS
TEST_MODEL=models/ggml-tiny.en.bin go test ./...

# Windows (PowerShell)
$env:TEST_MODEL = "D:\path\to\go-whisper.cpp\models\ggml-tiny.en.bin"
go test ./...
```

> **Windows / MSYS note**: `TEST_MODEL` must be a **native Windows path** (e.g. `D:\path\models\ggml-tiny.en.bin`), not an MSYS-style path (`/d/path/...`). `whisper_init_from_file` does not accept MSYS paths. Relative paths (e.g. `models/ggml-tiny.en.bin`) work fine from the repo root.

## GPU backends

Three GPU backends are supported, all opt-in at load time via `whisper.WithGPU(true)`.

### CUDA (Windows)

**Prerequisites:** VS2022 Community + CUDA toolkit (e.g. via Scoop: `scoop install cuda`).

```sh
task build:cuda   # builds whisper.cpp as MSVC DLLs into build-cuda/bin/
```

Build/run your Go code with the `cuda` tag:

```sh
go build -tags cuda ./...
go run -tags cuda ./examples/transcribe -m models/ggml-tiny.en.bin audio.wav
```

Enable GPU at load:

```go
m, err := whisper.New("models/ggml-tiny.en.bin", whisper.WithGPU(true))
```

**Runtime DLLs:** place these on PATH (or beside the exe):
- Built by `task build:cuda`: `whisper.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `ggml-cuda.dll` (from `build-cuda/bin/`)
- CUDA toolkit: `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` (Scoop puts the CUDA `bin\x64` on PATH automatically)

**CUDA architecture:** defaults to `75` (Turing/RTX 20xx). To target a different GPU, edit `CMAKE_CUDA_ARCHITECTURES` in `scripts/whispercpp-cuda.bat` before running `task build:cuda`.

### Vulkan (Windows)

**Prerequisites:** Vulkan SDK (e.g. via Scoop: `scoop install vulkan`).

```sh
task build:vulkan   # static build with GGML_VULKAN=ON into whisper.cpp/build-vulkan/
```

Build/run with the `vulkan` tag:

```sh
go build -tags vulkan ./...
go run -tags vulkan ./examples/transcribe -m models/ggml-tiny.en.bin audio.wav
```

Enable GPU at load:

```go
m, err := whisper.New("models/ggml-tiny.en.bin", whisper.WithGPU(true))
```

**Runtime DLLs:** only the system `vulkan-1.dll` is needed — no extra DLLs to ship.

### Metal (macOS)

Metal is enabled by default on macOS. No extra build step is needed:

```sh
task build:cpu   # Metal + Apple BLAS are ON by default; ggml-metal.a is linked automatically
```

Enable GPU at load:

```go
m, err := whisper.New("models/ggml-tiny.en.bin", whisper.WithGPU(true))
```

The `.metal` shader is embedded in `ggml-metal.a` at build time (`GGML_METAL_EMBED_LIBRARY=ON` default), so no `default.metallib` is needed at runtime.

### CI note

GitHub-hosted CI build-verifies the `cuda` and `vulkan` build tags (compile + link). GPU *execution* tests require a self-hosted runner with the appropriate hardware and are run locally.

## Notes / tips

- With `ggml-tiny.en`, always pass `WithLanguage("en")`. Auto-detect (`"auto"`) can mislabel the language on the tiny model (though the transcript text is still correct).

## Non-goals

This library is **speech-to-text only**. The following are LLM concepts that do not apply to whisper and will not be added:

- Grammar / GBNF constraints
- `logit_bias` / token bias
- Sampler chains

Whisper's only decoding choices are greedy vs. beam search plus threshold parameters — all of which are exposed via `TranscribeOption`.

## License

[BSD-3-Clause](LICENSE)
