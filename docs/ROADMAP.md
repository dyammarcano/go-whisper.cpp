# Roadmap

`go-whisper.cpp` — Go bindings for [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
(speech-to-text) with GPU backends and native speaker diarization.

**Status:** v0.2.0 shipped. Core capability complete; remaining work is audio-ingestion
ergonomics, diarization depth, and test coverage.

**Overall progress: ~85%** of the originally-scoped v1 surface (transcription, concurrency,
GPU backends, diarization, streaming) is implemented and released. The open items are
quality/ergonomics rather than missing core capability.

## Phase 1 — Foundation `[COMPLETE]` (v0.1.0)

- whisper.cpp v1.7.4 cgo binding via a thin C shim (`binding.cpp`).
- `Model` + `Session` split for real concurrent inference (shared read-only
  `whisper_context`, per-goroutine `whisper_state`).
- Full `TranscribeOption` set, segment/progress callbacks, context cancellation.
- Pure-Go `wav` package (16 kHz mono reader).

## Phase 2 — GPU backends `[COMPLETE]` (v0.1.0)

- CUDA (arch 75, MSVC DLLs) and Vulkan (MinGW static) — tested on Windows.
- Metal — build-verified for macOS. CPU static is the default.
- Build-tag link files (`link_{static,cuda,vulkan}_windows.go`, `link_linux.go`, `link_darwin.go`).

## Phase 3 — Speaker diarization `[COMPLETE]` (v0.1.0)

- `diarize` package over sherpa-onnx using pyannote segmentation-3.0 + WeSpeaker
  embeddings (no Python).
- Pure-Go `Label` merge helper (assigns each transcript segment the speaker with the
  greatest temporal overlap); decoupled from the whisper binding via a local `Segment` type.

## Phase 4 — Real-time / streaming `[COMPLETE]` (v0.2.0)

- `Stream` (sliding window over `Transcribe`): push `Write` + channel of partial/final
  `StreamResult`s, lossless backpressure by default, opt-in `WithDropOnOverrun` for live audio.

## Phase 5 — Audio ingestion `[PLANNED]` (proposed v0.3.0)

Today the `wav` package rejects non-16 kHz input (`ErrNot16kHz`); every real-world file
(44.1 kHz stereo, 48 kHz AAC) must be pre-converted. See [BACKLOG](BACKLOG.md) AI-1..AI-3.

- Pure-Go resample + downmix so the `wav` package accepts arbitrary rate/channels (16-bit PCM).
- Optional ffmpeg-decode path for compressed inputs (m4a/mp3/...) when ffmpeg is present.

## Phase 6 — Diarization depth `[PLANNED]`

- Token-level speaker labels (use whisper token timestamps instead of segment granularity).
- VAD-gated streaming finals (cleaner boundaries).
- GPU diarization (sherpa-onnx CUDA provider for the embedding model).

## Phase 7 — Hardening `[PLANNED]`

- Raise coverage to 80%+ (examples + `wav`); see Test Coverage below.
- Verify mid-inference cancellation on Linux/macOS under OpenMP (see [ISSUES](ISSUES.md)).
- Optionally nest `diarize/go.mod` so transcription-only consumers don't pull the ONNX tree.

## Test Coverage

**Current (per library package):**  |  **Target:** 80%

| Package | Coverage | Status |
|---------|----------|--------|
| `diarize` | 78.1% | Near target |
| `.` (whisper) | 68.3% | Needs improvement |
| `wav` | 56.8% | Needs improvement |
| `examples/*` | 0.0% | No tests (demo mains) |

Measured with `TEST_MODEL` + `DIARIZE_*` env set; `diarize` measured via a compiled test
binary beside the sherpa DLLs (see [CONTRIBUTORS](CONTRIBUTORS.md) for why). The integration
specs are env-gated, so CI (which only build-verifies) reports lower numbers than the above.
