# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-02

Adds real-time / streaming transcription on top of v0.1.0.

### Added

- **Streaming transcription** (`Stream`, `Model.NewStream`/`Session.NewStream`): incremental
  `Write` + channel of partial/final `StreamResult`s via a fixed-interval sliding window;
  lossless backpressure by default, opt-in `WithDropOnOverrun` for live audio.

## [0.1.0] - 2026-06-02

First usable release: whisper.cpp speech-to-text from Go, with GPU backends and
native speaker diarization. (Real-time streaming is planned for a later release.)

### Added

- **whisper.cpp binding** (v1.7.4, pinned git submodule) via a thin C shim over the
  C API, cgo, MinGW on Windows.
- **Model + Session split** for real concurrent inference: one shared read-only
  `whisper_context`, N per-goroutine `whisper_state` via `whisper_full_with_state`.
- **Transcription options**: language / auto-detect, greedy vs. beam search, temperature
  (+ fallback inc), entropy / log-prob / no-speech thresholds, token timestamps, segment
  and progress callbacks, offset / duration, audio context, initial prompt, max segment
  length / tokens, suppress-blank / suppress-non-speech.
- **Context-based cancellation** — both pre-flight and mid-inference (`context.Context`).
- **GPU backends**: CUDA (arch 75, MSVC DLLs) and Vulkan (MinGW static) tested on Windows,
  Metal build-verified for macOS; CPU static is the default.
- **`wav` package** — pure-Go 16 kHz mono WAV reader.
- **`diarize` package** — native speaker diarization over sherpa-onnx using pyannote
  segmentation-3.0 + WeSpeaker embeddings (no Python), plus a pure-Go `Label` helper that
  assigns each transcript segment the speaker with the greatest temporal overlap.
- **Examples**: `transcribe`, `diarize`, `transcribe-diarize`.
- **Tooling**: Taskfile + scripts to build each backend and provision models / runtime DLLs.

### Fixed

- **Windows/MinGW access violation (`0xC0000005`) when cancelling a transcription
  mid-compute.** ggml's `abort_callback` fired during `ggml_graph_compute`, and tearing
  down a libgomp parallel region from that cgo callback faulted. Built ggml with
  `GGML_OPENMP=OFF` on MinGW so its native Win32 threadpool runs the compute and aborts
  cleanly. The CUDA build was verified unaffected.

### Security

- **SHA256-pinned the diarization model downloads** (pyannote segmentation + WeSpeaker
  embedding); mismatches abort and re-fetch.

[0.2.0]: https://github.com/dyammarcano/go-whisper.cpp/releases/tag/v0.2.0
[0.1.0]: https://github.com/dyammarcano/go-whisper.cpp/releases/tag/v0.1.0
