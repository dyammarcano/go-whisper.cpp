# ADR-0002: Native speaker diarization via sherpa-onnx (no Python)

- **Status:** Accepted
- **Date:** 2026-06-01

## Context

whisper produces *what* was said but not *who* said it. We want speaker diarization in the
same module. The reference implementation, [pyannote-audio](https://github.com/pyannote/pyannote-audio),
is Python/PyTorch. Options: (a) embed/shell out to Python + pyannote, (b) call a Python
service, (c) run the pyannote ONNX models through a native runtime with Go bindings.

(a) and (b) impose a Python runtime and IPC on every consumer of a Go library — unacceptable
for a self-contained binding.

## Decision

Use **[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)** via its Go bindings
(`github.com/k2-fsa/sherpa-onnx-go`) to run **pyannote segmentation-3.0** (MIT) plus a
**WeSpeaker** embedding model through **ONNX Runtime** — no Python.

- `diarize.Diarizer` wraps `OfflineSpeakerDiarization`; `Diarize([]float32) []Turn`.
- Input contract matches whisper: **16 kHz mono float32**.
- The `diarize` package is **decoupled from the whisper binding**: `Label` operates on a
  package-local `diarize.Segment`, so transcription-only and diarization-only consumers can
  each link only what they need. Callers map `whisper.Segment` → `diarize.Segment` in two lines.
- Prebuilt sherpa/ONNX Runtime DLLs ship via the Go module (MinGW-compatible).

## Consequences

- **+** No Python; pure native execution, single-language consumer story.
- **+** Diarization and transcription are independently linkable (no import cycle, no forced
  ONNX dependency for STT-only users).
- **+** Uses the same audio format as whisper, so one decode path feeds both.
- **−** Adds the sherpa-onnx + ONNX Runtime dependency tree to the module (mitigation tracked:
  [BACKLOG](../BACKLOG.md) PK-1 — nest `diarize/go.mod`).
- **−** **Windows DLL-shadow gotcha:** a stale `System32\onnxruntime.dll` shadows the bundled
  Runtime and crashes temp-dir test binaries. Tests must compile the binary beside the DLLs
  (`task diarize:dlls` + `go test -c -o ... ./diarize/`). See [CONTRIBUTORS](../CONTRIBUTORS.md).
- **−** Default auto-clustering (threshold 0.5) can under-/over-cluster; callers tune via
  `WithNumSpeakers` / `WithThreshold`.
