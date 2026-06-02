# Implementation Tasks

Granular tasks for the planned work in [ROADMAP](ROADMAP.md), [BACKLOG](BACKLOG.md), and
[FEATURES](FEATURES.md), grouped by domain. Effort: Small / Medium / Large.

## Domain AI — Audio Ingestion (v0.3.0)

| ID | What | Files | Environment | Depends | Effort |
|----|------|-------|-------------|---------|--------|
| AI-1.1 | Add downmix (N channels → mono by averaging) to the WAV decoder | `wav/wav.go` | Go | — | Small |
| AI-1.2 | Add linear resampler (any rate → 16 kHz) | `wav/resample.go` (new) | Go | AI-1.1 | Medium |
| AI-1.3 | `ReadWAV` returns resampled 16 kHz mono instead of `ErrNot16kHz`; keep a strict `ReadWAV16k` for the old behavior | `wav/wav.go` | Go | AI-1.1, AI-1.2 | Small |
| AI-1.4 | Table tests: 44.1 kHz stereo, 48 kHz, 8 kHz fixtures → assert 16 kHz mono out, correct duration | `wav/resample_test.go` | Go | AI-1.3 | Medium |
| AI-3.1 | Detect ffmpeg; if present, decode compressed inputs (m4a/mp3) to PCM via subprocess | `wav/ffmpeg.go` (new, build-tagged or runtime-detected) | Go + ffmpeg | AI-1.3 | Medium |
| AI-3.2 | Doc + example: ingest arbitrary audio | `README.md`, `examples/transcribe` | Docs | AI-3.1 | Small |

## Domain DZ — Diarization & Streaming Depth (v0.4.0)

| ID | What | Files | Environment | Depends | Effort |
|----|------|-------|-------------|---------|--------|
| DZ-1.1 | Add token-granularity overlap merge alongside `Label` | `diarize/merge.go` | Go | — | Medium |
| DZ-1.2 | Tests for token-level labeling | `diarize/merge_test.go` | Go | DZ-1.1 | Small |
| DZ-2.1 | Optional VAD gate for streaming finals (sherpa VAD) | `stream.go`, `stream_options.go` | Go + ONNX | — | Large |
| DZ-3.1 | Expose sherpa CUDA provider for the embedding model | `diarize/diarize.go`, `diarize/options.go` | Go + CUDA | — | Medium |

## Domain TC — Test Coverage & Correctness (v0.3.0)

| ID | What | Files | Environment | Depends | Effort |
|----|------|-------|-------------|---------|--------|
| TC-1.1 | Build+smoke tests for each example (compile, `-h`/dry-run) | `examples/*/main_test.go` | Go | — | Small |
| TC-2.1 | `wav` edge-case tests (extra chunks, truncated data, bad rate) → 80% | `wav/wav_test.go` | Go | — | Small |
| TC-3.1 | Cover remaining option permutations + error paths in root pkg → 80% | `*_test.go` | Go | — | Medium |
| TC-4.1 | Run the mid-inference cancel spec on Linux + macOS (OpenMP on) | CI / local | Linux/macOS | — | Medium |

## Domain PK — Packaging (P3)

| ID | What | Files | Environment | Depends | Effort |
|----|------|-------|-------------|---------|--------|
| PK-1.1 | Nest a `diarize/go.mod` so the ONNX deps are opt-in | `diarize/go.mod`, `go.work` | Go modules | — | Medium |
| PK-2.1 | Add SHA256 pins for `base.en`/`small.en` in the model downloader | `scripts/download-model.sh` | Bash | — | Small |
| PK-3.1 | `examples/stream` variant wired to a mic capture lib (separate module) | `examples/stream-mic/` | Go | — | Medium |

## Suggested order

1. **TC-1.1, TC-2.1** (quick coverage wins, no deps) →
2. **AI-1.1 → AI-1.2 → AI-1.3 → AI-1.4** (resampling; the highest-impact ergonomics fix) →
3. **AI-3.1, AI-3.2** (ffmpeg ingestion) →
4. **TC-3.1**, cut **v0.3.0** →
5. **DZ-1.1/1.2, DZ-3.1** then **DZ-2.1**, cut **v0.4.0** →
6. **TC-4.1**, **PK-*** as capacity allows.
