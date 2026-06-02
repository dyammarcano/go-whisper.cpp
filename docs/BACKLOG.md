# Backlog

Prioritized future work and tech debt. **P1** = correctness/coverage, **P2** = improvements,
**P3** = nice-to-have. Effort: Small (<½ day) / Medium (1–2 days) / Large (½ week+).
No `TODO`/`FIXME`/`HACK` markers exist in the Go sources as of v0.2.0.

## P1 — Coverage & correctness

| ID | Item | Why | Effort |
|----|------|-----|--------|
| C-1 | Example smoke tests | `examples/*` are 0% covered; a build+`-help`/dry-run test guards them | Small |
| C-2 | Raise `wav` coverage (56.8% → 80%) | Header edge cases (extra chunks, bad rate, truncated data) untested | Small |
| C-3 | Raise root `whisper` coverage (68.3% → 80%) | Some option permutations and error paths uncovered | Medium |
| C-4 | Verify mid-inference cancel on Linux/macOS (OpenMP) | Residual risk in [ISSUES](ISSUES.md); only Windows/MinGW is proven + fixed | Medium |

## P2 — Audio ingestion (v0.3.0)

| ID | Item | Why | Effort |
|----|------|-----|--------|
| AI-1 | Pure-Go resample + downmix in `wav` | `wav` returns `ErrNot16kHz`; every real file (44.1 kHz stereo, 48 kHz) needs pre-conversion today | Medium |
| AI-2 | `wav` accepts arbitrary channel counts (downmix to mono) | Stereo/multi-channel inputs are common | Small |
| AI-3 | Optional ffmpeg-decode path for compressed audio | m4a/mp3/etc. can't be read at all without external conversion | Medium |

## P2 — Diarization & streaming depth

| ID | Item | Why | Effort |
|----|------|-----|--------|
| DZ-1 | Token-level speaker labels | `Label` is segment-granularity; finer attribution from token timestamps | Medium |
| DZ-2 | VAD-gated streaming finals | Fixed-interval windows can cut mid-word; VAD gives clean boundaries | Large |
| DZ-3 | GPU diarization (sherpa CUDA provider) | Embedding model runs on CPU today | Medium |

## P3 — Packaging & polish

| ID | Item | Why | Effort |
|----|------|-----|--------|
| PK-1 | Nest `diarize/go.mod` | Transcription-only consumers currently pull the sherpa-onnx/ONNX dep tree | Medium |
| PK-2 | Pin SHA256 for additional whisper models | Only `tiny.en` is checksum-pinned in `download-model.sh` | Small |
| PK-3 | `examples/stream` wired to a real mic capture | Demonstrate live low-latency use (kept out of the library module) | Medium |

Cross-referenced in [ROADMAP](ROADMAP.md), [FEATURES](FEATURES.md), and
[IMPLEMENTATION_TASKS](IMPLEMENTATION_TASKS.md).
