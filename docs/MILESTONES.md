# Milestones

Version milestones for `go-whisper.cpp`. Releases are git tags on `main`.

## v0.1.0 — Transcription + GPU + Diarization `[RELEASED]`

First usable release: whisper.cpp speech-to-text from Go.

- whisper.cpp v1.7.4 binding; `Model`/`Session` concurrency; full transcription options.
- GPU backends: CUDA (arch 75) + Vulkan tested on Windows, Metal CI; CPU static default.
- Native speaker diarization (`diarize`) over sherpa-onnx (pyannote + WeSpeaker, no Python).
- Pure-Go `wav` reader; `transcribe`/`diarize`/`transcribe-diarize` examples.
- **Fixed:** Windows/MinGW mid-inference cancel crash (`GGML_OPENMP=OFF`); CUDA verified unaffected.
- **Security:** SHA256-pinned diarization model downloads.

## v0.2.0 — Real-time / streaming `[RELEASED]`

- `Stream` (`Model.NewStream`/`Session.NewStream`): incremental `Write` + channel of
  partial/final `StreamResult`s via a fixed-interval sliding window; lossless backpressure
  by default, opt-in `WithDropOnOverrun` for live audio; `examples/stream`.
- **Fixed:** streaming goroutine/context leak on the graceful `CloseSend` path
  (`defer st.cancel()` in the worker), pinned by a leak regression test.
- **Test Coverage:** per-package — `diarize` 78.1%, `whisper` (root) 68.3%, `wav` 56.8%,
  `examples/*` 0% (demo mains).

## v0.3.0 — Audio ingestion `[PLANNED]`

Make the library accept ordinary audio instead of only 16 kHz mono WAV.

- Pure-Go resample + downmix in the `wav` package (any rate/channels, 16-bit PCM).
- Optional ffmpeg-decode path for compressed formats (m4a/mp3/...).
- **Coverage target:** 80%+ overall (raise `wav` and add example smoke tests).
- See [IMPLEMENTATION_TASKS](IMPLEMENTATION_TASKS.md) domain **AI** (Audio Ingestion).

## v0.4.0 — Diarization depth `[PLANNED]`

- Token-level speaker labels; VAD-gated streaming finals; GPU diarization (sherpa CUDA provider).
- **Coverage target:** maintain 80%+.
