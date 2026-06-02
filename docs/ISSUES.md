# Known Issues

## Resolved

### Mid-inference cancellation crash on Windows/MinGW (OpenMP) — FIXED

**Symptom:** Cancelling a transcription *while ggml compute was in flight* crashed the
process with an access violation (`Exception 0xC0000005`, "signal arrived during external
code execution") inside `whisper_bind_full`. Pre-canceling (aborting before compute starts)
was unaffected. Reproduced 100% by the `cancels mid-inference within a bound` spec.

**Root cause:** ggml's `abort_callback` is invoked by the compute threadpool *during*
`ggml_graph_compute`. When the cgo callback returns "abort" mid-graph, tearing down a
**libgomp** (`-fopenmp`) parallel region from inside that callback faults on Windows/MinGW.
whisper.cpp/ggml abort *logic* is clean (`return -6/-8/-9`); the fault is the
libgomp + cgo-callback-during-parallel-region interaction.

**Fix:** Build ggml with `GGML_OPENMP=OFF` on the MinGW build (`scripts/whispercpp.sh`),
so ggml's native Win32 threadpool runs the compute. It aborts mid-graph cleanly. With this
change the full 7-spec suite passes (0 skipped), cancellation included.

**CUDA verified (2026-06-02):** the same suite incl. the mid-inference cancel spec passes
on the CUDA build on a GTX 1650 (arch 75) — `8/8 specs, 0 skipped`. GPU inference offloads
the aborted compute, so the OpenMP/`vcomp` teardown path is not reached on the CUDA backend;
no fix was needed there.

## Open / residual risk

- **Linux / macOS still build with OpenMP.** The same `abort_callback`-during-compute path
  exists on those platforms but has **not** been observed to crash, and CI only
  build-verifies (it does not run the audio-gated cancellation tests). If a mid-inference
  cancellation crash ever surfaces on Linux/macOS, apply the same `-DGGML_OPENMP=OFF` there.

## Limitations (current)

- **Audio input must be 16 kHz mono.** The `wav` package returns `ErrNot16kHz` for other
  rates and does not downmix; callers must resample/downmix first (e.g. `ffmpeg -ar 16000
  -ac 1`, or the converter being tracked as [BACKLOG](BACKLOG.md) AI-1). The `diarize` and
  streaming paths share this contract.
- **No built-in decoding of compressed audio** (m4a/mp3/...). Decode to PCM WAV first
  (tracked as BACKLOG AI-3).
- **Transcription quality is bounded by the chosen model.** `ggml-tiny.en` is English-only;
  non-English or unclear/conversational audio produces poor output (and whisper's classic
  repetition/hallucination loop). Use a larger and/or multilingual model for such inputs.
- **Diarization speaker count depends on clustering settings.** Auto mode (default threshold
  0.5) can merge similar/short-segment speakers; pass `WithNumSpeakers` or tune
  `WithThreshold` when the auto count looks wrong.
- **GPU execution is verified only where hardware exists.** CI build-verifies the `cuda`/
  `vulkan` tags; runtime GPU paths are exercised on the maintainer's machine.
