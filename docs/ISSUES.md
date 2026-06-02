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

## Open / residual risk

- **Linux / macOS still build with OpenMP.** The same `abort_callback`-during-compute path
  exists on those platforms but has **not** been observed to crash, and CI only
  build-verifies (it does not run the audio-gated cancellation tests). If a mid-inference
  cancellation crash ever surfaces on Linux/macOS, apply the same `-DGGML_OPENMP=OFF` there.
- **CUDA build (`scripts/whispercpp-cuda.bat`, MSVC) is untested for this path.** GPU
  inference uses a different compute backend; the bundled `ggml-cpu` fallback still links
  MSVC's OpenMP (`vcomp`). Not yet exercised by the cancellation tests.
