# Bugs

Tracker for confirmed defects. Severity: Critical / High / Medium / Low.

## Open

_None._ No open bugs as of v0.2.0.

## Resolved

| ID | Severity | Summary | Fix | Resolved in |
|----|----------|---------|-----|-------------|
| B-1 | Critical | Access violation (`0xC0000005`) when cancelling a transcription **mid-compute** on Windows/MinGW — ggml's `abort_callback` tearing down a libgomp parallel region from a cgo callback faulted | Build ggml with `GGML_OPENMP=OFF` on MinGW (native Win32 threadpool aborts cleanly). CUDA build verified unaffected. See [ISSUES](ISSUES.md). | v0.1.0 |
| B-2 | Critical | `Stream` leaked a goroutine + `context.WithCancel` resources on the graceful `CloseSend` path (the documented happy path never cancelled the context) | `defer st.cancel()` in the worker `run()` — cancels on every exit, stopping the buffer's ctx-wakeup goroutine. Pinned by `TestStream_NoGoroutineLeakOnCloseSend`. | v0.2.0 |

Both were caught by testing during development (B-1 by running the env-gated cancellation
spec; B-2 by an adversarial code review) — not in production. The repro and root-cause notes
for B-1 live in [ISSUES.md](ISSUES.md).
