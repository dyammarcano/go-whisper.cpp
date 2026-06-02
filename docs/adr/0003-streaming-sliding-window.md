# ADR-0003: Streaming as a sliding-window layer over Transcribe

- **Status:** Accepted
- **Date:** 2026-06-02

## Context

Real-time transcription was a v1-scope goal. whisper.cpp is **not natively streaming** — it
decodes a bounded window (≤ 30 s) per call; there is no incremental, token-by-token decode
API. We need an API that ingests audio incrementally and emits partial + final results,
covering both live (microphone) and chunked/batch use, while reusing the existing,
tested `Transcribe` path. A C-level streaming shim was considered but rejected: whisper.cpp's
own `stream` example also reprocesses a window each step, so a custom C state-carry buys
little for a lot of added cgo surface.

## Decision

Implement `Stream` as a **pure-Go sliding-window orchestrator** in the root package, layered
over `Session.Transcribe`:

- A `Stream` owns one `Session` and a single worker goroutine.
- **Push + channel** API: `Write([]float32)` feeds a bounded buffer; `Results() <-chan
  StreamResult` delivers ordered partial/final updates; `CloseSend` flushes, `Close` aborts.
- **Fixed-interval window:** every `step` of new audio, re-run whisper over the trailing
  `window` (≤ 30 s), carrying committed text forward as `initial_prompt`. A pure
  `classifyWindow` function emits finals once (monotonic) and re-emits partials for the tail.
- **Overrun policy:** lossless backpressure by default (block when uncommitted audio would
  exceed the window); opt-in `WithDropOnOverrun` drops oldest + flags `Lagging` for live mics.
- No new cgo; reuses the hardened mid-inference cancellation.

## Consequences

- **+** Reuses the tested transcription path; no new native code; one design serves live + batch.
- **+** The commit logic is a pure function → unit-testable without a model.
- **+** Backpressure cap (`window`) keeps block-mode lossless; drop-mode bounds latency.
- **−** Not true incremental decoding: each window reprocesses overlapping audio (CPU cost
  scales with `window/step`). Acceptable and matches upstream's approach.
- **−** Partials are **segment-granularity** and re-emitted; consumers must replace prior
  partials. Token-level partials are future work ([BACKLOG](../BACKLOG.md) DZ-1).
- **−** Consumers **must drain `Results()`** (or call `Close()`); abandoning it blocks `Write`
  and strands the worker. A goroutine/context leak on the graceful `CloseSend` path was found
  and fixed (`defer st.cancel()`; see [BUGS](../BUGS.md) B-2).
- Fixed-interval was chosen over VAD-gated chunking to avoid coupling streaming to a VAD model;
  VAD-gated finals remain a future enhancement.
