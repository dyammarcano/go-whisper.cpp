# Real-time / Streaming Transcription — Design

**Date:** 2026-06-02
**Status:** Approved (brainstorming) — ready for implementation plan
**Depends on:** existing `Model`/`Session`/`runFull` transcription core (v0.1.0); the
hardened mid-inference cancellation (`GGML_OPENMP=OFF` on MinGW).

## Goal

Add a streaming transcription API so callers can feed audio incrementally and receive
**partial** (refining) and **final** (committed) transcription results as the audio
arrives — covering both **live low-latency** (microphone) and **chunked long-audio /
batch** use cases through one unified API.

## Background

whisper.cpp is **not natively streaming**: it decodes a bounded window (≤ 30 s) per call.
"Streaming" is therefore a **sliding-window orchestration** on top of the existing
`Transcribe`: keep a rolling audio buffer, periodically run whisper over the trailing
window, carry committed text forward as `initial_prompt` for continuity, and decide which
returned segments are stable enough to emit as final. This mirrors whisper.cpp's own
`stream` example (which also reprocesses the window each step rather than doing
model-level incremental decoding).

## Scope

- A `Stream` type in the **root `whisper` package** that orchestrates windowed inference.
- Push-based feeding (`Write`) + channel-based result delivery (`Results()`).
- Fixed-interval sliding window with configurable `step`/`window`/`keep`.
- Partial vs. final result semantics with `initial_prompt` carry-over.
- Overrun policy: backpressure by default, opt-in drop-oldest for live.

## Non-goals (this phase)

- **VAD-gated chunking** — fixed-interval windowing only; VAD is a future enhancement.
- **Token-level partials** — partials are at segment granularity.
- **Audio capture / resampling** — the library stays I/O-agnostic; callers supply
  16 kHz mono `[]float32` (same contract as `Transcribe`). Mic/file capture is the
  caller's concern (the `wav` package + examples show the format).
- **Diarized streaming** — combining live diarization with streaming is out of scope.

## Architecture

`Stream` lives in the root package (it is intrinsically coupled to `Session`/
`whisper_state`; a subpackage would require exporting internals). A `Stream`:

- **owns one `Session`** (a dedicated `whisper_state`) used for all its window runs;
- runs a single **worker goroutine** that performs the windowing loop by calling the
  existing `runFull` path each step (whisper inference is single-at-a-time, so one worker);
- exposes a **bounded buffer** fed by `Write` and drained by the worker, with a
  mutex+cond for backpressure;
- delivers results on a buffered `Results()` channel.

No new C/cgo surface — it reuses the tested transcription path and the hardened
cancellation. New files: `stream.go` (the `Stream` type + window loop), `stream_options.go`
(options), `stream_test.go` (pure-Go commit-logic tests + gated integration). Mirrors the
existing `session.go` / `options.go` split.

## Public API

```go
// StreamResult is one transcription update.
type StreamResult struct {
    Segment Segment // absolute timestamps measured from stream start (t=0)
    Partial bool    // true = provisional (may change/be superseded); false = committed final
    Lagging bool    // audio was dropped before this result (drop-on-overrun mode only)
}

// NewStream starts a streaming session. Model variant allocates and owns a fresh Session
// (freed on Close); Session variant uses the caller's session for the stream's lifetime.
func (m *Model)   NewStream(ctx context.Context, opts ...StreamOption) (*Stream, error)
func (s *Session) NewStream(ctx context.Context, opts ...StreamOption) (*Stream, error)

func (st *Stream) Write(samples []float32) error // 16kHz mono; blocks (backpressure) or
                                                 // drops oldest per policy; err if closed/ctx done
func (st *Stream) CloseSend() error              // no more audio: flush a final window, commit
                                                 // remaining segments, then close Results()
func (st *Stream) Results() <-chan StreamResult  // ordered partial+final updates
func (st *Stream) Err() error                    // terminal error (nil on clean EOF);
                                                 // valid once Results() is closed
func (st *Stream) Close() error                  // abort now: cancel in-flight inference,
                                                 // free owned state, drain; idempotent
```

### Options & defaults

```go
WithStreamStep(d time.Duration)            // run cadence            (default 700ms)
WithStreamWindow(d time.Duration)          // max audio per run, ≤30s (default 10s)
WithStreamKeep(d time.Duration)            // overlap kept past a commit (default 200ms)
WithDropOnOverrun(maxBuffer time.Duration) // live: bound buffer, drop oldest (default: off → block)
WithTranscribeOptions(opts ...TranscribeOption) // per-window opts (language, threads, …)
```

`NewStream` validates `0 < step ≤ window ≤ 30s` and `0 ≤ keep < window`, else `ErrConfig`.

## Windowing & commit algorithm

State (durations are absolute, measured from stream start):

- `consumed` — total audio that has entered the buffer (the absolute clock).
- `committed` — absolute time through which finals have been emitted.
- `buf` — samples covering `[windowStart, consumed)`, where `windowStart ≥ committed − keep`.
- `prompt` — accumulated committed text (capped, e.g. last ~200 chars) used as `initial_prompt`.

Each worker tick (every `step` of new audio, or immediately when buffered audio ≥ `window`):

1. If `consumed − windowStart > window`, advance `windowStart` to `consumed − window`.
   In **block** mode the backpressure cap (below) keeps `consumed − windowStart ≤ window`,
   so this only trims already-committed audio. In **drop-on-overrun** mode it may discard
   *un-committed* audio that fell off the back of the window — that audio is reported lost
   via the `Lagging` flag.
2. Run `runFull` over `buf` with `initial_prompt = prompt` + the caller's `TranscribeOptions`.
3. Convert each returned segment's relative timestamps to absolute (`+ windowStart`).
4. For each segment, in order:
   - **Final** if `segEnd ≤ consumed − keep` and `segStart ≥ committed`: emit
     `{Segment, Partial:false}` exactly once; set `committed = segEnd`; append text to `prompt`.
   - **Partial** if `segEnd > consumed − keep`: emit `{Segment, Partial:true}` (does not
     advance `committed`).
   - **Skip** if `segEnd ≤ committed` (duplicate from window overlap).
5. Slide the buffer: discard samples whose absolute end `≤ committed − keep`;
   set `windowStart = committed − keep`.

On `CloseSend`: run one final window over `[windowStart, consumed)` and commit **all**
remaining segments as final (even the tail), then close `Results()`.

**Invariant:** finals are emitted once, in non-decreasing time order; `committed` is
monotonic. **Consumer contract (documented):** a `Partial` is a provisional snapshot of the
un-committed tail and supersedes earlier partials; a `Partial` is eventually replaced by one
or more finals covering the same span.

## Overrun / backpressure

- **Default (block):** `Write` blocks while un-committed audio (`consumed − committed`)
  would exceed what a single window covers (`window − keep`); the worker committing +
  sliding unblocks it. This cap is what makes block mode **lossless**: all un-committed
  audio always stays inside the next processed window, so nothing falls off the back
  unprocessed. Correct for files/batch (the producer simply feeds slower).
- **`WithDropOnOverrun(maxBuffer)`:** when `consumed − committed > maxBuffer`, drop the
  oldest un-committed audio (advance `windowStart`/`committed` past it) and set `Lagging`
  on the next emitted result. Never blocks `Write` — correct for live mics.

## Concurrency & lifecycle

- One worker goroutine per `Stream`. `Write` and the worker share `buf` via mutex+cond.
- `ctx` cancellation (or `Close()`) flips the in-flight `runFull`'s abort flag (the
  hardened mid-inference cancel path) and tears the stream down: worker exits, `Results()`
  closes, `Err()` returns the `ctx` error (`ErrCanceled`).
- `Close()` is idempotent and frees the owned `Session` (Model variant). `CloseSend()` is
  the graceful end (flush + final); `Close()` is the abort.
- **Consumer must drain `Results()`** — the worker blocks sending results; an undrained
  channel applies backpressure to `Write` (documented; `Results()` is modestly buffered).

## Error handling

- New sentinel `ErrStreamClosed` (returned by `Write`/`CloseSend` after close).
- Reuses `ErrEmptyAudio` (empty `Write` is a no-op, not an error), `ErrCanceled`,
  `ErrTranscribe`, `ErrConfig`, `ErrClosed`.
- A whisper failure mid-stream is terminal: worker stops, `Results()` closes, the error is
  surfaced via `Err()`.

## Testing strategy

- **Pure-Go commit logic (no model):** factor the segment classify/commit/dedupe/offset
  math into a testable function over synthetic `(segments, consumed, committed, keep)`
  inputs. Table tests cover: final-once, partial in tail, dup-skip from overlap, monotonic
  `committed`, absolute-timestamp offsetting, `CloseSend` tail commit.
- **Overrun policy (no model):** unit-test the buffer's block vs. drop-oldest behavior and
  the `Lagging` flag with a stubbed/slow worker.
- **Integration (gated on `TEST_MODEL`):** feed `whisper.cpp/samples/jfk.wav` in small
  chunks (e.g. 0.5 s) into a `Stream`; assert (a) finals concatenate to contain "country",
  (b) at least one `Partial` was observed, (c) final segment timestamps are absolute and
  monotonic, (d) `CloseSend` drains and `Err()` is nil. Whisper CPU is statically linked,
  so plain `go test` works (no DLL-staging needed).

## Example

```go
m, _ := whisper.New(modelPath)
defer m.Close()

st, _ := m.NewStream(ctx,
    whisper.WithStreamStep(500*time.Millisecond),
    whisper.WithDropOnOverrun(8*time.Second), // live mic
    whisper.WithTranscribeOptions(whisper.WithLanguage("en")))

go func() {
    for chunk := range mic { _ = st.Write(chunk) } // []float32 16kHz mono
    _ = st.CloseSend()
}()

for r := range st.Results() {
    if r.Partial {
        redrawCurrentLine(r.Segment.Text)
    } else {
        commitLine(r.Segment.Text)
    }
}
if err := st.Err(); err != nil { log.Fatal(err) }
```

## Future work

- VAD-gated finals (cleaner boundaries) as an opt-in on top of the fixed-interval core.
- Token-level partials (word-by-word) using whisper token timestamps.
- A streaming `examples/stream` demo wired to a mic capture library (kept out of the
  library module to preserve I/O-agnosticism).
