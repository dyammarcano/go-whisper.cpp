# Real-time / Streaming Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `Stream` API to the root `whisper` package that feeds audio incrementally and emits partial + final transcription results via a sliding window over the existing `Transcribe`.

**Architecture:** A `Stream` owns a `Session` and a worker goroutine. `Write` appends 16 kHz mono `[]float32` into a bounded buffer (backpressure or drop-oldest). Every `step` of new audio (or buffer ≥ `window`) the worker re-runs `Session.Transcribe` over the trailing window with the committed text as `initial_prompt`, then a pure `classifyWindow` function splits returned segments into finals (emitted once) and partials (re-emitted). No new cgo.

**Tech Stack:** Go, the existing cgo whisper binding, `context`, `sync` (mutex + cond). Tests use the standard `testing` package (the suite also uses ginkgo, but these new tests are plain `testing.T`).

---

## Files

- **Create `errors.go`** (modify): add `ErrStreamClosed`.
- **Create `stream_options.go`**: `StreamResult`, `streamConfig`, `StreamOption`, defaults, `With*` options, `validate()`.
- **Create `stream_window.go`**: pure `classifyWindow` (segment → partial/final + new committed mark + prompt text). No goroutines, no cgo, no clock.
- **Create `stream_buffer.go`**: `streamBuffer` — bounded PCM buffer with block/drop write, `nextWindow`, `slide`, `closeSend`. No cgo.
- **Create `stream.go`**: `Stream` type, `(*Session).NewStream`, `(*Model).NewStream`, worker loop, `Write`/`CloseSend`/`Results`/`Err`/`Close`.
- **Create `stream_window_test.go`**: pure table tests for `classifyWindow`.
- **Create `stream_buffer_test.go`**: pure tests for block/drop/slide/close.
- **Create `stream_test.go`**: gated (`TEST_MODEL`) integration test feeding `jfk.wav` in chunks.
- **Create `examples/stream/main.go`**: file-fed streaming demo (no mic).
- **Modify `README.md`** + **`CHANGELOG.md`**: document the feature.

Audio↔duration uses 16000 Hz. Helper (defined in Task 3): `samplesFor(d) = int(d.Seconds()*16000)` and `durFor(n) = time.Duration(n) * time.Second / 16000`.

---

### Task 1: Add the `ErrStreamClosed` sentinel

**Files:**
- Modify: `errors.go`

- [ ] **Step 1: Read `errors.go`** to see the existing sentinel style (they are `errors.New("whisper: ...")` vars).

- [ ] **Step 2: Add the sentinel**

Add to the `var (...)` block in `errors.go`:

```go
	// ErrStreamClosed is returned by Stream.Write/CloseSend after the stream is closed.
	ErrStreamClosed = errors.New("whisper: stream closed")
```

- [ ] **Step 3: Verify it compiles**

Run: `go build ./...`
Expected: no output (success).

- [ ] **Step 4: Commit**

```bash
git add errors.go
git commit -m "feat(stream): add ErrStreamClosed sentinel"
```

---

### Task 2: Stream options, config, and `StreamResult`

**Files:**
- Create: `stream_options.go`
- Test: `stream_options_test.go`

- [ ] **Step 1: Write the failing test**

Create `stream_options_test.go`:

```go
package whisper

import (
	"testing"
	"time"
)

func TestStreamConfig_Defaults(t *testing.T) {
	c := defaultStreamConfig()
	if c.step != 700*time.Millisecond || c.window != 10*time.Second || c.keep != 200*time.Millisecond {
		t.Fatalf("unexpected defaults: %+v", c)
	}
	if c.dropMaxBuffer != 0 {
		t.Fatalf("drop should be off by default, got %v", c.dropMaxBuffer)
	}
}

func TestStreamConfig_OptionsAndValidate(t *testing.T) {
	c := defaultStreamConfig()
	for _, o := range []StreamOption{
		WithStreamStep(300 * time.Millisecond),
		WithStreamWindow(8 * time.Second),
		WithStreamKeep(150 * time.Millisecond),
		WithDropOnOverrun(5 * time.Second),
	} {
		o(&c)
	}
	if c.step != 300*time.Millisecond || c.window != 8*time.Second ||
		c.keep != 150*time.Millisecond || c.dropMaxBuffer != 5*time.Second {
		t.Fatalf("options not applied: %+v", c)
	}
	if err := c.validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
}

func TestStreamConfig_ValidateRejects(t *testing.T) {
	bad := []streamConfig{
		{step: 0, window: 10 * time.Second, keep: time.Second},                 // step <= 0
		{step: time.Second, window: 0, keep: 0},                                // window <= 0
		{step: 12 * time.Second, window: 10 * time.Second, keep: time.Second},  // step > window
		{step: time.Second, window: 31 * time.Second, keep: time.Second},       // window > 30s
		{step: time.Second, window: 10 * time.Second, keep: 10 * time.Second},  // keep >= window
		{step: time.Second, window: 10 * time.Second, keep: -1},                // keep < 0
	}
	for i, c := range bad {
		if err := c.validate(); err == nil {
			t.Errorf("case %d: expected error, got nil for %+v", i, c)
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test -run TestStreamConfig ./ -count=1`
Expected: FAIL — undefined `defaultStreamConfig`, `StreamOption`, etc.

- [ ] **Step 3: Write the implementation**

Create `stream_options.go`:

```go
package whisper

import (
	"fmt"
	"time"
)

// StreamResult is one transcription update from a Stream.
type StreamResult struct {
	Segment Segment // absolute timestamps measured from stream start (t=0)
	Partial bool    // true = provisional (may be superseded); false = committed final
	Lagging bool    // audio was dropped before this result (drop-on-overrun mode only)
}

// streamConfig holds resolved Stream settings.
type streamConfig struct {
	step          time.Duration       // run cadence (new audio per window run)
	window        time.Duration       // max audio per run (<= 30s)
	keep          time.Duration       // overlap kept past a commit
	dropMaxBuffer time.Duration       // 0 = block (lossless); >0 = drop oldest beyond this
	transcribe    []TranscribeOption  // per-window options (language, threads, ...)
}

// StreamOption configures a Stream.
type StreamOption func(*streamConfig)

func defaultStreamConfig() streamConfig {
	return streamConfig{
		step:   700 * time.Millisecond,
		window: 10 * time.Second,
		keep:   200 * time.Millisecond,
	}
}

// WithStreamStep sets how much new audio accumulates before each window run.
func WithStreamStep(d time.Duration) StreamOption { return func(c *streamConfig) { c.step = d } }

// WithStreamWindow sets the max audio (<= 30s) reprocessed each run.
func WithStreamWindow(d time.Duration) StreamOption { return func(c *streamConfig) { c.window = d } }

// WithStreamKeep sets the overlap retained past a committed segment (for decoder context).
func WithStreamKeep(d time.Duration) StreamOption { return func(c *streamConfig) { c.keep = d } }

// WithDropOnOverrun bounds the buffer to maxBuffer of audio and drops the oldest
// un-committed audio when exceeded (live mode). Default (0) blocks Write instead.
func WithDropOnOverrun(maxBuffer time.Duration) StreamOption {
	return func(c *streamConfig) { c.dropMaxBuffer = maxBuffer }
}

// WithTranscribeOptions sets the per-window transcription options.
func WithTranscribeOptions(opts ...TranscribeOption) StreamOption {
	return func(c *streamConfig) { c.transcribe = opts }
}

func (c *streamConfig) validate() error {
	switch {
	case c.step <= 0:
		return fmt.Errorf("%w: step must be > 0", ErrConfig)
	case c.window <= 0 || c.window > 30*time.Second:
		return fmt.Errorf("%w: window must be in (0, 30s]", ErrConfig)
	case c.step > c.window:
		return fmt.Errorf("%w: step must be <= window", ErrConfig)
	case c.keep < 0 || c.keep >= c.window:
		return fmt.Errorf("%w: keep must be in [0, window)", ErrConfig)
	case c.dropMaxBuffer < 0:
		return fmt.Errorf("%w: dropMaxBuffer must be >= 0", ErrConfig)
	}
	return nil
}
```

> Note: `ErrConfig` already exists in `errors.go` (used by diarize/options). If `go build` reports it undefined in this package, add `ErrConfig = errors.New("whisper: invalid config")` to `errors.go` in this task.

- [ ] **Step 4: Run test to verify it passes**

Run: `go test -run TestStreamConfig ./ -count=1`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add stream_options.go stream_options_test.go
git commit -m "feat(stream): StreamResult, options, config validation"
```

---

### Task 3: Pure window-commit logic (`classifyWindow`)

**Files:**
- Create: `stream_window.go`
- Test: `stream_window_test.go`

This is the heart of the feature and is **pure** (no whisper, no goroutines). `classifyWindow` converts a window run's segments (timestamps relative to `windowStart`) into ordered `StreamResult`s, advancing the committed mark.

Rules (segment end is absolute = relative + `windowStart`):
- **skip** if `absEnd <= committed` (already committed; overlap duplicate),
- **final** if `committed < absEnd <= commitBefore` → emit once, `committed = absEnd`, append text to prompt,
- **partial** otherwise (`absEnd > commitBefore`).

During streaming `commitBefore = consumed - keep`; on flush `commitBefore = consumed`.

- [ ] **Step 1: Write the failing test**

Create `stream_window_test.go`:

```go
package whisper

import (
	"testing"
	"time"
)

func seg(start, end float64, text string) Segment {
	return Segment{
		Start: time.Duration(start * float64(time.Second)),
		End:   time.Duration(end * float64(time.Second)),
		Text:  text,
	}
}

func TestClassifyWindow_FinalsPartialsSkips(t *testing.T) {
	// window starts at t=2s; segments are relative to that start.
	// committed=2s, commitBefore=5s (consumed 5.2s, keep .2s).
	segs := []Segment{
		seg(0.0, 1.0, " already"), // abs 2..3  -> end<=committed(2)? no, 3>2; end<=commitBefore(5) -> FINAL
		seg(1.0, 2.5, " hello"),   // abs 3..4.5 -> FINAL
		seg(2.5, 4.0, " world"),   // abs 4.5..6.0 -> end 6 > commitBefore 5 -> PARTIAL
	}
	got := classifyWindow(segs, 2*time.Second, 2*time.Second, 5*time.Second)
	if len(got.results) != 3 {
		t.Fatalf("want 3 results, got %d", len(got.results))
	}
	if got.results[0].Partial || got.results[1].Partial {
		t.Errorf("first two should be final: %+v", got.results)
	}
	if !got.results[2].Partial {
		t.Errorf("third should be partial: %+v", got.results[2])
	}
	if got.results[1].Segment.End != 4500*time.Millisecond {
		t.Errorf("absolute end wrong: %v", got.results[1].Segment.End)
	}
	if got.committed != 4500*time.Millisecond {
		t.Errorf("committed should advance to last final end (4.5s), got %v", got.committed)
	}
	if got.promptAdd != " hello" && got.promptAdd != " already hello" {
		t.Errorf("promptAdd should accumulate final text, got %q", got.promptAdd)
	}
}

func TestClassifyWindow_SkipAlreadyCommitted(t *testing.T) {
	// committed=3s; a re-decoded overlap segment ending at 3s must be skipped.
	segs := []Segment{
		seg(0.0, 1.0, " dup"),   // abs 2..3 -> end<=committed -> SKIP
		seg(1.0, 2.0, " fresh"), // abs 3..4 -> FINAL
	}
	got := classifyWindow(segs, 2*time.Second, 3*time.Second, 10*time.Second)
	if len(got.results) != 1 || got.results[0].Segment.Text != " fresh" {
		t.Fatalf("want only fresh segment, got %+v", got.results)
	}
	if got.committed != 4*time.Second {
		t.Errorf("committed should be 4s, got %v", got.committed)
	}
}

func TestClassifyWindow_FlushCommitsTail(t *testing.T) {
	// commitBefore == consumed (flush): the trailing segment becomes final.
	segs := []Segment{seg(0, 1.5, " tail")} // abs 0..1.5
	got := classifyWindow(segs, 0, 0, 1500*time.Millisecond)
	if len(got.results) != 1 || got.results[0].Partial {
		t.Fatalf("flush should commit tail as final: %+v", got.results)
	}
	if got.committed != 1500*time.Millisecond {
		t.Errorf("committed should be 1.5s, got %v", got.committed)
	}
}

func TestClassifyWindow_Empty(t *testing.T) {
	got := classifyWindow(nil, 0, 0, time.Second)
	if len(got.results) != 0 || got.committed != 0 || got.promptAdd != "" {
		t.Fatalf("empty input should yield empty result: %+v", got)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test -run TestClassifyWindow ./ -count=1`
Expected: FAIL — undefined `classifyWindow`.

- [ ] **Step 3: Write the implementation**

Create `stream_window.go`:

```go
package whisper

import "time"

// windowResult is the outcome of classifying one window run.
type windowResult struct {
	results   []StreamResult // ordered partial+final updates with absolute timestamps
	committed time.Duration  // updated committed mark
	promptAdd string         // concatenated text of newly-committed finals (carry-over)
}

// classifyWindow turns a window run's segments (timestamps relative to windowStart) into
// ordered StreamResults. A segment is final iff committed < absEnd <= commitBefore; one
// whose end is already <= committed is a re-decoded overlap and is skipped; everything
// past commitBefore is a (refining) partial. Pure: no whisper, no goroutines, no clock.
func classifyWindow(segs []Segment, windowStart, committed, commitBefore time.Duration) windowResult {
	out := windowResult{committed: committed}
	for _, s := range segs {
		absStart := s.Start + windowStart
		absEnd := s.End + windowStart
		if absEnd <= out.committed {
			continue // already committed (overlap duplicate)
		}
		abs := Segment{Start: absStart, End: absEnd, Text: s.Text, Tokens: s.Tokens}
		if absEnd <= commitBefore {
			out.results = append(out.results, StreamResult{Segment: abs, Partial: false})
			out.committed = absEnd
			out.promptAdd += s.Text
		} else {
			out.results = append(out.results, StreamResult{Segment: abs, Partial: true})
		}
	}
	return out
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test -run TestClassifyWindow ./ -count=1`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add stream_window.go stream_window_test.go
git commit -m "feat(stream): pure classifyWindow commit logic"
```

---

### Task 4: Bounded PCM buffer (block / drop / slide)

**Files:**
- Create: `stream_buffer.go`
- Test: `stream_buffer_test.go`

`streamBuffer` holds PCM from an absolute `start` time up to `consumed`. `write` appends, applying backpressure (block) or dropping the oldest audio (drop mode). `nextWindow` blocks until `step` of new audio (or close/cancel) and returns the trailing `window`. `slide` discards committed audio. A goroutine broadcasts on `ctx` cancel so blocked waiters wake.

- [ ] **Step 1: Write the failing test**

Create `stream_buffer_test.go`:

```go
package whisper

import (
	"context"
	"testing"
	"time"
)

const rate = 16000

func ones(d time.Duration) []float32 {
	n := int(d.Seconds() * float64(rate))
	s := make([]float32, n)
	for i := range s {
		s[i] = 1
	}
	return s
}

func TestStreamBuffer_NextWindowWaitsForStep(t *testing.T) {
	b := newStreamBuffer(rate, 0, 0, context.Background())
	_ = b.write(ones(300 * time.Millisecond)) // < step
	done := make(chan struct{})
	go func() {
		// blocks until 500ms of new audio total
		_, _, _, _, _, ok := b.nextWindow(0, 500*time.Millisecond, 10*time.Second)
		if !ok {
			t.Errorf("nextWindow returned !ok")
		}
		close(done)
	}()
	select {
	case <-done:
		t.Fatal("nextWindow returned before step reached")
	case <-time.After(50 * time.Millisecond):
	}
	_ = b.write(ones(300 * time.Millisecond)) // now 600ms >= step
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("nextWindow did not wake after step reached")
	}
}

func TestStreamBuffer_NextWindowTrailingWindow(t *testing.T) {
	b := newStreamBuffer(rate, 0, 0, context.Background())
	_ = b.write(ones(12 * time.Second)) // > window
	pcm, ws, consumed, _, closed, ok := b.nextWindow(0, time.Second, 10*time.Second)
	if !ok || closed {
		t.Fatal("unexpected ok/closed")
	}
	if consumed != 12*time.Second {
		t.Errorf("consumed = %v want 12s", consumed)
	}
	if ws != 2*time.Second { // trailing 10s window
		t.Errorf("windowStart = %v want 2s", ws)
	}
	if got := durFor(len(pcm)); got != 10*time.Second {
		t.Errorf("window len = %v want 10s", got)
	}
}

func TestStreamBuffer_DropOldestOnOverrun(t *testing.T) {
	b := newStreamBuffer(rate, 0, 4*time.Second, context.Background()) // drop beyond 4s
	_ = b.write(ones(3 * time.Second))
	_ = b.write(ones(3 * time.Second)) // total 6s > 4s -> drop oldest 2s
	pcm, ws, consumed, dropped, _, _ := b.nextWindow(0, time.Second, 10*time.Second)
	if !dropped {
		t.Error("expected dropped=true")
	}
	if consumed != 6*time.Second {
		t.Errorf("consumed = %v want 6s", consumed)
	}
	if ws != 2*time.Second || durFor(len(pcm)) != 4*time.Second {
		t.Errorf("after drop want start=2s len=4s, got start=%v len=%v", ws, durFor(len(pcm)))
	}
}

func TestStreamBuffer_SlideDiscards(t *testing.T) {
	b := newStreamBuffer(rate, 0, 0, context.Background())
	_ = b.write(ones(5 * time.Second))
	b.slide(3 * time.Second)
	pcm, ws, _, _, _, _ := b.nextWindow(0, time.Second, 10*time.Second)
	if ws != 3*time.Second || durFor(len(pcm)) != 2*time.Second {
		t.Errorf("after slide want start=3s len=2s, got start=%v len=%v", ws, durFor(len(pcm)))
	}
}

func TestStreamBuffer_CloseSendUnblocks(t *testing.T) {
	b := newStreamBuffer(rate, 0, 0, context.Background())
	_ = b.write(ones(100 * time.Millisecond)) // < step
	done := make(chan bool)
	go func() {
		_, _, _, _, closed, ok := b.nextWindow(0, time.Second, 10*time.Second)
		done <- (ok && closed)
	}()
	b.closeSend()
	select {
	case got := <-done:
		if !got {
			t.Error("expected ok && closed after CloseSend")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("nextWindow did not wake on CloseSend")
	}
	if err := b.write(ones(10 * time.Millisecond)); err == nil {
		t.Error("write after closeSend should error")
	}
}

func TestStreamBuffer_CtxCancelUnblocks(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	b := newStreamBuffer(rate, 0, 0, ctx)
	done := make(chan bool)
	go func() {
		_, _, _, _, _, ok := b.nextWindow(0, time.Second, 10*time.Second)
		done <- ok
	}()
	cancel()
	select {
	case ok := <-done:
		if ok {
			t.Error("expected ok=false after ctx cancel")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("nextWindow did not wake on ctx cancel")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test -run TestStreamBuffer ./ -count=1`
Expected: FAIL — undefined `newStreamBuffer`, `durFor`.

- [ ] **Step 3: Write the implementation**

Create `stream_buffer.go`:

```go
package whisper

import (
	"context"
	"sync"
	"time"
)

const sampleRate = 16000

func samplesFor(d time.Duration) int { return int(d.Seconds() * float64(sampleRate)) }
func durFor(n int) time.Duration      { return time.Duration(n) * time.Second / sampleRate }

// streamBuffer is a bounded PCM ring fed by Write and drained by the worker's nextWindow.
// All durations are absolute, measured from stream start.
type streamBuffer struct {
	mu        sync.Mutex
	cond      *sync.Cond
	pcm       []float32     // samples covering [start, consumed)
	start     time.Duration // absolute time of pcm[0]
	consumed  time.Duration // total audio that has entered the buffer
	blockCap  time.Duration // block mode: max (consumed-start) before Write blocks (0 in drop mode)
	dropCap   time.Duration // drop mode: max (consumed-start) before dropping oldest (0 in block mode)
	dropped   bool          // a drop happened since the last nextWindow
	closed    bool          // CloseSend called
	ctx       context.Context
}

// newStreamBuffer builds a buffer. blockCap>0 => block mode; dropCap>0 => drop mode.
func newStreamBuffer(_ int, blockCap, dropCap time.Duration, ctx context.Context) *streamBuffer {
	b := &streamBuffer{blockCap: blockCap, dropCap: dropCap, ctx: ctx}
	b.cond = sync.NewCond(&b.mu)
	// Wake blocked waiters when the context is canceled.
	go func() {
		<-ctx.Done()
		b.mu.Lock()
		b.cond.Broadcast()
		b.mu.Unlock()
	}()
	return b
}

func (b *streamBuffer) write(samples []float32) error {
	if len(samples) == 0 {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	// Block mode: wait until there is room (worker slides as it commits).
	for b.blockCap > 0 && (b.consumed-b.start) >= b.blockCap && !b.closed && b.ctx.Err() == nil {
		b.cond.Wait()
	}
	if b.closed {
		return ErrStreamClosed
	}
	if b.ctx.Err() != nil {
		return b.ctx.Err()
	}
	b.pcm = append(b.pcm, samples...)
	b.consumed += durFor(len(samples))
	// Drop mode: trim oldest audio beyond dropCap.
	if b.dropCap > 0 && (b.consumed-b.start) > b.dropCap {
		trim := samplesFor((b.consumed - b.start) - b.dropCap)
		if trim > len(b.pcm) {
			trim = len(b.pcm)
		}
		b.pcm = append(b.pcm[:0], b.pcm[trim:]...)
		b.start += durFor(trim)
		b.dropped = true
	}
	b.cond.Broadcast()
	return nil
}

func (b *streamBuffer) closeSend() {
	b.mu.Lock()
	b.closed = true
	b.cond.Broadcast()
	b.mu.Unlock()
}

// nextWindow blocks until >= step new audio beyond sinceConsumed, or CloseSend, or ctx
// cancel. Returns a COPY of the trailing window of audio, its absolute windowStart, the
// current consumed mark, whether audio was dropped since the last call, and closed. ok is
// false only on ctx cancel.
func (b *streamBuffer) nextWindow(sinceConsumed, step, window time.Duration) (
	pcm []float32, windowStart, consumed time.Duration, dropped, closed, ok bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for {
		if b.ctx.Err() != nil {
			return nil, 0, 0, false, b.closed, false
		}
		ready := (b.consumed - sinceConsumed) >= step
		if b.closed || ready {
			// Trailing window: cap to the last `window` seconds of buffered audio.
			ws := b.start
			off := 0
			if b.consumed-b.start > window {
				ws = b.consumed - window
				off = samplesFor(ws - b.start)
				if off > len(b.pcm) {
					off = len(b.pcm)
				}
			}
			out := make([]float32, len(b.pcm)-off)
			copy(out, b.pcm[off:])
			dropped = b.dropped
			b.dropped = false
			return out, ws, b.consumed, dropped, b.closed, true
		}
		b.cond.Wait()
	}
}

// slide discards buffered audio whose absolute time is before newStart.
func (b *streamBuffer) slide(newStart time.Duration) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if newStart <= b.start {
		return
	}
	trim := samplesFor(newStart - b.start)
	if trim > len(b.pcm) {
		trim = len(b.pcm)
	}
	b.pcm = append(b.pcm[:0], b.pcm[trim:]...)
	b.start += durFor(trim)
	b.cond.Broadcast() // unblock any Write waiting on backpressure
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test -run TestStreamBuffer ./ -count=1`
Expected: PASS (6 tests). If the drop/slide sample arithmetic is off by one sample at boundaries, prefer the assertions' durations — `durFor`/`samplesFor` round via integer division, so a 1-sample slack is acceptable; adjust the test tolerance with `±durFor(1)` if needed.

- [ ] **Step 5: Commit**

```bash
git add stream_buffer.go stream_buffer_test.go
git commit -m "feat(stream): bounded PCM buffer with block/drop/slide"
```

---

### Task 5: The `Stream` type and worker loop

**Files:**
- Create: `stream.go`

Ties Tasks 2–4 together: `NewStream` builds the buffer + owned/borrowed Session + spawns the worker; the worker loops `nextWindow → Session.Transcribe → classifyWindow → slide → emit`.

- [ ] **Step 1: Write the implementation**

Create `stream.go`:

```go
package whisper

import (
	"context"
	"errors"
	"sync"
	"time"
)

// Stream performs sliding-window streaming transcription. Create with Model.NewStream or
// Session.NewStream. Feed 16 kHz mono audio with Write; read updates from Results; signal
// end of audio with CloseSend; abort with Close. Not safe for concurrent Write calls.
type Stream struct {
	ctx        context.Context
	cancel     context.CancelFunc
	session    *Session
	ownSession bool
	cfg        streamConfig
	buf        *streamBuffer
	results    chan StreamResult
	wg         sync.WaitGroup
	closeOnce  sync.Once

	committed time.Duration // worker-only
	prompt    string        // worker-only carry-over (capped)

	errMu sync.Mutex
	err   error
}

// NewStream starts a streaming session that allocates and owns a fresh Session (freed on Close).
func (m *Model) NewStream(ctx context.Context, opts ...StreamOption) (*Stream, error) {
	s, err := m.NewSession()
	if err != nil {
		return nil, err
	}
	st, err := s.newStream(ctx, true, opts...)
	if err != nil {
		_ = s.Close()
		return nil, err
	}
	return st, nil
}

// NewStream starts a streaming session over the caller's Session (not freed on Close).
func (s *Session) NewStream(ctx context.Context, opts ...StreamOption) (*Stream, error) {
	return s.newStream(ctx, false, opts...)
}

func (s *Session) newStream(ctx context.Context, own bool, opts ...StreamOption) (*Stream, error) {
	if s == nil || s.state == nil {
		return nil, ErrClosed
	}
	cfg := defaultStreamConfig()
	for _, o := range opts {
		o(&cfg)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	cctx, cancel := context.WithCancel(ctx)
	// Cap is on (consumed-start); start==committed-keep after a slide, so capping at
	// `window` keeps uncommitted audio (consumed-committed) at <= window-keep — lossless.
	blockCap := cfg.window
	dropCap := time.Duration(0)
	if cfg.dropMaxBuffer > 0 {
		blockCap, dropCap = 0, cfg.dropMaxBuffer
	}
	st := &Stream{
		ctx:        cctx,
		cancel:     cancel,
		session:    s,
		ownSession: own,
		cfg:        cfg,
		buf:        newStreamBuffer(sampleRate, blockCap, dropCap, cctx),
		results:    make(chan StreamResult, 16),
	}
	st.wg.Add(1)
	go st.run()
	return st, nil
}

// Write feeds 16 kHz mono samples. Blocks (backpressure) or drops oldest per policy.
func (st *Stream) Write(samples []float32) error {
	if st.ctx.Err() != nil {
		return st.ctx.Err()
	}
	return st.buf.write(samples)
}

// CloseSend signals end of audio: the worker flushes a final window then closes Results.
func (st *Stream) CloseSend() error {
	if st.ctx.Err() != nil {
		return st.ctx.Err()
	}
	st.buf.closeSend()
	return nil
}

// Results returns the channel of ordered partial+final updates. Closed when the stream ends.
func (st *Stream) Results() <-chan StreamResult { return st.results }

// Err returns the terminal error (nil on clean EOF). Valid once Results is closed.
func (st *Stream) Err() error {
	st.errMu.Lock()
	defer st.errMu.Unlock()
	return st.err
}

// Close aborts the stream: cancels in-flight inference, waits for the worker, frees an
// owned Session. Idempotent.
func (st *Stream) Close() error {
	st.closeOnce.Do(func() {
		st.cancel()
		st.wg.Wait()
		if st.ownSession {
			_ = st.session.Close()
		}
	})
	return nil
}

func (st *Stream) setErr(err error) {
	st.errMu.Lock()
	if st.err == nil {
		st.err = err
	}
	st.errMu.Unlock()
}

func (st *Stream) appendPrompt(s string) {
	if s == "" {
		return
	}
	st.prompt += s
	const maxRunes = 224 // modest carry-over budget
	if r := []rune(st.prompt); len(r) > maxRunes {
		st.prompt = string(r[len(r)-maxRunes:])
	}
}

func (st *Stream) run() {
	defer st.wg.Done()
	defer close(st.results)

	var since time.Duration // consumed mark at the last run
	for {
		pcm, windowStart, consumed, dropped, closed, ok := st.buf.nextWindow(since, st.cfg.step, st.cfg.window)
		if !ok {
			st.setErr(st.ctx.Err())
			return
		}
		since = consumed
		if len(pcm) == 0 {
			if closed {
				return
			}
			continue
		}
		commitBefore := consumed - st.cfg.keep
		if closed {
			commitBefore = consumed // flush: commit the tail too
		}
		opts := append([]TranscribeOption{WithInitialPrompt(st.prompt)}, st.cfg.transcribe...)
		res, err := st.session.Transcribe(st.ctx, pcm, opts...)
		if err != nil {
			if errors.Is(err, ErrCanceled) {
				st.setErr(st.ctx.Err())
			} else {
				st.setErr(err)
			}
			return
		}
		cr := classifyWindow(res.Segments, windowStart, st.committed, commitBefore)
		st.committed = cr.committed
		st.appendPrompt(cr.promptAdd)
		if st.committed > st.cfg.keep {
			st.buf.slide(st.committed - st.cfg.keep)
		}
		for _, r := range cr.results {
			if dropped {
				r.Lagging = true
			}
			select {
			case st.results <- r:
			case <-st.ctx.Done():
				st.setErr(st.ctx.Err())
				return
			}
		}
		if closed {
			return
		}
	}
}
```

- [ ] **Step 2: Verify it builds**

Run: `go build ./...`
Expected: success. (`WithInitialPrompt` and `Session.Transcribe` already exist; `Segment.Tokens` exists.)

- [ ] **Step 3: Vet**

Run: `go vet ./...`
Expected: no findings.

- [ ] **Step 4: Run all existing pure tests still pass**

Run: `go test -run 'TestStream|TestClassifyWindow' ./ -count=1`
Expected: PASS (no regressions).

- [ ] **Step 5: Commit**

```bash
git add stream.go
git commit -m "feat(stream): Stream type, NewStream, sliding-window worker"
```

---

### Task 6: Gated integration test (jfk.wav in chunks)

**Files:**
- Create: `stream_test.go`

- [ ] **Step 1: Write the test**

Create `stream_test.go`:

```go
package whisper_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	whisper "github.com/dyammarcano/go-whisper.cpp"
	"github.com/dyammarcano/go-whisper.cpp/wav"
)

func TestStream_Integration(t *testing.T) {
	mp := os.Getenv("TEST_MODEL")
	if mp == "" {
		t.Skip("set TEST_MODEL (task models:tiny) to run streaming integration")
	}
	m, err := whisper.New(mp)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = m.Close() }()

	samples, err := wav.ReadFile("whisper.cpp/samples/jfk.wav")
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}

	st, err := m.NewStream(context.Background(),
		whisper.WithStreamStep(500*time.Millisecond),
		whisper.WithStreamWindow(10*time.Second),
		whisper.WithTranscribeOptions(whisper.WithLanguage("en")))
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}

	// Feeder: push the clip in 0.5s chunks, then signal EOF.
	go func() {
		const chunk = 16000 / 2 // 0.5s @ 16kHz
		for i := 0; i < len(samples); i += chunk {
			end := i + chunk
			if end > len(samples) {
				end = len(samples)
			}
			if err := st.Write(samples[i:end]); err != nil {
				return
			}
		}
		_ = st.CloseSend()
	}()

	var finals []whisper.StreamResult
	sawPartial := false
	var lastFinalEnd time.Duration
	for r := range st.Results() {
		if r.Partial {
			sawPartial = true
			continue
		}
		if r.Segment.End < lastFinalEnd {
			t.Errorf("finals not monotonic: %v after %v", r.Segment.End, lastFinalEnd)
		}
		lastFinalEnd = r.Segment.End
		finals = append(finals, r)
	}
	if err := st.Err(); err != nil {
		t.Fatalf("stream err: %v", err)
	}
	if len(finals) == 0 {
		t.Fatal("no final segments produced")
	}
	var b strings.Builder
	for _, r := range finals {
		b.WriteString(r.Segment.Text)
	}
	full := strings.ToLower(b.String())
	if !strings.Contains(full, "country") {
		t.Errorf("streamed transcript missing 'country'; got: %q", full)
	}
	if !sawPartial {
		t.Log("warning: no partial observed (acceptable for a short clip, but unexpected)")
	}
}
```

- [ ] **Step 2: Run it (gated)**

Run (PowerShell):
```powershell
$env:TEST_MODEL = "$PWD\models\ggml-tiny.en.bin"
go test -run TestStream_Integration ./ -count=1 -timeout 120s -v
```
Expected: PASS. The concatenated finals contain "country". If it crashes, the OpenMP cancel fix is required — confirm `whisper.cpp/build-cpu` was built by the current `scripts/whispercpp.sh` (which sets `GGML_OPENMP=OFF`).

- [ ] **Step 3: Commit**

```bash
git add stream_test.go
git commit -m "test(stream): gated jfk.wav streaming integration"
```

---

### Task 7: File-fed streaming example

**Files:**
- Create: `examples/stream/main.go`

- [ ] **Step 1: Write the example**

Create `examples/stream/main.go`:

```go
// Command stream demonstrates streaming transcription by feeding a WAV file in chunks.
// Usage: go run ./examples/stream <model.bin> <audio.wav>
package main

import (
	"context"
	"fmt"
	"os"
	"time"

	whisper "github.com/dyammarcano/go-whisper.cpp"
	"github.com/dyammarcano/go-whisper.cpp/wav"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: stream <model.bin> <audio.wav>")
		os.Exit(2)
	}
	m, err := whisper.New(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "load model:", err)
		os.Exit(1)
	}
	defer func() { _ = m.Close() }()

	samples, err := wav.ReadFile(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "read wav:", err)
		os.Exit(1)
	}

	st, err := m.NewStream(context.Background(),
		whisper.WithStreamStep(500*time.Millisecond),
		whisper.WithTranscribeOptions(whisper.WithLanguage("en")))
	if err != nil {
		fmt.Fprintln(os.Stderr, "new stream:", err)
		os.Exit(1)
	}

	go func() {
		const chunk = 16000 / 2 // 0.5s
		for i := 0; i < len(samples); i += chunk {
			end := i + chunk
			if end > len(samples) {
				end = len(samples)
			}
			_ = st.Write(samples[i:end])
			time.Sleep(50 * time.Millisecond) // simulate real-time-ish arrival
		}
		_ = st.CloseSend()
	}()

	for r := range st.Results() {
		tag := "FINAL  "
		if r.Partial {
			tag = "partial"
		}
		fmt.Printf("[%s %6.2fs-%6.2fs]%s\n", tag, r.Segment.Start.Seconds(), r.Segment.End.Seconds(), r.Segment.Text)
	}
	if err := st.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "stream error:", err)
		os.Exit(1)
	}
}
```

- [ ] **Step 2: Verify it builds**

Run: `go build ./examples/stream`
Expected: success.

- [ ] **Step 3: (Optional) run it**

Run (PowerShell):
```powershell
go run ./examples/stream models\ggml-tiny.en.bin whisper.cpp\samples\jfk.wav
```
Expected: a stream of `partial`/`FINAL` lines ending with the JFK quote.

- [ ] **Step 4: Commit**

```bash
git add examples/stream/main.go
git commit -m "docs(example): file-fed streaming demo"
```

---

### Task 8: Documentation (README + CHANGELOG)

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a README section**

Insert after the "Speaker diarization" section (before "## Running tests"):

````markdown
## Streaming transcription

Feed audio incrementally and receive refining **partial** results plus committed **final**
segments, via a sliding window over `Transcribe` (whisper itself decodes ≤30 s windows, so
streaming is a windowing layer — no native incremental decoding).

```go
st, _ := m.NewStream(ctx,
    whisper.WithStreamStep(500*time.Millisecond),
    whisper.WithDropOnOverrun(8*time.Second),                 // live mic: bound latency
    whisper.WithTranscribeOptions(whisper.WithLanguage("en")))

go func() {
    for chunk := range mic { _ = st.Write(chunk) }            // []float32, 16 kHz mono
    _ = st.CloseSend()
}()

for r := range st.Results() {
    if r.Partial { redraw(r.Segment.Text) } else { commit(r.Segment.Text) }
}
if err := st.Err(); err != nil { /* handle */ }
```

Tuning: `WithStreamStep` (cadence), `WithStreamWindow` (≤30 s reprocessed each run),
`WithStreamKeep` (decoder-context overlap). Default is lossless backpressure (`Write`
blocks); `WithDropOnOverrun` switches to bounded drop-oldest with `StreamResult.Lagging`.
See [`examples/stream`](examples/stream).
````

- [ ] **Step 2: Add a CHANGELOG entry**

Add an `## [Unreleased]` section above `## [0.1.0]` in `CHANGELOG.md`:

```markdown
## [Unreleased]

### Added

- **Streaming transcription** (`Stream`, `Model.NewStream`/`Session.NewStream`): incremental
  `Write` + channel of partial/final `StreamResult`s via a fixed-interval sliding window;
  lossless backpressure by default, opt-in `WithDropOnOverrun` for live audio.
```

- [ ] **Step 3: Verify docs build / links**

Run: `go build ./...`
Expected: success (sanity; docs are markdown).

- [ ] **Step 4: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: README streaming section + CHANGELOG entry"
```

---

## Final verification

- [ ] Run the full pure-Go suite: `go test -run 'TestStream|TestClassifyWindow' ./ -count=1` → PASS.
- [ ] Run the gated integration (PowerShell): set `TEST_MODEL`, then
  `go test -run TestStream_Integration ./ -count=1 -timeout 120s` → PASS.
- [ ] Run the full whisper suite to confirm no regressions (compiled-exe form to bypass the test-reporter hook):
  `go test -c -o whisper.test.exe . ; .\whisper.test.exe -test.v ; Remove-Item whisper.test.exe` → all specs pass, 0 skipped.
- [ ] `go vet ./...` and `golangci-lint run ./...` → clean.
- [ ] Then use **superpowers:finishing-a-development-branch** to merge `feat/whisper-streaming`.
```
