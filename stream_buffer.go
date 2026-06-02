package whisper

import (
	"context"
	"sync"
	"time"
)

const sampleRate = 16000

func samplesFor(d time.Duration) int { return int(d.Seconds() * float64(sampleRate)) }
func durFor(n int) time.Duration     { return time.Duration(n) * time.Second / sampleRate }

// streamBuffer is a bounded PCM ring fed by write and drained by nextWindow.
// All durations are absolute, measured from stream start.
type streamBuffer struct {
	mu       sync.Mutex
	cond     *sync.Cond
	pcm      []float32
	start    time.Duration
	consumed time.Duration
	blockCap time.Duration
	dropCap  time.Duration
	dropped  bool
	closed   bool
	ctx      context.Context //nolint:containedctx // goroutine waiting on ctx.Done() is bound to this ctx for the buffer's lifetime
}

// newStreamBuffer builds a buffer. blockCap>0 => block mode; dropCap>0 => drop mode.
func newStreamBuffer(_ int, blockCap, dropCap time.Duration, ctx context.Context) *streamBuffer {
	b := &streamBuffer{blockCap: blockCap, dropCap: dropCap, ctx: ctx}
	b.cond = sync.NewCond(&b.mu)
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
	if b.dropCap > 0 && (b.consumed-b.start) > b.dropCap {
		trim := min(samplesFor((b.consumed-b.start)-b.dropCap), len(b.pcm))
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

// nextWindow blocks until >= step new audio beyond sinceConsumed, or closeSend, or ctx
// cancel. Returns a COPY of the trailing window of audio, its absolute windowStart, the
// current consumed mark, whether audio was dropped since the last call, closed, and ok.
// ok is false only on ctx cancel.
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
			ws := b.start
			off := 0
			if b.consumed-b.start > window {
				ws = b.consumed - window
				off = min(samplesFor(ws-b.start), len(b.pcm))
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
	trim := min(samplesFor(newStart-b.start), len(b.pcm))
	b.pcm = append(b.pcm[:0], b.pcm[trim:]...)
	b.start += durFor(trim)
	b.cond.Broadcast()
}
