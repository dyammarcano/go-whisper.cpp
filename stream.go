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
