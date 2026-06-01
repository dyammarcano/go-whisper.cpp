package whisper

// #include <stdlib.h>
// #include "binding.h"
import "C"

import (
	"context"
	"fmt"
	"runtime/cgo"
	"unsafe"
)

// Session owns a whisper_state for a single inference at a time. Not safe for
// concurrent use; create one Session per goroutine via Model.NewSession.
type Session struct {
	model *Model
	state unsafe.Pointer // whisper_state*
}

// NewSession allocates an independent inference state over the shared Model.
func (m *Model) NewSession() (*Session, error) {
	if m == nil || m.ptr == nil {
		return nil, ErrClosed
	}
	st := C.whisper_bind_new_state(m.ptr)
	if st == nil {
		return nil, ErrStateInit
	}
	return &Session{model: m, state: st}, nil
}

// Close frees the session's state. Idempotent.
func (s *Session) Close() error {
	if s == nil || s.state == nil {
		return nil
	}
	C.whisper_bind_free_state(s.state)
	s.state = nil
	return nil
}

// Transcribe runs whisper over the session's own state. ctx cancels mid-inference.
func (s *Session) Transcribe(ctx context.Context, samples []float32, opts ...TranscribeOption) (*Result, error) {
	if s == nil || s.state == nil || s.model == nil || s.model.ptr == nil {
		return nil, ErrClosed
	}
	return runFull(ctx, s.model, s, samples, opts...)
}

// Transcribe on the Model uses the context's internal state, serialized by a mutex.
// Convenient for one-shot/CLI use; for concurrency, use Sessions.
func (m *Model) Transcribe(ctx context.Context, samples []float32, opts ...TranscribeOption) (*Result, error) {
	if m == nil || m.ptr == nil {
		return nil, ErrClosed
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return runFull(ctx, m, nil, samples, opts...)
}

// runFull is the shared inference core. session == nil -> use the model's internal state.
func runFull(ctx context.Context, m *Model, session *Session, samples []float32, opts ...TranscribeOption) (*Result, error) {
	if len(samples) == 0 {
		return nil, ErrEmptyAudio
	}
	to := defaultTranscribeOptions()
	for _, o := range opts {
		o(&to)
	}

	aborted := &abortFlag{}
	if ctx.Err() != nil { // already canceled — don't start any work
		aborted.set()
	}
	bridge := &callbackBridge{
		onSegment:  to.onSegment,
		onProgress: to.onProgress,
		aborted:    aborted,
	}
	h := cgo.NewHandle(bridge)
	defer h.Delete() // exactly-once; whisper_full joins all worker threads before returning

	// Cancellation watcher: flips the abort flag when ctx is done.
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		select {
		case <-ctx.Done():
			aborted.set()
		case <-stop:
		}
	}()

	cp := buildCParams(&to, h)
	// cp.language / cp.initial_prompt are C strings that whisper reads DURING whisper_full;
	// they MUST stay alive until whisper_bind_full returns. These defers run at runFull's
	// return (after the call + get_result), so the invariant holds — do not move them earlier.
	var langC *C.char
	if cp.language != nil {
		langC = cp.language
		defer C.free(unsafe.Pointer(langC))
	}
	var promptC *C.char
	if cp.initial_prompt != nil {
		promptC = cp.initial_prompt
		defer C.free(unsafe.Pointer(promptC))
	}

	var statePtr unsafe.Pointer
	if session != nil {
		statePtr = session.state
	}
	//nolint:gocritic // dupSubExpr is a false positive on cgo-expanded args (&samples[0] vs len(samples)).
	rc := C.whisper_bind_full(m.ptr, statePtr, &cp,
		(*C.float)(unsafe.Pointer(&samples[0])), C.int(len(samples)))

	if perr := bridge.panicErr.Load(); perr != nil {
		return nil, *perr
	}
	if aborted.isSet() {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("%w: %w", ErrCanceled, err)
		}
		return nil, ErrCanceled
	}
	if rc != 0 {
		return nil, fmt.Errorf("%w: rc=%d", ErrTranscribe, int(rc))
	}

	want := 0
	if to.tokenTimestamps {
		want = 1
	}
	res := marshalResult(unsafe.Pointer(C.whisper_bind_get_result(m.ptr, statePtr, C.int(want))))
	return res, nil
}

// buildCParams fills the flat C params struct. Caller frees cp.language/initial_prompt.
func buildCParams(to *transcribeOptions, h cgo.Handle) C.whisper_bind_params {
	var cp C.whisper_bind_params
	if to.beamSearch {
		cp.strategy = 1
	}
	cp.n_threads = C.int(to.threads)
	if to.translate {
		cp.translate = 1
	}
	cp.language = C.CString(to.language)
	if to.detectLanguage {
		cp.detect_language = 1
	}
	cp.beam_size = C.int(to.beamSize)
	cp.best_of = C.int(to.bestOf)
	cp.temperature = C.float(to.temperature)
	cp.temperature_inc = C.float(to.temperatureInc)
	cp.entropy_thold = C.float(to.entropyThold)
	cp.logprob_thold = C.float(to.logProbThold)
	cp.no_speech_thold = C.float(to.noSpeechThold)
	if to.noContext {
		cp.no_context = 1
	}
	if to.singleSegment {
		cp.single_segment = 1
	}
	if to.tokenTimestamps {
		cp.token_timestamps = 1
	}
	cp.max_len = C.int(to.maxLen)
	if to.splitOnWord {
		cp.split_on_word = 1
	}
	cp.max_tokens = C.int(to.maxTokens)
	cp.offset_ms = C.int(to.offsetMs)
	cp.duration_ms = C.int(to.durationMs)
	cp.audio_ctx = C.int(to.audioCtx)
	if to.suppressBlank {
		cp.suppress_blank = 1
	}
	if to.suppressNST {
		cp.suppress_nst = 1
	}
	if to.initialPrompt != "" {
		cp.initial_prompt = C.CString(to.initialPrompt)
	}
	if to.onSegment != nil {
		cp.segment_cb = C.uintptr_t(h)
	}
	if to.onProgress != nil {
		cp.progress_cb = C.uintptr_t(h)
	}
	cp.abort_cb = C.uintptr_t(h) // always installed for cancellation
	return cp
}
