package whisper

import (
	"errors"
	"fmt"
	"sync/atomic"
)

var (
	ErrModelLoad  = errors.New("whisper: failed to load model")
	ErrStateInit  = errors.New("whisper: failed to init state")
	ErrTranscribe = errors.New("whisper: transcription failed")
	ErrCanceled   = errors.New("whisper: transcription canceled")
	ErrEmptyAudio = errors.New("whisper: empty audio (no samples)")
	ErrClosed     = errors.New("whisper: use of closed model or session")

	// ErrStreamClosed is returned by Stream.Write/CloseSend after the stream is closed.
	ErrStreamClosed = errors.New("whisper: stream closed")

	// ErrConfig indicates invalid configuration.
	ErrConfig = errors.New("whisper: invalid config")
)

// abortFlag is a goroutine-safe one-shot flag read by the abort trampoline.
type abortFlag struct{ v atomic.Bool }

func (a *abortFlag) set()        { a.v.Store(true) }
func (a *abortFlag) isSet() bool { return a.v.Load() }

func panicToError(r any) error {
	if e, ok := r.(error); ok {
		return fmt.Errorf("panic in callback: %w", e)
	}
	return fmt.Errorf("panic in callback: %v", r)
}
