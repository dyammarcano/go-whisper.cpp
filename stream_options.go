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
	step          time.Duration
	window        time.Duration
	keep          time.Duration
	dropMaxBuffer time.Duration
	transcribe    []TranscribeOption
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
