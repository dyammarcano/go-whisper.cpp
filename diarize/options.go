package diarize

import "time"

// Option configures a Diarizer.
type Option func(*options)

type options struct {
	numSpeakers int     // >0 => known count (overrides threshold)
	threshold   float32 // used when numSpeakers == 0
	minOn       time.Duration
	minOff      time.Duration
	threads     int
	debug       bool
}

func defaultOptions() options {
	return options{
		numSpeakers: 0,
		threshold:   0.5, // unknown-count default; larger => fewer speakers
		minOn:       300 * time.Millisecond,
		minOff:      500 * time.Millisecond,
		threads:     1,
	}
}

// WithNumSpeakers forces exactly n speakers (use when the count is known). Mutually
// exclusive with WithThreshold; whichever is applied last wins.
func WithNumSpeakers(n int) Option { return func(o *options) { o.numSpeakers = n } }

// WithThreshold sets the clustering threshold for unknown speaker count (larger =>
// fewer speakers; default 0.5). Clears any WithNumSpeakers.
func WithThreshold(t float32) Option {
	return func(o *options) { o.numSpeakers = 0; o.threshold = t }
}

// WithMinDuration sets minimum on/off speech durations (defaults 300ms / 500ms).
func WithMinDuration(on, off time.Duration) Option {
	return func(o *options) { o.minOn = on; o.minOff = off }
}

// WithThreads sets ONNX intra-op threads for both models (default 1).
func WithThreads(n int) Option { return func(o *options) { o.threads = n } }

// WithDebug enables sherpa-onnx debug logging.
var WithDebug Option = func(o *options) { o.debug = true }
