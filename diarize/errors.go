package diarize

import "errors"

var (
	// ErrConfig means sherpa rejected the config (e.g. missing/invalid model files).
	ErrConfig = errors.New("diarize: invalid configuration (bad or missing models)")
	// ErrEmptyAudio means Diarize was called with no samples.
	ErrEmptyAudio = errors.New("diarize: empty audio (no samples)")
	// ErrClosed means the diarizer was used after Close.
	ErrClosed = errors.New("diarize: use of closed diarizer")
)
