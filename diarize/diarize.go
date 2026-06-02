// Package diarize provides native speaker diarization ("who spoke when") via
// sherpa-onnx + pyannote models (ONNX Runtime), with no Python dependency. It
// depends only on sherpa-onnx-go (never on the whisper.cpp binding); use Label to
// fuse Turns with a transcript.
package diarize

import (
	"fmt"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// Diarizer wraps a sherpa-onnx OfflineSpeakerDiarization. Not safe for concurrent
// Diarize calls; create one per goroutine if needed. Close frees native memory.
type Diarizer struct {
	sd *sherpa.OfflineSpeakerDiarization
}

// New builds a diarizer from segmentation + embedding ONNX model paths.
func New(segmentationModel, embeddingModel string, opts ...Option) (*Diarizer, error) {
	o := defaultOptions()
	for _, opt := range opts {
		opt(&o)
	}
	var c sherpa.OfflineSpeakerDiarizationConfig
	c.Segmentation.Pyannote.Model = segmentationModel
	c.Segmentation.NumThreads = o.threads
	c.Segmentation.Provider = "cpu"
	c.Embedding.Model = embeddingModel
	c.Embedding.NumThreads = o.threads
	c.Embedding.Provider = "cpu"
	if o.debug {
		c.Segmentation.Debug = 1
		c.Embedding.Debug = 1
	}
	if o.numSpeakers > 0 {
		c.Clustering.NumClusters = o.numSpeakers
	} else {
		c.Clustering.Threshold = o.threshold
	}
	c.MinDurationOn = float32(o.minOn.Seconds())
	c.MinDurationOff = float32(o.minOff.Seconds())

	sd := sherpa.NewOfflineSpeakerDiarization(&c)
	if sd == nil {
		return nil, fmt.Errorf("%w: segmentation=%q embedding=%q", ErrConfig, segmentationModel, embeddingModel)
	}
	return &Diarizer{sd: sd}, nil
}

// Close frees the underlying native diarizer. Idempotent.
func (d *Diarizer) Close() error {
	if d == nil || d.sd == nil {
		return nil
	}
	sherpa.DeleteOfflineSpeakerDiarization(d.sd)
	d.sd = nil
	return nil
}

// SampleRate is the required input rate (16000). Returns 0 if closed.
func (d *Diarizer) SampleRate() int {
	if d == nil || d.sd == nil {
		return 0
	}
	return d.sd.SampleRate()
}

// Diarize runs offline diarization over 16 kHz mono float32 samples in [-1,1] and
// returns speaker turns ordered by Start.
func (d *Diarizer) Diarize(samples []float32) ([]Turn, error) {
	if d == nil || d.sd == nil {
		return nil, ErrClosed
	}
	if len(samples) == 0 {
		return nil, ErrEmptyAudio
	}
	segs := d.sd.Process(samples)
	turns := make([]Turn, len(segs))
	for i, s := range segs {
		turns[i] = Turn{
			Start:   secondsToDuration(s.Start),
			End:     secondsToDuration(s.End),
			Speaker: s.Speaker,
		}
	}
	return turns, nil
}

// Turn is one speaker's contiguous span. Speaker is a 0-based id, per-file (NOT
// stable across different recordings).
type Turn struct {
	Start, End time.Duration
	Speaker    int
}

func secondsToDuration(s float32) time.Duration {
	return time.Duration(float64(s) * float64(time.Second))
}
