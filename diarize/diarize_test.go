package diarize_test

import (
	"os"
	"testing"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"github.com/dyammarcano/go-whisper.cpp/diarize"
)

// gated: set DIARIZE_SEG_MODEL + DIARIZE_EMB_MODEL + DIARIZE_WAV (a 16kHz mono multi-
// speaker wav). Provision with `task models:diarize` + `task diarize:dlls`.
func TestDiarize_Integration(t *testing.T) {
	seg := os.Getenv("DIARIZE_SEG_MODEL")
	emb := os.Getenv("DIARIZE_EMB_MODEL")
	wavPath := os.Getenv("DIARIZE_WAV")
	if seg == "" || emb == "" || wavPath == "" {
		t.Skip("set DIARIZE_SEG_MODEL, DIARIZE_EMB_MODEL, DIARIZE_WAV to run")
	}

	d, err := diarize.New(seg, emb, diarize.WithThreshold(0.5))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = d.Close() }()
	if d.SampleRate() != 16000 {
		t.Fatalf("sample rate = %d want 16000", d.SampleRate())
	}

	w := sherpa.ReadWave(wavPath)
	if w == nil {
		t.Fatalf("ReadWave(%q) failed", wavPath)
	}
	if w.SampleRate != d.SampleRate() {
		t.Fatalf("wav rate %d != required %d", w.SampleRate, d.SampleRate())
	}
	turns, err := d.Diarize(w.Samples)
	if err != nil {
		t.Fatalf("Diarize: %v", err)
	}
	if len(turns) == 0 {
		t.Fatal("no turns produced")
	}
	speakers := map[int]bool{}
	for _, tn := range turns {
		if tn.End <= tn.Start {
			t.Errorf("non-positive turn: %+v", tn)
		}
		speakers[tn.Speaker] = true
	}
	t.Logf("turns=%d speakers=%d", len(turns), len(speakers))
	if len(speakers) < 2 {
		t.Errorf("expected >=2 speakers in a multi-speaker clip, got %d", len(speakers))
	}
}

func TestDiarize_EmptyAndClosed(t *testing.T) {
	// pure-API guards — no model needed for the closed path
	var d *diarize.Diarizer
	if _, err := d.Diarize([]float32{0.1}); err == nil {
		t.Error("nil diarizer Diarize should error")
	}
}
