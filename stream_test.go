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
