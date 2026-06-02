package whisper_test

import (
	"context"
	"os"
	"runtime"
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
			end := min(i+chunk, len(samples))
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

func TestStream_NoGoroutineLeakOnCloseSend(t *testing.T) {
	mp := os.Getenv("TEST_MODEL")
	if mp == "" {
		t.Skip("set TEST_MODEL to run the leak regression")
	}
	m, err := whisper.New(mp)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = m.Close() }()

	silence := make([]float32, 16000) // 1s of silence @16kHz
	runOne := func() {
		st, err := m.NewStream(context.Background(), whisper.WithTranscribeOptions(whisper.WithLanguage("en")))
		if err != nil {
			t.Fatalf("NewStream: %v", err)
		}
		go func() { _ = st.Write(silence); _ = st.CloseSend() }()
		for range st.Results() {
		}
		if err := st.Err(); err != nil {
			t.Fatalf("stream err: %v", err)
		}
		// NOTE: deliberately NOT calling st.Close() — graceful path must self-clean.
	}

	runOne() // warm up (lazy runtime goroutines)
	time.Sleep(100 * time.Millisecond)
	runtime.GC()
	base := runtime.NumGoroutine()
	for range 10 {
		runOne()
	}
	// allow goroutines to wind down
	var n int
	for range 40 {
		runtime.GC()
		n = runtime.NumGoroutine()
		if n <= base+2 {
			return // OK
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("goroutine leak: baseline %d, after 10 CloseSend-only streams %d (want <= %d)", base, n, base+2)
}
