// Command stream demonstrates streaming transcription by feeding a WAV file in chunks.
// Usage: go run ./examples/stream <model.bin> <audio.wav>
package main

import (
	"context"
	"fmt"
	"os"
	"time"

	whisper "github.com/dyammarcano/go-whisper.cpp"
	"github.com/dyammarcano/go-whisper.cpp/wav"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: stream <model.bin> <audio.wav>")
		os.Exit(2)
	}
	m, err := whisper.New(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "load model:", err)
		os.Exit(1)
	}
	defer func() { _ = m.Close() }()

	samples, err := wav.ReadFile(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "read wav:", err)
		os.Exit(1)
	}

	st, err := m.NewStream(context.Background(),
		whisper.WithStreamStep(500*time.Millisecond),
		whisper.WithTranscribeOptions(whisper.WithLanguage("en")))
	if err != nil {
		fmt.Fprintln(os.Stderr, "new stream:", err)
		os.Exit(1)
	}

	go func() {
		const chunk = 16000 / 2 // 0.5s
		for i := 0; i < len(samples); i += chunk {
			end := i + chunk
			if end > len(samples) {
				end = len(samples)
			}
			_ = st.Write(samples[i:end])
			time.Sleep(50 * time.Millisecond) // simulate real-time-ish arrival
		}
		_ = st.CloseSend()
	}()

	for r := range st.Results() {
		tag := "FINAL  "
		if r.Partial {
			tag = "partial"
		}
		fmt.Printf("[%s %6.2fs-%6.2fs]%s\n", tag, r.Segment.Start.Seconds(), r.Segment.End.Seconds(), r.Segment.Text)
	}
	if err := st.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "stream error:", err)
		os.Exit(1)
	}
}
