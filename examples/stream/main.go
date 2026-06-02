// Command stream demonstrates streaming transcription by feeding a WAV file in chunks.
// Usage: go run ./examples/stream <model.bin> <audio.wav>
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	whisper "github.com/dyammarcano/go-whisper.cpp"
	"github.com/dyammarcano/go-whisper.cpp/wav"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	if len(os.Args) != 3 {
		return errors.New("usage: stream <model.bin> <audio.wav>")
	}
	m, err := whisper.New(os.Args[1])
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	defer func() { _ = m.Close() }()

	samples, err := wav.ReadFile(os.Args[2])
	if err != nil {
		return fmt.Errorf("read wav: %w", err)
	}

	st, err := m.NewStream(context.Background(),
		whisper.WithStreamStep(500*time.Millisecond),
		whisper.WithTranscribeOptions(whisper.WithLanguage("en")))
	if err != nil {
		return fmt.Errorf("new stream: %w", err)
	}

	go func() {
		const chunk = 16000 / 2 // 0.5s
		for i := 0; i < len(samples); i += chunk {
			end := min(i+chunk, len(samples))
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
		_, _ = fmt.Fprintf(os.Stdout, "[%s %6.2fs-%6.2fs]%s\n", tag, r.Segment.Start.Seconds(), r.Segment.End.Seconds(), r.Segment.Text)
	}
	if err := st.Err(); err != nil {
		return fmt.Errorf("stream error: %w", err)
	}
	return nil
}
