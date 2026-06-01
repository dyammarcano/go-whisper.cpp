package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"runtime"

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
	model := flag.String("m", "models/ggml-tiny.en.bin", "path to ggml model")
	audio := flag.String("f", "whisper.cpp/samples/jfk.wav", "path to 16 kHz mono WAV")
	lang := flag.String("l", "auto", "language ('auto' to detect)")
	translate := flag.Bool("translate", false, "translate to English")
	threads := flag.Int("t", runtime.NumCPU(), "threads")
	flag.Parse()

	m, err := whisper.New(*model)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	defer func() { _ = m.Close() }()

	samples, err := wav.ReadFile(*audio)
	if err != nil {
		return fmt.Errorf("read wav: %w", err)
	}

	opts := []whisper.TranscribeOption{whisper.WithLanguage(*lang), whisper.WithThreads(*threads)}
	if *translate {
		opts = append(opts, whisper.WithTranslate)
	}
	res, err := m.Transcribe(context.Background(), samples, opts...)
	if err != nil {
		return fmt.Errorf("transcribe: %w", err)
	}
	_, _ = fmt.Fprintf(os.Stdout, "[language: %s]\n", res.Language)
	for _, s := range res.Segments {
		_, _ = fmt.Fprintf(os.Stdout, "[%s -> %s] %s\n", s.Start, s.End, s.Text)
	}
	return nil
}
