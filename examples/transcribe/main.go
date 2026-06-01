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
	model := flag.String("m", "models/ggml-tiny.en.bin", "path to ggml model")
	audio := flag.String("f", "whisper.cpp/samples/jfk.wav", "path to 16 kHz mono WAV")
	lang := flag.String("l", "auto", "language ('auto' to detect)")
	translate := flag.Bool("translate", false, "translate to English")
	threads := flag.Int("t", runtime.NumCPU(), "threads")
	flag.Parse()

	m, err := whisper.New(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load model:", err)
		os.Exit(1)
	}
	defer m.Close()

	samples, err := wav.ReadFile(*audio)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read wav:", err)
		os.Exit(1)
	}

	opts := []whisper.TranscribeOption{whisper.WithLanguage(*lang), whisper.WithThreads(*threads)}
	if *translate {
		opts = append(opts, whisper.WithTranslate)
	}
	res, err := m.Transcribe(context.Background(), samples, opts...)
	if err != nil {
		fmt.Fprintln(os.Stderr, "transcribe:", err)
		os.Exit(1)
	}
	fmt.Printf("[language: %s]\n", res.Language)
	for _, s := range res.Segments {
		fmt.Printf("[%s -> %s] %s\n", s.Start, s.End, s.Text)
	}
}
