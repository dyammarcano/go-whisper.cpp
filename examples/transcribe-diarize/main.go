package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	whisper "github.com/dyammarcano/go-whisper.cpp"
	"github.com/dyammarcano/go-whisper.cpp/diarize"
	"github.com/dyammarcano/go-whisper.cpp/wav"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	model := flag.String("m", "models/ggml-tiny.en.bin", "whisper ggml model")
	seg := flag.String("seg", "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx", "segmentation model")
	emb := flag.String("emb", "models/wespeaker_en_voxceleb_resnet34_LM.onnx", "embedding model")
	audio := flag.String("f", "models/4speakers.wav", "input wav (16 kHz mono, or any with -resample)")
	n := flag.Int("n", 0, "number of speakers (0 = auto)")
	resample := flag.Bool("resample", false, "downmix + resample non-16 kHz input to 16 kHz mono")
	flag.Parse()

	var rdopts []wav.ReadOption
	if *resample {
		rdopts = append(rdopts, wav.WithResample())
	}
	samples, err := wav.ReadFile(*audio, rdopts...)
	if err != nil {
		return fmt.Errorf("read wav: %w", err)
	}

	m, err := whisper.New(*model)
	if err != nil {
		return fmt.Errorf("whisper: %w", err)
	}
	defer func() { _ = m.Close() }()

	res, err := m.Transcribe(context.Background(), samples)
	if err != nil {
		return fmt.Errorf("transcribe: %w", err)
	}

	opts := []diarize.Option{diarize.WithThreshold(0.5)}
	if *n > 0 {
		opts = []diarize.Option{diarize.WithNumSpeakers(*n)}
	}
	d, err := diarize.New(*seg, *emb, opts...)
	if err != nil {
		return fmt.Errorf("diarize: %w", err)
	}
	defer func() { _ = d.Close() }()

	turns, err := d.Diarize(samples)
	if err != nil {
		return fmt.Errorf("diarize: %w", err)
	}

	segs := make([]diarize.Segment, len(res.Segments))
	for i, s := range res.Segments {
		segs[i] = diarize.Segment{Start: s.Start, End: s.End, Text: s.Text}
	}
	for _, ls := range diarize.Label(segs, turns) {
		_, _ = fmt.Fprintf(os.Stdout, "[Speaker %d] %s\n", ls.Speaker, ls.Text)
	}
	return nil
}
