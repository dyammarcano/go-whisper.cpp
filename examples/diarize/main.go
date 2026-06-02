package main

import (
	"flag"
	"fmt"
	"os"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"github.com/dyammarcano/go-whisper.cpp/diarize"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	seg := flag.String("seg", "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx", "segmentation model")
	emb := flag.String("emb", "models/wespeaker_en_voxceleb_resnet34_LM.onnx", "embedding model")
	wavPath := flag.String("f", "models/4speakers.wav", "16 kHz mono wav")
	n := flag.Int("n", 0, "number of speakers (0 = auto)")
	thr := flag.Float64("t", 0.5, "clustering threshold when n=0")
	flag.Parse()

	opts := []diarize.Option{}
	if *n > 0 {
		opts = append(opts, diarize.WithNumSpeakers(*n))
	} else {
		opts = append(opts, diarize.WithThreshold(float32(*thr)))
	}
	d, err := diarize.New(*seg, *emb, opts...)
	if err != nil {
		return fmt.Errorf("new: %w", err)
	}
	defer func() { _ = d.Close() }()

	w := sherpa.ReadWave(*wavPath)
	if w == nil {
		return fmt.Errorf("read wav failed: %s", *wavPath)
	}
	if w.SampleRate != d.SampleRate() {
		return fmt.Errorf("wav rate %d != required %d", w.SampleRate, d.SampleRate())
	}
	turns, err := d.Diarize(w.Samples)
	if err != nil {
		return fmt.Errorf("diarize: %w", err)
	}
	for _, t := range turns {
		_, _ = fmt.Fprintf(os.Stdout, "[%s -> %s] speaker %d\n", t.Start, t.End, t.Speaker)
	}
	return nil
}
