package main

import (
	"flag"
	"fmt"
	"os"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"github.com/dyammarcano/go-whisper.cpp/diarize"
)

func main() {
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
		fmt.Fprintln(os.Stderr, "new:", err)
		os.Exit(1)
	}
	defer d.Close()

	w := sherpa.ReadWave(*wavPath)
	if w == nil {
		fmt.Fprintln(os.Stderr, "read wav failed:", *wavPath)
		os.Exit(1)
	}
	if w.SampleRate != d.SampleRate() {
		fmt.Fprintf(os.Stderr, "wav rate %d != required %d\n", w.SampleRate, d.SampleRate())
		os.Exit(1)
	}
	turns, err := d.Diarize(w.Samples)
	if err != nil {
		fmt.Fprintln(os.Stderr, "diarize:", err)
		os.Exit(1)
	}
	for _, t := range turns {
		fmt.Printf("[%s -> %s] speaker %d\n", t.Start, t.End, t.Speaker)
	}
}
