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
	model := flag.String("m", "models/ggml-tiny.en.bin", "whisper ggml model")
	seg := flag.String("seg", "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx", "segmentation model")
	emb := flag.String("emb", "models/wespeaker_en_voxceleb_resnet34_LM.onnx", "embedding model")
	audio := flag.String("f", "models/4speakers.wav", "16 kHz mono wav")
	n := flag.Int("n", 0, "number of speakers (0 = auto)")
	flag.Parse()

	samples, err := wav.ReadFile(*audio)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read wav:", err)
		os.Exit(1)
	}

	m, err := whisper.New(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "whisper:", err)
		os.Exit(1)
	}
	defer m.Close()
	res, err := m.Transcribe(context.Background(), samples)
	if err != nil {
		fmt.Fprintln(os.Stderr, "transcribe:", err)
		os.Exit(1)
	}

	opts := []diarize.Option{diarize.WithThreshold(0.5)}
	if *n > 0 {
		opts = []diarize.Option{diarize.WithNumSpeakers(*n)}
	}
	d, err := diarize.New(*seg, *emb, opts...)
	if err != nil {
		fmt.Fprintln(os.Stderr, "diarize:", err)
		os.Exit(1)
	}
	defer d.Close()
	turns, err := d.Diarize(samples)
	if err != nil {
		fmt.Fprintln(os.Stderr, "diarize:", err)
		os.Exit(1)
	}

	segs := make([]diarize.Segment, len(res.Segments))
	for i, s := range res.Segments {
		segs[i] = diarize.Segment{Start: s.Start, End: s.End, Text: s.Text}
	}
	for _, ls := range diarize.Label(segs, turns) {
		fmt.Printf("[Speaker %d] %s\n", ls.Speaker, ls.Text)
	}
}
