package diarize_test

import (
	"testing"
	"time"

	"github.com/dyammarcano/go-whisper.cpp/diarize"
)

func ms(n int) time.Duration { return time.Duration(n) * time.Millisecond }

func TestLabel(t *testing.T) {
	turns := []diarize.Turn{
		{Start: ms(0), End: ms(1000), Speaker: 0},
		{Start: ms(1000), End: ms(2000), Speaker: 1},
	}
	cases := []struct {
		name string
		seg  diarize.Segment
		want int
	}{
		{"inside speaker 0", diarize.Segment{Start: ms(100), End: ms(500), Text: "a"}, 0},
		{"inside speaker 1", diarize.Segment{Start: ms(1200), End: ms(1800), Text: "b"}, 1},
		{"overlaps both, more in 1", diarize.Segment{Start: ms(900), End: ms(1600), Text: "c"}, 1},
		{"no overlap", diarize.Segment{Start: ms(5000), End: ms(6000), Text: "d"}, -1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := diarize.Label([]diarize.Segment{tc.seg}, turns)
			if len(got) != 1 {
				t.Fatalf("len=%d want 1", len(got))
			}
			if got[0].Speaker != tc.want {
				t.Errorf("speaker=%d want %d", got[0].Speaker, tc.want)
			}
			if got[0].Text != tc.seg.Text || got[0].Start != tc.seg.Start || got[0].End != tc.seg.End {
				t.Errorf("segment not preserved: %+v", got[0])
			}
		})
	}
}

func TestLabel_Empty(t *testing.T) {
	if got := diarize.Label(nil, nil); len(got) != 0 {
		t.Fatalf("want empty, got %d", len(got))
	}
}
