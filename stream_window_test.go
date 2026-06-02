package whisper

import (
	"testing"
	"time"
)

func seg(start, end float64, text string) Segment {
	return Segment{
		Start: time.Duration(start * float64(time.Second)),
		End:   time.Duration(end * float64(time.Second)),
		Text:  text,
	}
}

func TestClassifyWindow_FinalsPartialsSkips(t *testing.T) {
	segs := []Segment{
		seg(0.0, 1.0, " already"),
		seg(1.0, 2.5, " hello"),
		seg(2.5, 4.0, " world"),
	}
	got := classifyWindow(segs, 2*time.Second, 2*time.Second, 5*time.Second)
	if len(got.results) != 3 {
		t.Fatalf("want 3 results, got %d", len(got.results))
	}
	if got.results[0].Partial || got.results[1].Partial {
		t.Errorf("first two should be final: %+v", got.results)
	}
	if !got.results[2].Partial {
		t.Errorf("third should be partial: %+v", got.results[2])
	}
	if got.results[1].Segment.End != 4500*time.Millisecond {
		t.Errorf("absolute end wrong: %v", got.results[1].Segment.End)
	}
	if got.committed != 4500*time.Millisecond {
		t.Errorf("committed should advance to last final end (4.5s), got %v", got.committed)
	}
	if got.promptAdd != " hello" && got.promptAdd != " already hello" {
		t.Errorf("promptAdd should accumulate final text, got %q", got.promptAdd)
	}
}

func TestClassifyWindow_SkipAlreadyCommitted(t *testing.T) {
	segs := []Segment{
		seg(0.0, 1.0, " dup"),
		seg(1.0, 2.0, " fresh"),
	}
	got := classifyWindow(segs, 2*time.Second, 3*time.Second, 10*time.Second)
	if len(got.results) != 1 || got.results[0].Segment.Text != " fresh" {
		t.Fatalf("want only fresh segment, got %+v", got.results)
	}
	if got.committed != 4*time.Second {
		t.Errorf("committed should be 4s, got %v", got.committed)
	}
}

func TestClassifyWindow_FlushCommitsTail(t *testing.T) {
	segs := []Segment{seg(0, 1.5, " tail")}
	got := classifyWindow(segs, 0, 0, 1500*time.Millisecond)
	if len(got.results) != 1 || got.results[0].Partial {
		t.Fatalf("flush should commit tail as final: %+v", got.results)
	}
	if got.committed != 1500*time.Millisecond {
		t.Errorf("committed should be 1.5s, got %v", got.committed)
	}
}

func TestClassifyWindow_Empty(t *testing.T) {
	got := classifyWindow(nil, 0, 0, time.Second)
	if len(got.results) != 0 || got.committed != 0 || got.promptAdd != "" {
		t.Fatalf("empty input should yield empty result: %+v", got)
	}
}
