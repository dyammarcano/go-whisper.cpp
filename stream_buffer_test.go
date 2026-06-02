package whisper

import (
	"context"
	"testing"
	"time"
)

const rate = 16000

func ones(d time.Duration) []float32 {
	n := int(d.Seconds() * float64(rate))
	s := make([]float32, n)
	for i := range s {
		s[i] = 1
	}
	return s
}

func TestStreamBuffer_NextWindowWaitsForStep(t *testing.T) {
	b := newStreamBuffer(rate, 0, 0, context.Background())
	_ = b.write(ones(300 * time.Millisecond))
	done := make(chan struct{})
	go func() {
		_, _, _, _, _, ok := b.nextWindow(0, 500*time.Millisecond, 10*time.Second)
		if !ok {
			t.Errorf("nextWindow returned !ok")
		}
		close(done)
	}()
	select {
	case <-done:
		t.Fatal("nextWindow returned before step reached")
	case <-time.After(50 * time.Millisecond):
	}
	_ = b.write(ones(300 * time.Millisecond))
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("nextWindow did not wake after step reached")
	}
}

func TestStreamBuffer_NextWindowTrailingWindow(t *testing.T) {
	b := newStreamBuffer(rate, 0, 0, context.Background())
	_ = b.write(ones(12 * time.Second))
	pcm, ws, consumed, _, closed, ok := b.nextWindow(0, time.Second, 10*time.Second)
	if !ok || closed {
		t.Fatal("unexpected ok/closed")
	}
	if consumed != 12*time.Second {
		t.Errorf("consumed = %v want 12s", consumed)
	}
	if ws != 2*time.Second {
		t.Errorf("windowStart = %v want 2s", ws)
	}
	if got := durFor(len(pcm)); got != 10*time.Second {
		t.Errorf("window len = %v want 10s", got)
	}
}

func TestStreamBuffer_DropOldestOnOverrun(t *testing.T) {
	b := newStreamBuffer(rate, 0, 4*time.Second, context.Background())
	_ = b.write(ones(3 * time.Second))
	_ = b.write(ones(3 * time.Second))
	pcm, ws, consumed, dropped, _, _ := b.nextWindow(0, time.Second, 10*time.Second)
	if !dropped {
		t.Error("expected dropped=true")
	}
	if consumed != 6*time.Second {
		t.Errorf("consumed = %v want 6s", consumed)
	}
	if ws != 2*time.Second || durFor(len(pcm)) != 4*time.Second {
		t.Errorf("after drop want start=2s len=4s, got start=%v len=%v", ws, durFor(len(pcm)))
	}
}

func TestStreamBuffer_SlideDiscards(t *testing.T) {
	b := newStreamBuffer(rate, 0, 0, context.Background())
	_ = b.write(ones(5 * time.Second))
	b.slide(3 * time.Second)
	pcm, ws, _, _, _, _ := b.nextWindow(0, time.Second, 10*time.Second) //nolint:dogsled
	if ws != 3*time.Second || durFor(len(pcm)) != 2*time.Second {
		t.Errorf("after slide want start=3s len=2s, got start=%v len=%v", ws, durFor(len(pcm)))
	}
}

func TestStreamBuffer_CloseSendUnblocks(t *testing.T) {
	b := newStreamBuffer(rate, 0, 0, context.Background())
	_ = b.write(ones(100 * time.Millisecond))
	done := make(chan bool)
	go func() {
		_, _, _, _, closed, ok := b.nextWindow(0, time.Second, 10*time.Second)
		done <- (ok && closed)
	}()
	b.closeSend()
	select {
	case got := <-done:
		if !got {
			t.Error("expected ok && closed after CloseSend")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("nextWindow did not wake on CloseSend")
	}
	if err := b.write(ones(10 * time.Millisecond)); err == nil {
		t.Error("write after closeSend should error")
	}
}

func TestStreamBuffer_CtxCancelUnblocks(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	b := newStreamBuffer(rate, 0, 0, ctx)
	done := make(chan bool)
	go func() {
		_, _, _, _, _, ok := b.nextWindow(0, time.Second, 10*time.Second)
		done <- ok
	}()
	cancel()
	select {
	case ok := <-done:
		if ok {
			t.Error("expected ok=false after ctx cancel")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("nextWindow did not wake on ctx cancel")
	}
}
