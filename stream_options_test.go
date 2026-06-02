package whisper

import (
	"testing"
	"time"
)

func TestStreamConfig_Defaults(t *testing.T) {
	c := defaultStreamConfig()
	if c.step != 700*time.Millisecond || c.window != 10*time.Second || c.keep != 200*time.Millisecond {
		t.Fatalf("unexpected defaults: %+v", c)
	}
	if c.dropMaxBuffer != 0 {
		t.Fatalf("drop should be off by default, got %v", c.dropMaxBuffer)
	}
}

func TestStreamConfig_OptionsAndValidate(t *testing.T) {
	c := defaultStreamConfig()
	for _, o := range []StreamOption{
		WithStreamStep(300 * time.Millisecond),
		WithStreamWindow(8 * time.Second),
		WithStreamKeep(150 * time.Millisecond),
		WithDropOnOverrun(5 * time.Second),
	} {
		o(&c)
	}
	if c.step != 300*time.Millisecond || c.window != 8*time.Second ||
		c.keep != 150*time.Millisecond || c.dropMaxBuffer != 5*time.Second {
		t.Fatalf("options not applied: %+v", c)
	}
	if err := c.validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
}

func TestStreamConfig_ValidateRejects(t *testing.T) {
	bad := []streamConfig{
		{step: 0, window: 10 * time.Second, keep: time.Second},
		{step: time.Second, window: 0, keep: 0},
		{step: 12 * time.Second, window: 10 * time.Second, keep: time.Second},
		{step: time.Second, window: 31 * time.Second, keep: time.Second},
		{step: time.Second, window: 10 * time.Second, keep: 10 * time.Second},
		{step: time.Second, window: 10 * time.Second, keep: -1},
	}
	for i, c := range bad {
		if err := c.validate(); err == nil {
			t.Errorf("case %d: expected error, got nil for %+v", i, c)
		}
	}
}
