package wav

import (
	"math"
	"testing"
)

func TestResampleLinear_Identity(t *testing.T) {
	in := []float32{0, 0.5, -0.5, 1}
	got := resampleLinear(in, 16000, 16000)
	if len(got) != len(in) {
		t.Fatalf("len=%d want %d", len(got), len(in))
	}
}

func TestResampleLinear_Empty(t *testing.T) {
	if got := resampleLinear(nil, 44100, 16000); len(got) != 0 {
		t.Fatalf("len=%d want 0", len(got))
	}
}

func TestResampleLinear_Downsample2to1(t *testing.T) {
	in := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	got := resampleLinear(in, 2, 1) // halve the rate -> every other sample
	want := []float32{0, 2, 4, 6}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Errorf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestResampleLinear_Upsample(t *testing.T) {
	in := []float32{0, 2} // doubling the rate
	got := resampleLinear(in, 1, 2)
	if len(got) != 4 {
		t.Fatalf("len=%d want 4", len(got))
	}
	// positions 0,0.5,1.0,1.5 -> 0, 1, 2, 2(clamped to last)
	want := []float32{0, 1, 2, 2}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Errorf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestResampleLinear_ConstantPreserved(t *testing.T) {
	in := make([]float32, 441)
	for i := range in {
		in[i] = 0.25
	}
	got := resampleLinear(in, 44100, 16000)
	if got := len(got); got != 160 { // 441*16000/44100
		t.Fatalf("len=%d want 160", got)
	}
	for i, v := range got {
		if math.Abs(float64(v-0.25)) > 1e-5 {
			t.Fatalf("[%d]=%v want 0.25 (constant must be preserved)", i, v)
		}
	}
}
