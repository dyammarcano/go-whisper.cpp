package wav_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/dyammarcano/go-whisper.cpp/wav"
)

// buildWAV creates a minimal PCM16 WAV in memory.
func buildWAV(sampleRate uint32, channels uint16, samples []int16) []byte {
	var b bytes.Buffer
	// *bytes.Buffer never errors, so muting binary.Write/WriteString returns is correct.
	le := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	dataLen := len(samples) * 2
	_, _ = b.WriteString("RIFF")
	le(uint32(36 + dataLen))
	_, _ = b.WriteString("WAVE")
	_, _ = b.WriteString("fmt ")
	le(uint32(16))
	le(uint16(1)) // PCM
	le(channels)
	le(sampleRate)
	le(sampleRate * uint32(channels) * 2) // byte rate
	le(channels * 2)                      // block align
	le(uint16(16))                        // bits
	_, _ = b.WriteString("data")
	le(uint32(dataLen))
	for _, s := range samples {
		le(s)
	}
	return b.Bytes()
}

func TestReadWAV_Mono16k(t *testing.T) {
	in := buildWAV(16000, 1, []int16{0, 16384, -16384, 32767})
	got, err := wav.ReadWAV(bytes.NewReader(in))
	if err != nil {
		t.Fatalf("ReadWAV: %v", err)
	}
	want := []float32{0, 0.5, -0.5, 32767.0 / 32768.0}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Errorf("sample %d = %v want %v", i, got[i], want[i])
		}
	}
}

func TestReadWAV_StereoDownmix(t *testing.T) {
	// L=1.0(32767), R=-1.0(-32768) -> mono ~0
	in := buildWAV(16000, 2, []int16{32767, -32768})
	got, err := wav.ReadWAV(bytes.NewReader(in))
	if err != nil {
		t.Fatalf("ReadWAV: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("frames=%d want 1", len(got))
	}
	if math.Abs(float64(got[0])) > 1e-2 {
		t.Errorf("downmix = %v want ~0", got[0])
	}
}

func TestReadWAV_RejectsNon16k(t *testing.T) {
	in := buildWAV(44100, 1, []int16{1, 2, 3})
	_, err := wav.ReadWAV(bytes.NewReader(in))
	if !errors.Is(err, wav.ErrNot16kHz) {
		t.Fatalf("err=%v want ErrNot16kHz", err)
	}
}

func TestReadWAV_RejectsBadHeader(t *testing.T) {
	_, err := wav.ReadWAV(bytes.NewReader([]byte("NOTAWAVfile!!")))
	if !errors.Is(err, wav.ErrBadHeader) {
		t.Fatalf("err=%v want ErrBadHeader", err)
	}
}

func TestReadWAV_ResampleDownmix44kStereo(t *testing.T) {
	const frames = 441 // 441 @ 44.1 kHz -> 160 @ 16 kHz
	samples := make([]int16, frames*2)
	for i := range samples {
		samples[i] = 8192 // L=R=0.25 -> mono 0.25
	}
	got, err := wav.ReadWAV(bytes.NewReader(buildWAV(44100, 2, samples)), wav.WithResample())
	if err != nil {
		t.Fatalf("ReadWAV: %v", err)
	}
	if len(got) != 160 {
		t.Fatalf("len=%d want 160 (16 kHz mono)", len(got))
	}
	for i, v := range got {
		if math.Abs(float64(v-0.25)) > 1e-2 {
			t.Fatalf("sample %d=%v want ~0.25 (constant must survive downmix+resample)", i, v)
		}
	}
}

func TestReadWAV_Resample8kMonoUpsample(t *testing.T) {
	const frames = 800 // 800 @ 8 kHz -> 1600 @ 16 kHz
	samples := make([]int16, frames)
	for i := range samples {
		samples[i] = 16384
	}
	got, err := wav.ReadWAV(bytes.NewReader(buildWAV(8000, 1, samples)), wav.WithResample())
	if err != nil {
		t.Fatalf("ReadWAV: %v", err)
	}
	if len(got) != 1600 {
		t.Fatalf("len=%d want 1600 (16 kHz upsample)", len(got))
	}
}

func TestReadWAV_ResampleNoOpAt16k(t *testing.T) {
	// WithResample on already-16 kHz audio must be a no-op (resampleLinear is not invoked).
	got, err := wav.ReadWAV(bytes.NewReader(buildWAV(16000, 1, []int16{0, 16384, -16384, 32767})), wav.WithResample())
	if err != nil {
		t.Fatalf("ReadWAV: %v", err)
	}
	if len(got) != 4 {
		t.Fatalf("len=%d want 4 (no resample at 16 kHz)", len(got))
	}
}
