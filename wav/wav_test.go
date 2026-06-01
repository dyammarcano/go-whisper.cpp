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
