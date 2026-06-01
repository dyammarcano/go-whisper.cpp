// Package wav decodes RIFF/WAVE audio to 16 kHz mono float32 in [-1,1] for whisper.
// Pure Go, no cgo. Non-16 kHz input is rejected (resample with ffmpeg -ar 16000 -ac 1).
package wav

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

const SampleRate = 16000

var (
	ErrNot16kHz       = errors.New("wav: sample rate is not 16000 Hz")
	ErrBadHeader      = errors.New("wav: not a RIFF/WAVE file")
	ErrUnsupportedFmt = errors.New("wav: unsupported audio format / bit depth")
)

type fmtChunk struct {
	audioFormat   uint16
	numChannels   uint16
	sampleRate    uint32
	bitsPerSample uint16
}

// ReadFile decodes a WAV file at path.
func ReadFile(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return ReadWAV(f)
}

// ReadWAV decodes a RIFF/WAVE stream to 16 kHz mono float32.
func ReadWAV(r io.Reader) ([]float32, error) {
	var riff [12]byte
	if _, err := io.ReadFull(r, riff[:]); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrBadHeader, err)
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return nil, ErrBadHeader
	}
	var fc fmtChunk
	gotFmt := false
	for {
		var hdr [8]byte
		if _, err := io.ReadFull(r, hdr[:]); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil, errors.New("wav: no data chunk found")
			}
			return nil, fmt.Errorf("read chunk header: %w", err)
		}
		id := string(hdr[0:4])
		size := binary.LittleEndian.Uint32(hdr[4:8])
		switch id {
		case "fmt ":
			body := make([]byte, size)
			if _, err := io.ReadFull(r, body); err != nil {
				return nil, fmt.Errorf("read fmt chunk: %w", err)
			}
			if len(body) < 16 {
				return nil, ErrUnsupportedFmt
			}
			fc.audioFormat = binary.LittleEndian.Uint16(body[0:2])
			fc.numChannels = binary.LittleEndian.Uint16(body[2:4])
			fc.sampleRate = binary.LittleEndian.Uint32(body[4:8])
			fc.bitsPerSample = binary.LittleEndian.Uint16(body[14:16])
			gotFmt = true
			if size%2 == 1 {
				if _, err := io.CopyN(io.Discard, r, 1); err != nil {
					return nil, err
				}
			}
		case "data":
			if !gotFmt {
				return nil, errors.New("wav: data chunk before fmt chunk")
			}
			if fc.sampleRate != SampleRate {
				return nil, fmt.Errorf("%w: got %d Hz (resample: ffmpeg -i in -ar 16000 -ac 1 out.wav)", ErrNot16kHz, fc.sampleRate)
			}
			data := make([]byte, size)
			if _, err := io.ReadFull(r, data); err != nil {
				return nil, fmt.Errorf("read data chunk: %w", err)
			}
			return decodeSamples(data, &fc)
		default:
			n := int64(size)
			if size%2 == 1 {
				n++
			}
			if _, err := io.CopyN(io.Discard, r, n); err != nil {
				return nil, fmt.Errorf("skip %q chunk: %w", id, err)
			}
		}
	}
}

func decodeSamples(data []byte, fc *fmtChunk) ([]float32, error) {
	ch := int(fc.numChannels)
	if ch < 1 {
		return nil, ErrUnsupportedFmt
	}
	bps := int(fc.bitsPerSample) / 8
	blockAlign := bps * ch
	if blockAlign == 0 || len(data)%blockAlign != 0 {
		return nil, fmt.Errorf("%w: data not aligned to block %d", ErrUnsupportedFmt, blockAlign)
	}
	frames := len(data) / blockAlign
	out := make([]float32, frames)

	var conv func(b []byte) float32
	switch {
	case fc.audioFormat == 3 && fc.bitsPerSample == 32:
		conv = func(b []byte) float32 { return math.Float32frombits(binary.LittleEndian.Uint32(b)) }
	case (fc.audioFormat == 1 || fc.audioFormat == 0xFFFE) && fc.bitsPerSample == 16:
		conv = func(b []byte) float32 { return float32(int16(binary.LittleEndian.Uint16(b))) / 32768.0 }
	case fc.audioFormat == 1 && fc.bitsPerSample == 8:
		conv = func(b []byte) float32 { return (float32(b[0]) - 128) / 128.0 }
	case (fc.audioFormat == 1 || fc.audioFormat == 0xFFFE) && fc.bitsPerSample == 24:
		conv = func(b []byte) float32 {
			u := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16
			if u&0x800000 != 0 {
				u |= 0xFF000000
			}
			return float32(int32(u)) / 8388608.0
		}
	case (fc.audioFormat == 1 || fc.audioFormat == 0xFFFE) && fc.bitsPerSample == 32:
		conv = func(b []byte) float32 { return float32(int32(binary.LittleEndian.Uint32(b))) / 2147483648.0 }
	default:
		return nil, fmt.Errorf("%w: fmt=%d bits=%d", ErrUnsupportedFmt, fc.audioFormat, fc.bitsPerSample)
	}

	for f := 0; f < frames; f++ {
		base := f * blockAlign
		var sum float32
		for c := 0; c < ch; c++ {
			off := base + c*bps
			sum += conv(data[off : off+bps])
		}
		out[f] = sum / float32(ch)
	}
	return out, nil
}
