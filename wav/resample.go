package wav

// resampleLinear converts a mono signal (values in [-1,1]) from fromRate to toRate using
// linear interpolation. Output length is len(in)*toRate/fromRate. It returns in unchanged
// when the rates match or the input is empty.
//
// Linear interpolation applies no anti-alias low-pass before downsampling. This is adequate
// for 16 kHz speech-to-text (whisper is robust to it) and avoids a heavier polyphase/sinc
// resampler; it is not hi-fi audio resampling.
func resampleLinear(in []float32, fromRate, toRate int) []float32 {
	if len(in) == 0 || fromRate <= 0 || toRate <= 0 || fromRate == toRate {
		return in
	}
	outN := int(int64(len(in)) * int64(toRate) / int64(fromRate))
	if outN <= 0 {
		return []float32{}
	}
	out := make([]float32, outN)
	ratio := float64(fromRate) / float64(toRate)
	last := len(in) - 1
	for i := range out {
		pos := float64(i) * ratio
		j := int(pos)
		if j >= last {
			out[i] = in[last]
			continue
		}
		frac := float32(pos - float64(j))
		out[i] = in[j]*(1-frac) + in[j+1]*frac
	}
	return out
}
