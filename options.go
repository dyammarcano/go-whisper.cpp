package whisper

// ModelOption configures Model loading (whisper_context_params).
type ModelOption func(*modelOptions)

type modelOptions struct {
	gpu       bool
	flashAttn bool
	gpuDevice int
}

func defaultModelOptions() modelOptions {
	return modelOptions{gpu: false, flashAttn: false, gpuDevice: 0}
}

// TranscribeOption configures a single Transcribe call (whisper_full_params).
type TranscribeOption func(*transcribeOptions)

type transcribeOptions struct {
	beamSearch      bool
	beamSize        int
	bestOf          int
	threads         int
	translate       bool
	language        string // "" / "auto" -> autodetect
	detectLanguage  bool
	temperature     float32
	temperatureInc  float32
	entropyThold    float32
	logProbThold    float32
	noSpeechThold   float32
	noContext       bool
	singleSegment   bool
	tokenTimestamps bool
	maxLen          int
	splitOnWord     bool
	maxTokens       int
	offsetMs        int
	durationMs      int
	audioCtx        int
	suppressBlank   bool
	suppressNST     bool
	initialPrompt   string
	onSegment       func(Segment)
	onProgress      func(int)
}

func defaultTranscribeOptions() transcribeOptions {
	return transcribeOptions{
		language:      "auto",
		suppressBlank: true,
		suppressNST:   true,
		bestOf:        2,
		beamSize:      5,
	}
}
