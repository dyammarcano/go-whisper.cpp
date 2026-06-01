package whisper

import "time"

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

// WithGPU enables/disables GPU offload (default off).
func WithGPU(on bool) ModelOption { return func(o *modelOptions) { o.gpu = on } }

// WithGPUDevice selects the GPU device index.
func WithGPUDevice(d int) ModelOption { return func(o *modelOptions) { o.gpuDevice = d } }

// WithFlashAttn enables flash attention.
var WithFlashAttn ModelOption = func(o *modelOptions) { o.flashAttn = true }

func WithLanguage(lang string) TranscribeOption {
	return func(o *transcribeOptions) { o.language = lang }
}

var WithTranslate TranscribeOption = func(o *transcribeOptions) { o.translate = true }
var WithDetectLanguage TranscribeOption = func(o *transcribeOptions) { o.detectLanguage = true }

func WithThreads(n int) TranscribeOption { return func(o *transcribeOptions) { o.threads = n } }
func WithBeamSearch(size int) TranscribeOption {
	return func(o *transcribeOptions) { o.beamSearch = true; o.beamSize = size }
}
func WithGreedy(bestOf int) TranscribeOption {
	return func(o *transcribeOptions) { o.beamSearch = false; o.bestOf = bestOf }
}
func WithTemperature(t float32) TranscribeOption {
	return func(o *transcribeOptions) { o.temperature = t }
}
func WithTemperatureInc(t float32) TranscribeOption {
	return func(o *transcribeOptions) { o.temperatureInc = t }
}
func WithEntropyThold(t float32) TranscribeOption {
	return func(o *transcribeOptions) { o.entropyThold = t }
}
func WithLogProbThold(t float32) TranscribeOption {
	return func(o *transcribeOptions) { o.logProbThold = t }
}
func WithNoSpeechThold(t float32) TranscribeOption {
	return func(o *transcribeOptions) { o.noSpeechThold = t }
}

var WithTokenTimestamps TranscribeOption = func(o *transcribeOptions) { o.tokenTimestamps = true }

func WithMaxSegmentLen(n int) TranscribeOption { return func(o *transcribeOptions) { o.maxLen = n } }

var WithSplitOnWord TranscribeOption = func(o *transcribeOptions) { o.splitOnWord = true }

func WithMaxTokens(n int) TranscribeOption { return func(o *transcribeOptions) { o.maxTokens = n } }

var WithNoContext TranscribeOption = func(o *transcribeOptions) { o.noContext = true }
var WithSingleSegment TranscribeOption = func(o *transcribeOptions) { o.singleSegment = true }

func WithInitialPrompt(s string) TranscribeOption {
	return func(o *transcribeOptions) { o.initialPrompt = s }
}
func WithSegmentCallback(f func(Segment)) TranscribeOption {
	return func(o *transcribeOptions) { o.onSegment = f }
}
func WithProgressCallback(f func(int)) TranscribeOption {
	return func(o *transcribeOptions) { o.onProgress = f }
}

func WithOffset(d time.Duration) TranscribeOption {
	return func(o *transcribeOptions) { o.offsetMs = int(d / time.Millisecond) }
}
func WithDuration(d time.Duration) TranscribeOption {
	return func(o *transcribeOptions) { o.durationMs = int(d / time.Millisecond) }
}
func WithAudioCtx(n int) TranscribeOption { return func(o *transcribeOptions) { o.audioCtx = n } }

// suppress_blank / suppress_nst default to true (see defaultTranscribeOptions); pass false to disable.
func WithSuppressBlank(on bool) TranscribeOption {
	return func(o *transcribeOptions) { o.suppressBlank = on }
}
func WithSuppressNonSpeech(on bool) TranscribeOption {
	return func(o *transcribeOptions) { o.suppressNST = on }
}
