# Features

## Completed

| Feature | API | Since |
|---------|-----|-------|
| whisper.cpp transcription | `whisper.New`, `(*Model).Transcribe` | v0.1.0 |
| Concurrent inference (Model + Session) | `(*Model).NewSession`, `(*Session).Transcribe` | v0.1.0 |
| Transcription options | `WithLanguage`, `WithBeamSearch`, `WithGreedy`, `WithTemperature`, thresholds, `WithMaxSegmentLen`, `WithMaxTokens`, `WithOffset`, `WithDuration`, `WithAudioCtx`, `WithInitialPrompt`, `WithSuppressBlank`, `WithSuppressNonSpeech` | v0.1.0 |
| Token timestamps | `WithTokenTimestamps` → `Segment.Tokens` | v0.1.0 |
| Segment / progress callbacks | `WithSegmentCallback`, `WithProgressCallback` | v0.1.0 |
| Context cancellation (pre-flight + mid-inference) | `context.Context` arg | v0.1.0 |
| GPU backends | build tags `cuda`, `vulkan`; `WithGPU`, `WithGPUDevice` | v0.1.0 |
| Language helpers | `(*Model).Languages`, `(*Model).IsMultilingual` | v0.1.0 |
| WAV decoding (16 kHz mono) | `wav.ReadFile`, `wav.ReadWAV` | v0.1.0 |
| Speaker diarization | `diarize.New`, `(*Diarizer).Diarize` → `[]Turn` | v0.1.0 |
| Diarization options | `WithNumSpeakers`, `WithThreshold`, `WithMinDuration`, `WithThreads` | v0.1.0 |
| Transcript ↔ speaker merge | `diarize.Label(segs, turns) []LabeledSegment` | v0.1.0 |
| Streaming transcription | `(*Model).NewStream`/`(*Session).NewStream`, `Write`/`CloseSend`/`Results`/`Err`/`Close` | v0.2.0 |
| Streaming options | `WithStreamStep`, `WithStreamWindow`, `WithStreamKeep`, `WithDropOnOverrun`, `WithTranscribeOptions` | v0.2.0 |

## Proposed

| Feature | Rationale | Status |
|---------|-----------|--------|
| WAV resample + downmix | `wav` rejects non-16 kHz (`ErrNot16kHz`); real files need pre-conversion | Proposed (v0.3.0) |
| ffmpeg-decode ingestion path | Accept m4a/mp3/etc. when ffmpeg is present | Proposed (v0.3.0) |
| Token-level speaker labels | `Label` is segment-granularity; whisper exposes token timestamps | Proposed (v0.4.0) |
| VAD-gated streaming finals | Cleaner segment boundaries than fixed-interval windows | Proposed (v0.4.0) |
| GPU diarization | sherpa-onnx CUDA provider for the embedding model | Proposed (v0.4.0) |
| Nested `diarize/go.mod` | Let transcription-only consumers skip the ONNX dependency tree | Proposed |

See [ROADMAP](ROADMAP.md) and [IMPLEMENTATION_TASKS](IMPLEMENTATION_TASKS.md) for sequencing.
