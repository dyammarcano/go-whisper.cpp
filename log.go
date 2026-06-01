package whisper

// extern void goWhisperLog(int, char*);
import "C"

import "log/slog"

// goWhisperLog routes whisper/ggml C logs into slog at debug level (quiet by default).
//
//export goWhisperLog
func goWhisperLog(level C.int, text *C.char) {
	slog.Debug("whisper", "level", int(level), "msg", C.GoString(text))
}
