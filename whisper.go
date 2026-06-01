package whisper

// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR}/whisper.cpp/include -I${SRCDIR}/whisper.cpp/ggml/include
// #cgo CFLAGS:   -I${SRCDIR}/whisper.cpp/include -I${SRCDIR}/whisper.cpp/ggml/include
// #cgo LDFLAGS:  -L${SRCDIR}/ -lbinding
// #include <stdlib.h>
// #include "binding.h"
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// Model wraps a loaded whisper_context. Safe to share across goroutines by
// creating one Session per goroutine. Close once when done.
// Model.Transcribe is mutex-serialized (single-flight); for parallel
// transcription create one Session per goroutine via NewSession.
type Model struct {
	ptr unsafe.Pointer // whisper_context*
	mu  sync.Mutex     // guards the context's internal state for Model.Transcribe
	mo  modelOptions
}

var logOnce sync.Once

// New loads a ggml whisper model from disk.
func New(modelPath string, opts ...ModelOption) (*Model, error) {
	logOnce.Do(func() { C.whisper_bind_install_log() })
	mo := defaultModelOptions()
	for _, o := range opts {
		o(&mo)
	}
	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))

	useGPU, flash := 0, 0
	if mo.gpu {
		useGPU = 1
	}
	if mo.flashAttn {
		flash = 1
	}
	ptr := C.whisper_bind_load_model(cpath, C.int(useGPU), C.int(flash), C.int(mo.gpuDevice))
	if ptr == nil {
		return nil, fmt.Errorf("%w: %s", ErrModelLoad, modelPath)
	}
	return &Model{ptr: ptr, mo: mo}, nil
}

// Close frees the underlying model. Idempotent.
func (m *Model) Close() error {
	if m == nil || m.ptr == nil {
		return nil
	}
	C.whisper_bind_free_model(m.ptr)
	m.ptr = nil
	return nil
}

// Languages returns all language names whisper knows (id 0..max).
func (m *Model) Languages() []string {
	maxID := int(C.whisper_bind_lang_max_id())
	out := make([]string, 0, maxID+1)
	for id := 0; id <= maxID; id++ {
		if s := C.whisper_bind_lang_str(C.int(id)); s != nil {
			out = append(out, C.GoString(s))
		}
	}
	return out
}

// IsMultilingual reports whether the model supports languages other than English.
func (m *Model) IsMultilingual() bool {
	if m == nil || m.ptr == nil {
		return false
	}
	return C.whisper_bind_is_multilingual(m.ptr) != 0
}

// langStr maps a language id to its name ("" if unknown).
func langStr(id int) string {
	if id < 0 {
		return ""
	}
	if s := C.whisper_bind_lang_str(C.int(id)); s != nil {
		return C.GoString(s)
	}
	return ""
}
