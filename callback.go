package whisper

// The preamble needs <stdint.h> so cgo can resolve C.uintptr_t / C.int64_t used in
// the //export signatures below. binding.cpp declares these funcs itself (it is
// compiled outside cgo), so NO extern decls are needed here. import "C" is required
// for //export to take effect.
//
// #include <stdint.h>
import "C"

import (
	"runtime/cgo"
	"sync/atomic"
)

// callbackBridge carries Go closures + cancellation/panic state through whisper's
// void* user_data (as a cgo.Handle). Trampolines run on ggml worker threads, so
// mutable state must be goroutine/thread-safe: aborted is an atomic flag and
// panicErr is an atomic pointer (read by runFull on the caller goroutine).
type callbackBridge struct {
	onSegment  func(seg Segment)
	onProgress func(percent int)
	aborted    *abortFlag
	panicErr   atomic.Pointer[error]
}

// goWhisperSegment is called by whisper for each NEW segment; the shim passes that
// segment's timings + text directly (no re-collection, no reentry via the Model).
//
//export goWhisperSegment
func goWhisperSegment(handle C.uintptr_t, t0 C.int64_t, t1 C.int64_t, text *C.char) {
	defer recoverInto(handle)
	b := cgo.Handle(uintptr(handle)).Value().(*callbackBridge)
	if b.onSegment == nil {
		return
	}
	b.onSegment(Segment{
		Start: csToDuration(int64(t0)),
		End:   csToDuration(int64(t1)),
		Text:  C.GoString(text),
	})
}

//export goWhisperProgress
func goWhisperProgress(handle C.uintptr_t, progress C.int) {
	defer recoverInto(handle)
	b := cgo.Handle(uintptr(handle)).Value().(*callbackBridge)
	if b.onProgress != nil {
		b.onProgress(int(progress))
	}
}

//export goWhisperAbort
func goWhisperAbort(handle C.uintptr_t) C.int {
	b, ok := cgo.Handle(uintptr(handle)).Value().(*callbackBridge)
	if !ok || b.aborted == nil {
		return 0
	}
	if b.aborted.isSet() {
		return 1
	}
	return 0
}

// recoverInto converts a panic inside a trampoline into a stored error + abort,
// so a Go panic never unwinds into C (undefined behavior).
func recoverInto(handle C.uintptr_t) {
	if r := recover(); r != nil {
		if b, ok := cgo.Handle(uintptr(handle)).Value().(*callbackBridge); ok {
			err := panicToError(r)
			b.panicErr.Store(&err)
			if b.aborted != nil {
				b.aborted.set()
			}
		}
	}
}
