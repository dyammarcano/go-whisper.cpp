package whisper

// #include "binding.h"
import "C"

import (
	"time"
	"unsafe"
)

// Result is the outcome of a transcription.
type Result struct {
	Segments []Segment
	Language string // detected/used language name (e.g. "en"); "" if unknown
}

// Segment is a transcribed span of audio.
type Segment struct {
	Start, End time.Duration
	Text       string
	Tokens     []Token // populated only when WithTokenTimestamps() is set
}

// Token is a single decoded token with timing + probability.
type Token struct {
	Text       string
	P          float32
	Start, End time.Duration
}

func csToDuration(cs int64) time.Duration { return time.Duration(cs) * 10 * time.Millisecond }

// marshalResult converts a C whisper_bind_result into a Go Result and frees the C memory.
func marshalResult(ptr unsafe.Pointer) *Result {
	if ptr == nil {
		return &Result{}
	}
	r := (*C.whisper_bind_result)(ptr)
	defer C.whisper_bind_free_result(r)

	n := int(r.n_segments)
	res := &Result{Segments: make([]Segment, 0, n), Language: langStr(int(r.lang_id))}
	if n == 0 {
		return res
	}
	segs := unsafe.Slice(r.segments, n)
	for i := range n {
		cs := segs[i]
		seg := Segment{
			Start: csToDuration(int64(cs.t0)),
			End:   csToDuration(int64(cs.t1)),
			Text:  C.GoString(cs.text),
		}
		if nt := int(cs.n_tokens); nt > 0 && cs.tokens != nil {
			toks := unsafe.Slice(cs.tokens, nt)
			seg.Tokens = make([]Token, nt)
			for j := range nt {
				seg.Tokens[j] = Token{
					Text:  C.GoString(toks[j].text),
					P:     float32(toks[j].p),
					Start: csToDuration(int64(toks[j].t0)),
					End:   csToDuration(int64(toks[j].t1)),
				}
			}
		}
		res.Segments = append(res.Segments, seg)
	}
	return res
}
