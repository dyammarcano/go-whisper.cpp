package whisper

import "time"

// windowResult is the outcome of classifying one window run.
type windowResult struct {
	results   []StreamResult
	committed time.Duration
	promptAdd string
}

// classifyWindow turns a window run's segments (timestamps relative to windowStart) into
// ordered StreamResults. A segment is final iff committed < absEnd <= commitBefore; one
// whose end is already <= committed is a re-decoded overlap and is skipped; everything
// past commitBefore is a (refining) partial. Pure: no whisper, no goroutines, no clock.
func classifyWindow(segs []Segment, windowStart, committed, commitBefore time.Duration) windowResult {
	out := windowResult{committed: committed}
	for _, s := range segs {
		absStart := s.Start + windowStart
		absEnd := s.End + windowStart
		if absEnd <= out.committed {
			continue
		}
		abs := Segment{Start: absStart, End: absEnd, Text: s.Text, Tokens: s.Tokens}
		if absEnd <= commitBefore {
			out.results = append(out.results, StreamResult{Segment: abs, Partial: false})
			out.committed = absEnd
			out.promptAdd += s.Text
		} else {
			out.results = append(out.results, StreamResult{Segment: abs, Partial: true})
		}
	}
	return out
}
