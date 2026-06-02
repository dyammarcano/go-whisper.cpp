package diarize

import "time"

// Segment is a transcript span to be labeled. Kept local (structurally identical to
// whisper.Segment's timing/text) so the diarize package does NOT import the
// whisper.cpp binding — callers map whisper.Segment -> diarize.Segment in 2 lines.
type Segment struct {
	Start, End time.Duration
	Text       string
}

// LabeledSegment is a Segment annotated with the dominant speaker (-1 if no turn overlaps).
type LabeledSegment struct {
	Segment
	Speaker int
}

// Label assigns each segment the speaker whose turn has the greatest temporal overlap
// (-1 if none). Turns need not be sorted. Pure Go; O(len(segs)*len(turns)).
func Label(segs []Segment, turns []Turn) []LabeledSegment {
	out := make([]LabeledSegment, len(segs))
	for i, s := range segs {
		best, bestOverlap := -1, time.Duration(0)
		for _, t := range turns {
			if ov := overlap(s.Start, s.End, t.Start, t.End); ov > bestOverlap {
				best, bestOverlap = t.Speaker, ov
			}
		}
		out[i] = LabeledSegment{Segment: s, Speaker: best}
	}
	return out
}

// overlap returns the duration [aStart,aEnd] and [bStart,bEnd] share (0 if disjoint).
func overlap(aStart, aEnd, bStart, bEnd time.Duration) time.Duration {
	start := aStart
	if bStart > start {
		start = bStart
	}
	end := aEnd
	if bEnd < end {
		end = bEnd
	}
	if end > start {
		return end - start
	}
	return 0
}
