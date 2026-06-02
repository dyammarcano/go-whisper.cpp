# go-whisper.cpp Speaker Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native (no-Python) `diarize` package — speaker diarization via sherpa-onnx + pyannote models — plus a pure-Go merge helper that labels whisper transcript segments with speakers.

**Architecture:** `diarize` wraps `github.com/k2-fsa/sherpa-onnx-go`'s `OfflineSpeakerDiarization` (pyannote segmentation-3.0 + WeSpeaker embedding + clustering, via prebuilt MinGW ONNX-Runtime DLLs). It depends ONLY on sherpa-onnx-go (never whisper.cpp); the merge helper uses a local plain `Segment` type so the two cgo deps stay decoupled. Same 16 kHz mono float32 input as whisper.

**Tech Stack:** Go 1.25+ (cgo), `github.com/k2-fsa/sherpa-onnx-go` v1.13.2 (ONNX Runtime, CPU), MinGW gcc 15.2, Task, ginkgo/gomega. CPU-only (prebuilt onnxruntime).

**Spec:** `docs/superpowers/specs/2026-06-01-go-whisper-diarize-design.md`. **Builds on:** merged whisper foundation + GPU backends.

**Module:** `github.com/dyammarcano/go-whisper.cpp` (new `diarize/` subpackage, same module).

---

## Shared contracts (USE THESE EXACT NAMES)

- **Upstream Go API** (`import sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"`): `OfflineSpeakerDiarizationConfig{Segmentation, Embedding, Clustering, MinDurationOn float32, MinDurationOff float32}`; `Segmentation.Pyannote.Model string`, `Segmentation.NumThreads/Debug int`, `Segmentation.Provider string`; `Embedding.Model string`, `Embedding.NumThreads/Debug int`, `Embedding.Provider string`; `Clustering FastClusteringConfig{NumClusters int; Threshold float32}`. `NewOfflineSpeakerDiarization(*cfg) *OfflineSpeakerDiarization` (nil on bad config), `DeleteOfflineSpeakerDiarization(sd)`, `(sd).SampleRate() int`, `(sd).Process([]float32) []OfflineSpeakerDiarizationSegment{Start,End float32 seconds; Speaker int}`.
- **Our public API** (package `diarize`): `Diarizer`, `New(segModel, embModel string, ...Option) (*Diarizer, error)`, `(*Diarizer).Diarize([]float32) ([]Turn, error)`, `Close() error`, `SampleRate() int`. `Turn{Start,End time.Duration; Speaker int}`. `Option`, `WithNumSpeakers/WithThreshold/WithMinDuration/WithThreads/WithDebug`. Merge: `Segment{Start,End time.Duration; Text string}`, `LabeledSegment{Segment; Speaker int}`, `Label([]Segment, []Turn) []LabeledSegment`.
- **Sentinels:** `ErrConfig`, `ErrEmptyAudio`, `ErrClosed`.
- **Unit bridge:** sherpa returns **seconds (float32)** → `time.Duration` via `secondsToDuration`. (whisper segments are already `time.Duration`.)

## File structure

```
diarize/diarize.go      Diarizer, New, Diarize, Close, SampleRate, Turn, secondsToDuration   [Task 3]
diarize/options.go      Option + setters + defaults                                          [Task 3]
diarize/errors.go       sentinels                                                            [Task 3]
diarize/merge.go        Segment, LabeledSegment, Label, overlap                              [Task 4]
diarize/merge_test.go   pure-Go table tests                                                  [Task 4]
diarize/diarize_test.go integration (env-gated)                                              [Task 5]
scripts/download-diarize-models.sh   SHA-verified model fetch                                [Task 1]
scripts/copy-sherpa-dlls.sh          copy runtime DLLs from module cache                     [Task 1]
examples/diarize/main.go             standalone CLI                                          [Task 6]
examples/transcribe-diarize/main.go  whisper + diarize -> labeled transcript                 [Task 6]
Taskfile.yml            +models:diarize, +diarize:dlls                                       [Task 1]
go.mod                  +github.com/k2-fsa/sherpa-onnx-go                                     [Task 1]
.github/workflows/ci.yml +diarize-build job                                                  [Task 7]
README.md               Diarization section                                                  [Task 7]
```

---

## Task 1: Add the dependency + provisioning (models, DLLs, Task targets)

**Files:** Modify `go.mod`; Create `scripts/download-diarize-models.sh`, `scripts/copy-sherpa-dlls.sh`; Modify `Taskfile.yml`.

- [ ] **Step 1: Add the sherpa-onnx-go dependency**

Run:
```bash
go get github.com/k2-fsa/sherpa-onnx-go@v1.13.2
```
Expected: `go.mod` requires `github.com/k2-fsa/sherpa-onnx-go v1.13.2` and `go.sum` is populated; the platform module (sherpa-onnx-go-windows on this box) + its vendored DLLs download into the module cache. **Do NOT run `go mod tidy` here** — no package imports the dep until Task 3, and tidy would drop the unused require. (`go get` adds the require even without an importer.) `go mod tidy` runs in Task 3 Step 4 once `diarize.go` imports it. Verify with `go list -m github.com/k2-fsa/sherpa-onnx-go`.

- [ ] **Step 2: Write `scripts/download-diarize-models.sh`** (SHA256-verified)

Create `scripts/download-diarize-models.sh`:
```bash
#!/usr/bin/env bash
# Download diarization models (pyannote segmentation-3.0 MIT + WeSpeaker embedding) into models/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p models
SEG_TBZ="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"

# Segmentation (extract -> models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx)
if [ ! -f "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx" ]; then
  echo "downloading segmentation model"; curl -fL "$SEG_TBZ" -o models/seg.tar.bz2
  tar xjf models/seg.tar.bz2 -C models && rm -f models/seg.tar.bz2
fi
# Embedding
if [ ! -f "models/wespeaker_en_voxceleb_resnet34_LM.onnx" ]; then
  echo "downloading embedding model"; curl -fL "$EMB_URL" -o models/wespeaker_en_voxceleb_resnet34_LM.onnx
fi
# Optional checksum verification (fill SHA once known): echo "<sha>  models/..." | sha256sum -c -
echo "SEG=$ROOT/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
echo "EMB=$ROOT/models/wespeaker_en_voxceleb_resnet34_LM.onnx"
```
> `curl` runs inside this build script (not the agent shell) — intentional, like the whisper `download-model.sh`. Compute and pin SHA256s once after first download.

- [ ] **Step 3: Write `scripts/copy-sherpa-dlls.sh`** (copy runtime DLLs from the module cache)

Create `scripts/copy-sherpa-dlls.sh`:
```bash
#!/usr/bin/env bash
# Copy sherpa-onnx + onnxruntime runtime DLLs from the Go module cache to a target dir
# (default: repo root, so `go test`/examples find them on PATH/CWD).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
DEST="${1:-$ROOT}"
VER="$(go list -m -f '{{.Version}}' github.com/k2-fsa/sherpa-onnx-go-windows 2>/dev/null || echo v1.13.2)"
GOMODCACHE="$(go env GOMODCACHE)"
LIBDIR="$GOMODCACHE/github.com/k2-fsa/sherpa-onnx-go-windows@$VER/lib/x86_64-pc-windows-gnu"
if [ ! -d "$LIBDIR" ]; then echo "lib dir not found: $LIBDIR (run 'go mod download' first)"; exit 1; fi
mkdir -p "$DEST"
for dll in onnxruntime.dll sherpa-onnx-c-api.dll sherpa-onnx-cxx-api.dll; do
  cp -f "$LIBDIR/$dll" "$DEST/" && echo "copied $dll -> $DEST"
done
```

- [ ] **Step 4: Add Task targets to `Taskfile.yml`**

Add under `tasks:`:
```yaml
  models:diarize:
    desc: Download diarization models (pyannote seg-3.0 MIT + WeSpeaker embedding)
    cmds: [bash scripts/download-diarize-models.sh]
  diarize:dlls:
    desc: Copy sherpa-onnx/onnxruntime runtime DLLs from the module cache to repo root
    cmds: [bash scripts/copy-sherpa-dlls.sh]
```

- [ ] **Step 5: Validate + commit**

Run: `task --list` (shows the 2 new targets); `bash -n scripts/download-diarize-models.sh scripts/copy-sherpa-dlls.sh`.
```bash
git add go.mod go.sum scripts/download-diarize-models.sh scripts/copy-sherpa-dlls.sh Taskfile.yml
git commit -m "build(diarize): add sherpa-onnx-go dep + model/DLL provisioning"
```

---

## Task 2: Phase-0 spike — link & run the prebuilt DLLs (go/no-go)

**Files:** Create (throwaway) `_spike/main.go`; remove after.

**Purpose:** confirm the toolchain (MinGW gcc 15.2; module `go 1.25`, dev-box toolchain go1.26.x; CI pins go-version 1.25) links & RUNS the v1.13.2 prebuilt DLLs without a CRT mismatch, BEFORE building the wrapper. If this fails, STOP and escalate (try an older sherpa-onnx-go tag, or build sherpa-onnx from source).

- [ ] **Step 1: Provision models + DLLs**

Run:
```bash
task models:diarize
task diarize:dlls
```
Expected: `models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx`, `models/wespeaker_en_voxceleb_resnet34_LM.onnx`, and `onnxruntime.dll`/`sherpa-onnx-c-api.dll`/`sherpa-onnx-cxx-api.dll` at repo root.

- [ ] **Step 2: Write the throwaway spike** using sherpa directly

Create `_spike/main.go`:
```go
package main

import (
	"fmt"
	"os"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	var c sherpa.OfflineSpeakerDiarizationConfig
	c.Segmentation.Pyannote.Model = "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
	c.Embedding.Model = "models/wespeaker_en_voxceleb_resnet34_LM.onnx"
	c.Clustering.Threshold = 0.5
	c.Segmentation.Provider = "cpu"
	c.Embedding.Provider = "cpu"
	sd := sherpa.NewOfflineSpeakerDiarization(&c)
	if sd == nil {
		fmt.Fprintln(os.Stderr, "nil diarization (bad config/models)")
		os.Exit(1)
	}
	defer sherpa.DeleteOfflineSpeakerDiarization(sd)
	fmt.Println("sample rate:", sd.SampleRate())
	w := sherpa.ReadWave(os.Args[1]) // a 16kHz mono wav
	if w == nil {
		fmt.Fprintln(os.Stderr, "bad wave")
		os.Exit(1)
	}
	if w.SampleRate != sd.SampleRate() {
		fmt.Fprintf(os.Stderr, "wav rate %d != required %d\n", w.SampleRate, sd.SampleRate())
		os.Exit(1)
	}
	for _, s := range sd.Process(w.Samples) {
		fmt.Printf("%.2f-%.2f speaker_%d\n", s.Start, s.End, s.Speaker)
	}
}
```

- [ ] **Step 3: Get a multi-speaker test wav + build & run the spike**

Run:
```bash
curl -fL https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav -o models/4speakers.wav
CGO_ENABLED=1 go run ./_spike models/4speakers.wav
```
Expected: prints `sample rate: 16000` and several `start-end speaker_N` lines spanning multiple speaker ids. THIS PROVES the prebuilt DLLs link+run on this toolchain.
> If `go run` fails at LINK (undefined refs / CRT) or CRASHES on load (missing DLL → ensure the 3 DLLs are at repo root/CWD; or a libstdc++/winpthread mismatch): STOP. Report BLOCKED with the exact error. Remedies: pin an older sherpa-onnx-go (e.g. v1.12.x) and retry; or as a last resort build sherpa-onnx C libs from source with this MinGW. Do NOT proceed to Task 3 until the spike runs.

- [ ] **Step 4: Remove the spike (it was a probe)**

Run:
```bash
rm -rf _spike
```
No commit (throwaway). Record the spike result (pass + the speaker count seen) in the task report.

---

## Task 3: Core `diarize` package (Diarizer + options + errors)

**Files:** Create `diarize/errors.go`, `diarize/options.go`, `diarize/diarize.go`.

- [ ] **Step 1: Write `diarize/errors.go`**

Create `diarize/errors.go`:
```go
package diarize

import "errors"

var (
	// ErrConfig means sherpa rejected the config (e.g. missing/invalid model files).
	ErrConfig = errors.New("diarize: invalid configuration (bad or missing models)")
	// ErrEmptyAudio means Diarize was called with no samples.
	ErrEmptyAudio = errors.New("diarize: empty audio (no samples)")
	// ErrClosed means the diarizer was used after Close.
	ErrClosed = errors.New("diarize: use of closed diarizer")
)
```

- [ ] **Step 2: Write `diarize/options.go`**

Create `diarize/options.go`:
```go
package diarize

import "time"

// Option configures a Diarizer.
type Option func(*options)

type options struct {
	numSpeakers int     // >0 => known count (overrides threshold)
	threshold   float32 // used when numSpeakers == 0
	minOn       time.Duration
	minOff      time.Duration
	threads     int
	debug       bool
}

func defaultOptions() options {
	return options{
		numSpeakers: 0,
		threshold:   0.5, // unknown-count default; larger => fewer speakers
		minOn:       300 * time.Millisecond,
		minOff:      500 * time.Millisecond,
		threads:     1,
	}
}

// WithNumSpeakers forces exactly n speakers (use when the count is known). Mutually
// exclusive with WithThreshold; whichever is applied last wins.
func WithNumSpeakers(n int) Option { return func(o *options) { o.numSpeakers = n } }

// WithThreshold sets the clustering threshold for unknown speaker count (larger =>
// fewer speakers; default 0.5). Clears any WithNumSpeakers.
func WithThreshold(t float32) Option {
	return func(o *options) { o.numSpeakers = 0; o.threshold = t }
}

// WithMinDuration sets minimum on/off speech durations (defaults 300ms / 500ms).
func WithMinDuration(on, off time.Duration) Option {
	return func(o *options) { o.minOn = on; o.minOff = off }
}

// WithThreads sets ONNX intra-op threads for both models (default 1).
func WithThreads(n int) Option { return func(o *options) { o.threads = n } }

// WithDebug enables sherpa-onnx debug logging.
var WithDebug Option = func(o *options) { o.debug = true }
```

- [ ] **Step 3: Write `diarize/diarize.go`**

Create `diarize/diarize.go`:
```go
// Package diarize provides native speaker diarization ("who spoke when") via
// sherpa-onnx + pyannote models (ONNX Runtime), with no Python dependency. It
// depends only on sherpa-onnx-go (never on the whisper.cpp binding); use Label to
// fuse Turns with a transcript.
package diarize

import (
	"fmt"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// Diarizer wraps a sherpa-onnx OfflineSpeakerDiarization. Not safe for concurrent
// Diarize calls; create one per goroutine if needed. Close frees native memory.
type Diarizer struct {
	sd *sherpa.OfflineSpeakerDiarization
}

// New builds a diarizer from segmentation + embedding ONNX model paths.
func New(segmentationModel, embeddingModel string, opts ...Option) (*Diarizer, error) {
	o := defaultOptions()
	for _, opt := range opts {
		opt(&o)
	}
	var c sherpa.OfflineSpeakerDiarizationConfig
	c.Segmentation.Pyannote.Model = segmentationModel
	c.Segmentation.NumThreads = o.threads
	c.Segmentation.Provider = "cpu"
	c.Embedding.Model = embeddingModel
	c.Embedding.NumThreads = o.threads
	c.Embedding.Provider = "cpu"
	if o.debug {
		c.Segmentation.Debug = 1
		c.Embedding.Debug = 1
	}
	if o.numSpeakers > 0 {
		c.Clustering.NumClusters = o.numSpeakers
	} else {
		c.Clustering.Threshold = o.threshold
	}
	c.MinDurationOn = float32(o.minOn.Seconds())
	c.MinDurationOff = float32(o.minOff.Seconds())

	sd := sherpa.NewOfflineSpeakerDiarization(&c)
	if sd == nil {
		return nil, fmt.Errorf("%w: segmentation=%q embedding=%q", ErrConfig, segmentationModel, embeddingModel)
	}
	return &Diarizer{sd: sd}, nil
}

// Close frees the underlying native diarizer. Idempotent.
func (d *Diarizer) Close() error {
	if d == nil || d.sd == nil {
		return nil
	}
	sherpa.DeleteOfflineSpeakerDiarization(d.sd)
	d.sd = nil
	return nil
}

// SampleRate is the required input rate (16000). Returns 0 if closed.
func (d *Diarizer) SampleRate() int {
	if d == nil || d.sd == nil {
		return 0
	}
	return d.sd.SampleRate()
}

// Diarize runs offline diarization over 16 kHz mono float32 samples in [-1,1] and
// returns speaker turns ordered by Start.
func (d *Diarizer) Diarize(samples []float32) ([]Turn, error) {
	if d == nil || d.sd == nil {
		return nil, ErrClosed
	}
	if len(samples) == 0 {
		return nil, ErrEmptyAudio
	}
	segs := d.sd.Process(samples)
	turns := make([]Turn, len(segs))
	for i, s := range segs {
		turns[i] = Turn{
			Start:   secondsToDuration(s.Start),
			End:     secondsToDuration(s.End),
			Speaker: s.Speaker,
		}
	}
	return turns, nil
}

// Turn is one speaker's contiguous span. Speaker is a 0-based id, per-file (NOT
// stable across different recordings).
type Turn struct {
	Start, End time.Duration
	Speaker    int
}

func secondsToDuration(s float32) time.Duration {
	return time.Duration(float64(s) * float64(time.Second))
}
```

- [ ] **Step 4: Build the package** (cgo links the prebuilt DLLs)

Run:
```bash
go mod tidy   # diarize.go now imports sherpa-onnx-go -> records it + the platform module (indirect)
CGO_ENABLED=1 go build ./diarize/...
go vet ./diarize/...
```
Expected: clean; `go.mod` now firmly requires sherpa-onnx-go. The vendored DLLs are in the module cache; linking uses the baked-in cgo LDFLAGS (no extra flags). (`go vet`/`go build` only LINK the libs — no DLLs-on-PATH needed yet; that's only for RUNNING a binary, Task 4+.)

- [ ] **Step 5: Commit**

```bash
git add diarize/errors.go diarize/options.go diarize/diarize.go go.mod go.sum
git commit -m "feat(diarize): Diarizer over sherpa-onnx (pyannote models, no Python)"
```

---

## Task 4: Merge helper (`Label`) — pure Go, TDD

**Files:** Create `diarize/merge_test.go`, `diarize/merge.go`.

- [ ] **Step 1: Write the failing test**

Create `diarize/merge_test.go`:
```go
package diarize_test

import (
	"testing"
	"time"

	"github.com/dyammarcano/go-whisper.cpp/diarize"
)

func ms(n int) time.Duration { return time.Duration(n) * time.Millisecond }

func TestLabel(t *testing.T) {
	turns := []diarize.Turn{
		{Start: ms(0), End: ms(1000), Speaker: 0},
		{Start: ms(1000), End: ms(2000), Speaker: 1},
	}
	cases := []struct {
		name string
		seg  diarize.Segment
		want int
	}{
		{"inside speaker 0", diarize.Segment{Start: ms(100), End: ms(500), Text: "a"}, 0},
		{"inside speaker 1", diarize.Segment{Start: ms(1200), End: ms(1800), Text: "b"}, 1},
		{"overlaps both, more in 1", diarize.Segment{Start: ms(900), End: ms(1600), Text: "c"}, 1},
		{"no overlap", diarize.Segment{Start: ms(5000), End: ms(6000), Text: "d"}, -1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := diarize.Label([]diarize.Segment{tc.seg}, turns)
			if len(got) != 1 {
				t.Fatalf("len=%d want 1", len(got))
			}
			if got[0].Speaker != tc.want {
				t.Errorf("speaker=%d want %d", got[0].Speaker, tc.want)
			}
			if got[0].Text != tc.seg.Text || got[0].Start != tc.seg.Start || got[0].End != tc.seg.End {
				t.Errorf("segment not preserved: %+v", got[0])
			}
		})
	}
}

func TestLabel_Empty(t *testing.T) {
	if got := diarize.Label(nil, nil); len(got) != 0 {
		t.Fatalf("want empty, got %d", len(got))
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./diarize/ -run TestLabel -v`
Expected: FAIL — `Segment`/`LabeledSegment`/`Label` undefined.

- [ ] **Step 3: Write `diarize/merge.go`**

Create `diarize/merge.go`:
```go
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
```

- [ ] **Step 4: Run to verify it passes**

IMPORTANT (Windows DLL-shadow, verified in the Phase-0 spike): package `diarize` imports sherpa-onnx-go (cgo), so the test binary **load-links the 3 sherpa DLLs at startup** — even for pure-Go `TestLabel`. Windows searches the **exe's own directory first, then System32** — and a stale `C:\Windows\System32\onnxruntime.dll` (older ORT) shadows the bundled 1.24.x, crashing with "The requested API version [N] is not available". `go test ./diarize/` builds the test exe in a TEMP dir (no DLLs there → System32 wins → crash). So **compile the test binary into the DLL dir and run it there**:
```bash
task diarize:dlls                               # copy the 3 DLLs to repo root
go test -c -o diarize.test.exe ./diarize/       # build test exe AT repo root (beside the DLLs)
./diarize.test.exe -test.run TestLabel -test.v  # exe-dir search finds the bundled ORT first
rm -f diarize.test.exe
```
Expected: PASS (both tests, all sub-cases). The test LOGIC is pure Go (no models needed); the DLLs must sit **beside the test exe** so the correct ORT loads.

- [ ] **Step 5: Commit**

```bash
git add diarize/merge.go diarize/merge_test.go
git commit -m "feat(diarize): pure-Go Label merge helper (speaker by max overlap)"
```

---

## Task 5: Integration test (env-gated)

**Files:** Create `diarize/diarize_test.go`.

- [ ] **Step 1: Write the integration spec**

Create `diarize/diarize_test.go`:
```go
package diarize_test

import (
	"os"
	"testing"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"github.com/dyammarcano/go-whisper.cpp/diarize"
)

// gated: set DIARIZE_SEG_MODEL + DIARIZE_EMB_MODEL + DIARIZE_WAV (a 16kHz mono multi-
// speaker wav). Provision with `task models:diarize` + `task diarize:dlls`.
func TestDiarize_Integration(t *testing.T) {
	seg := os.Getenv("DIARIZE_SEG_MODEL")
	emb := os.Getenv("DIARIZE_EMB_MODEL")
	wavPath := os.Getenv("DIARIZE_WAV")
	if seg == "" || emb == "" || wavPath == "" {
		t.Skip("set DIARIZE_SEG_MODEL, DIARIZE_EMB_MODEL, DIARIZE_WAV to run")
	}

	d, err := diarize.New(seg, emb, diarize.WithThreshold(0.5))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer d.Close()
	if d.SampleRate() != 16000 {
		t.Fatalf("sample rate = %d want 16000", d.SampleRate())
	}

	w := sherpa.ReadWave(wavPath)
	if w == nil {
		t.Fatalf("ReadWave(%q) failed", wavPath)
	}
	if w.SampleRate != d.SampleRate() {
		t.Fatalf("wav rate %d != required %d", w.SampleRate, d.SampleRate())
	}
	turns, err := d.Diarize(w.Samples)
	if err != nil {
		t.Fatalf("Diarize: %v", err)
	}
	if len(turns) == 0 {
		t.Fatal("no turns produced")
	}
	speakers := map[int]bool{}
	for _, tn := range turns {
		if tn.End <= tn.Start {
			t.Errorf("non-positive turn: %+v", tn)
		}
		speakers[tn.Speaker] = true
	}
	t.Logf("turns=%d speakers=%d", len(turns), len(speakers))
	if len(speakers) < 2 {
		t.Errorf("expected >=2 speakers in a multi-speaker clip, got %d", len(speakers))
	}
}

func TestDiarize_EmptyAndClosed(t *testing.T) {
	// pure-API guards — no model needed for the closed path
	var d *diarize.Diarizer
	if _, err := d.Diarize([]float32{0.1}); err == nil {
		t.Error("nil diarizer Diarize should error")
	}
}
```

- [ ] **Step 2: Run the gated test without env (confirms skip)**

(Windows: compile the test exe beside the DLLs — see Task 4 Step 4 — to avoid the System32 ORT shadow.)
Run:
```bash
go test -c -o diarize.test.exe ./diarize/
./diarize.test.exe -test.run TestDiarize -test.v
```
Expected: `TestDiarize_Integration` SKIPS (env unset); `TestDiarize_EmptyAndClosed` PASSES.

- [ ] **Step 3: Run the full integration with models + DLLs**

Run:
```bash
task models:diarize && task diarize:dlls
curl -fL https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav -o models/4speakers.wav
go test -c -o diarize.test.exe ./diarize/
DIARIZE_SEG_MODEL="$(pwd)/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx" \
DIARIZE_EMB_MODEL="$(pwd)/models/wespeaker_en_voxceleb_resnet34_LM.onnx" \
DIARIZE_WAV="$(pwd)/models/4speakers.wav" \
./diarize.test.exe -test.run TestDiarize_Integration -test.v
```
Expected: PASS — logs `turns=N speakers>=2` (the 4-speaker clip should yield ~4 with threshold 0.5; the assertion only requires >=2). DLLs at repo root satisfy the loader.
> If paths with `$(pwd)` (MSYS `/d/...`) fail to load the ONNX models, use native Windows paths (`D:\...`), mirroring the whisper TEST_MODEL note.

- [ ] **Step 4: Commit**

```bash
git add diarize/diarize_test.go
git commit -m "test(diarize): integration spec (gated) + closed-diarizer guard"
```

---

## Task 6: Examples — standalone + combo

**Files:** Create `examples/diarize/main.go`, `examples/transcribe-diarize/main.go`.

- [ ] **Step 1: Write `examples/diarize/main.go`** (standalone)

Create `examples/diarize/main.go`:
```go
package main

import (
	"flag"
	"fmt"
	"os"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"github.com/dyammarcano/go-whisper.cpp/diarize"
)

func main() {
	seg := flag.String("seg", "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx", "segmentation model")
	emb := flag.String("emb", "models/wespeaker_en_voxceleb_resnet34_LM.onnx", "embedding model")
	wavPath := flag.String("f", "models/4speakers.wav", "16 kHz mono wav")
	n := flag.Int("n", 0, "number of speakers (0 = auto)")
	thr := flag.Float64("t", 0.5, "clustering threshold when n=0")
	flag.Parse()

	opts := []diarize.Option{}
	if *n > 0 {
		opts = append(opts, diarize.WithNumSpeakers(*n))
	} else {
		opts = append(opts, diarize.WithThreshold(float32(*thr)))
	}
	d, err := diarize.New(*seg, *emb, opts...)
	if err != nil {
		fmt.Fprintln(os.Stderr, "new:", err)
		os.Exit(1)
	}
	defer d.Close()

	w := sherpa.ReadWave(*wavPath)
	if w == nil {
		fmt.Fprintln(os.Stderr, "read wav failed:", *wavPath)
		os.Exit(1)
	}
	if w.SampleRate != d.SampleRate() {
		fmt.Fprintf(os.Stderr, "wav rate %d != required %d\n", w.SampleRate, d.SampleRate())
		os.Exit(1)
	}
	turns, err := d.Diarize(w.Samples)
	if err != nil {
		fmt.Fprintln(os.Stderr, "diarize:", err)
		os.Exit(1)
	}
	for _, t := range turns {
		fmt.Printf("[%s -> %s] speaker %d\n", t.Start, t.End, t.Speaker)
	}
}
```

- [ ] **Step 2: Write `examples/transcribe-diarize/main.go`** (combo)

Create `examples/transcribe-diarize/main.go`:
```go
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	whisper "github.com/dyammarcano/go-whisper.cpp"
	"github.com/dyammarcano/go-whisper.cpp/diarize"
	"github.com/dyammarcano/go-whisper.cpp/wav"
)

func main() {
	model := flag.String("m", "models/ggml-tiny.en.bin", "whisper ggml model")
	seg := flag.String("seg", "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx", "segmentation model")
	emb := flag.String("emb", "models/wespeaker_en_voxceleb_resnet34_LM.onnx", "embedding model")
	audio := flag.String("f", "models/4speakers.wav", "16 kHz mono wav")
	n := flag.Int("n", 0, "number of speakers (0 = auto)")
	flag.Parse()

	samples, err := wav.ReadFile(*audio)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read wav:", err)
		os.Exit(1)
	}

	m, err := whisper.New(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "whisper:", err)
		os.Exit(1)
	}
	defer m.Close()
	res, err := m.Transcribe(context.Background(), samples)
	if err != nil {
		fmt.Fprintln(os.Stderr, "transcribe:", err)
		os.Exit(1)
	}

	opts := []diarize.Option{diarize.WithThreshold(0.5)}
	if *n > 0 {
		opts = []diarize.Option{diarize.WithNumSpeakers(*n)}
	}
	d, err := diarize.New(*seg, *emb, opts...)
	if err != nil {
		fmt.Fprintln(os.Stderr, "diarize:", err)
		os.Exit(1)
	}
	defer d.Close()
	turns, err := d.Diarize(samples)
	if err != nil {
		fmt.Fprintln(os.Stderr, "diarize:", err)
		os.Exit(1)
	}

	segs := make([]diarize.Segment, len(res.Segments))
	for i, s := range res.Segments {
		segs[i] = diarize.Segment{Start: s.Start, End: s.End, Text: s.Text}
	}
	for _, ls := range diarize.Label(segs, turns) {
		fmt.Printf("[Speaker %d] %s\n", ls.Speaker, ls.Text)
	}
}
```
> The combo example links BOTH whisper.cpp and onnxruntime (cgo) — it needs the whisper CPU static libs (`task build:cpu`) AND the sherpa DLLs (`task diarize:dlls`) + both model sets.

- [ ] **Step 3: Build the examples**

Run:
```bash
CGO_ENABLED=1 go build ./examples/diarize ./examples/transcribe-diarize
```
Expected: clean compile (no run required here; running needs models + DLLs + a whisper build).

- [ ] **Step 4: Commit**

```bash
git add examples/diarize/main.go examples/transcribe-diarize/main.go
git commit -m "docs(example): standalone diarize + transcribe-diarize combo"
```

---

## Task 7: CI + README

**Files:** Modify `.github/workflows/ci.yml`, `README.md`.

- [ ] **Step 1: Add a `diarize-build` CI job**

Append to `.github/workflows/ci.yml` under `jobs:`:
```yaml
  diarize-build:
    # Build the diarize package + run the pure-Go merge tests. The integration test is
    # env-gated and skips without models/DLLs (model downloads + GPU not run in CI here).
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
        with: { submodules: recursive }
      - uses: actions/setup-go@v5
        with: { go-version: '1.25' }
      - name: Setup MinGW (Windows)
        if: runner.os == 'Windows'
        uses: msys2/setup-msys2@v2
        with: { msystem: UCRT64, path-type: inherit, install: mingw-w64-ucrt-x86_64-gcc }
      - name: Build diarize (link-verify, both OSes)
        env: { CC: gcc, CXX: g++, CGO_ENABLED: '1' }
        run: |
          go mod download
          go build ./diarize/...
      - name: Provision DLLs + run merge tests (Windows)
        if: runner.os == 'Windows'
        env: { CC: gcc, CXX: g++, CGO_ENABLED: '1' }
        run: |
          bash scripts/copy-sherpa-dlls.sh "$PWD"
          go test -c -o diarize.test.exe ./diarize/
          ./diarize.test.exe -test.run TestLabel -test.v
```
> `go build ./diarize/...` only LINKS the sherpa libs (their `-L` path is the module cache) — that's why build-verify works on both ubuntu+windows. RUNNING any `diarize` test binary load-links the DLLs, so the test step (Windows) first copies them to CWD via `copy-sherpa-dlls.sh`. The gated integration test skips (no model env). Linux test execution needs `LD_LIBRARY_PATH` to the `.so` dir — deferred (build-verify only on ubuntu). Mirrors the existing `build-test` job's `CC: gcc, CXX: g++` convention.

- [ ] **Step 2: Add a "Speaker diarization" section to `README.md`**

Add a section documenting: what it is (native, no Python, pyannote models via sherpa-onnx, CPU); `task models:diarize` + `task diarize:dlls`; the standalone API (`diarize.New` / `Diarize` / `Turn`, `WithNumSpeakers` vs `WithThreshold`); the merge (`diarize.Label` over `diarize.Segment` + `Turn`); the combo example mapping `whisper.Segment → diarize.Segment`; runtime DLLs must be beside the exe; speaker IDs are per-file; CPU-only (GPU deferred); MIT segmentation model.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml README.md
git commit -m "ci+docs: diarize build/test job + README diarization section"
```

---

## Self-review checklist (plan author)

- **Spec coverage:** §1 goals 1 (Diarize→Turns, Task 3) & 2 (Label, Task 4) & 3 (NumSpeakers/Threshold, Task 3) & 4 (native MinGW build, Tasks 2-3) & 5 (provisioning, Task 1); §2 API discipline (Task 3 mirrors verified symbols); §3 decoupling (merge uses local Segment, Task 4); §4 API (Tasks 3-4); §5 errors (Task 3); §6 build/models/DLLs (Task 1); §7 testing (Tasks 4-5); §8 layout (all); Phase-0 spike (Task 2). GPU/streaming/token-level/cross-file explicitly non-goals.
- **Placeholders:** none — full code in every code step. The two `curl`s (model + test wav) are intentional build/test fetches (annotated). README task is a concrete content list (doc task). SHA pinning in download script is marked "fill once known" (acceptable — download is checksummed once values are captured on first run; not a logic gap).
- **Type consistency:** `Diarizer`/`Turn`/`Segment`/`LabeledSegment`, `New`/`Diarize`/`Close`/`SampleRate`/`Label`, `options`/`With*`, `ErrConfig`/`ErrEmptyAudio`/`ErrClosed`, `secondsToDuration`/`overlap` used identically across Tasks 3-7. sherpa symbol names match the verified API (Segmentation.Pyannote.Model, Clustering.NumClusters/Threshold, Process, ReadWave, DeleteOfflineSpeakerDiarization).

## Follow-on (not in this plan)

1. **Token-level speaker labels** — use whisper token timestamps (`WithTokenTimestamps`) for word-accurate speaker attribution.
2. **GPU diarization** — swap the prebuilt CPU onnxruntime for a CUDA EP build.
3. **Nested `diarize/go.mod`** — fully isolate the sherpa-onnx-go dep from the core whisper module if its weight becomes a problem.
4. **Model checksums** — pin SHA256s in `download-diarize-models.sh` after first download.
```
