# go-whisper.cpp — Speaker Diarization (Design Spec)

- **Status:** Draft for review
- **Date:** 2026-06-01
- **Module:** `github.com/dyammarcano/go-whisper.cpp` (new `diarize/` subpackage, **same module**)
- **Go:** 1.25+ · **License:** BSD-3-Clause (binding) — model/runtime licenses noted below
- **Upstream:** `github.com/k2-fsa/sherpa-onnx-go` v1.13.2 (ONNX Runtime + pyannote models)
- **Builds on:** the merged whisper foundation + GPU backends.

---

## 1. Purpose & Goals

Add **speaker diarization** ("who spoke when") to go-whisper.cpp, natively (cgo, no Python), and a helper to fuse it with whisper transcription into **speaker-labeled transcripts**. "Incorporate pyannote-audio" is realized via pyannote's **models** (segmentation-3.0) run through `sherpa-onnx` + ONNX Runtime — not the Python `pyannote.audio` library — preserving the single-binary, native ethos.

**Goals (v1):**
1. **Standalone diarization API:** `[]float32` (16 kHz mono) → ordered speaker turns `{Start, End, Speaker}`.
2. **Merge helper:** combine whisper segments + diarization turns → per-segment speaker labels (overlap-max assignment). Pure Go, no cgo.
3. Known-speaker-count (`NumSpeakers`) **and** unknown-count (`Threshold`) clustering.
4. Native build on the dev box (MinGW gcc + `CGO_ENABLED=1 go build`) — sherpa-onnx-go ships **MinGW-built** prebuilt DLLs; no MSVC, no Python, no C++ source build.
5. Model + runtime-DLL provisioning via Task targets (download + copy), SHA-verified.

**Non-goals (v1):**
- No Python / PyTorch / HuggingFace-token path (that was the rejected alternative).
- No GPU diarization — the prebuilt onnxruntime is **CPU-only**; GPU (CUDA EP) is deferred.
- No streaming/online diarization (offline `Process` over a full buffer only).
- No cross-file speaker identity (speaker IDs are per-file, arbitrary; documented).
- No word/token-level speaker labels in v1 (segment-level merge only; token-level is a follow-on using whisper token timestamps).

---

## 2. Key upstream facts (verified)

- **Go API** (`github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx`, alias `sherpa`; platform module `sherpa-onnx-go-windows`/`-linux`/`-macos` auto-selected by build tags):
  - `OfflineSpeakerDiarizationConfig{ Segmentation OfflineSpeakerSegmentationModelConfig; Embedding SpeakerEmbeddingExtractorConfig; Clustering FastClusteringConfig; MinDurationOn float32; MinDurationOff float32 }`
  - `Segmentation.Pyannote.Model string`, `Embedding.Model string`, `*.NumThreads int`, `*.Provider string` ("cpu"), `*.Debug int`
  - `FastClusteringConfig{ NumClusters int; Threshold float32 }` — set **exactly one**: `NumClusters>0` (known) **xor** `Threshold` (unknown; default 0.5; **larger ⇒ fewer speakers**).
  - `NewOfflineSpeakerDiarization(*Config) *OfflineSpeakerDiarization`, `DeleteOfflineSpeakerDiarization(sd)` (frees C memory — must defer), `(sd).SampleRate() int` (= 16000), `(sd).Process([]float32) []OfflineSpeakerDiarizationSegment`.
  - `OfflineSpeakerDiarizationSegment{ Start float32; End float32; Speaker int }` — **Start/End in seconds**, `Speaker` 0-based.
- **Input contract** matches whisper.cpp exactly: 16 kHz, mono, float32 in [-1,1]. The **same `[]float32`** feeds both transcription and diarization (no resample/convert).
- **Windows/MinGW linkage:** the platform module embeds `lib/x86_64-pc-windows-gnu/` with `-cgo LDFLAGS: -lsherpa-onnx-c-api -lonnxruntime` baked in. MinGW ld links the DLLs directly (no import libs, no MSVC). `go build ./...` with `CGO_ENABLED=1` is sufficient once the dep is required.
- **Runtime DLLs** (must be beside the exe / on PATH): `onnxruntime.dll` (~15 MB CPU), `sherpa-onnx-c-api.dll` (~4 MB), `sherpa-onnx-cxx-api.dll` (~0.2 MB), from `$GOMODCACHE/github.com/k2-fsa/sherpa-onnx-go-windows@v1.13.2/lib/x86_64-pc-windows-gnu/`.
- **Models** (download once; ~32 MB; not embedded): segmentation `sherpa-onnx-pyannote-segmentation-3-0` (**MIT** — commercial-safe) and embedding `wespeaker_en_voxceleb_resnet34_LM.onnx`. **Avoid** `sherpa-onnx-reverb-diarization-v1` (non-commercial). The zh-CN 3D-Speaker model in the docs example is wrong for English audio.
- **Residual risk (spike):** confirm MinGW gcc 15.2 / Go 1.26 link & run the v1.13.2 prebuilt DLLs without a libstdc++/libgcc/winpthread CRT mismatch — a ~15-min link+run smoke test, before building the package.

---

## 3. Architecture

```
caller ─► diarize (Go pkg) ─► sherpa-onnx-go ─► libsherpa-onnx-c-api + onnxruntime (prebuilt DLLs, cgo)
            │  Diarizer / Turn / Options                    (pyannote seg-3.0 + WeSpeaker embed + clustering)
            └─► Label(segs, turns) ─► []LabeledSegment   (pure Go, no cgo, no whisper.cpp dep)

combo example: whisper.Transcribe ──┐
                                    ├─► map whisper.Segment → diarize.Segment ─► diarize.Label ─► labeled transcript
            diarize.Diarize ────────┘
```

**Decoupling rule:** `diarize` depends **only** on `sherpa-onnx-go` (ONNX) — **not** on the root `whisper` package (whisper.cpp). The merge helper uses a local plain `diarize.Segment{Start,End time.Duration; Text string}`, so importing `diarize` never drags in whisper.cpp, and importing `whisper` never drags in ONNX. The combo example is the only place both are imported; it does the 2-line `whisper.Segment → diarize.Segment` map.

**Same-module decision (v1):** `diarize/` lives in the `go-whisper.cpp` module. This adds `sherpa-onnx-go` to `go.mod`, but Go only compiles/links it when a package importing `diarize` is built — transcription-only users never link ONNX. (Trade-off: `go mod download all` / `go build ./...` in CI pulls the ~20 MB vendored DLLs. A nested `diarize/go.mod` to fully isolate the dep is recorded as a follow-on if the weight becomes a problem.)

---

## 4. Public Go API (`diarize` package)

```go
package diarize

import "time"

// Diarizer wraps a sherpa-onnx OfflineSpeakerDiarization. Not safe for concurrent
// Diarize calls on one instance; create one per goroutine if needed. Close when done.
type Diarizer struct { /* sd *sherpa.OfflineSpeakerDiarization; closed bool */ }

// New builds a diarizer from model paths + options. Returns ErrConfig if sherpa
// rejects the config (e.g. missing/invalid model files).
func New(segmentationModel, embeddingModel string, opts ...Option) (*Diarizer, error)

func (d *Diarizer) Close() error
func (d *Diarizer) SampleRate() int   // 16000

// Diarize runs offline diarization over 16 kHz mono float32 samples in [-1,1].
// Returns speaker turns ordered by Start. ErrEmptyAudio if len(samples)==0.
func (d *Diarizer) Diarize(samples []float32) ([]Turn, error)

type Turn struct {
    Start, End time.Duration
    Speaker    int // 0-based, per-file (NOT stable across files)
}
```

**Options** (`options.go`, mirroring the whisper package's functional-option idiom):
`WithNumSpeakers(n int)` (known count) · `WithThreshold(t float32)` (unknown count; mutually exclusive with NumSpeakers — last-set wins, documented) · `WithMinDuration(on, off time.Duration)` · `WithThreads(n int)` · `WithDebug` (var toggle).

**Merge helper** (`merge.go`, pure Go):
```go
// Segment is a transcript span to be labeled (structurally identical to whisper.Segment's
// timing/text — kept local so diarize does not import the whisper.cpp package).
type Segment struct {
    Start, End time.Duration
    Text       string
}
type LabeledSegment struct {
    Segment
    Speaker int // -1 if no diarization turn overlaps
}

// Label assigns each segment the speaker whose turn has the greatest temporal overlap
// (-1 if none). Turns need not be sorted; Label is O(len(segs)*len(turns)) — fine for
// transcript sizes. Pure Go.
func Label(segs []Segment, turns []Turn) []LabeledSegment
```

**Usage (combo):**
```go
samples, _ := wav.ReadFile("audio.wav")            // 16 kHz mono — feeds both
res, _ := model.Transcribe(ctx, samples)           // whisper
d, _ := diarize.New(segModel, embModel, diarize.WithNumSpeakers(2)); defer d.Close()
turns, _ := d.Diarize(samples)                     // sherpa-onnx

segs := make([]diarize.Segment, len(res.Segments))
for i, s := range res.Segments { segs[i] = diarize.Segment{Start: s.Start, End: s.End, Text: s.Text} }
for _, ls := range diarize.Label(segs, turns) {
    fmt.Printf("[Speaker %d] %s\n", ls.Speaker, ls.Text)
}
```

---

## 5. Errors

Sentinels (`errors.go`): `ErrConfig` (nil from `NewOfflineSpeakerDiarization` — bad/missing models), `ErrEmptyAudio`, `ErrClosed`. All wrapped with `%w`.

---

## 6. Build, models & runtime DLLs

- **go.mod:** add `github.com/k2-fsa/sherpa-onnx-go v1.13.2`; `go mod tidy` resolves the platform module (indirect, build-tag-gated) and downloads the vendored DLLs into the module cache.
- **Build:** `CGO_ENABLED=1 go build -tags '' ./diarize/...` — MinGW gcc links the prebuilt C-API + onnxruntime DLLs (no extra flags/tags).
- **Models** — `scripts/download-diarize-models.sh` (SHA256-verified) into `models/`:
  - seg: `sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` → `models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx`
  - embed: `wespeaker_en_voxceleb_resnet34_LM.onnx`
  - Task target `models:diarize`.
- **Runtime DLLs** — `scripts/copy-sherpa-dlls.sh` copies the 3 DLLs from `$(go env GOMODCACHE)/github.com/k2-fsa/sherpa-onnx-go-windows@<ver>/lib/x86_64-pc-windows-gnu/` to a target dir (beside the exe / a `dist/` dir / the test working dir). Task target `diarize:dlls`. Tests prepend that dir to PATH.

---

## 7. Testing

- **Pure-Go, always run:** `merge_test.go` — table-driven `Label` cases (overlap assignment, no-overlap → -1, boundary/seconds↔Duration, multi-speaker). Plus option-applier tests. No cgo, no models, no DLLs.
- **Integration (cgo + models + DLLs), gated:** `diarize_test.go` reads `DIARIZE_SEG_MODEL` / `DIARIZE_EMB_MODEL` env (or skips); `task models:diarize` + `task diarize:dlls` provision them. Diarize a known multi-speaker WAV (sherpa's `0-four-speakers-zh.wav` or an English fixture) and assert the distinct `Speaker` count matches expectation (with `WithNumSpeakers` and with `WithThreshold`). DLLs on PATH.
- **Combo (gated):** transcribe + diarize + Label on a 2-speaker clip; assert each labeled segment has a non-negative speaker and text.
- ginkgo/gomega; `go test -race` on the pure-Go path; `golangci-lint run` clean.

---

## 8. Project layout

```
go-whisper.cpp/
├── diarize/
│   ├── diarize.go      Diarizer, New, Diarize, Close, SampleRate, Turn (cgo via sherpa-onnx-go)
│   ├── options.go      Option + functional setters + defaults
│   ├── merge.go        Segment, LabeledSegment, Label (pure Go)
│   ├── errors.go       sentinels
│   ├── diarize_test.go integration (env-gated)
│   └── merge_test.go   pure-Go
├── scripts/download-diarize-models.sh   SHA-verified model fetch
├── scripts/copy-sherpa-dlls.sh          copy runtime DLLs from module cache
├── examples/diarize/main.go             standalone diarization CLI
├── examples/transcribe-diarize/main.go  whisper + diarize → labeled transcript
├── Taskfile.yml        +models:diarize, +diarize:dlls, +build:diarize
├── go.mod              +github.com/k2-fsa/sherpa-onnx-go
├── .github/workflows/ci.yml   +diarize build + pure-Go merge test (ubuntu/windows)
└── README.md           Diarization section
```

---

## 9. Milestones

- **Phase 0 — Spike:** add the dep; smoke-test that MinGW gcc 15.2 / Go 1.26 link+run a trivial `OfflineSpeakerDiarization` (CRT compatibility) with a downloaded model + the 3 DLLs. Gate the rest on this passing.
- **Phase 1 — Core `diarize`:** `diarize.go`/`options.go`/`errors.go` + `Diarize` working on a known WAV; integration test (gated).
- **Phase 2 — Merge:** `merge.go` + `merge_test.go` (pure Go, TDD).
- **Phase 3 — Provisioning:** `download-diarize-models.sh`, `copy-sherpa-dlls.sh`, Task targets.
- **Phase 4 — Examples + CI + docs:** both examples; CI diarize-build + merge test; README.

---

## 10. Risks

- **R1 (HIGH→spike): CRT/ABI** — prebuilt DLLs built with k2-fsa's (possibly older) MinGW vs gcc 15.2. Mitigation: Phase-0 link+run smoke test; if it mismatches, pin an older sherpa-onnx-go that matches, or (last resort) build sherpa-onnx C libs from source.
- **R2 (MED): model assets** — ~32 MB downloads, not embedded; first-run/CI must provision. Mitigation: SHA-verified Task target; integration tests skip if absent.
- **R3 (MED): diarization quality** — wrong embedding model (zh-CN) or mis-tuned `Threshold` degrades results. Mitigation: English WeSpeaker default; document `NumSpeakers` vs `Threshold` tuning.
- **R4 (LOW): dependency weight** — sherpa-onnx-go DLLs in the module. Mitigation: lazy linkage; nested-module split recorded as follow-on.
- **R5 (LOW): unit mismatch** — diar seconds vs whisper centiseconds. Mitigation: both normalized to `time.Duration` at the boundary; `Label` operates on Durations.

---

## 11. Acceptance criteria (v1)

1. `CGO_ENABLED=1 go build ./...` builds the `diarize` package on the dev box (MinGW); the Phase-0 spike runs a real diarization.
2. `Diarizer.Diarize` returns correct-count speaker turns on a known multi-speaker WAV (both `WithNumSpeakers` and `WithThreshold`).
3. `Label` correctly assigns speakers by max overlap (pure-Go tests, including no-overlap → -1), `-race` clean.
4. The combo example prints a speaker-labeled transcript from one 16 kHz buffer (whisper + diarize).
5. `diarize` imports neither the whisper package nor Python; `whisper`-only builds don't link ONNX.
6. Model + DLL Task targets provision everything; integration tests pass with them and skip cleanly without.
7. `golangci-lint run` clean; MIT segmentation model used (commercial-safe).
```
