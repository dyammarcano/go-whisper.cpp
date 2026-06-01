# go-whisper.cpp — Go binding for whisper.cpp (Design Spec)

- **Status:** Draft for review
- **Date:** 2026-06-01
- **Module:** `github.com/dyammarcano/go-whisper.cpp`
- **Go:** 1.25 · **License:** BSD-3-Clause
- **Upstream:** `https://github.com/ggml-org/whisper.cpp` (git submodule)
- **Reference pattern:** `go-skynet/go-llama.cpp` (sibling binding; mirror its conventions)

---

## 1. Purpose & Goals

A high-level, idiomatic Go binding for whisper.cpp that mirrors the proven structure of
go-llama.cpp (thin C shim + cgo + build-tag link files + Taskfile/scripts + a pure-Go helper
subpackage), adapted for **speech-to-text** rather than text generation.

**Goals (v1):**

1. Load a ggml whisper model and transcribe 16 kHz mono float32 PCM to timestamped text segments.
2. Translate-to-English mode, automatic language detection, and per-segment **and** per-token timestamps.
3. Streaming (sliding-window) transcription over a sample channel — *the user owns audio capture*.
4. Progress / new-segment callbacks and **mid-inference cancellation** via `context.Context`.
5. **Cross-platform from v1:** Windows (CPU/CUDA/Vulkan), Linux (CPU/CUDA/Vulkan), macOS (CPU/Metal) —
   **every platform×backend built in CI**. CPU integration tests run on all three OS runners; GPU paths
   (CUDA/Vulkan/Metal) are **build- and link-verified** in CI with execution tests on self-hosted/local
   GPU runners (GitHub-hosted runners have no GPU).
6. A pure-Go (`no cgo`) `wav/` subpackage: WAV → 16 kHz mono `[]float32` (analog of go-llama's `gguf/`).

**Non-goals (explicit — do NOT "mirror" these from go-llama):**

- No grammar/GBNF, no `logit_bias`, no sampler chains, no top-k/top-p/repeat-penalty knobs — these are
  LLM-generation concepts with **no analog in whisper STT**.
- No audio resampling in v1 (`wav/` rejects non-16 kHz with an actionable error). Deferred to v1.1.
- No microphone/PortAudio dependency — `stream/` consumes a `<-chan []float32`; capture is the caller's job.
- No model quantization/conversion tooling.

---

## 2. Key upstream facts (verified against `include/whisper.h`, master @ 2026-06)

These drive the binding and are easy to get wrong:

- `whisper_full(ctx, params, samples, n)` takes `whisper_full_params` **by value**; `whisper_init_*_with_params`
  takes `whisper_context_params` **by value**. Both contain anonymous nested structs (`greedy{best_of}`,
  `beam_search{beam_size,patience}`), `const char*` fields, and **five** function-pointer + `void* *_user_data`
  callback pairs. → **Never construct these structs from Go cgo.** The C shim owns their construction.
- **Always** call `whisper_full_default_params(strategy)` / `whisper_context_default_params()` and then mutate
  fields. `whisper_full_params` is high version-volatility (recently added: `carry_initial_prompt`, `suppress_nst`,
  `suppress_regex`, `tdrz_enable`, the trailing `vad`/`vad_model_path`/`vad_params` block; removed: `speed_up`;
  renamed: `suppress_non_speech_tokens` → `suppress_nst`). Zero-init + partial-set risks silent memory corruption
  if header and linked lib drift.
- Bind only the `*_with_params` initializers — the bare `whisper_init_from_file/buffer/init` are `WHISPER_DEPRECATED`.
- `whisper_full_get_segment_text` / `whisper_full_get_token_text` return `const char*` **owned by the context and
  invalidated by the next `whisper_full`**. Copy immediately (`C.GoString` / `strdup` in the shim); never free, never retain.
- **Timestamp units:** `whisper_full_get_segment_t0/t1` and `whisper_token_data.t0/t1` are `int64_t` **centiseconds (10 ms)**.
  The VAD API (`whisper_vad_segments_get_segment_t0/t1`) returns `float` **seconds**. Do not mix scales.
- **Cancellation:** the only in-flight cancel mechanism is `abort_callback` (type `ggml_abort_callback`,
  signature `bool(*)(void* user_data)` — **only** `user_data`, returns `true` to abort). There is **no**
  `whisper_abort_callback` typedef.
- **Concurrency:** `whisper_context` is the heavyweight, read-only model (loaded once). `whisper_state`
  (`whisper_init_state` + `whisper_full_with_state`) is the per-inference scratch. One context → N states →
  real concurrency without reloading weights. This is whisper's native scaling primitive and shapes our type split (§5).
- **VAD** is a full standalone API (`whisper_vad_*`) usable for streaming silence-gating, but requires a
  **separate** Silero VAD ggml model file. Integrated VAD (`whisper_full_params.vad=true`) needs `vad_model_path`.
  → VAD-gated streaming is **v1.1** (asset dependency); v1 streaming uses fixed-step sliding windows.

---

## 3. Architecture overview

```
caller ──► whisper (Go pkg) ──► binding.cpp/.h (thin C shim) ──► libwhisper + ggml (cgo-linked)
              │  Model / Session / Result / Options
              ├──► wav/   (pure Go, no cgo)   — decode WAV → []float32
              └──► stream/ (Go, uses a Session) — sliding-window over <-chan []float32
```

**Why a C shim** (precise scope — only two load-bearing reasons + one optimization):

1. **Own `whisper_full_params` / `whisper_context_params` construction.** The by-value structs with nested
   anonymous sub-structs and char* never cross cgo. Go passes a flat `whisper_bind_params` (primitive fields only);
   the shim does `whisper_full_default_params(...)` then mutates. This is the ABI-stability firewall.
2. **Host the callback trampolines.** A file with `//export` may only *declare* C funcs; the trampoline *bodies*
   live in `binding.cpp` and call back into exported Go functions, with the `cgo.Handle` passed through `user_data`.
3. *(optimization)* **Batch-marshal results.** One shim call returns a flat array of all segments (+ optional tokens)
   with strings `strdup`'d into the result, so Go crosses cgo **once** instead of per-segment/`per-token`. Freed by one shim call.

The shim uses **only `extern "C"` whisper_* symbols**, so (as in go-llama) the MinGW cgo host can link a
MSVC-built `whisper.dll` for the CUDA path. (Risk + spike: §11.)

---

## 4. C shim ABI (`binding.h`)

All structs below use **primitive fields only** so Go can build them safely. `int` is used for booleans (0/1).
Handles are `uintptr_t` (a `runtime/cgo.Handle`), passed to whisper as `(void*)(uintptr_t)handle`.

```c
#ifndef GO_WHISPER_BINDING_H
#define GO_WHISPER_BINDING_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

// ---- model / state lifecycle ----
void* whisper_bind_load_model(const char* path, int use_gpu, int flash_attn, int gpu_device); // -> whisper_context* (NULL on fail)
void  whisper_bind_free_model(void* ctx);
void* whisper_bind_new_state(void* ctx);        // -> whisper_state*  (NULL on fail)
void  whisper_bind_free_state(void* state);

// ---- transcription params (flat; shim assembles whisper_full_params) ----
typedef struct {
    int          strategy;          // 0 = greedy, 1 = beam_search
    int          n_threads;
    int          translate;         // bool
    const char*  language;          // "auto"/""/NULL = autodetect; else e.g. "en"
    int          detect_language;   // bool
    int          beam_size;         // <=0 -> default
    int          best_of;           // <=0 -> default
    float        temperature;
    float        temperature_inc;
    float        entropy_thold;
    float        logprob_thold;
    float        no_speech_thold;
    int          no_context;        // bool
    int          single_segment;    // bool
    int          token_timestamps;  // bool
    int          max_len;           // chars (0 = no limit)
    int          split_on_word;     // bool
    int          max_tokens;        // per segment (0 = no limit)
    int          offset_ms;
    int          duration_ms;
    int          audio_ctx;         // 0 = default
    int          suppress_blank;    // bool
    int          suppress_nst;      // bool (non-speech tokens)
    const char*  initial_prompt;    // NULL = none
    int          print_to_stderr;   // 0 = silence whisper's internal prints

    uintptr_t    segment_cb;        // cgo.Handle (0 = none)
    uintptr_t    progress_cb;       // cgo.Handle (0 = none)
    uintptr_t    abort_cb;          // cgo.Handle (0 = none)
} whisper_bind_params;

// Run transcription. If state != NULL uses whisper_full_with_state, else whisper_full.
// samples: 16 kHz mono f32. Returns 0 on success, whisper rc otherwise, -100 if aborted.
int whisper_bind_full(void* ctx, void* state, const whisper_bind_params* p,
                      const float* samples, int n_samples);

// ---- batched result marshalling (one crossing) ----
typedef struct { int64_t t0, t1; float p; const char* text; } whisper_bind_token;
typedef struct {
    int64_t t0, t1;                 // centiseconds
    const char* text;               // strdup'd; owned by result
    int n_tokens;
    whisper_bind_token* tokens;     // NULL unless want_tokens
} whisper_bind_segment;
typedef struct {
    int n_segments;
    whisper_bind_segment* segments;
    int lang_id;                    // detected/used language id (-1 if n/a)
} whisper_bind_result;

whisper_bind_result* whisper_bind_get_result(void* ctx, void* state, int want_tokens); // ctx/state mirror the full call
void whisper_bind_free_result(whisper_bind_result* r);

// ---- language helpers ----
int         whisper_bind_lang_auto_detect(void* ctx, void* state, int offset_ms, int n_threads); // -> lang id or <0
int         whisper_bind_lang_id(const char* lang);   // name -> id (-1 unknown)
const char* whisper_bind_lang_str(int id);            // id -> name
int         whisper_bind_lang_max_id(void);

// ---- logging: route whisper/ggml logs into an exported Go sink ----
void whisper_bind_install_log(void);

#ifdef __cplusplus
}
#endif
#endif
```

**Trampolines (in `binding.cpp`, declared `extern` in the `//export` Go file):**

```c
// each calls back into an //export'd Go function; user_data carries the cgo.Handle
static void whisper_bind_seg_tramp (struct whisper_context*, struct whisper_state*, int n_new, void* ud); // -> goWhisperSegment
static void whisper_bind_prog_tramp(struct whisper_context*, struct whisper_state*, int prog,  void* ud); // -> goWhisperProgress
static bool whisper_bind_abort_tramp(void* ud);                                                           // -> goWhisperAbort
```

The shim sets each callback on `whisper_full_params` **only** when the matching handle in `whisper_bind_params`
is non-zero, and sets `*_user_data = (void*)(uintptr_t)handle`.

---

## 5. Public Go API

Two types, reflecting whisper's context/state split — the **central design decision** (§10):

```go
package whisper

// Model wraps a loaded whisper_context. Immutable after load; safe to SHARE across goroutines
// by creating one Session per goroutine. Cheap to share (holds the model weights once).
type Model struct { /* ptr unsafe.Pointer; closed atomic.Bool; ... */ }

// New loads a model. Mirrors go-llama's New(...) entry point.
func New(modelPath string, opts ...ModelOption) (*Model, error)
func (m *Model) Close() error
func (m *Model) Languages() []string            // from whisper_bind_lang_str over 0..max_id
func (m *Model) IsMultilingual() bool

// Session owns a whisper_state (per-inference scratch). NOT safe for concurrent use;
// one Session per goroutine. Created from a Model.
type Session struct { /* model *Model; state unsafe.Pointer; ... */ }
func (m *Model) NewSession() (*Session, error)
func (s *Session) Close() error

// Transcribe runs whisper_full_with_state on the session. ctx cancels mid-inference (abort_callback).
func (s *Session) Transcribe(ctx context.Context, samples []float32, opts ...TranscribeOption) (*Result, error)

// DetectLanguage returns the most probable language id+name for the given samples.
func (s *Session) DetectLanguage(ctx context.Context, samples []float32, opts ...TranscribeOption) (string, error)

// Convenience single-flight path on the Model's internal context state (mutex-guarded).
// For one-shot/CLI use; concurrent callers should use Sessions.
func (m *Model) Transcribe(ctx context.Context, samples []float32, opts ...TranscribeOption) (*Result, error)
```

```go
// result.go — pure Go
type Result struct {
    Segments []Segment
    Language string        // detected/used language (e.g. "en")
}
type Segment struct {
    Start, End time.Duration   // converted from centiseconds (×10ms)
    Text       string
    Tokens     []Token         // populated only if WithTokenTimestamps()
}
type Token struct {
    Text       string
    P          float32         // probability
    Start, End time.Duration
}
```

**Example:**

```go
m, err := whisper.New("ggml-base.en.bin", whisper.WithGPU(true))
defer m.Close()

samples, _ := wav.ReadFile("audio.wav")           // []float32, 16 kHz mono
res, err := m.Transcribe(context.Background(), samples,
    whisper.WithLanguage("auto"),
    whisper.WithBeamSearch(5),
    whisper.WithTokenTimestamps(),
    whisper.WithSegmentCallback(func(seg whisper.Segment) {
        fmt.Printf("[%s → %s] %s\n", seg.Start, seg.End, seg.Text)
    }))
```

---

## 6. Options (`options.go`) — mirror go-llama's idiom exactly

Two option types; closure-returning setters for values, package-level **vars** for bare boolean toggles
(used without parens, as go-llama does):

```go
type ModelOption      func(*ModelOptions)
type TranscribeOption func(*TranscribeOptions)

// value setter
func WithGPUDevice(d int) ModelOption { return func(o *ModelOptions) { o.GPUDevice = d } }
// bare toggle (var, no parens at call site)
var WithFlashAttn ModelOption = func(o *ModelOptions) { o.FlashAttn = true }

func NewModelOptions(opts ...ModelOption) ModelOptions { p := DefaultModelOptions; for _, o := range opts { o(&p) }; return p }
```

**ModelOption surface (→ `whisper_context_params`):** `WithGPU(bool)` / `WithGPUDevice(int)` / `WithFlashAttn` /
`WithoutGPU`. (DTW token-timestamp context params deferred — `dtw_mem_size` is upstream-marked "TODO: remove".)

**TranscribeOption surface (→ `whisper_full_params`):**
`WithLanguage(string)`, `WithTranslate`, `WithDetectLanguage`, `WithThreads(int)`,
`WithGreedy(bestOf int)`, `WithBeamSearch(beamSize int)`,
`WithTemperature(float32)`, `WithTemperatureInc(float32)`, `WithEntropyThold(float32)`,
`WithLogProbThold(float32)`, `WithNoSpeechThold(float32)`,
`WithTokenTimestamps`, `WithMaxSegmentLen(int)`, `WithSplitOnWord`, `WithMaxTokens(int)`,
`WithOffset(time.Duration)`, `WithDuration(time.Duration)`, `WithAudioCtx(int)`,
`WithNoContext`, `WithSingleSegment`, `WithSuppressBlank`, `WithSuppressNonSpeech`,
`WithInitialPrompt(string)`,
`WithSegmentCallback(func(Segment))`, `WithProgressCallback(func(percent int))`.

---

## 7. Concurrency, cancellation & safety

- **Model = shared, Session = per-goroutine.** `Model.Transcribe` serialises on a `sync.Mutex` over the context's
  internal state (single-flight convenience). For parallelism, callers create N `Session`s from one `Model` and run
  `Session.Transcribe` concurrently (each uses its own `whisper_state` via `whisper_full_with_state`). One model load, N concurrent inferences.
- **Cancellation:** every `Transcribe` takes `context.Context`. We install `abort_callback`; the trampoline checks a
  flag set when `ctx` is `Done()` (a watcher goroutine flips an `atomic.Bool` the abort trampoline reads via its
  `cgo.Handle`). On abort, `whisper_bind_full` returns `-100` → `Transcribe` returns `ctx.Err()` wrapped.
- **Panic safety:** every `//export` trampoline `defer`s a `recover()` that records a sticky error on the bridge and
  returns abort=`true` (for `abort_callback`) — a Go panic must never unwind into C. Acceptance criterion, not a follow-up.
- **Handle lifetime:** `cgo.Handle` is created in `Transcribe`, `defer handle.Delete()` immediately after creation so
  it cannot leak on early-return error paths. Callbacks fire synchronously inside `whisper_full`, so the deferred
  delete after the call returns is correct.
- **Empty input guard:** reject `len(samples)==0` before the cgo call (`&samples[0]` would panic).

---

## 8. `wav/` subpackage (pure Go, no cgo) — analog of go-llama `gguf/`

`func ReadWAV(r io.Reader) ([]float32, error)` and `func ReadFile(path string) ([]float32, error)` returning
**16 kHz mono float32 in [-1,1]**. Stdlib only.

- Parse RIFF/WAVE; **loop over chunks** (`fmt `, `data`, skipping `LIST`/`fact`/`JUNK`/`bext`…), honoring the
  word-alignment **pad byte** on odd-sized chunks. Never assume `data` immediately follows `fmt `.
- Convert: 16-bit signed `/32768.0`; 8-bit **unsigned** `(b-128)/128`; 24-bit sign-extended `/8388608`;
  32-bit int `/2147483648`; 32-bit IEEE float pass-through. Handle `WAVE_FORMAT_EXTENSIBLE` (0xFFFE).
- **Downmix stereo→mono** by averaging channels.
- Reject `data` not a multiple of `BlockAlign` (truncated/malformed).
- **Sample rate:** if `!= 16000`, return `fmt.Errorf("%w: got %d Hz", ErrNot16kHz, sr)` — typed, `errors.Is`-checkable,
  naming the required rate and pointing at `ffmpeg -i in -ar 16000 -ac 1 out.wav`. (Silent mis-rate → garbage transcripts,
  which is worse than an explicit error.) Resampling is v1.1.

Carries an attribution header if any logic is derived from another decoder; otherwise original, mirroring gguf's
package-doc + tri-function convenience shape. Fully unit-tested with synthetic in-code WAV fixtures (no cgo, no model).

---

## 9. `stream/` subpackage — sliding-window streaming

Built on a dedicated `Session` owned by a single worker goroutine (honest concurrency, not lock-bound). Algorithm
ported from upstream `examples/stream/stream.cpp`:

```go
type Options struct {
    StepMs, LengthMs, KeepMs int   // defaults 3000 / 10000 / 200
    NoContext bool                 // default true (each window fresh — matches stream.cpp default)
    Transcribe []whisper.TranscribeOption
}
type Event struct { Segment whisper.Segment; Final bool }   // Final at the n_new_line boundary
func New(s *whisper.Session, opts Options) *Stream
func (st *Stream) Run(ctx context.Context, in <-chan []float32) (<-chan Event, <-chan error)
```

Window math (SR=16000 → samples_per_ms=16):
`nStep=StepMs*16`, `nLen=LengthMs*16`, `nKeep=KeepMs*16`, `nNewLine=max(1, LengthMs/StepMs - 1)`.
Per step: accumulate `nStep` new samples; `take = min(len(old), max(0, nKeep+nLen-nNew))`;
`window = old[len-take:] ++ new`; run `Transcribe(window, WithNoContext...)`; emit segments as non-Final;
set `old = window`; every `nNewLine` steps, emit boundary segments as `Final` and trim `old = window[len-nKeep:]`.

**Dedup policy** (documented): overlapping windows re-emit text; consumers treat non-`Final` events as the live
hypothesis for the current window and `Final` events as committed. v1.1 adds VAD-gated stepping (`StepMs<=0`,
requires a Silero VAD model). Mic capture stays the caller's responsibility — `Run` consumes a `<-chan []float32`.

---

## 10. Errors & logging

- **Sentinels** (`errors.go`): `ErrModelLoad`, `ErrStateInit`, `ErrTranscribe`, `ErrCanceled`, `ErrEmptyAudio`,
  and `wav.ErrNot16kHz` / `wav.ErrBadHeader` / `wav.ErrUnsupportedFmt`. Every cgo status code becomes a `%w`-wrapped error.
- **Logging:** `whisper_bind_install_log()` routes whisper/ggml's log callback into an exported Go sink that forwards
  to `log/slog` (quiet by default; `WithVerbose`/env to raise). `print_to_stderr=0` silences whisper's internal prints.

---

## 11. Build system

**Layout & link strategy mirror go-llama** (CPU/Vulkan = MinGW **static** `.a` via `--start-group`; CUDA = MSVC
**shared** `whisper.dll` linked by path). Build tags are mutually exclusive.

**`whisper.go` cgo preamble (platform-independent flags):**
```go
// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR}/whisper.cpp/include -I${SRCDIR}/whisper.cpp/ggml/include
// #cgo CFLAGS:   -I${SRCDIR}/whisper.cpp/include -I${SRCDIR}/whisper.cpp/ggml/include
// #cgo LDFLAGS:  -L${SRCDIR}/ -lbinding
// #include "binding.h"
import "C"
```

**Windows link files (verified `.a` paths — note `build/src` vs `build/ggml/src`):**
```go
// link_static_windows.go   //go:build windows && !cuda && !vulkan
#cgo LDFLAGS: -L${SRCDIR}/whisper.cpp/build-cpu/src -L${SRCDIR}/whisper.cpp/build-cpu/ggml/src
#cgo LDFLAGS: -Wl,--start-group -lwhisper -lggml -lggml-cpu -lggml-base -Wl,--end-group -fopenmp -lstdc++ -lm

// link_vulkan_windows.go   //go:build windows && vulkan && !cuda
#cgo LDFLAGS: -L${SRCDIR}/whisper.cpp/build-vulkan/src -L${SRCDIR}/whisper.cpp/build-vulkan/ggml/src
#cgo LDFLAGS: -Wl,--start-group -lwhisper -lggml -lggml-cpu -lggml-base -lggml-vulkan -Wl,--end-group -fopenmp -lstdc++ -lm C:/Windows/System32/vulkan-1.dll

// link_cuda_windows.go     //go:build windows && cuda
// NOTE: CUDA builds to repo-root build-cuda/ (cmake -S whisper.cpp -B build-cuda), mirroring go-llama;
// CPU/Vulkan build under whisper.cpp/build-<backend>/. Keep these output dirs distinct.
#cgo LDFLAGS: ${SRCDIR}/build-cuda/bin/whisper.dll -lstdc++
```
**Linux/macOS link files (first-class, CI-built):** `link_linux.go` (CPU `--start-group`; `linux && cuda` and
`linux && vulkan` variants), `link_darwin.go` (Metal: `-framework Metal -framework MetalKit -framework Accelerate
-framework Foundation`, links static `.a`). Enumerated in the build matrix below.

**Build matrix (all built in CI from v1):** `{windows, linux} × {cpu, cuda, vulkan}` + `darwin × {cpu, metal}`.
CI compiles every cell and runs **CPU integration tests** on windows/ubuntu/macos runners; CUDA/Vulkan/Metal cells are
**build- and link-verified** in CI (GitHub-hosted runners lack GPUs) with execution tests on self-hosted/local GPU runners.

**Scripts** (`scripts/`, ported from go-llama; the MinGW `WINVER`/`GGML_NO_THREAD_POWER_THROTTLING` workaround +
`ggml-cpu.c` sed-patch is replicated):
- `whispercpp.sh [cpu|vulkan]` → cmake (MinGW Makefiles / Ninja, `BUILD_SHARED_LIBS=OFF`,
  `WHISPER_BUILD_EXAMPLES/TESTS/SERVER=OFF`, `WHISPER_SDL2=OFF`, `GGML_NATIVE=OFF` for redistributables,
  `GGML_BACKEND_DL=OFF`, `+ -DGGML_VULKAN=ON` for vulkan) → `whisper.cpp/build-<backend>/`.
- `whispercpp-cuda.bat` → vcvars64 + cmake (`BUILD_SHARED_LIBS=ON`, `GGML_CUDA=ON`,
  `CMAKE_CUDA_ARCHITECTURES=...`, `GGML_CUDA_NCCL=OFF`, MSVC `cl`/`nvcc`, `--config Release`) → `build-cuda/bin/whisper.dll`.
- `binding.sh` → `g++ -std=c++17 -O3 $WINVER -I whisper.cpp/include -I whisper.cpp/ggml/include -c binding.cpp` → `ar rcs libbinding.a`.
- `download-model.sh <name>` → fetch a pinned ggml model from HuggingFace **with SHA256 verification** into a cache dir.

> **Build flag gotchas (must honor):** `WHISPER_BUILD_*` default ON when whisper is top-level → must force OFF.
> `BUILD_SHARED_LIBS` auto-OFF under MinGW but pass it explicitly. `GGML_BACKEND_DL` must stay OFF (else backends
> become dlopen plugins with no static `.a`). Static `.a` for `ggml-cuda`/`ggml-vulkan` land flat in `build/ggml/src/`
> (no per-backend subdir). MSVC multi-config emits `Release/*.lib` not `.a` (CUDA-DLL path uses the DLL by path, so this is moot).

**Taskfile.yml:** `default`(list), `deps`(`git submodule update --init --recursive`),
`build:cpu`/`build:vulkan` (`whispercpp.sh` + `binding.sh`), `build:cuda` (`whispercpp-cuda.bat` + `binding.sh`),
`build`(=cpu), `models:tiny`(download), `test`, `fmt`, `fix`, `lint`, `clean`.

---

## 12. Project layout

```
go-whisper.cpp/
├── whisper.cpp/                 submodule (ggml-org/whisper.cpp)
├── binding.cpp / binding.h      thin C shim (params build, trampolines, batched results, log install)
├── whisper.go                   cgo preamble; Model; New/Close; lang helpers
├── session.go                   Session; NewSession; Transcribe; DetectLanguage (whisper_full_with_state)
├── callback.go                  //export goWhisperSegment/Progress/Abort + extern decls (no C defs here)
├── options.go                   ModelOption / TranscribeOption + defaults + appliers
├── result.go                    Result / Segment / Token + marshalling from whisper_bind_result
├── errors.go                    sentinel errors
├── log.go                       whisper_bind_install_log + //export goWhisperLog → slog
├── link_static_windows.go       windows && !cuda && !vulkan
├── link_cuda_windows.go         windows && cuda
├── link_vulkan_windows.go       windows && vulkan && !cuda
├── link_linux.go                linux (cpu; cuda variant via tag)
├── link_darwin.go               darwin (Metal)
├── wav/        wav.go  wav_test.go            (pure Go, no cgo)
├── stream/     stream.go  stream_test.go      (uses a Session)
├── examples/transcribe/main.go
├── examples/stream/main.go
├── scripts/    whispercpp.sh  whispercpp-cuda.bat  binding.sh  download-model.sh
├── whisper_suite_test.go        ginkgo RunSpecs
├── whisper_test.go              specs; model via TEST_MODEL env, Skip if unset; Label("gpu")
├── Taskfile.yml  go.mod  go.sum  README.md  LICENSE(BSD-3)  .gitignore  .golangci.yml
```

`.golangci.yml` mirrors go-llama's (`default: all` minus the same disable set). `.gitignore` covers
`/binding.o`, `/libbinding.a`, `whisper.cpp/build-*`, `build-cuda/`, `*.exe`, `.scripts/`, downloaded models.

---

## 13. Testing

- **Pure-Go, always run:** `wav/` (synthetic in-code WAV fixtures), options appliers, result marshalling helpers,
  stream window-math (table-driven, no model).
- **Integration (cgo + model), gated:** `whisper_test.go` reads `TEST_MODEL`; each model-dependent `It` does
  `if testModelPath == "" { Skip(...) }` (mirrors go-llama). `task models:tiny` populates a pinned `ggml-tiny.en`
  (SHA256-checked) and exports `TEST_MODEL` for hermetic local/CI runs. GPU specs carry `Label("gpu")`.
- **Equivalence:** a known short WAV → assert transcript contains expected words; cancellation test asserts a
  cancelled `ctx` aborts within a bound; concurrency test runs N Sessions over one Model in parallel.
- ginkgo v2 / gomega, `golangci-lint run --fix`, 80%+ on pure-Go packages.

---

## 14. Design decisions & deviations from a strict go-llama mirror

These changes were adopted from an adversarial architecture review; flagged here for sign-off:

1. **Model + Session split** (vs go-llama's single `LLama`). Rationale: whisper's native scaling primitive is one
   read-only `whisper_context` + N `whisper_state`. A single mutex-guarded type would make "concurrency" and
   "streaming" dishonest (lock-bound single-flight). We keep a `Model.Transcribe` convenience for the simple path.
2. **`context.Context` + `abort_callback` cancellation** in v1 (not in go-llama's surface). Multi-minute audio must
   be cancellable; this is whisper's only in-flight cancel mechanism.
3. **Full cross-platform, all built in CI from v1** — **confirmed by product decision** (go-llama is Windows-only in
   practice). Windows/Linux (CPU/CUDA/Vulkan) + macOS (CPU/Metal); CI builds every cell and runs CPU integration tests
   on all three OS runners. GPU **execution** tests require self-hosted/local GPU runners (GitHub-hosted runners have no GPU).
4. **Streaming kept in v1** per product decision, but made honest via a dedicated Session and a concretely-specified
   window/keep/dedup policy from `stream.cpp`. *Risk flagged:* it is the least-settled API; VAD-gated mode is v1.1.
5. **`wav/` errors on non-16 kHz** (rather than silently passing or auto-resampling). Stereo IS downmixed. Resampler is v1.1.

---

## 15. Risks & required spikes

- **R1 (HIGH): MinGW-host ↔ MSVC `whisper.dll` ABI.** go-llama links an MSVC `llama.dll` from MinGW cgo successfully,
  and whisper exposes the same `extern "C"` surface — strong precedent. **Spike S1 (before committing CUDA path):**
  build `whisper.dll` (MSVC), compile `binding.cpp` (MinGW), link & load; confirm no C++ objects/exceptions cross the
  boundary and the correct C++ runtime loads. If friction → build the shim with `cl` for the CUDA profile.
- **R2 (MED): `whisper_full_params` drift.** Mitigated by `default_params`+mutate and pinning the submodule SHA.
- **R3 (MED): streaming API churn.** Mitigated by basing it on public `Session.Transcribe` + a documented dedup contract;
  `stream/` can change in v1.x without touching the core.
- **R4 (LOW): GPU arch coverage** — `CMAKE_CUDA_ARCHITECTURES` must match the dev GPU; documented in README.

---

## 16. Milestones

- **Phase 0 — Spike & scaffold:** S1 ABI spike; submodule pinned; `go.mod`; Taskfile/scripts; CI matrix skeleton
  (windows/ubuntu/macos × cpu, + GPU build-only cells).
- **Phase 1 — Core CPU path (3 OSes):** `binding.cpp/.h`; `whisper.go`/`session.go`/`options.go`/`result.go`/`errors.go`/`callback.go`/`log.go`;
  CPU build green on Windows + Linux + macOS; transcribe a WAV end-to-end; cancellation + panic-safety; ginkgo gated tests.
- **Phase 2 — `wav/`:** pure-Go decoder + tests; `examples/transcribe`.
- **Phase 3 — GPU backends:** Windows CUDA (MSVC DLL) + Vulkan; Linux CUDA + Vulkan; macOS Metal — link files & scripts;
  GPU-labelled execution tests on self-hosted/local runners; build+link cells verified in CI.
- **Phase 4 — `stream/`:** sliding-window over a Session; `examples/stream`; dedup contract tests.
- **Phase 5 — Docs & polish:** README (mirrors go-llama section order), model download, `docs/` (ARCHITECTURE/ROADMAP/BACKLOG).

---

## 17. Acceptance criteria (v1)

1. `task deps && task build:cpu && go test ./...` passes on **Windows, Linux, and macOS** in CI (integration tests run with `TEST_MODEL`); every platform×backend cell compiles & links in CI.
2. `Model`/`Session` transcribe a 16 kHz WAV to correct timestamped segments; translate + autodetect + token timestamps work.
3. A cancelled `context.Context` aborts an in-flight `Transcribe` promptly; no goroutine/handle leaks; no panics cross cgo.
4. N concurrent `Session`s over one `Model` run in parallel (one model load) without data races (`-race` clean on pure-Go + a stubbed path).
5. `wav/` decodes mono/stereo 8/16/24/32-int + 32-float WAVs, downmixes stereo, and returns `ErrNot16kHz` for off-rate input.
6. Windows/Linux CUDA + Vulkan and macOS Metal builds produce working binaries on GPU-equipped runners; required DLLs/dylibs documented.
7. `stream/` transcribes a chunked WAV fed over a channel, emitting incremental + `Final` events per the documented policy.
8. `golangci-lint run` clean; non-goals (grammar/logit_bias/samplers) absent.
```
