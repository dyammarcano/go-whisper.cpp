# go-whisper.cpp Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working, cross-platform (CPU) Go binding for whisper.cpp that loads a ggml model and transcribes 16 kHz mono float32 PCM into timestamped segments/tokens, plus a pure-Go WAV reader — built via a thin C shim + cgo, mirroring go-llama.cpp.

**Architecture:** Upstream whisper.cpp is a git submodule built to static libs by a script. A thin C shim (`binding.cpp`/`binding.h`, compiled to `libbinding.a`) owns construction of whisper's by-value param structs and hosts callback trampolines; Go calls the shim over cgo. Public Go types split into `Model` (shared `whisper_context`) and `Session` (per-inference `whisper_state`). A separate pure-Go `wav/` package decodes WAV with no cgo.

**Tech Stack:** Go 1.25 (cgo), C++17, whisper.cpp + ggml, MinGW/Clang/GCC, CMake, Task (Taskfile), ginkgo/gomega, golangci-lint, `runtime/cgo.Handle`.

**Spec:** `docs/superpowers/specs/2026-06-01-go-whisper-cpp-binding-design.md` (this plan covers spec §1–§8, §10–§13 for the CPU path; GPU §11 CUDA/Vulkan/Metal and `stream/` §9 are follow-on plans).

**Module:** `github.com/dyammarcano/go-whisper.cpp`

---

## Shared contracts (USE THESE EXACT NAMES across all tasks)

**C shim ABI** — defined in Task 3 (`binding.h`). Key symbols: `whisper_bind_load_model`, `whisper_bind_free_model`, `whisper_bind_new_state`, `whisper_bind_free_state`, `whisper_bind_full`, `whisper_bind_get_result`, `whisper_bind_free_result`, `whisper_bind_lang_str`, `whisper_bind_lang_id`, `whisper_bind_lang_max_id`, `whisper_bind_install_log`. Structs: `whisper_bind_params`, `whisper_bind_token`, `whisper_bind_segment`, `whisper_bind_result`.

**Exported Go trampoline targets** (defined in Task 7 `callback.go` / Task 8 `log.go`): `goWhisperSegment(handle C.uintptr_t, nNew C.int)`, `goWhisperProgress(handle C.uintptr_t, progress C.int)`, `goWhisperAbort(handle C.uintptr_t) C.int`, `goWhisperLog(level C.int, text *C.char)`.

**Go public types** (package `whisper`):
- `type Model struct` — `New(modelPath string, opts ...ModelOption) (*Model, error)`, `(*Model) Close() error`, `(*Model) Transcribe(ctx context.Context, samples []float32, opts ...TranscribeOption) (*Result, error)`, `(*Model) NewSession() (*Session, error)`, `(*Model) Languages() []string`.
- `type Session struct` — `(*Session) Transcribe(ctx context.Context, samples []float32, opts ...TranscribeOption) (*Result, error)`, `(*Session) Close() error`.
- `type Result struct { Segments []Segment; Language string }`
- `type Segment struct { Start, End time.Duration; Text string; Tokens []Token }`
- `type Token struct { Text string; P float32; Start, End time.Duration }`
- Options: `type ModelOption func(*modelOptions)`, `type TranscribeOption func(*transcribeOptions)`.

**Sentinel errors** (Task 9 `errors.go`): `ErrModelLoad`, `ErrStateInit`, `ErrTranscribe`, `ErrCanceled`, `ErrEmptyAudio`, `ErrClosed`.

**Timestamp conversion:** whisper returns centiseconds (10 ms units). `csToDuration(cs int64) time.Duration { return time.Duration(cs) * 10 * time.Millisecond }`.

---

## File structure

```
go-whisper.cpp/
├── whisper.cpp/                 submodule (ggml-org/whisper.cpp)            [Task 1]
├── go.mod  go.sum                                                          [Task 2]
├── Taskfile.yml                                                            [Task 4]
├── .gitignore  .golangci.yml                                              [Task 2,4]
├── scripts/whispercpp.sh  scripts/binding.sh  scripts/download-model.sh   [Task 4,5]
├── binding.h  binding.cpp                                                  [Task 3,6]
├── doc.go                       package doc + cgo preamble (no exports)    [Task 7]
├── whisper.go                   Model, New, Close, Languages, lang helpers [Task 7,10]
├── session.go                   Session, NewSession, Transcribe core       [Task 11,12]
├── callback.go                  //export trampolines + cgo.Handle bridge   [Task 7,12]
├── log.go                       whisper_bind_install_log + //export log    [Task 8]
├── options.go                   modelOptions/transcribeOptions + With*     [Task 9,10]
├── result.go                    Result/Segment/Token + marshalling         [Task 11]
├── errors.go                    sentinel errors                            [Task 9]
├── whisper_suite_test.go        ginkgo RunSpecs                            [Task 13]
├── whisper_test.go              integration specs (TEST_MODEL gated)       [Task 13]
├── examples/transcribe/main.go  CLI demo                                   [Task 15]
├── wav/wav.go  wav/wav_test.go  pure-Go WAV reader (no cgo)               [Task 14]
└── .github/workflows/ci.yml     CPU build+test matrix skeleton             [Task 16]
```

---

## Task 1: Add whisper.cpp as a git submodule (Phase 0)

**Files:**
- Create: `.gitmodules`, `whisper.cpp/` (submodule)

- [ ] **Step 1: Add the submodule pinned to a known-good tag**

Run:
```bash
git submodule add https://github.com/ggml-org/whisper.cpp whisper.cpp
cd whisper.cpp && git fetch --tags && git checkout v1.7.4 && cd ..
```
Expected: `whisper.cpp/include/whisper.h` and `whisper.cpp/CMakeLists.txt` exist; `.gitmodules` created.
(If `v1.7.4` is unavailable, use the latest `v1.7.x` tag — record the exact tag in the commit message.)

- [ ] **Step 2: Verify the header has the expected API**

Run:
```bash
grep -c "whisper_full_with_state\|whisper_init_state\|whisper_full_lang_id" whisper.cpp/include/whisper.h
```
Expected: a number `>= 3` (these symbols are required by the shim).

- [ ] **Step 3: Commit**

```bash
git add .gitmodules whisper.cpp
git commit -m "build: vendor whisper.cpp as a pinned git submodule"
```

---

## Task 2: Initialize the Go module and base config (Phase 0)

**Files:**
- Create: `go.mod`, `.gitignore` (append), `LICENSE`

- [ ] **Step 1: Create the Go module**

Run:
```bash
go mod init github.com/dyammarcano/go-whisper.cpp
go mod edit -go=1.25.0
```

- [ ] **Step 2: Add test dependencies**

Run:
```bash
go get github.com/onsi/ginkgo/v2@v2.13.0 github.com/onsi/gomega@v1.28.0
```
Expected: `go.mod` lists ginkgo + gomega; `go.sum` populated.

- [ ] **Step 3: Write `.gitignore`** (append to existing — `.scripts/` is already present)

Create/append `.gitignore`:
```gitignore
# build artifacts
/binding.o
/libbinding.a
/build-cuda/
whisper.cpp/build-*/
*.exe
*.dll
*.dylib
*.so

# models downloaded for tests
/models/
*.bin

# os / editor
.DS_Store
.idea/
.vscode/
Thumbs.db
```

- [ ] **Step 4: Write `LICENSE`** (BSD-3-Clause, per project policy)

Create `LICENSE` with the standard BSD 3-Clause text, copyright `2026 Dyam Marcano`.

- [ ] **Step 5: Commit**

```bash
git add go.mod go.sum .gitignore LICENSE
git commit -m "build: init go module (1.25) + ginkgo/gomega + BSD-3 license"
```

---

## Task 3: Write the C shim header (`binding.h`) (Phase 1)

**Files:**
- Create: `binding.h`

- [ ] **Step 1: Write `binding.h`**

Create `binding.h` (verbatim — primitive fields only so Go can build the structs):
```c
#ifndef GO_WHISPER_BINDING_H
#define GO_WHISPER_BINDING_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

void* whisper_bind_load_model(const char* path, int use_gpu, int flash_attn, int gpu_device);
void  whisper_bind_free_model(void* ctx);
void* whisper_bind_new_state(void* ctx);
void  whisper_bind_free_state(void* state);

typedef struct {
    int          strategy;          // 0 = greedy, 1 = beam_search
    int          n_threads;
    int          translate;
    const char*  language;          // "auto"/""/NULL -> autodetect
    int          detect_language;
    int          beam_size;         // <=0 -> default
    int          best_of;           // <=0 -> default
    float        temperature;
    float        temperature_inc;
    float        entropy_thold;
    float        logprob_thold;
    float        no_speech_thold;
    int          no_context;
    int          single_segment;
    int          token_timestamps;
    int          max_len;
    int          split_on_word;
    int          max_tokens;
    int          offset_ms;
    int          duration_ms;
    int          audio_ctx;
    int          suppress_blank;
    int          suppress_nst;
    const char*  initial_prompt;    // NULL -> none
    uintptr_t    segment_cb;        // cgo.Handle (0 = none)
    uintptr_t    progress_cb;       // cgo.Handle (0 = none)
    uintptr_t    abort_cb;          // cgo.Handle (0 = none)
} whisper_bind_params;

// Returns 0 on success, whisper rc on failure, -100 if aborted via abort_cb.
int whisper_bind_full(void* ctx, void* state, const whisper_bind_params* p,
                      const float* samples, int n_samples);

typedef struct { int64_t t0, t1; float p; const char* text; } whisper_bind_token;
typedef struct {
    int64_t t0, t1;                 // centiseconds
    const char* text;               // owned by result (strdup'd)
    int n_tokens;
    whisper_bind_token* tokens;     // NULL unless want_tokens
} whisper_bind_segment;
typedef struct {
    int n_segments;
    whisper_bind_segment* segments;
    int lang_id;                    // detected/used language id (-1 if n/a)
} whisper_bind_result;

whisper_bind_result* whisper_bind_get_result(void* ctx, void* state, int want_tokens);
void whisper_bind_free_result(whisper_bind_result* r);

int         whisper_bind_lang_id(const char* lang);
const char* whisper_bind_lang_str(int id);
int         whisper_bind_lang_max_id(void);

void whisper_bind_install_log(void);

#ifdef __cplusplus
}
#endif
#endif
```

> Note: spec §4 listed a `print_to_stderr` field; it is intentionally omitted here — Task 6 hardcodes whisper's `print_progress/realtime/timestamps/special` to `false` (logs go through `whisper_bind_install_log` → slog instead). A `WithVerbose` toggle (spec §10) is deferred to a follow-on.

- [ ] **Step 2: Commit**

```bash
git add binding.h
git commit -m "feat(shim): add C ABI header for whisper binding"
```

---

## Task 4: Write the build scripts and Taskfile (Phase 0)

**Files:**
- Create: `scripts/whispercpp.sh`, `scripts/binding.sh`, `Taskfile.yml`, `.golangci.yml`

- [ ] **Step 1: Write `scripts/whispercpp.sh`** (CPU/Vulkan static build; mirrors go-llama)

Create `scripts/whispercpp.sh`:
```bash
#!/usr/bin/env bash
# Build whisper.cpp static libs (CPU or Vulkan). Output: whisper.cpp/build-<backend>/
set -euo pipefail
BACKEND="${1:-cpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
SRC="whisper.cpp"; BUILD="$SRC/build-$BACKEND"

GEN="Unix Makefiles"; EXTRA=""
case "$(uname -s)" in MINGW*|MSYS*) GEN="MinGW Makefiles";; esac
[ "$BACKEND" = "vulkan" ] && EXTRA="-DGGML_VULKAN=ON"

# MinGW workaround (same as go-llama): gate THREAD_POWER_THROTTLING_STATE.
WINVER=""
case "$(uname -s)" in
  MINGW*|MSYS*)
    WINVER="-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00 -DNTDDI_VERSION=0x0A000007 -DGGML_NO_THREAD_POWER_THROTTLING"
    CPUC="$SRC/ggml/src/ggml-cpu/ggml-cpu.c"
    if [ -f "$CPUC" ] && ! grep -q "GGML_NO_THREAD_POWER_THROTTLING" "$CPUC"; then
      sed -i 's/#if _WIN32_WINNT >= 0x0602$/#if _WIN32_WINNT >= 0x0602 \&\& !defined(GGML_NO_THREAD_POWER_THROTTLING)/' "$CPUC"
    fi
  ;;
esac

rm -rf "$BUILD"
cmake -S "$SRC" -B "$BUILD" -G "$GEN" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF \
  -DWHISPER_SDL2=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF $EXTRA \
  ${WINVER:+-DCMAKE_C_FLAGS="$WINVER" -DCMAKE_CXX_FLAGS="$WINVER"}
cmake --build "$BUILD" -j
echo "=== $BACKEND static libs ==="; find "$BUILD" -name '*.a' | sort
```

- [ ] **Step 2: Write `scripts/binding.sh`** (compile the shim → `libbinding.a`)

Create `scripts/binding.sh`:
```bash
#!/usr/bin/env bash
# Compile binding.cpp -> libbinding.a (binding.o only; whisper libs linked by cgo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CXX="${CXX:-g++}"
WINVER=""
case "$(uname -s)" in
  MINGW*|MSYS*) WINVER="-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00 -DNTDDI_VERSION=0x0A000007 -DGGML_NO_THREAD_POWER_THROTTLING";;
esac
"$CXX" -std=c++17 -O3 $WINVER \
  -I whisper.cpp/include -I whisper.cpp/ggml/include \
  -c binding.cpp -o binding.o
rm -f libbinding.a
ar rcs libbinding.a binding.o
echo "built libbinding.a"
```

- [ ] **Step 3: Write `Taskfile.yml`**

Create `Taskfile.yml`:
```yaml
version: '3'
tasks:
  default: { cmds: [task --list], silent: true }
  deps:
    desc: Initialise the whisper.cpp submodule
    cmds: [git submodule update --init --recursive]
  build:cpu:
    desc: Build whisper.cpp (CPU static) and the binding
    cmds: [bash scripts/whispercpp.sh cpu, bash scripts/binding.sh]
  build: { desc: Alias for build:cpu, cmds: [{task: build:cpu}] }
  models:tiny:
    desc: Download the pinned ggml-tiny.en test model (SHA256-checked)
    cmds: [bash scripts/download-model.sh tiny.en]
  test: { desc: Run Go tests, cmds: [go test ./...] }
  fmt:  { desc: Format Go sources, cmds: [go fmt ./...] }
  fix:  { desc: Apply go fix, cmds: [go fix ./...] }
  lint: { desc: Run golangci-lint (auto-fix), cmds: [golangci-lint run --fix ./...] }
  clean:
    desc: Remove build artifacts
    cmds: ["bash -c 'rm -rf whisper.cpp/build-* build-cuda binding.o libbinding.a'"]
```

- [ ] **Step 4: Write `.golangci.yml`** (mirror go-llama: `default: all` minus the same disables)

Create `.golangci.yml`:
```yaml
version: "2"
run:
  timeout: 2m
linters:
  default: all
  disable:
    - cyclop
    - depguard
    - dupl
    - err113
    - exhaustruct
    - funlen
    - gochecknoglobals
    - gochecknoinits
    - gocognit
    - gocyclo
    - godot
    - godox
    - gosec
    - ireturn
    - lll
    - mnd
    - nestif
    - nlreturn
    - nonamedreturns
    - paralleltest
    - revive
    - tagliatelle
    - testpackage
    - varnamelen
    - wrapcheck
    - wsl
```

- [ ] **Step 5: Commit**

```bash
git add scripts/whispercpp.sh scripts/binding.sh Taskfile.yml .golangci.yml
git commit -m "build: add whisper.cpp/binding build scripts, Taskfile, linter config"
```

---

## Task 5: Write the model download script (Phase 0)

**Files:**
- Create: `scripts/download-model.sh`

- [ ] **Step 1: Write `scripts/download-model.sh`** (SHA256-verified)

Create `scripts/download-model.sh`:
```bash
#!/usr/bin/env bash
# Download a pinned ggml whisper model with SHA256 verification into ./models/.
set -euo pipefail
NAME="${1:-tiny.en}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p models
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${NAME}.bin"
OUT="models/ggml-${NAME}.bin"
# Pinned SHA256 for ggml-tiny.en.bin (update if NAME changes).
declare -A SHA=( ["tiny.en"]="921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f" )
if [ -f "$OUT" ]; then echo "exists: $OUT"; else
  echo "downloading $URL"; curl -fL "$URL" -o "$OUT"
fi
if [ -n "${SHA[$NAME]:-}" ]; then
  echo "${SHA[$NAME]}  $OUT" | sha256sum -c - || { echo "checksum FAILED"; rm -f "$OUT"; exit 1; }
fi
echo "model ready: $OUT"
echo "export TEST_MODEL=$ROOT/$OUT"
```
> Note: this is the one place `curl` is intentionally used inside a build script (not via the agent's Bash tool). If the pinned SHA for `tiny.en` differs at implementation time, compute it once with `sha256sum` and update the map.

- [ ] **Step 2: Commit**

```bash
git add scripts/download-model.sh
git commit -m "build: add SHA256-verified model download script"
```

---

## Task 6: Implement the C shim (`binding.cpp`) (Phase 1)

**Files:**
- Create: `binding.cpp`

- [ ] **Step 1: Write `binding.cpp`** (full implementation)

Create `binding.cpp`:
```cpp
// go-whisper.cpp binding — thin C shim over whisper.h's C API.
// Owns whisper_full_params/whisper_context_params construction (by-value structs
// never cross cgo) and hosts callback trampolines that call exported Go funcs.
#include "binding.h"
#include "whisper.h"
#include "ggml.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" {
    // These prototypes MUST match cgo's generated _cgoexp_ signatures for the
    // //export funcs in callback.go / log.go. cgo emits non-const char* for *C.char,
    // uintptr_t for C.uintptr_t, int64_t for C.int64_t, int for C.int. binding.cpp is
    // compiled OUTSIDE cgo (by scripts/binding.sh), so it needs its own decls.
    void goWhisperSegment(uintptr_t handle, int64_t t0, int64_t t1, char* text);
    void goWhisperProgress(uintptr_t handle, int progress);
    int  goWhisperAbort(uintptr_t handle);
    void goWhisperLog(int level, char* text);
}

// seg_tramp reads ONLY the newly-added segments [total-n_new, total) from the live
// state/context it is handed (the documented, upstream-idiomatic use of
// new_segment_callback) and hands each to Go directly. It does NOT re-collect all
// segments and never re-enters via the Go Model pointer — avoiding the reentrancy
// hazard and the O(n^2) allocation of snapshotting on every callback.
static void seg_tramp(struct whisper_context* c, struct whisper_state* st, int n_new, void* ud) {
    int total = st ? whisper_full_n_segments_from_state(st) : whisper_full_n_segments(c);
    for (int i = total - n_new; i < total; ++i) {
        if (i < 0) continue;
        int64_t t0 = st ? whisper_full_get_segment_t0_from_state(st, i) : whisper_full_get_segment_t0(c, i);
        int64_t t1 = st ? whisper_full_get_segment_t1_from_state(st, i) : whisper_full_get_segment_t1(c, i);
        const char* txt = st ? whisper_full_get_segment_text_from_state(st, i) : whisper_full_get_segment_text(c, i);
        goWhisperSegment((uintptr_t)ud, t0, t1, (char*)(txt ? txt : ""));
    }
}
static void prog_tramp(struct whisper_context*, struct whisper_state*, int progress, void* ud) {
    goWhisperProgress((uintptr_t)ud, progress);
}
static bool abort_tramp(void* ud) {
    return goWhisperAbort((uintptr_t)ud) != 0;
}
static void log_tramp(enum ggml_log_level level, const char* text, void*) {
    goWhisperLog((int)level, (char*)text);
}

extern "C" void whisper_bind_install_log(void) { whisper_log_set(log_tramp, nullptr); }

extern "C" void* whisper_bind_load_model(const char* path, int use_gpu, int flash_attn, int gpu_device) {
    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu    = use_gpu != 0;
    cp.flash_attn = flash_attn != 0;
    cp.gpu_device = gpu_device;
    return (void*) whisper_init_from_file_with_params(path, cp);
}
extern "C" void  whisper_bind_free_model(void* ctx) { if (ctx) whisper_free((struct whisper_context*)ctx); }
extern "C" void* whisper_bind_new_state(void* ctx) {
    if (!ctx) return nullptr;
    return (void*) whisper_init_state((struct whisper_context*)ctx);
}
extern "C" void  whisper_bind_free_state(void* st) { if (st) whisper_free_state((struct whisper_state*)st); }

static whisper_full_params build_params(const whisper_bind_params* p) {
    whisper_full_params wp = whisper_full_default_params(
        p->strategy == 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
    if (p->n_threads > 0) wp.n_threads = p->n_threads;
    wp.translate       = p->translate != 0;
    wp.language        = (p->language && p->language[0]) ? p->language : "auto";
    wp.detect_language = p->detect_language != 0;
    if (p->beam_size > 0) wp.beam_search.beam_size = p->beam_size;
    if (p->best_of   > 0) wp.greedy.best_of        = p->best_of;
    wp.temperature      = p->temperature;
    wp.temperature_inc  = p->temperature_inc;
    wp.entropy_thold    = p->entropy_thold;
    wp.logprob_thold    = p->logprob_thold;
    wp.no_speech_thold  = p->no_speech_thold;
    wp.no_context       = p->no_context != 0;
    wp.single_segment   = p->single_segment != 0;
    wp.token_timestamps = p->token_timestamps != 0;
    if (p->max_len   > 0) wp.max_len    = p->max_len;
    wp.split_on_word    = p->split_on_word != 0;
    if (p->max_tokens> 0) wp.max_tokens = p->max_tokens;
    wp.offset_ms        = p->offset_ms;
    wp.duration_ms      = p->duration_ms;
    if (p->audio_ctx > 0) wp.audio_ctx  = p->audio_ctx;
    wp.suppress_blank   = p->suppress_blank != 0;
    wp.suppress_nst     = p->suppress_nst != 0;
    if (p->initial_prompt && p->initial_prompt[0]) wp.initial_prompt = p->initial_prompt;
    wp.print_progress = false; wp.print_realtime = false;
    wp.print_timestamps = false; wp.print_special = false;
    if (p->segment_cb)  { wp.new_segment_callback   = seg_tramp;  wp.new_segment_callback_user_data   = (void*)(uintptr_t)p->segment_cb; }
    if (p->progress_cb) { wp.progress_callback      = prog_tramp; wp.progress_callback_user_data      = (void*)(uintptr_t)p->progress_cb; }
    if (p->abort_cb)    { wp.abort_callback         = abort_tramp;wp.abort_callback_user_data         = (void*)(uintptr_t)p->abort_cb; }
    return wp;
}

extern "C" int whisper_bind_full(void* ctx, void* state, const whisper_bind_params* p,
                                 const float* samples, int n_samples) {
    whisper_full_params wp = build_params(p);
    if (state) return whisper_full_with_state((struct whisper_context*)ctx, (struct whisper_state*)state, wp, samples, n_samples);
    return whisper_full((struct whisper_context*)ctx, wp, samples, n_samples);
}

extern "C" whisper_bind_result* whisper_bind_get_result(void* ctx, void* state, int want_tokens) {
    struct whisper_context* c = (struct whisper_context*)ctx;
    struct whisper_state*   s = (struct whisper_state*)state;
    int n = s ? whisper_full_n_segments_from_state(s) : whisper_full_n_segments(c);
    whisper_bind_result* r = (whisper_bind_result*)calloc(1, sizeof(whisper_bind_result));
    r->n_segments = n;
    r->lang_id = s ? whisper_full_lang_id_from_state(s) : whisper_full_lang_id(c);
    r->segments = (whisper_bind_segment*)calloc(n > 0 ? (size_t)n : 1, sizeof(whisper_bind_segment));
    for (int i = 0; i < n; ++i) {
        const char* txt = s ? whisper_full_get_segment_text_from_state(s, i) : whisper_full_get_segment_text(c, i);
        r->segments[i].t0   = s ? whisper_full_get_segment_t0_from_state(s, i) : whisper_full_get_segment_t0(c, i);
        r->segments[i].t1   = s ? whisper_full_get_segment_t1_from_state(s, i) : whisper_full_get_segment_t1(c, i);
        r->segments[i].text = strdup(txt ? txt : "");
        if (want_tokens) {
            int nt = s ? whisper_full_n_tokens_from_state(s, i) : whisper_full_n_tokens(c, i);
            r->segments[i].n_tokens = nt;
            r->segments[i].tokens = (whisper_bind_token*)calloc(nt > 0 ? (size_t)nt : 1, sizeof(whisper_bind_token));
            for (int j = 0; j < nt; ++j) {
                whisper_token_data td = s ? whisper_full_get_token_data_from_state(s, i, j) : whisper_full_get_token_data(c, i, j);
                r->segments[i].tokens[j].t0 = td.t0;
                r->segments[i].tokens[j].t1 = td.t1;
                r->segments[i].tokens[j].p  = td.p;
                // whisper_token_to_str(ctx, token) is state-independent; td.id came from token_data.
                const char* tt = whisper_token_to_str(c, td.id);
                r->segments[i].tokens[j].text = strdup(tt ? tt : "");
            }
        }
    }
    return r;
}
extern "C" void whisper_bind_free_result(whisper_bind_result* r) {
    if (!r) return;
    for (int i = 0; i < r->n_segments; ++i) {
        free((void*)r->segments[i].text);
        if (r->segments[i].tokens) {
            for (int j = 0; j < r->segments[i].n_tokens; ++j) free((void*)r->segments[i].tokens[j].text);
            free(r->segments[i].tokens);
        }
    }
    free(r->segments);
    free(r);
}

extern "C" int         whisper_bind_lang_id(const char* lang) { return whisper_lang_id(lang); }
extern "C" const char* whisper_bind_lang_str(int id)          { return whisper_lang_str(id); }
extern "C" int         whisper_bind_lang_max_id(void)         { return whisper_lang_max_id(); }
```
> Verified against whisper.cpp v1.7.x: every `whisper_*`/`ggml_*` symbol above matches the pinned header. Token text uses `whisper_token_to_str(ctx, token)` (state-independent) — the wrong-arity `whisper_full_get_token_text_from_state` call was removed. `ggml_log_level` / `ggml_log_callback` come from `ggml.h`, which `whisper.h` includes transitively (binding.cpp also `#include "ggml.h"`).

- [ ] **Step 2: Build whisper.cpp + the shim to verify it compiles & links symbols**

Run:
```bash
task deps && task build:cpu
```
Expected: `whisper.cpp/build-cpu/src/libwhisper.a` exists; `libbinding.a` built; no undefined-symbol errors.

- [ ] **Step 3: Commit**

```bash
git add binding.cpp
git commit -m "feat(shim): implement whisper C shim (params build, results, callbacks, log)"
```

---

## Task 7: cgo preamble, Model type, and trampoline declarations (Phase 1)

**Files:**
- Create: `doc.go`, `whisper.go`, `callback.go`, `link_static_windows.go`, `link_linux.go`, `link_darwin.go`

- [ ] **Step 1: Write `doc.go`** (package doc only)

Create `doc.go`:
```go
// Package whisper provides Go bindings for whisper.cpp speech-to-text.
//
// A Model wraps a loaded whisper_context (shared, read-only). A Session wraps a
// whisper_state for a single inference; create one Session per goroutine for
// concurrent transcription over one shared Model.
package whisper
```

- [ ] **Step 2: Write `whisper.go`** (cgo preamble + Model + New + Close + Languages)

Create `whisper.go`:
```go
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
```

- [ ] **Step 3: Write `callback.go`** (extern decls + exported trampolines + bridge)

Create `callback.go`:
```go
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
```
> `callback.go` references `Segment` (Task 11) and `csToDuration` (Task 11) — same package, so the package only fully builds once Task 11/12 land (see the build-order note in Tasks 9–12).

- [ ] **Step 4: Write the link files** (CPU static + POSIX)

Create `link_static_windows.go`:
```go
//go:build windows && !cuda && !vulkan

package whisper

// #cgo LDFLAGS: -L${SRCDIR}/whisper.cpp/build-cpu/src -L${SRCDIR}/whisper.cpp/build-cpu/ggml/src
// #cgo LDFLAGS: -Wl,--start-group -lwhisper -lggml -lggml-cpu -lggml-base -Wl,--end-group -lstdc++ -lm
import "C"
```
Create `link_linux.go`:
```go
//go:build linux && !cuda && !vulkan

package whisper

// #cgo LDFLAGS: -L${SRCDIR}/whisper.cpp/build-cpu/src -L${SRCDIR}/whisper.cpp/build-cpu/ggml/src
// #cgo LDFLAGS: -Wl,--start-group -lwhisper -lggml -lggml-cpu -lggml-base -Wl,--end-group -lstdc++ -lm -lpthread -ldl
import "C"
```
Create `link_darwin.go`:
```go
//go:build darwin

package whisper

// #cgo LDFLAGS: -L${SRCDIR}/whisper.cpp/build-cpu/src -L${SRCDIR}/whisper.cpp/build-cpu/ggml/src
// #cgo LDFLAGS: -lwhisper -lggml -lggml-cpu -lggml-base -lstdc++
// #cgo LDFLAGS: -framework Accelerate -framework Metal -framework MetalKit -framework Foundation
import "C"
```
> Note: on macOS, the default CPU build still links the Metal framework (whisper.cpp's default ggml-metal backend). If a pure-CPU macOS build is needed, a `!metal` variant can be added in the GPU plan.

- [ ] **Step 5: Commit**

```bash
git add doc.go whisper.go callback.go link_static_windows.go link_linux.go link_darwin.go
git commit -m "feat: cgo preamble, Model lifecycle, callback bridge, link files"
```

---

## Task 8: Logging bridge (`log.go`) (Phase 1)

**Files:**
- Create: `log.go`

- [ ] **Step 1: Write `log.go`**

Create `log.go`:
```go
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
```

- [ ] **Step 2: Commit**

```bash
git add log.go
git commit -m "feat: route whisper C logs into slog"
```

---

## Task 9: Errors, abort flag, and option types (Phase 1)

**Files:**
- Create: `errors.go`, `options.go` (types + defaults; With* funcs in Task 10)

- [ ] **Step 1: Write `errors.go`**

Create `errors.go`:
```go
package whisper

import (
	"errors"
	"fmt"
	"sync/atomic"
)

var (
	ErrModelLoad  = errors.New("whisper: failed to load model")
	ErrStateInit  = errors.New("whisper: failed to init state")
	ErrTranscribe = errors.New("whisper: transcription failed")
	ErrCanceled   = errors.New("whisper: transcription canceled")
	ErrEmptyAudio = errors.New("whisper: empty audio (no samples)")
	ErrClosed     = errors.New("whisper: use of closed model or session")
)

// abortFlag is a goroutine-safe one-shot flag read by the abort trampoline.
type abortFlag struct{ v atomic.Bool }

func (a *abortFlag) set()        { a.v.Store(true) }
func (a *abortFlag) isSet() bool { return a.v.Load() }

func panicToError(r any) error {
	if e, ok := r.(error); ok {
		return fmt.Errorf("panic in callback: %w", e)
	}
	return fmt.Errorf("panic in callback: %v", r)
}
```

- [ ] **Step 2: Write option type scaffolding in `options.go`**

Create `options.go`:
```go
package whisper

// ModelOption configures Model loading (whisper_context_params).
type ModelOption func(*modelOptions)

type modelOptions struct {
	gpu       bool
	flashAttn bool
	gpuDevice int
}

func defaultModelOptions() modelOptions {
	return modelOptions{gpu: false, flashAttn: false, gpuDevice: 0}
}

// TranscribeOption configures a single Transcribe call (whisper_full_params).
type TranscribeOption func(*transcribeOptions)

type transcribeOptions struct {
	beamSearch      bool
	beamSize        int
	bestOf          int
	threads         int
	translate       bool
	language        string // "" / "auto" -> autodetect
	detectLanguage  bool
	temperature     float32
	temperatureInc  float32
	entropyThold    float32
	logProbThold    float32
	noSpeechThold   float32
	noContext       bool
	singleSegment   bool
	tokenTimestamps bool
	maxLen          int
	splitOnWord     bool
	maxTokens       int
	offsetMs        int
	durationMs      int
	audioCtx        int
	suppressBlank   bool
	suppressNST     bool
	initialPrompt   string
	onSegment       func(Segment)
	onProgress      func(int)
}

func defaultTranscribeOptions() transcribeOptions {
	return transcribeOptions{
		language:      "auto",
		suppressBlank: true,
		suppressNST:   true,
		bestOf:        2,
		beamSize:      5,
	}
}
```

- [ ] **Step 3: Verify the C side builds and the new Go files are syntactically valid**

Run:
```bash
task build:cpu
gofmt -l errors.go options.go
```
Expected: `task build:cpu` produces `libbinding.a`; `gofmt -l` prints nothing (files formatted).
> NOTE: do NOT run `go build ./...` yet — the cgo package does not compile until Task 12 (callback.go references `Segment`/`csToDuration` created in Task 11, and `session.go` in Task 12). The first real package build gate is **Task 12 Step 2**.

- [ ] **Step 4: Commit**

```bash
git add errors.go options.go
git commit -m "feat: sentinel errors, abort flag, option types + defaults"
```

---

## Task 10: Functional option setters (`options.go`) (Phase 1)

**Files:**
- Modify: `options.go` (append `With*` funcs); `whisper.go` (Model options)

- [ ] **Step 1: Append Model options to `options.go`**

Append to `options.go`:
```go
// WithGPU enables/disables GPU offload (default off).
func WithGPU(on bool) ModelOption { return func(o *modelOptions) { o.gpu = on } }

// WithGPUDevice selects the GPU device index.
func WithGPUDevice(d int) ModelOption { return func(o *modelOptions) { o.gpuDevice = d } }

// WithFlashAttn enables flash attention.
var WithFlashAttn ModelOption = func(o *modelOptions) { o.flashAttn = true }
```

- [ ] **Step 2: Append Transcribe options to `options.go`**

Append to `options.go`:
```go
func WithLanguage(lang string) TranscribeOption { return func(o *transcribeOptions) { o.language = lang } }
var WithTranslate TranscribeOption       = func(o *transcribeOptions) { o.translate = true }
var WithDetectLanguage TranscribeOption  = func(o *transcribeOptions) { o.detectLanguage = true }
func WithThreads(n int) TranscribeOption  { return func(o *transcribeOptions) { o.threads = n } }
func WithBeamSearch(size int) TranscribeOption {
	return func(o *transcribeOptions) { o.beamSearch = true; o.beamSize = size }
}
func WithGreedy(bestOf int) TranscribeOption {
	return func(o *transcribeOptions) { o.beamSearch = false; o.bestOf = bestOf }
}
func WithTemperature(t float32) TranscribeOption    { return func(o *transcribeOptions) { o.temperature = t } }
func WithTemperatureInc(t float32) TranscribeOption { return func(o *transcribeOptions) { o.temperatureInc = t } }
func WithEntropyThold(t float32) TranscribeOption   { return func(o *transcribeOptions) { o.entropyThold = t } }
func WithLogProbThold(t float32) TranscribeOption   { return func(o *transcribeOptions) { o.logProbThold = t } }
func WithNoSpeechThold(t float32) TranscribeOption  { return func(o *transcribeOptions) { o.noSpeechThold = t } }
var WithTokenTimestamps TranscribeOption = func(o *transcribeOptions) { o.tokenTimestamps = true }
func WithMaxSegmentLen(n int) TranscribeOption { return func(o *transcribeOptions) { o.maxLen = n } }
var WithSplitOnWord TranscribeOption     = func(o *transcribeOptions) { o.splitOnWord = true }
func WithMaxTokens(n int) TranscribeOption     { return func(o *transcribeOptions) { o.maxTokens = n } }
var WithNoContext TranscribeOption       = func(o *transcribeOptions) { o.noContext = true }
var WithSingleSegment TranscribeOption   = func(o *transcribeOptions) { o.singleSegment = true }
func WithInitialPrompt(s string) TranscribeOption { return func(o *transcribeOptions) { o.initialPrompt = s } }
func WithSegmentCallback(f func(Segment)) TranscribeOption { return func(o *transcribeOptions) { o.onSegment = f } }
func WithProgressCallback(f func(int)) TranscribeOption    { return func(o *transcribeOptions) { o.onProgress = f } }

func WithOffset(d time.Duration) TranscribeOption   { return func(o *transcribeOptions) { o.offsetMs = int(d / time.Millisecond) } }
func WithDuration(d time.Duration) TranscribeOption { return func(o *transcribeOptions) { o.durationMs = int(d / time.Millisecond) } }
func WithAudioCtx(n int) TranscribeOption           { return func(o *transcribeOptions) { o.audioCtx = n } }
// suppress_blank / suppress_nst default to true (see defaultTranscribeOptions); pass false to disable.
func WithSuppressBlank(on bool) TranscribeOption     { return func(o *transcribeOptions) { o.suppressBlank = on } }
func WithSuppressNonSpeech(on bool) TranscribeOption { return func(o *transcribeOptions) { o.suppressNST = on } }
```
> These cover the remaining spec §6 options (`WithOffset/WithDuration/WithAudioCtx/WithSuppressBlank/WithSuppressNonSpeech`) so no `transcribeOptions` field set in `buildCParams` (Task 12) is unreachable from the public API. **Add `import "time"` to the top of `options.go`** (the file otherwise has no imports).

- [ ] **Step 3: Verify the new Go file is syntactically valid**

Run: `gofmt -l options.go`
Expected: prints nothing. (Full package build is still gated at Task 12 Step 2.)

- [ ] **Step 4: Commit**

```bash
git add options.go
git commit -m "feat: functional options for model + transcription"
```

---

## Task 11: Result types + marshalling (`result.go`) (Phase 1)

**Files:**
- Create: `result.go`

- [ ] **Step 1: Write `result.go`**

Create `result.go`:
```go
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
	r := (*C.whisper_bind_result)(ptr)
	defer C.whisper_bind_free_result(r)

	n := int(r.n_segments)
	res := &Result{Segments: make([]Segment, 0, n), Language: langStr(int(r.lang_id))}
	if n == 0 {
		return res
	}
	segs := unsafe.Slice(r.segments, n)
	for i := 0; i < n; i++ {
		cs := segs[i]
		seg := Segment{
			Start: csToDuration(int64(cs.t0)),
			End:   csToDuration(int64(cs.t1)),
			Text:  C.GoString(cs.text),
		}
		if nt := int(cs.n_tokens); nt > 0 && cs.tokens != nil {
			toks := unsafe.Slice(cs.tokens, nt)
			seg.Tokens = make([]Token, nt)
			for j := 0; j < nt; j++ {
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
```

- [ ] **Step 2: Verify the new Go file is syntactically valid**

Run: `gofmt -l result.go`
Expected: prints nothing. (`result.go` defines `Segment`/`csToDuration` that `callback.go` already references; the full cgo package build gate is Task 12 Step 2.)

- [ ] **Step 3: Commit**

```bash
git add result.go
git commit -m "feat: Result/Segment/Token types and C result marshalling"
```

---

## Task 12: Session + Transcribe (the core inference path) (Phase 1)

**Files:**
- Create: `session.go`; Modify: `callback.go` (add `collectNewSegments`)

- [ ] **Step 1: Write `session.go`**

Create `session.go`:
```go
package whisper

// #include <stdlib.h>
// #include "binding.h"
import "C"

import (
	"context"
	"fmt"
	"runtime/cgo"
	"unsafe"
)

// Session owns a whisper_state for a single inference at a time. Not safe for
// concurrent use; create one Session per goroutine via Model.NewSession.
type Session struct {
	model *Model
	state unsafe.Pointer // whisper_state*
}

// NewSession allocates an independent inference state over the shared Model.
func (m *Model) NewSession() (*Session, error) {
	if m == nil || m.ptr == nil {
		return nil, ErrClosed
	}
	st := C.whisper_bind_new_state(m.ptr)
	if st == nil {
		return nil, ErrStateInit
	}
	return &Session{model: m, state: st}, nil
}

// Close frees the session's state. Idempotent.
func (s *Session) Close() error {
	if s == nil || s.state == nil {
		return nil
	}
	C.whisper_bind_free_state(s.state)
	s.state = nil
	return nil
}

// Transcribe runs whisper over the session's own state. ctx cancels mid-inference.
func (s *Session) Transcribe(ctx context.Context, samples []float32, opts ...TranscribeOption) (*Result, error) {
	if s == nil || s.state == nil || s.model == nil || s.model.ptr == nil {
		return nil, ErrClosed
	}
	return runFull(ctx, s.model, s, samples, opts...)
}

// Transcribe on the Model uses the context's internal state, serialized by a mutex.
// Convenient for one-shot/CLI use; for concurrency, use Sessions.
func (m *Model) Transcribe(ctx context.Context, samples []float32, opts ...TranscribeOption) (*Result, error) {
	if m == nil || m.ptr == nil {
		return nil, ErrClosed
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return runFull(ctx, m, nil, samples, opts...)
}

// runFull is the shared inference core. session == nil -> use the model's internal state.
func runFull(ctx context.Context, m *Model, session *Session, samples []float32, opts ...TranscribeOption) (*Result, error) {
	if len(samples) == 0 {
		return nil, ErrEmptyAudio
	}
	to := defaultTranscribeOptions()
	for _, o := range opts {
		o(&to)
	}

	aborted := &abortFlag{}
	if ctx.Err() != nil { // already canceled — don't start any work
		aborted.set()
	}
	bridge := &callbackBridge{
		onSegment:  to.onSegment,
		onProgress: to.onProgress,
		aborted:    aborted,
	}
	h := cgo.NewHandle(bridge)
	defer h.Delete() // exactly-once; whisper_full joins all worker threads before returning

	// Cancellation watcher: flips the abort flag when ctx is done.
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		select {
		case <-ctx.Done():
			aborted.set()
		case <-stop:
		}
	}()

	cp := buildCParams(&to, h)
	// cp.language / cp.initial_prompt are C strings that whisper reads DURING whisper_full;
	// they MUST stay alive until whisper_bind_full returns. These defers run at runFull's
	// return (after the call + get_result), so the invariant holds — do not move them earlier.
	var langC *C.char
	if cp.language != nil {
		langC = cp.language
		defer C.free(unsafe.Pointer(langC))
	}
	var promptC *C.char
	if cp.initial_prompt != nil {
		promptC = cp.initial_prompt
		defer C.free(unsafe.Pointer(promptC))
	}

	var statePtr unsafe.Pointer
	if session != nil {
		statePtr = session.state
	}
	rc := C.whisper_bind_full(m.ptr, statePtr, &cp,
		(*C.float)(unsafe.Pointer(&samples[0])), C.int(len(samples)))

	if perr := bridge.panicErr.Load(); perr != nil {
		return nil, *perr
	}
	if aborted.isSet() {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrCanceled, err)
		}
		return nil, ErrCanceled
	}
	if rc != 0 {
		return nil, fmt.Errorf("%w: rc=%d", ErrTranscribe, int(rc))
	}

	want := 0
	if to.tokenTimestamps {
		want = 1
	}
	res := marshalResult(unsafe.Pointer(C.whisper_bind_get_result(m.ptr, statePtr, C.int(want))))
	return res, nil
}

// buildCParams fills the flat C params struct. Caller frees cp.language/initial_prompt.
func buildCParams(to *transcribeOptions, h cgo.Handle) C.whisper_bind_params {
	var cp C.whisper_bind_params
	if to.beamSearch {
		cp.strategy = 1
	}
	cp.n_threads = C.int(to.threads)
	if to.translate {
		cp.translate = 1
	}
	cp.language = C.CString(to.language)
	if to.detectLanguage {
		cp.detect_language = 1
	}
	cp.beam_size = C.int(to.beamSize)
	cp.best_of = C.int(to.bestOf)
	cp.temperature = C.float(to.temperature)
	cp.temperature_inc = C.float(to.temperatureInc)
	cp.entropy_thold = C.float(to.entropyThold)
	cp.logprob_thold = C.float(to.logProbThold)
	cp.no_speech_thold = C.float(to.noSpeechThold)
	if to.noContext {
		cp.no_context = 1
	}
	if to.singleSegment {
		cp.single_segment = 1
	}
	if to.tokenTimestamps {
		cp.token_timestamps = 1
	}
	cp.max_len = C.int(to.maxLen)
	if to.splitOnWord {
		cp.split_on_word = 1
	}
	cp.max_tokens = C.int(to.maxTokens)
	cp.offset_ms = C.int(to.offsetMs)
	cp.duration_ms = C.int(to.durationMs)
	cp.audio_ctx = C.int(to.audioCtx)
	if to.suppressBlank {
		cp.suppress_blank = 1
	}
	if to.suppressNST {
		cp.suppress_nst = 1
	}
	if to.initialPrompt != "" {
		cp.initial_prompt = C.CString(to.initialPrompt)
	}
	if to.onSegment != nil {
		cp.segment_cb = C.uintptr_t(h)
	}
	if to.onProgress != nil {
		cp.progress_cb = C.uintptr_t(h)
	}
	cp.abort_cb = C.uintptr_t(h) // always installed for cancellation
	return cp
}
```

- [ ] **Step 2: Build & vet the FULL cgo package (the first real build gate)**

This is the first point at which every Go file exists and the package can compile end-to-end.

Run:
```bash
task build:cpu && go build ./... && go vet ./...
```
Expected: clean build + vet. If `go build` fails on `callback.go` referencing `Segment`/`csToDuration`, confirm `result.go` (Task 11) exists. The package links against `libbinding.a` + the whisper static libs via the link file for the host OS.
> The new-segment callback is delivered entirely by the shim's `seg_tramp` (Task 6) passing each new segment's `t0/t1/text` to `goWhisperSegment` — there is no Go-side `collectNewSegments`, so no reentry into the live whisper state and no O(n²) snapshotting.

- [ ] **Step 3: Commit**

```bash
git add session.go
git commit -m "feat: Session + Transcribe core (whisper_full/_with_state, cancellation)"
```

---

## Task 13: Integration tests (ginkgo, TEST_MODEL-gated) (Phase 1)

**Files:**
- Create: `whisper_suite_test.go`, `whisper_test.go`

- [ ] **Step 1: Write the ginkgo suite**

Create `whisper_suite_test.go`:
```go
package whisper_test

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestWhisper(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "go-whisper.cpp suite")
}
```

- [ ] **Step 2: Write the failing integration spec**

Create `whisper_test.go`:
```go
package whisper_test

import (
	"context"
	"os"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	whisper "github.com/dyammarcano/go-whisper.cpp"
	"github.com/dyammarcano/go-whisper.cpp/wav"
)

var _ = Describe("whisper binding", func() {
	modelPath := os.Getenv("TEST_MODEL")
	audioPath := os.Getenv("TEST_AUDIO") // a 16 kHz mono WAV; defaults to whisper.cpp's sample

	It("fails to load a missing model", func() {
		_, err := whisper.New("/no/such/model.bin")
		Expect(err).To(MatchError(whisper.ErrModelLoad))
	})

	It("transcribes a known WAV", func() {
		if modelPath == "" {
			Skip("set TEST_MODEL to run integration tests (task models:tiny)")
		}
		if audioPath == "" {
			audioPath = "whisper.cpp/samples/jfk.wav"
		}
		m, err := whisper.New(modelPath)
		Expect(err).NotTo(HaveOccurred())
		defer m.Close()

		samples, err := wav.ReadFile(audioPath)
		Expect(err).NotTo(HaveOccurred())

		res, err := m.Transcribe(context.Background(), samples, whisper.WithLanguage("en"))
		Expect(err).NotTo(HaveOccurred())
		Expect(res.Segments).NotTo(BeEmpty())
		full := ""
		for _, s := range res.Segments {
			full += s.Text
		}
		Expect(full).To(ContainSubstring("country")) // jfk.wav: "...what you can do for your country"
	})

	It("cancels an in-flight transcription", func() {
		if modelPath == "" {
			Skip("set TEST_MODEL")
		}
		m, err := whisper.New(modelPath)
		Expect(err).NotTo(HaveOccurred())
		defer m.Close()
		samples, err := wav.ReadFile("whisper.cpp/samples/jfk.wav")
		Expect(err).NotTo(HaveOccurred())

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // already canceled
		_, err = m.Transcribe(ctx, samples, whisper.WithLanguage("en"))
		Expect(err).To(MatchError(whisper.ErrCanceled))
	})

	It("runs concurrent sessions over one model", func() {
		if modelPath == "" {
			Skip("set TEST_MODEL")
		}
		m, err := whisper.New(modelPath)
		Expect(err).NotTo(HaveOccurred())
		defer m.Close()
		samples, err := wav.ReadFile("whisper.cpp/samples/jfk.wav")
		Expect(err).NotTo(HaveOccurred())

		done := make(chan error, 3)
		for i := 0; i < 3; i++ {
			go func() {
				s, e := m.NewSession()
				if e != nil {
					done <- e
					return
				}
				defer s.Close()
				_, e = s.Transcribe(context.Background(), samples, whisper.WithLanguage("en"))
				done <- e
			}()
		}
		for i := 0; i < 3; i++ {
			Eventually(done, 60*time.Second).Should(Receive(BeNil()))
		}
	})
})
```

- [ ] **Step 3: Run the model-less spec to verify the suite wires up**

Run:
```bash
task build:cpu && go test ./... -run TestWhisper -v
```
Expected: the "fails to load a missing model" spec PASSES; model-gated specs SKIP (TEST_MODEL unset). No build/link errors.

- [ ] **Step 4: Run the full integration suite with a model**

Run:
```bash
task models:tiny
TEST_MODEL="$(pwd)/models/ggml-tiny.en.bin" go test -race ./... -v
```
Expected: all specs PASS under `-race` (transcription contains "country"; cancellation returns ErrCanceled; 3 concurrent sessions succeed). `-race` specifically guards the cross-thread `aborted`/`panicErr` access from ggml worker threads — both are atomic, so it must be clean.

- [ ] **Step 5: Commit**

```bash
git add whisper_suite_test.go whisper_test.go
git commit -m "test: integration specs (load, transcribe, cancel, concurrency)"
```

---

## Task 14: Pure-Go WAV reader (`wav/`) (Phase 2)

**Files:**
- Create: `wav/wav.go`, `wav/wav_test.go`

- [ ] **Step 1: Write the failing test**

Create `wav/wav_test.go`:
```go
package wav_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/dyammarcano/go-whisper.cpp/wav"
)

// buildWAV creates a minimal PCM16 WAV in memory.
func buildWAV(sampleRate uint32, channels uint16, samples []int16) []byte {
	var b bytes.Buffer
	dataLen := len(samples) * 2
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+dataLen))
	b.WriteString("WAVE")
	b.WriteString("fmt ")
	binary.Write(&b, binary.LittleEndian, uint32(16))
	binary.Write(&b, binary.LittleEndian, uint16(1)) // PCM
	binary.Write(&b, binary.LittleEndian, channels)
	binary.Write(&b, binary.LittleEndian, sampleRate)
	binary.Write(&b, binary.LittleEndian, sampleRate*uint32(channels)*2) // byte rate
	binary.Write(&b, binary.LittleEndian, channels*2)                    // block align
	binary.Write(&b, binary.LittleEndian, uint16(16))                    // bits
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(dataLen))
	for _, s := range samples {
		binary.Write(&b, binary.LittleEndian, s)
	}
	return b.Bytes()
}

func TestReadWAV_Mono16k(t *testing.T) {
	in := buildWAV(16000, 1, []int16{0, 16384, -16384, 32767})
	got, err := wav.ReadWAV(bytes.NewReader(in))
	if err != nil {
		t.Fatalf("ReadWAV: %v", err)
	}
	want := []float32{0, 0.5, -0.5, 32767.0 / 32768.0}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Errorf("sample %d = %v want %v", i, got[i], want[i])
		}
	}
}

func TestReadWAV_StereoDownmix(t *testing.T) {
	// L=1.0(32767), R=-1.0(-32768) -> mono ~0
	in := buildWAV(16000, 2, []int16{32767, -32768})
	got, err := wav.ReadWAV(bytes.NewReader(in))
	if err != nil {
		t.Fatalf("ReadWAV: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("frames=%d want 1", len(got))
	}
	if math.Abs(float64(got[0])) > 1e-2 {
		t.Errorf("downmix = %v want ~0", got[0])
	}
}

func TestReadWAV_RejectsNon16k(t *testing.T) {
	in := buildWAV(44100, 1, []int16{1, 2, 3})
	_, err := wav.ReadWAV(bytes.NewReader(in))
	if !errors.Is(err, wav.ErrNot16kHz) {
		t.Fatalf("err=%v want ErrNot16kHz", err)
	}
}

func TestReadWAV_RejectsBadHeader(t *testing.T) {
	_, err := wav.ReadWAV(bytes.NewReader([]byte("NOTAWAVfile!!")))
	if !errors.Is(err, wav.ErrBadHeader) {
		t.Fatalf("err=%v want ErrBadHeader", err)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./wav/ -v`
Expected: FAIL — `wav` package / `ReadWAV` undefined.

- [ ] **Step 3: Write `wav/wav.go`**

Create `wav/wav.go`:
```go
// Package wav decodes RIFF/WAVE audio to 16 kHz mono float32 in [-1,1] for whisper.
// Pure Go, no cgo. Non-16 kHz input is rejected (resample with ffmpeg -ar 16000 -ac 1).
package wav

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

const SampleRate = 16000

var (
	ErrNot16kHz       = errors.New("wav: sample rate is not 16000 Hz")
	ErrBadHeader      = errors.New("wav: not a RIFF/WAVE file")
	ErrUnsupportedFmt = errors.New("wav: unsupported audio format / bit depth")
)

type fmtChunk struct {
	audioFormat   uint16
	numChannels   uint16
	sampleRate    uint32
	bitsPerSample uint16
}

// ReadFile decodes a WAV file at path.
func ReadFile(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return ReadWAV(f)
}

// ReadWAV decodes a RIFF/WAVE stream to 16 kHz mono float32.
func ReadWAV(r io.Reader) ([]float32, error) {
	var riff [12]byte
	if _, err := io.ReadFull(r, riff[:]); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrBadHeader, err)
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return nil, ErrBadHeader
	}
	var fc fmtChunk
	gotFmt := false
	for {
		var hdr [8]byte
		if _, err := io.ReadFull(r, hdr[:]); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil, errors.New("wav: no data chunk found")
			}
			return nil, fmt.Errorf("read chunk header: %w", err)
		}
		id := string(hdr[0:4])
		size := binary.LittleEndian.Uint32(hdr[4:8])
		switch id {
		case "fmt ":
			body := make([]byte, size)
			if _, err := io.ReadFull(r, body); err != nil {
				return nil, fmt.Errorf("read fmt chunk: %w", err)
			}
			if len(body) < 16 {
				return nil, ErrUnsupportedFmt
			}
			fc.audioFormat = binary.LittleEndian.Uint16(body[0:2])
			fc.numChannels = binary.LittleEndian.Uint16(body[2:4])
			fc.sampleRate = binary.LittleEndian.Uint32(body[4:8])
			fc.bitsPerSample = binary.LittleEndian.Uint16(body[14:16])
			gotFmt = true
			if size%2 == 1 {
				if _, err := io.CopyN(io.Discard, r, 1); err != nil {
					return nil, err
				}
			}
		case "data":
			if !gotFmt {
				return nil, errors.New("wav: data chunk before fmt chunk")
			}
			if fc.sampleRate != SampleRate {
				return nil, fmt.Errorf("%w: got %d Hz (resample: ffmpeg -i in -ar 16000 -ac 1 out.wav)", ErrNot16kHz, fc.sampleRate)
			}
			data := make([]byte, size)
			if _, err := io.ReadFull(r, data); err != nil {
				return nil, fmt.Errorf("read data chunk: %w", err)
			}
			return decodeSamples(data, &fc)
		default:
			n := int64(size)
			if size%2 == 1 {
				n++
			}
			if _, err := io.CopyN(io.Discard, r, n); err != nil {
				return nil, fmt.Errorf("skip %q chunk: %w", id, err)
			}
		}
	}
}

func decodeSamples(data []byte, fc *fmtChunk) ([]float32, error) {
	ch := int(fc.numChannels)
	if ch < 1 {
		return nil, ErrUnsupportedFmt
	}
	bps := int(fc.bitsPerSample) / 8
	blockAlign := bps * ch
	if blockAlign == 0 || len(data)%blockAlign != 0 {
		return nil, fmt.Errorf("%w: data not aligned to block %d", ErrUnsupportedFmt, blockAlign)
	}
	frames := len(data) / blockAlign
	out := make([]float32, frames)

	var conv func(b []byte) float32
	switch {
	case fc.audioFormat == 3 && fc.bitsPerSample == 32:
		conv = func(b []byte) float32 { return math.Float32frombits(binary.LittleEndian.Uint32(b)) }
	case (fc.audioFormat == 1 || fc.audioFormat == 0xFFFE) && fc.bitsPerSample == 16:
		conv = func(b []byte) float32 { return float32(int16(binary.LittleEndian.Uint16(b))) / 32768.0 }
	case fc.audioFormat == 1 && fc.bitsPerSample == 8:
		conv = func(b []byte) float32 { return (float32(b[0]) - 128) / 128.0 }
	case (fc.audioFormat == 1 || fc.audioFormat == 0xFFFE) && fc.bitsPerSample == 24:
		conv = func(b []byte) float32 {
			u := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16
			if u&0x800000 != 0 {
				u |= 0xFF000000
			}
			return float32(int32(u)) / 8388608.0
		}
	case (fc.audioFormat == 1 || fc.audioFormat == 0xFFFE) && fc.bitsPerSample == 32:
		conv = func(b []byte) float32 { return float32(int32(binary.LittleEndian.Uint32(b))) / 2147483648.0 }
	default:
		return nil, fmt.Errorf("%w: fmt=%d bits=%d", ErrUnsupportedFmt, fc.audioFormat, fc.bitsPerSample)
	}

	for f := 0; f < frames; f++ {
		base := f * blockAlign
		var sum float32
		for c := 0; c < ch; c++ {
			off := base + c*bps
			sum += conv(data[off : off+bps])
		}
		out[f] = sum / float32(ch)
	}
	return out, nil
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `go test ./wav/ -v`
Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add wav/wav.go wav/wav_test.go
git commit -m "feat(wav): pure-Go WAV decoder to 16 kHz mono float32"
```

---

## Task 15: Transcribe example (Phase 2)

**Files:**
- Create: `examples/transcribe/main.go`

- [ ] **Step 1: Write `examples/transcribe/main.go`**

Create `examples/transcribe/main.go`:
```go
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"runtime"

	whisper "github.com/dyammarcano/go-whisper.cpp"
	"github.com/dyammarcano/go-whisper.cpp/wav"
)

func main() {
	model := flag.String("m", "models/ggml-tiny.en.bin", "path to ggml model")
	audio := flag.String("f", "whisper.cpp/samples/jfk.wav", "path to 16 kHz mono WAV")
	lang := flag.String("l", "auto", "language ('auto' to detect)")
	translate := flag.Bool("translate", false, "translate to English")
	threads := flag.Int("t", runtime.NumCPU(), "threads")
	flag.Parse()

	m, err := whisper.New(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load model:", err)
		os.Exit(1)
	}
	defer m.Close()

	samples, err := wav.ReadFile(*audio)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read wav:", err)
		os.Exit(1)
	}

	opts := []whisper.TranscribeOption{whisper.WithLanguage(*lang), whisper.WithThreads(*threads)}
	if *translate {
		opts = append(opts, whisper.WithTranslate)
	}
	res, err := m.Transcribe(context.Background(), samples, opts...)
	if err != nil {
		fmt.Fprintln(os.Stderr, "transcribe:", err)
		os.Exit(1)
	}
	fmt.Printf("[language: %s]\n", res.Language)
	for _, s := range res.Segments {
		fmt.Printf("[%s -> %s] %s\n", s.Start, s.End, s.Text)
	}
}
```

- [ ] **Step 2: Verify it builds**

Run: `go build ./examples/transcribe`
Expected: clean build (produces no binary with `go build ./...`; this just compiles).

- [ ] **Step 3: Run it (smoke, requires model)**

Run:
```bash
task models:tiny
go run ./examples/transcribe -m models/ggml-tiny.en.bin -f whisper.cpp/samples/jfk.wav
```
Expected: prints a language line + segments containing "country".

- [ ] **Step 4: Commit**

```bash
git add examples/transcribe/main.go
git commit -m "docs(example): add transcribe CLI example"
```

---

## Task 16: CPU CI matrix skeleton (Phase 0/1) (Phase 2)

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the CI workflow** (windows/ubuntu/macos × CPU)

Create `.github/workflows/ci.yml`:
```yaml
name: ci
on: [push, pull_request]
jobs:
  build-test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
        with: { submodules: recursive }
      - uses: actions/setup-go@v5
        with: { go-version: '1.25' }
      - name: Install Task
        uses: arduino/setup-task@v2
        with: { version: 3.x, repo-token: "${{ secrets.GITHUB_TOKEN }}" }
      - name: Setup MinGW + CMake (Windows)
        if: runner.os == 'Windows'
        uses: msys2/setup-msys2@v2
        with:
          msystem: UCRT64
          path-type: inherit
          install: >-
            mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake make
      - name: Build whisper.cpp + binding (CPU)
        env: { CC: gcc, CXX: g++ }
        run: task build:cpu
      - name: Download test model
        run: task models:tiny
      - name: Test (race)
        env:
          TEST_MODEL: ${{ github.workspace }}/models/ggml-tiny.en.bin
        run: go test -race ./... -v
      - name: Lint
        uses: golangci/golangci-lint-action@v6
        env: { CGO_ENABLED: '1' }
        with: { version: latest }
```
> macOS/Linux runners have a system compiler; Windows uses MSYS2 UCRT64 MinGW (set up above, `path-type: inherit` puts `gcc`/`g++`/`cmake`/`make` on PATH for later steps). The lint step runs **after** `build:cpu` so `libbinding.a` + whisper static libs exist for cgo typechecking; `CGO_ENABLED=1` is required since the root package is cgo. If Windows lint is flaky, scope it to `./wav/...` on Windows and run full lint on Linux.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: CPU build+test matrix (ubuntu/macos/windows)"
```

---

## Task 17: README (Phase 2)

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`** (mirror go-llama's section order: title/badge, intro, usage/clone, build, quickstart, wav, notes)

Create `README.md` covering: project intro; `git clone --recurse-submodules`; `task deps && task build:cpu`; the `Model`/`Session` quickstart (the §5 example); `wav.ReadFile`; the `TEST_MODEL` test flow; a "Backends" note that CUDA/Vulkan/Metal and streaming arrive in follow-on work; and a "Non-goals" note (no grammar/logit_bias/samplers — STT only).

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with build + quickstart"
```

---

## Self-review checklist (completed by plan author)

- **Spec coverage:** §1 goals 1,2 (transcribe/translate/detect/timestamps) → Tasks 6,12,11; §3 shim → Tasks 3,6; §4 ABI → Task 3; §5 API → Tasks 7,11,12; §6 options → Tasks 9,10; §7 concurrency/cancel/panic-safety → Tasks 12,7,9; §8 wav → Task 14; §10 errors/log → Tasks 9,8; §11 CPU build (Win/Linux/mac) → Tasks 1,4,7; §12 layout → all; §13 testing → Task 13. **Deferred to follow-on plans (noted):** §9 stream/, §11 CUDA/Vulkan/Metal builds, §15 R1 ABI spike for the MSVC-DLL CUDA path, §1 goal 3 streaming.
- **Placeholder scan:** no TBD/TODO; every code step shows full code. The one `curl` in `download-model.sh` is intentional (build script, not agent Bash) and annotated.
- **Type consistency:** `Model`/`Session`/`Result`/`Segment`/`Token`, `modelOptions`/`transcribeOptions`, `whisper_bind_*` symbols, `goWhisper*` trampolines, `csToDuration`, `abortFlag`, `callbackBridge` used identically across Tasks 3,6,7,9,10,11,12. `cp.language`/`cp.initial_prompt` freed by caller (Task 12) matching `C.CString` allocation.

## Post-verification fixes (applied)

This plan was adversarially verified by three reviewers (C-shim vs the pinned `whisper.h`, Go/cgo correctness, spec-coverage/quality). All findings were applied:

- **[BLOCKER] token text:** removed the wrong-arity `whisper_full_get_token_text_from_state(c,s,j)` call (real sig is 4-arg); use `whisper_token_to_str(ctx, td.id)` (Task 6). Every other `whisper_*`/`ggml_*` symbol verified against v1.7.x.
- **[BLOCKER] callback.go preamble/imports:** explicit `#include <stdint.h>` + `import ("runtime/cgo"; "sync/atomic")`; no stale extern decls (Task 7).
- **[BLOCKER] reentrancy + O(n²):** the new-segment callback now receives each new segment's `t0/t1/text` directly from the shim's `seg_tramp`; the Go-side `collectNewSegments` snapshot path was removed (Tasks 6, 7, 12).
- **[HIGH] data race:** `callbackBridge.panicErr` is `atomic.Pointer[error]`; integration suite runs under `-race` (Tasks 7, 12, 13).
- **[HIGH] missing options:** added `WithOffset/WithDuration/WithAudioCtx/WithSuppressBlank/WithSuppressNonSpeech` + `import "time"` (Task 10).
- **[BLOCKER] build order:** intermediate `go build ./...` gates (Tasks 9–11) replaced with `gofmt`/C-side checks; the single real package build gate is Task 12 Step 2 (`go build ./... && go vet ./...`).
- **[LOW] lint/compat:** `max`→`maxID` (predeclared-identifier shadowing); log trampoline uses non-const `char*` to match cgo's generated prototype; Windows CI gets an MSYS2 MinGW setup step; `print_to_stderr` omission documented (Task 3).

## Known follow-on plans (not in this plan)

1. **GPU backends + full CI matrix** — `whispercpp-cuda.bat`, `link_cuda_windows.go`, `link_vulkan_windows.go`, Linux CUDA/Vulkan, macOS Metal CPU/GPU split; the R1 MinGW↔MSVC-DLL ABI spike (spec §15); GPU-labelled execution tests on self-hosted runners.
2. **`stream/` package** — sliding-window streaming over `<-chan []float32` per spec §9; `examples/stream`; dedup contract tests; optional VAD mode (v1.1).
3. **v1.1 polish** — optional resampler in `wav/`, standalone `DetectLanguage`, `docs/ARCHITECTURE.md`/`ROADMAP.md`/`BACKLOG.md`.
```
