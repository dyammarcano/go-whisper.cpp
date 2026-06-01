# go-whisper.cpp GPU Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CUDA (Windows, MSVC DLL) and Vulkan (Windows, MinGW static) GPU backends with build+run verification on the dev box, fix the macOS Metal link path (CI build-verified), and extend CI — all behind mutually-exclusive build tags, with zero change to the public Go API.

**Architecture:** Same shim/cgo binding; only the link layer + build scripts change per backend. CUDA mirrors go-llama.cpp's proven flow: build whisper.cpp as MSVC shared DLLs (`whisper.dll` + `ggml*.dll` incl. `ggml-cuda.dll`) via Ninja+`cl`+`nvcc`, and `link_cuda_windows.go` links `build-cuda/bin/whisper.dll` directly by path (C-ABI only, so MinGW cgo links an MSVC DLL — precedent: go-llama runs CUDA this way). Vulkan is a MinGW **static** build (`GGML_VULKAN=ON` → adds `ggml-vulkan.a`) linked against the system `vulkan-1.dll`. Metal is the macOS default static build (`ggml-metal.a`+`ggml-blas.a`) with the shader embedded.

**Tech Stack:** whisper.cpp v1.7.4, CUDA 13.3 (scoop) + VS2022 Community, Vulkan SDK 1.4.350 (scoop), Ninja, MinGW gcc/g++ 15.2, Go 1.26, Task. Build tags: `cuda`, `vulkan` (Windows); `darwin` (Metal).

**Spec:** `docs/superpowers/specs/2026-06-01-go-whisper-cpp-binding-design.md` §11, §15. **Builds on:** the foundation plan (CPU path, already merged to main).

**Decisions (confirmed):** CUDA arch **75** (Turing, matching go-llama). Scope: CUDA + Vulkan tested locally; Metal CI build-only.

---

## Environment facts (verified on the dev box)

- CUDA 13.3 at `%USERPROFILE%\scoop\apps\cuda\current`; `nvcc.exe` in `…\bin\`; runtime DLLs in `…\bin\x64\` (`cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`) — scoop puts both `bin` and `bin\x64` on PATH.
- VS2022 Community at `C:\Program Files\Microsoft Visual Studio\2022\Community` (vcvars64.bat).
- Vulkan SDK **already installed**: `VULKAN_SDK=%USERPROFILE%\scoop\apps\vulkan\current` (1.4.350) with `Include\vulkan\vulkan.h`, `Lib\vulkan-1.lib`, `Bin\glslc.exe`; `C:\Windows\System32\vulkan-1.dll` present.
- go-llama.cpp precedent: `scripts/llamacpp-cuda.bat` (Ninja, `--target llama`) + `link_cuda_windows.go` (`${SRCDIR}/build-cuda/bin/llama.dll -lstdc++`) → working CUDA.
- v1.7.4: `GGML_CUDA_FA` and `GGML_CUDA_NCCL` do **not** exist — do not pass them. `GGML_BACKEND_DL` stays OFF.

---

## Shared contracts

- **Build tags (mutually exclusive):** CPU `windows && !cuda && !vulkan` (exists); CUDA `windows && cuda` (new); Vulkan `windows && vulkan && !cuda` (new); Linux CPU `linux && !cuda && !vulkan` (exists); macOS `darwin` (exists, fixed here). Building `-tags cuda` selects the CUDA link file and excludes the static CPU one.
- **CUDA output:** repo-root `build-cuda/bin/` (Ninja single-config). Runtime DLLs (ship on PATH): `whisper.dll, ggml.dll, ggml-base.dll, ggml-cpu.dll, ggml-cuda.dll` (built) + `cudart64_13.dll, cublas64_13.dll, cublasLt64_13.dll` (toolkit, already on PATH via scoop).
- **Vulkan output:** `whisper.cpp/build-vulkan/` static archives incl. `ggml/src/ggml-vulkan.a`.
- **GPU is opt-in at load:** GPU is only used when the caller passes `whisper.WithGPU(true)` (ModelOptions.gpu → whisper_context_params.use_gpu). The existing CPU specs pass no GPU option, so GPU tests MUST explicitly use `WithGPU(true)`.

---

## File structure

```
go-whisper.cpp/
├── scripts/whispercpp-cuda.bat      CUDA MSVC-DLL build (Ninja, target whisper)     [Task 1]
├── scripts/whispercpp.sh            +VULKAN_SDK fallback (vulkan arg already exists) [Task 4]
├── link_cuda_windows.go             windows && cuda  -> build-cuda/bin/whisper.dll   [Task 2]
├── link_vulkan_windows.go           windows && vulkan && !cuda -> static + vulkan-1  [Task 4]
├── link_darwin.go                   FIX: add ggml-metal.a, ggml-blas.a, QuartzCore   [Task 6]
├── whisper_gpu_test.go              //go:build cuda||vulkan — WithGPU(true) smoke     [Task 3]
├── Taskfile.yml                     +build:cuda, +build:vulkan                       [Task 1,4]
├── .github/workflows/ci.yml         +gpu-build job; fix macOS Metal leg              [Task 7]
└── README.md / docs                 backends section: tags, DLL PATH, GPU run        [Task 8]
```

---

## Task 1: CUDA build script + Taskfile target

**Files:** Create `scripts/whispercpp-cuda.bat`; Modify `Taskfile.yml`.

- [ ] **Step 1: Write `scripts/whispercpp-cuda.bat`** (mirror go-llama's canonical bat, whisper-adapted)

Create `scripts/whispercpp-cuda.bat`:
```bat
@echo off
REM Build whisper.cpp with CUDA as MSVC shared DLLs (whisper.dll + ggml*.dll incl.
REM ggml-cuda.dll) into repo-root build-cuda\bin\. CUDA/MSVC stays sealed in the DLLs;
REM the MinGW cgo host links only whisper.dll's C API. Needs VS2022 + CUDA toolkit (nvcc).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "PATH=%USERPROFILE%\scoop\shims;%PATH%"
set "CUDA=%USERPROFILE%\scoop\apps\cuda\current"
cd /d "%~dp0.." || exit /b 1
rmdir /s /q build-cuda 2>nul
cmake -S whisper.cpp -B build-cuda -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ^
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 ^
  -DCMAKE_CUDA_COMPILER="%CUDA%\bin\nvcc.exe" ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  -DCMAKE_CUDA_FLAGS="-Xcompiler=/Zc:preprocessor" ^
  -DCMAKE_C_FLAGS="/Zc:preprocessor" -DCMAKE_CXX_FLAGS="/Zc:preprocessor" ^
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF ^
  -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_SERVER=OFF || exit /b 1
REM Cap parallel jobs: the heavy fattn-vec hs256 CUDA template instances can fail
REM (nvcc subprocess exit 1, no diagnostic) under full 8-way Ninja parallelism on
REM CUDA 13 + arch 75. -j 4 builds them reliably.
cmake --build build-cuda --config Release --target whisper -j 4 || exit /b 1
echo === CUDA DLLs ===
dir /s /b build-cuda\bin\*.dll
```
> VERIFIED on the dev box (CUDA 13.3 + VS2022 + arch 75): builds all 5 DLLs; `go build -tags cuda` links; the GPU smoke test transcribes on the GPU. Two CUDA-13-specific flags are MANDATORY: `/Zc:preprocessor` (CUDA 13's CCCL headers `#error` on MSVC's legacy preprocessor) and `-j 4` (the `fattn-vec hs256` flash-attn templates fail under full 8-way Ninja). Mirrors go-llama's Ninja flow otherwise. If a future toolchain breaks Ninja configure, the fallback is the VS generator (`-G "Visual Studio 17 2022" -A x64 -T "cuda=%CUDA%"`) → DLLs in `build-cuda\bin\Release\`, requiring the `Release\` segment in Task 2's link path.

- [ ] **Step 2: Add `build:cuda` to `Taskfile.yml`**

Add under `tasks:` (after `build:cpu`):
```yaml
  build:cuda:
    desc: Build whisper.cpp (CUDA, MSVC DLLs) and the binding — needs VS2022 + CUDA toolkit
    cmds:
      - cmd /c "scripts\\whispercpp-cuda.bat"
      - bash scripts/binding.sh
```
> `binding.sh` (foundation) compiles `binding.cpp -> libbinding.a` against headers only; it is backend-agnostic and reused as-is for CUDA.

- [ ] **Step 3: Commit** (no build yet — that's Task 3)

```bash
git add scripts/whispercpp-cuda.bat Taskfile.yml
git commit -m "build(cuda): add MSVC-DLL CUDA build script + task"
```

---

## Task 2: CUDA link file

**Files:** Create `link_cuda_windows.go`.

- [ ] **Step 1: Write `link_cuda_windows.go`**

Create `link_cuda_windows.go`:
```go
//go:build windows && cuda

package whisper

// CUDA build (-tags cuda): links the MSVC-built whisper.dll (which pulls
// ggml.dll -> ggml-cuda.dll at runtime) via the C ABI only — so the MinGW cgo host
// links an MSVC DLL. Build with scripts\whispercpp-cuda.bat (task build:cuda).
// At RUNTIME, build-cuda\bin\*.dll (whisper, ggml, ggml-base, ggml-cpu, ggml-cuda)
// AND the CUDA toolkit DLLs (cudart64_13.dll, cublas64_13.dll, cublasLt64_13.dll)
// must be on PATH or beside the .exe. (scoop already puts the CUDA bin\x64 on PATH.)
// (Blank line below is REQUIRED so this prose is not part of the cgo preamble.)

// #cgo LDFLAGS: ${SRCDIR}/build-cuda/bin/whisper.dll -lstdc++
import "C"
```
> `${SRCDIR}` resolves to the package dir (repo root), so the DLL must be at `<repo>/build-cuda/bin/whisper.dll`. Linking the DLL by path (no import lib) matches go-llama exactly.

- [ ] **Step 2: Confirm tag exclusivity** (no build)

Run: `gofmt -l link_cuda_windows.go` (empty). Verify `link_static_windows.go` is `windows && !cuda && !vulkan` (so `-tags cuda` deselects it) — already true from the foundation.

- [ ] **Step 3: Commit**

```bash
git add link_cuda_windows.go
git commit -m "build(cuda): add windows cuda link file (whisper.dll by path)"
```

---

## Task 3: Build CUDA, link `-tags cuda`, and GPU smoke test

**Files:** Create `whisper_gpu_test.go`.

- [ ] **Step 1: Write the GPU smoke spec** (shared by cuda + vulkan tags)

Create `whisper_gpu_test.go`:
```go
//go:build cuda || vulkan

package whisper_test

import (
	"context"
	"os"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	whisper "github.com/dyammarcano/go-whisper.cpp"
	"github.com/dyammarcano/go-whisper.cpp/wav"
)

var _ = Describe("GPU backend", Label("gpu"), func() {
	modelPath := os.Getenv("TEST_MODEL")

	It("transcribes on the GPU (WithGPU)", func() {
		if modelPath == "" {
			Skip("set TEST_MODEL to run GPU tests (and build with -tags cuda or -tags vulkan)")
		}
		m, err := whisper.New(modelPath, whisper.WithGPU(true))
		Expect(err).NotTo(HaveOccurred())
		defer m.Close()

		samples, err := wav.ReadFile("whisper.cpp/samples/jfk.wav")
		Expect(err).NotTo(HaveOccurred())

		res, err := m.Transcribe(context.Background(), samples, whisper.WithLanguage("en"))
		Expect(err).NotTo(HaveOccurred())
		full := ""
		for _, s := range res.Segments {
			full += s.Text
		}
		Expect(strings.ToLower(full)).To(ContainSubstring("country"))
	})
})
```

- [ ] **Step 2: Build whisper CUDA DLLs + the binding**

Run (large output → use ctx_execute):
```bash
task build:cuda
```
Expected: `build-cuda/bin/` contains `whisper.dll, ggml.dll, ggml-base.dll, ggml-cpu.dll, ggml-cuda.dll` (list them); `libbinding.a` rebuilt. If Ninja+CUDA13 configure fails, see the Task 1 VS-generator fallback and adjust the Task 2 link path to `build-cuda/bin/Release/whisper.dll`, then re-commit Task 2.

- [ ] **Step 3: Link-build with `-tags cuda`**

Run:
```bash
go build -tags cuda ./...
```
Expected: clean (the cgo package now links `build-cuda/bin/whisper.dll`). If it can't find `whisper.dll` at link time, confirm the path in `link_cuda_windows.go`.

- [ ] **Step 4: Run the GPU smoke test with DLLs on PATH**

Run (TEST_MODEL must be a NATIVE Windows path; prepend the build-cuda DLL dir + CUDA bin\x64 to PATH):
```bash
PATH="$(pwd)/build-cuda/bin:$HOME/scoop/apps/cuda/current/bin/x64:$PATH" \
TEST_MODEL='D:\weaver-sync\development\personal\projects\go-whisper.cpp\models\ggml-tiny.en.bin' \
go test -tags cuda -run TestWhisper ./... -v
```
Expected: the `GPU backend` spec PASSES (transcript contains "country"). whisper's log (via slog at debug, or stderr) should mention the CUDA device. If it loads but runs on CPU, confirm `ggml-cuda.dll` is present in build-cuda/bin and on PATH. If a DLL is missing at runtime you'll get exit code 0xC0000135 / "DLL not found" — add the missing DLL's dir to PATH.
> Report the actual outcome. If CUDA execution cannot be validated on this machine (driver/GPU issue) but the `-tags cuda` BUILD+LINK succeeds, that is DONE_WITH_CONCERNS (build verified; record the runtime blocker) — the build+link is the hard part the binding owns.

- [ ] **Step 5: Commit**

```bash
git add whisper_gpu_test.go
git commit -m "test(gpu): WithGPU smoke spec for cuda/vulkan tags"
```

---

## Task 4: Vulkan build support + link file

**Files:** Modify `scripts/whispercpp.sh` (VULKAN_SDK fallback); Modify `Taskfile.yml`; Create `link_vulkan_windows.go`.

- [ ] **Step 1: Add a VULKAN_SDK fallback to `scripts/whispercpp.sh`**

The foundation `whispercpp.sh` already handles the `vulkan` arg (`EXTRA="-DGGML_VULKAN=ON"`). Add, near the top of the script (after `BACKEND=...`), a fallback so the SDK is found even if the env var isn't exported into the shell:
```bash
# Vulkan: cmake's find_package(Vulkan) needs VULKAN_SDK. It is set at the Windows
# user level by scoop; export a fallback for shells that didn't inherit it.
# cmake on Windows needs a NATIVE path (C:/...), not the MSYS form (/c/...), so
# convert via cygpath when available.
if [ "$BACKEND" = "vulkan" ] && [ -z "${VULKAN_SDK:-}" ]; then
  if [ -d "$HOME/scoop/apps/vulkan/current" ]; then
    if command -v cygpath >/dev/null 2>&1; then
      export VULKAN_SDK="$(cygpath -m "$HOME/scoop/apps/vulkan/current")"
    else
      export VULKAN_SDK="$HOME/scoop/apps/vulkan/current"
    fi
    echo "VULKAN_SDK fallback -> $VULKAN_SDK"
  fi
fi
```
> VERIFIED: without `cygpath -m`, cmake 4.3's FindVulkan fails ("Could NOT find Vulkan") on the MSYS path. With it: "Found Vulkan: …/Lib/vulkan-1.lib (version 1.4.350)".

- [ ] **Step 2: Add `build:vulkan` to `Taskfile.yml`**

Add under `tasks:`:
```yaml
  build:vulkan:
    desc: Build whisper.cpp (Vulkan, MinGW static) and the binding — needs the Vulkan SDK
    cmds:
      - bash scripts/whispercpp.sh vulkan
      - bash scripts/binding.sh
```

- [ ] **Step 3: Write `link_vulkan_windows.go`**

Create `link_vulkan_windows.go`:
```go
//go:build windows && vulkan && !cuda

package whisper

// Vulkan build (-tags vulkan): MinGW static libs from `task build:vulkan`
// (GGML_VULKAN=ON), linked against the system Vulkan loader (vulkan-1.dll in
// System32). ggml*.a have no lib prefix -> full paths; --start-group resolves the
// whisper<->ggml<->ggml-vulkan circular refs. Run `task build:vulkan` first.
// (Blank line below is REQUIRED so this prose is not part of the cgo preamble.)

// #cgo LDFLAGS: -Wl,--start-group ${SRCDIR}/whisper.cpp/build-vulkan/src/libwhisper.a ${SRCDIR}/whisper.cpp/build-vulkan/ggml/src/ggml-vulkan/ggml-vulkan.a ${SRCDIR}/whisper.cpp/build-vulkan/ggml/src/ggml-cpu.a ${SRCDIR}/whisper.cpp/build-vulkan/ggml/src/ggml.a ${SRCDIR}/whisper.cpp/build-vulkan/ggml/src/ggml-base.a -Wl,--end-group C:/Windows/System32/vulkan-1.dll -fopenmp -lstdc++ -lm
import "C"
```
> The Vulkan loader is linked via the system `vulkan-1.dll` directly (MinGW ld accepts a `.dll`), avoiding the MinGW-vs-MSVC `vulkan-1.lib` question. VERIFIED on the dev box: `ggml-vulkan.a` lands in a **nested** subdir (`ggml/src/ggml-vulkan/ggml-vulkan.a`), unlike the other ggml archives — the path above reflects that.

- [ ] **Step 4: Commit** (no build yet)

```bash
git add scripts/whispercpp.sh Taskfile.yml link_vulkan_windows.go
git commit -m "build(vulkan): add Vulkan build task + windows vulkan link file"
```

---

## Task 5: Build Vulkan, link `-tags vulkan`, run GPU smoke

- [ ] **Step 1: Build whisper Vulkan static libs + the binding**

Run (ctx_execute):
```bash
task build:vulkan
```
Expected: `whisper.cpp/build-vulkan/src/libwhisper.a` and `whisper.cpp/build-vulkan/ggml/src/{ggml-vulkan.a, ggml-cpu.a, ggml.a, ggml-base.a}` exist (list them with `ls`). If `find_package(Vulkan)` fails, confirm `VULKAN_SDK` is set (the Step-1 fallback) and that `glslc` is on PATH.

- [ ] **Step 2: Reconcile the link file with actual archive names**

Run: `ls whisper.cpp/build-vulkan/src/*.a whisper.cpp/build-vulkan/ggml/src/*.a`
If any name differs from `link_vulkan_windows.go`, fix the LDFLAGS to match exactly and re-commit. (Expected to match the CPU layout + `ggml-vulkan.a`.)

- [ ] **Step 3: Link-build with `-tags vulkan`**

Run:
```bash
go build -tags vulkan ./...
```
Expected: clean link. Undefined Vulkan symbols → confirm the `C:/Windows/System32/vulkan-1.dll` entry in the LDFLAGS.

- [ ] **Step 4: Run the GPU smoke test (Vulkan)**

Run:
```bash
TEST_MODEL='D:\weaver-sync\development\personal\projects\go-whisper.cpp\models\ggml-tiny.en.bin' \
go test -tags vulkan -run TestWhisper ./... -v
```
Expected: the `GPU backend` spec PASSES (transcript contains "country"); whisper log mentions a Vulkan device. (Vulkan needs no extra runtime DLL beyond the system `vulkan-1.dll`.) Same DONE_WITH_CONCERNS rule as Task 3 Step 4 if the device can't run but build+link succeed.

- [ ] **Step 5: Commit any link reconciliation**

```bash
git add link_vulkan_windows.go
git commit -m "build(vulkan): reconcile link archives with build output" --allow-empty
```

---

## Task 6: Fix macOS Metal link file

**Files:** Modify `link_darwin.go`.

- [ ] **Step 1: Correct `link_darwin.go`** (the default macOS build enables Metal + BLAS, which the foundation's link line omits → it would fail on macOS CI)

Replace the cgo preamble in `link_darwin.go` with:
```go
//go:build darwin

package whisper

// macOS default build enables Metal + Apple BLAS, producing ggml-metal.a + ggml-blas.a.
// GGML_METAL_EMBED_LIBRARY is ON by default, so the .metal shader is embedded in
// ggml-metal.a (no default.metallib needed at runtime). ld64 has no --start-group; list
// archives in dependency order. Frameworks: Foundation/Metal/MetalKit (ggml-metal),
// Accelerate (ggml-blas), QuartzCore (Metal runtime).
// (Blank line below is REQUIRED so this prose is not part of the cgo preamble.)

// #cgo LDFLAGS: ${SRCDIR}/whisper.cpp/build-cpu/src/libwhisper.a ${SRCDIR}/whisper.cpp/build-cpu/ggml/src/ggml-cpu.a ${SRCDIR}/whisper.cpp/build-cpu/ggml/src/ggml-metal.a ${SRCDIR}/whisper.cpp/build-cpu/ggml/src/ggml-blas.a ${SRCDIR}/whisper.cpp/build-cpu/ggml/src/ggml.a ${SRCDIR}/whisper.cpp/build-cpu/ggml/src/ggml-base.a -lstdc++
// #cgo LDFLAGS: -framework Foundation -framework Metal -framework MetalKit -framework Accelerate -framework QuartzCore
import "C"
```
> Cannot be built/verified on the Windows dev box — the macOS CI runner (Task 7) is the verification. If the macOS build emits different archive names (e.g. no `ggml-blas.a` when BLAS is off), CI surfaces it and the names get corrected from the CI log.

- [ ] **Step 2: gofmt + commit**

Run: `gofmt -l link_darwin.go` (empty).
```bash
git add link_darwin.go
git commit -m "build(metal): link ggml-metal/ggml-blas + frameworks on darwin"
```

---

## Task 7: Extend CI — GPU build-verify + fix macOS leg

**Files:** Modify `.github/workflows/ci.yml`.

- [ ] **Step 1: Add a `gpu-build` job** (build/link-verify only; GitHub-hosted runners have no GPU)

Append this job to `.github/workflows/ci.yml` (under `jobs:`):
```yaml
  gpu-build:
    # Build/link-verify GPU tags. GitHub-hosted runners have NO GPU, so this proves the
    # cgo build + link, not execution. Run GPU *execution* tests on a self-hosted runner.
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
        with: { submodules: recursive }
      - uses: actions/setup-go@v5
        with: { go-version: '1.25' }
      - uses: arduino/setup-task@v2
        with: { version: 3.x, repo-token: "${{ secrets.GITHUB_TOKEN }}" }
      - name: Setup MinGW + CMake
        uses: msys2/setup-msys2@v2
        with:
          msystem: UCRT64
          path-type: inherit
          install: >-
            mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake make
      - name: Install Vulkan SDK
        uses: humbletim/install-vulkan-sdk@v1.2
        with: { version: latest, cache: true }
      - name: Build + link-verify Vulkan
        env: { CC: gcc, CXX: g++ }
        run: |
          task build:vulkan
          go build -tags vulkan ./...
      # NOTE: CUDA build-verify needs the CUDA toolkit + MSVC on the runner; deferred to a
      # self-hosted GPU runner (it also runs the GPU execution tests). Documented in README.
```
> The CPU `build-test` matrix job (foundation) already covers ubuntu/macos/windows CPU + the macOS Metal link (now fixed in Task 6). This adds Vulkan build-verify. CUDA build-verify + GPU execution tests run on a self-hosted runner (out of scope for GitHub-hosted CI; documented).

- [ ] **Step 2: Validate YAML + commit**

Validate the YAML parses (python `yaml.safe_load` or equivalent).
```bash
git add .github/workflows/ci.yml
git commit -m "ci: add Vulkan build-verify job; macOS leg now links Metal"
```

---

## Task 8: Docs — backends, build tags, runtime DLLs

**Files:** Modify `README.md`.

- [ ] **Step 1: Add a "GPU backends" section to `README.md`**

Add a section documenting:
- Build: `task build:cuda` (Windows, needs VS2022 + CUDA toolkit) / `task build:vulkan` (Windows, needs Vulkan SDK). macOS Metal builds via the default `task build:cpu` (Metal is on by default).
- Run/build with the tag: `go build -tags cuda` / `go run -tags cuda ./examples/transcribe -m … (WithGPU at load)`; same for `-tags vulkan`.
- Enable the GPU at load: `whisper.New(model, whisper.WithGPU(true))`.
- CUDA runtime: ship `build-cuda/bin/*.dll` (whisper, ggml, ggml-base, ggml-cpu, ggml-cuda) + the toolkit `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` on PATH (or beside the exe). CUDA arch is 75 (Turing) by default — edit `whispercpp-cuda.bat` `CMAKE_CUDA_ARCHITECTURES` for other GPUs.
- Vulkan runtime: only the system `vulkan-1.dll` is needed (no extra DLLs).
- A note that GitHub-hosted CI build-verifies GPU tags but GPU execution tests require a self-hosted GPU runner.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: GPU backends (cuda/vulkan tags, build, runtime DLLs)"
```

---

## Self-review checklist (plan author)

- **Coverage:** CUDA build (T1) + link (T2) + build/run/test (T3); Vulkan build (T4) + build/run/test (T5); Metal link fix (T6); CI (T7); docs (T8). Matches the confirmed scope (CUDA+Vulkan local, Metal CI).
- **Placeholders:** none — bat/link files/test are full content; README task is a concrete section list (doc task).
- **Tag consistency:** `windows && cuda`, `windows && vulkan && !cuda`, `windows && !cuda && !vulkan` (CPU, exists), `darwin`. `-tags cuda` deselects CPU + Vulkan; `-tags vulkan` deselects CPU + (via `!cuda`) yields Vulkan only. `whisper_gpu_test.go` is `cuda || vulkan`.
- **No API change:** GPU is purely a link-layer + `WithGPU(true)` concern; the Go public API is unchanged.

## Risks

- **CUDA Ninja+13 configure:** if it fails, use the VS-generator fallback (Task 1 note) and adjust the link path to `build-cuda/bin/Release/`. The go-llama Ninja precedent makes failure unlikely.
- **Vulkan archive names / loader link:** reconciled at Task 5 Step 2 from the actual build output; loader linked via the system DLL to dodge the MSVC-lib question.
- **Metal link names:** unverifiable on Windows; macOS CI is the gate. `ggml-blas.a` only exists if BLAS is enabled (default) — CI corrects if not.
- **GPU execution in CI:** not possible on GitHub-hosted runners; execution tests are self-hosted/local (documented), build+link is CI-verified.
```
