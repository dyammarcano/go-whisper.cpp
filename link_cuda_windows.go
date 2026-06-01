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
