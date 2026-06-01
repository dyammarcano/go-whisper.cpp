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
