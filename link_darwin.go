//go:build darwin

package whisper

// macOS: ggml*.a have no lib prefix -> full paths. ld64 has no --start-group; list in
// dependency order (whisper -> ggml-cpu -> ggml -> ggml-base). NOTE: the default macOS
// ggml build also enables the Metal + BLAS backends, which emit ADDITIONAL archives
// (ggml-metal.a, ggml-blas.a) under build-cpu/ggml/src/ and need the Metal/MetalKit/
// Accelerate frameworks. The macos CI runner validates this; if those archives are
// present the implementer adds them to this list (full Metal wiring is in the GPU plan).

// #cgo LDFLAGS: ${SRCDIR}/whisper.cpp/build-cpu/src/libwhisper.a ${SRCDIR}/whisper.cpp/build-cpu/ggml/src/ggml-cpu.a ${SRCDIR}/whisper.cpp/build-cpu/ggml/src/ggml.a ${SRCDIR}/whisper.cpp/build-cpu/ggml/src/ggml-base.a -lstdc++
// #cgo LDFLAGS: -framework Accelerate -framework Metal -framework MetalKit -framework Foundation
import "C"
