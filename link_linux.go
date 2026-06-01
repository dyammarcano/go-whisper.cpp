//go:build linux && !cuda && !vulkan

package whisper

// Same cmake layout as Windows: ggml*.a have no lib prefix -> full paths. Linux adds
// -lpthread -ldl. CI (ubuntu runner) validates this exact line.
//
// #cgo LDFLAGS: -Wl,--start-group ${SRCDIR}/whisper.cpp/build-cpu/src/libwhisper.a ${SRCDIR}/whisper.cpp/build-cpu/ggml/src/ggml-cpu.a ${SRCDIR}/whisper.cpp/build-cpu/ggml/src/ggml.a ${SRCDIR}/whisper.cpp/build-cpu/ggml/src/ggml-base.a -Wl,--end-group -fopenmp -lstdc++ -lm -lpthread -ldl
import "C"
