//go:build windows && vulkan && !cuda

package whisper

// Vulkan build (-tags vulkan): MinGW static libs from `task build:vulkan`
// (GGML_VULKAN=ON), linked against the system Vulkan loader (vulkan-1.dll in
// System32). ggml*.a have no lib prefix -> full paths; --start-group resolves the
// whisper<->ggml<->ggml-vulkan circular refs. Run `task build:vulkan` first.
// (Blank line below is REQUIRED so this prose is not part of the cgo preamble.)

// #cgo LDFLAGS: -Wl,--start-group ${SRCDIR}/whisper.cpp/build-vulkan/src/libwhisper.a ${SRCDIR}/whisper.cpp/build-vulkan/ggml/src/ggml-vulkan.a ${SRCDIR}/whisper.cpp/build-vulkan/ggml/src/ggml-cpu.a ${SRCDIR}/whisper.cpp/build-vulkan/ggml/src/ggml.a ${SRCDIR}/whisper.cpp/build-vulkan/ggml/src/ggml-base.a -Wl,--end-group C:/Windows/System32/vulkan-1.dll -fopenmp -lstdc++ -lm
import "C"
