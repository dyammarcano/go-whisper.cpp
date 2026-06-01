//go:build windows && !cuda && !vulkan

package whisper

// ggml static archives are emitted WITHOUT a lib prefix (ggml.a, ggml-base.a,
// ggml-cpu.a), so -lggml would NOT resolve — give them by FULL PATH. Only
// libwhisper.a has the prefix. --start-group resolves whisper<->ggml circular refs.
// VERIFIED on the dev box: this exact line links + runs whisper_bind_lang_max_id() (=99).

// #cgo LDFLAGS: -Wl,--start-group ${SRCDIR}/whisper.cpp/build-cpu/src/libwhisper.a ${SRCDIR}/whisper.cpp/build-cpu/ggml/src/ggml-cpu.a ${SRCDIR}/whisper.cpp/build-cpu/ggml/src/ggml.a ${SRCDIR}/whisper.cpp/build-cpu/ggml/src/ggml-base.a -Wl,--end-group -fopenmp -lstdc++ -lm
import "C"
