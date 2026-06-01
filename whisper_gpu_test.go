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
