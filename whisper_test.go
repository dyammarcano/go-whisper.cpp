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
