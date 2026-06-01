package whisper_test

import (
	"context"
	"os"
	"strings"
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
		defer func() { _ = m.Close() }()

		samples, err := wav.ReadFile(audioPath)
		Expect(err).NotTo(HaveOccurred())

		res, err := m.Transcribe(context.Background(), samples, whisper.WithLanguage("en"))
		Expect(err).NotTo(HaveOccurred())
		Expect(res.Segments).NotTo(BeEmpty())
		var full strings.Builder
		for _, s := range res.Segments {
			full.WriteString(s.Text)
		}
		Expect(full.String()).To(ContainSubstring("country")) // jfk.wav: "...what you can do for your country"
	})

	It("produces non-zero monotonic timestamps", func() {
		if modelPath == "" {
			Skip("set TEST_MODEL")
		}
		m, err := whisper.New(modelPath)
		Expect(err).NotTo(HaveOccurred())
		defer func() { _ = m.Close() }()
		samples, err := wav.ReadFile("whisper.cpp/samples/jfk.wav")
		Expect(err).NotTo(HaveOccurred())

		res, err := m.Transcribe(context.Background(), samples, whisper.WithLanguage("en"))
		Expect(err).NotTo(HaveOccurred())
		Expect(res.Segments).NotTo(BeEmpty())
		// Locks the centisecond->Duration (x10ms) conversion: end > start > 0.
		Expect(res.Segments[0].End).To(BeNumerically(">", res.Segments[0].Start))
		Expect(res.Segments[0].End).To(BeNumerically(">", time.Duration(0)))
	})

	It("emits token timestamps", func() {
		if modelPath == "" {
			Skip("set TEST_MODEL")
		}
		m, err := whisper.New(modelPath)
		Expect(err).NotTo(HaveOccurred())
		defer func() { _ = m.Close() }()
		samples, err := wav.ReadFile("whisper.cpp/samples/jfk.wav")
		Expect(err).NotTo(HaveOccurred())

		res, err := m.Transcribe(context.Background(), samples,
			whisper.WithLanguage("en"), whisper.WithTokenTimestamps)
		Expect(err).NotTo(HaveOccurred())
		Expect(res.Segments).NotTo(BeEmpty())
		Expect(res.Segments[0].Tokens).ToNot(BeEmpty())
	})

	It("cancels an in-flight transcription", func() {
		if modelPath == "" {
			Skip("set TEST_MODEL")
		}
		m, err := whisper.New(modelPath)
		Expect(err).NotTo(HaveOccurred())
		defer func() { _ = m.Close() }()
		samples, err := wav.ReadFile("whisper.cpp/samples/jfk.wav")
		Expect(err).NotTo(HaveOccurred())

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // already canceled
		_, err = m.Transcribe(ctx, samples, whisper.WithLanguage("en"))
		Expect(err).To(MatchError(whisper.ErrCanceled))
	})

	It("cancels mid-inference within a bound", func() {
		if modelPath == "" {
			Skip("set TEST_MODEL")
		}
		m, err := whisper.New(modelPath)
		Expect(err).NotTo(HaveOccurred())
		defer func() { _ = m.Close() }()
		samples, err := wav.ReadFile("whisper.cpp/samples/jfk.wav")
		Expect(err).NotTo(HaveOccurred())

		ctx, cancel := context.WithCancel(context.Background())
		time.AfterFunc(50*time.Millisecond, cancel)
		errCh := make(chan error, 1)
		go func() {
			_, e := m.Transcribe(ctx, samples, whisper.WithLanguage("en"))
			errCh <- e
		}()
		Eventually(errCh, 10*time.Second).Should(Receive(MatchError(whisper.ErrCanceled)))
	})

	It("runs concurrent sessions over one model", func() {
		if modelPath == "" {
			Skip("set TEST_MODEL")
		}
		m, err := whisper.New(modelPath)
		Expect(err).NotTo(HaveOccurred())
		defer func() { _ = m.Close() }()
		samples, err := wav.ReadFile("whisper.cpp/samples/jfk.wav")
		Expect(err).NotTo(HaveOccurred())

		done := make(chan error, 3)
		for range 3 {
			go func() {
				s, e := m.NewSession()
				if e != nil {
					done <- e
					return
				}
				defer func() { _ = s.Close() }()
				_, e = s.Transcribe(context.Background(), samples, whisper.WithLanguage("en"))
				done <- e
			}()
		}
		for range 3 {
			Eventually(done, 60*time.Second).Should(Receive(BeNil()))
		}
	})
})
