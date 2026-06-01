package whisper_test

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestWhisper(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "go-whisper.cpp suite")
}
