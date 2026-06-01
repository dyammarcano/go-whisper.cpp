// Package whisper provides Go bindings for whisper.cpp speech-to-text.
//
// A Model wraps a loaded whisper_context (shared, read-only). A Session wraps a
// whisper_state for a single inference; create one Session per goroutine for
// concurrent transcription over one shared Model.
package whisper
