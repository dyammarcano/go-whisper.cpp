# Architecture

`go-whisper.cpp` is a cgo binding over [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
plus an independent sherpa-onnx diarization package. Diagrams below reflect the actual code.

## System overview

```mermaid
flowchart TB
    subgraph app["Consumer code / examples"]
        EX["examples: transcribe · diarize · transcribe-diarize · stream"]
    end

    subgraph gowhisper["module github.com/dyammarcano/go-whisper.cpp"]
        subgraph core["package whisper (root)"]
            MODEL["Model (shared whisper_context)"]
            SESSION["Session (per-goroutine whisper_state)"]
            STREAM["Stream (sliding-window worker)"]
            OPTS["options / result / errors / callback"]
        end
        WAV["package wav (16 kHz mono PCM reader)"]
        subgraph dia["package diarize"]
            DZ["Diarizer"]
            LABEL["Label (overlap merge, local Segment type)"]
        end
    end

    subgraph cgo["cgo boundary"]
        SHIM["binding.cpp / binding.h (thin C shim)"]
        LINK["link_*.go (build-tag LDFLAGS)"]
    end

    subgraph native["native libraries"]
        WCPP["whisper.cpp v1.7.4"]
        GGML["ggml backends: CPU · CUDA · Vulkan · Metal"]
        SHERPA["sherpa-onnx + ONNX Runtime"]
        MODELS["pyannote seg-3.0 + WeSpeaker (ONNX)"]
    end

    EX --> MODEL & WAV & DZ & LABEL
    EX --> STREAM
    MODEL --> SESSION --> STREAM
    core --> SHIM
    SHIM --> WCPP --> GGML
    LINK -. selects backend .-> GGML
    DZ --> SHERPA --> MODELS
    WAV -- "[]float32 16kHz" --> MODEL
    WAV -- "[]float32 16kHz" --> DZ
    LABEL -. "whisper.Segment -> diarize.Segment" .-> DZ
```

The `diarize` package never imports `whisper`; callers map `whisper.Segment` →
`diarize.Segment` in two lines, keeping transcription and diarization independently linkable.

## Transcription flow (sequence)

```mermaid
sequenceDiagram
    participant C as Caller
    participant S as Session
    participant R as runFull
    participant Shim as binding.cpp
    participant W as whisper.cpp/ggml

    C->>S: Transcribe(ctx, samples, opts...)
    S->>R: runFull(ctx, model, session, samples, opts)
    R->>R: build C params + cgo.Handle (callbacks/abort)
    R->>R: spawn ctx watcher goroutine
    R->>Shim: whisper_bind_full(ctx, state, params, samples, n)
    Shim->>W: whisper_full_with_state(...)
    loop per ggml graph node (ith==0)
        W->>Shim: abort_callback
        Shim->>R: goWhisperAbort -> aborted flag?
    end
    alt ctx cancelled mid-inference
        R-->>C: ErrCanceled
    else success
        W-->>Shim: rc=0
        Shim->>W: whisper_bind_get_result(state)
        Shim-->>R: segments (+token timestamps)
        R-->>C: *Result
    end
```

## Streaming flow (sequence)

```mermaid
sequenceDiagram
    participant P as Producer goroutine
    participant B as streamBuffer
    participant Wk as Stream worker
    participant S as Session
    participant Cons as Consumer

    P->>B: Write(chunk)  %% blocks (backpressure) or drops oldest
    P->>B: CloseSend()   %% after last chunk
    loop every step of new audio (or buffer >= window)
        Wk->>B: nextWindow(since, step, window)
        B-->>Wk: trailing window + windowStart + consumed
        Wk->>S: Transcribe(window, initial_prompt=committed text)
        S-->>Wk: segments
        Wk->>Wk: classifyWindow -> finals (once) + partials
        Wk->>B: slide(committed - keep)
        Wk-->>Cons: StreamResult{Partial|final}
    end
    Note over Wk: defer st.cancel() on exit -> stops buffer ctx goroutine (no leak)
    Wk-->>Cons: close(Results); Err() valid
```

## Diarization + merge flow (sequence)

```mermaid
sequenceDiagram
    participant C as Caller
    participant D as Diarizer
    participant Sh as sherpa-onnx
    participant L as Label

    C->>D: Diarize(samples 16kHz mono)
    D->>Sh: OfflineSpeakerDiarization (pyannote seg + WeSpeaker embed)
    Sh-->>D: speaker turns
    D-->>C: []Turn{Start,End,Speaker}
    C->>L: Label(segs []Segment, turns []Turn)
    L-->>C: []LabeledSegment (each seg -> speaker of max temporal overlap, -1 if none)
```

## Concurrency model

```mermaid
flowchart LR
    M["Model — one read-only whisper_context (whisper_init_from_file)"]
    M --> S1["Session 1 — whisper_state (goroutine 1)"]
    M --> S2["Session 2 — whisper_state (goroutine 2)"]
    M --> SN["Session N — whisper_state (goroutine N)"]
    S1 --> ST1["Stream (optional) — 1 worker, owns its Session"]
    note["Model.Transcribe uses an internal state behind a mutex (one-shot/CLI).<br/>Sessions give true parallel inference via whisper_full_with_state."]
```

## Build-tag backend selection

CPU is the default. GPU backends are opt-in via build tags, each with its own link file:

| Build tag | File | Backend | Linkage |
|-----------|------|---------|---------|
| _(none)_ `windows` | `link_static_windows.go` | CPU static | `whisper.cpp/build-cpu/*.a`, `-fopenmp` off on MinGW |
| `cuda` | `link_cuda_windows.go` | CUDA | `build-cuda/bin/whisper.dll` (MSVC) |
| `vulkan` | `link_vulkan_windows.go` | Vulkan | `build-vulkan/*.a` + `vulkan-1.dll` |
| `linux` | `link_linux.go` | CPU static | `build-cpu/*.a` |
| `darwin` | `link_darwin.go` | Metal/Accelerate | `build-cpu/*.a` + frameworks |

See [ADR-0001](adr/0001-cgo-binding-over-whisper-cpp.md) for the binding strategy,
[ADR-0002](adr/0002-native-diarization-sherpa-onnx.md) for diarization, and
[ADR-0003](adr/0003-streaming-sliding-window.md) for streaming.
