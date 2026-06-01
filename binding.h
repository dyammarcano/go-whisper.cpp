#ifndef GO_WHISPER_BINDING_H
#define GO_WHISPER_BINDING_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

void* whisper_bind_load_model(const char* path, int use_gpu, int flash_attn, int gpu_device);
void  whisper_bind_free_model(void* ctx);
void* whisper_bind_new_state(void* ctx);
void  whisper_bind_free_state(void* state);

typedef struct {
    int          strategy;          // 0 = greedy, 1 = beam_search
    int          n_threads;
    int          translate;
    const char*  language;          // "auto"/""/NULL -> autodetect
    int          detect_language;
    int          beam_size;         // <=0 -> default
    int          best_of;           // <=0 -> default
    float        temperature;
    float        temperature_inc;
    float        entropy_thold;
    float        logprob_thold;
    float        no_speech_thold;
    int          no_context;
    int          single_segment;
    int          token_timestamps;
    int          max_len;
    int          split_on_word;
    int          max_tokens;
    int          offset_ms;
    int          duration_ms;
    int          audio_ctx;
    int          suppress_blank;
    int          suppress_nst;
    const char*  initial_prompt;    // NULL -> none
    uintptr_t    segment_cb;        // cgo.Handle (0 = none)
    uintptr_t    progress_cb;       // cgo.Handle (0 = none)
    uintptr_t    abort_cb;          // cgo.Handle (0 = none)
} whisper_bind_params;

// Returns 0 on success, whisper rc on failure, -100 if aborted via abort_cb.
int whisper_bind_full(void* ctx, void* state, const whisper_bind_params* p,
                      const float* samples, int n_samples);

typedef struct { int64_t t0, t1; float p; const char* text; } whisper_bind_token;
typedef struct {
    int64_t t0, t1;                 // centiseconds
    const char* text;               // owned by result (strdup'd)
    int n_tokens;
    whisper_bind_token* tokens;     // NULL unless want_tokens
} whisper_bind_segment;
typedef struct {
    int n_segments;
    whisper_bind_segment* segments;
    int lang_id;                    // detected/used language id (-1 if n/a)
} whisper_bind_result;

whisper_bind_result* whisper_bind_get_result(void* ctx, void* state, int want_tokens);
void whisper_bind_free_result(whisper_bind_result* r);

int         whisper_bind_lang_id(const char* lang);
const char* whisper_bind_lang_str(int id);
int         whisper_bind_lang_max_id(void);

void whisper_bind_install_log(void);

#ifdef __cplusplus
}
#endif
#endif
