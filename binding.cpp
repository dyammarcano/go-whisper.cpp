// go-whisper.cpp binding — thin C shim over whisper.h's C API.
// Owns whisper_full_params/whisper_context_params construction (by-value structs
// never cross cgo) and hosts callback trampolines that call exported Go funcs.
#include "binding.h"
#include "whisper.h"
#include "ggml.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" {
    // These prototypes MUST match cgo's generated _cgoexp_ signatures for the
    // //export funcs in callback.go / log.go. cgo emits non-const char* for *C.char,
    // uintptr_t for C.uintptr_t, int64_t for C.int64_t, int for C.int. binding.cpp is
    // compiled OUTSIDE cgo (by scripts/binding.sh), so it needs its own decls.
    void goWhisperSegment(uintptr_t handle, int64_t t0, int64_t t1, char* text);
    void goWhisperProgress(uintptr_t handle, int progress);
    int  goWhisperAbort(uintptr_t handle);
    void goWhisperLog(int level, char* text);
}

// seg_tramp reads ONLY the newly-added segments [total-n_new, total) from the live
// state/context it is handed (the documented, upstream-idiomatic use of
// new_segment_callback) and hands each to Go directly. It does NOT re-collect all
// segments and never re-enters via the Go Model pointer — avoiding the reentrancy
// hazard and the O(n^2) allocation of snapshotting on every callback.
static void seg_tramp(struct whisper_context* c, struct whisper_state* st, int n_new, void* ud) {
    int total = st ? whisper_full_n_segments_from_state(st) : whisper_full_n_segments(c);
    for (int i = total - n_new; i < total; ++i) {
        if (i < 0) continue;
        int64_t t0 = st ? whisper_full_get_segment_t0_from_state(st, i) : whisper_full_get_segment_t0(c, i);
        int64_t t1 = st ? whisper_full_get_segment_t1_from_state(st, i) : whisper_full_get_segment_t1(c, i);
        const char* txt = st ? whisper_full_get_segment_text_from_state(st, i) : whisper_full_get_segment_text(c, i);
        goWhisperSegment((uintptr_t)ud, t0, t1, (char*)(txt ? txt : ""));
    }
}
static void prog_tramp(struct whisper_context*, struct whisper_state*, int progress, void* ud) {
    goWhisperProgress((uintptr_t)ud, progress);
}
static bool abort_tramp(void* ud) {
    return goWhisperAbort((uintptr_t)ud) != 0;
}
static void log_tramp(enum ggml_log_level level, const char* text, void*) {
    goWhisperLog((int)level, (char*)text);
}

extern "C" void whisper_bind_install_log(void) { whisper_log_set(log_tramp, nullptr); }

extern "C" void* whisper_bind_load_model(const char* path, int use_gpu, int flash_attn, int gpu_device) {
    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu    = use_gpu != 0;
    cp.flash_attn = flash_attn != 0;
    cp.gpu_device = gpu_device;
    return (void*) whisper_init_from_file_with_params(path, cp);
}
extern "C" void  whisper_bind_free_model(void* ctx) { if (ctx) whisper_free((struct whisper_context*)ctx); }
extern "C" void* whisper_bind_new_state(void* ctx) {
    if (!ctx) return nullptr;
    return (void*) whisper_init_state((struct whisper_context*)ctx);
}
extern "C" void  whisper_bind_free_state(void* st) { if (st) whisper_free_state((struct whisper_state*)st); }

static whisper_full_params build_params(const whisper_bind_params* p) {
    whisper_full_params wp = whisper_full_default_params(
        p->strategy == 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
    if (p->n_threads > 0) wp.n_threads = p->n_threads;
    wp.translate       = p->translate != 0;
    wp.language        = (p->language && p->language[0]) ? p->language : "auto";
    wp.detect_language = p->detect_language != 0;
    if (p->beam_size > 0) wp.beam_search.beam_size = p->beam_size;
    if (p->best_of   > 0) wp.greedy.best_of        = p->best_of;
    wp.temperature      = p->temperature;
    wp.temperature_inc  = p->temperature_inc;
    wp.entropy_thold    = p->entropy_thold;
    wp.logprob_thold    = p->logprob_thold;
    wp.no_speech_thold  = p->no_speech_thold;
    wp.no_context       = p->no_context != 0;
    wp.single_segment   = p->single_segment != 0;
    wp.token_timestamps = p->token_timestamps != 0;
    if (p->max_len   > 0) wp.max_len    = p->max_len;
    wp.split_on_word    = p->split_on_word != 0;
    if (p->max_tokens> 0) wp.max_tokens = p->max_tokens;
    wp.offset_ms        = p->offset_ms;
    wp.duration_ms      = p->duration_ms;
    if (p->audio_ctx > 0) wp.audio_ctx  = p->audio_ctx;
    wp.suppress_blank   = p->suppress_blank != 0;
    wp.suppress_nst     = p->suppress_nst != 0;
    if (p->initial_prompt && p->initial_prompt[0]) wp.initial_prompt = p->initial_prompt;
    wp.print_progress = false; wp.print_realtime = false;
    wp.print_timestamps = false; wp.print_special = false;
    if (p->segment_cb)  { wp.new_segment_callback   = seg_tramp;  wp.new_segment_callback_user_data   = (void*)(uintptr_t)p->segment_cb; }
    if (p->progress_cb) { wp.progress_callback      = prog_tramp; wp.progress_callback_user_data      = (void*)(uintptr_t)p->progress_cb; }
    if (p->abort_cb)    { wp.abort_callback         = abort_tramp;wp.abort_callback_user_data         = (void*)(uintptr_t)p->abort_cb; }
    return wp;
}

extern "C" int whisper_bind_full(void* ctx, void* state, const whisper_bind_params* p,
                                 const float* samples, int n_samples) {
    whisper_full_params wp = build_params(p);
    if (state) return whisper_full_with_state((struct whisper_context*)ctx, (struct whisper_state*)state, wp, samples, n_samples);
    return whisper_full((struct whisper_context*)ctx, wp, samples, n_samples);
}

extern "C" whisper_bind_result* whisper_bind_get_result(void* ctx, void* state, int want_tokens) {
    struct whisper_context* c = (struct whisper_context*)ctx;
    struct whisper_state*   s = (struct whisper_state*)state;
    int n = s ? whisper_full_n_segments_from_state(s) : whisper_full_n_segments(c);
    whisper_bind_result* r = (whisper_bind_result*)calloc(1, sizeof(whisper_bind_result));
    r->n_segments = n;
    r->lang_id = s ? whisper_full_lang_id_from_state(s) : whisper_full_lang_id(c);
    r->segments = (whisper_bind_segment*)calloc(n > 0 ? (size_t)n : 1, sizeof(whisper_bind_segment));
    for (int i = 0; i < n; ++i) {
        const char* txt = s ? whisper_full_get_segment_text_from_state(s, i) : whisper_full_get_segment_text(c, i);
        r->segments[i].t0   = s ? whisper_full_get_segment_t0_from_state(s, i) : whisper_full_get_segment_t0(c, i);
        r->segments[i].t1   = s ? whisper_full_get_segment_t1_from_state(s, i) : whisper_full_get_segment_t1(c, i);
        r->segments[i].text = strdup(txt ? txt : "");
        if (want_tokens) {
            int nt = s ? whisper_full_n_tokens_from_state(s, i) : whisper_full_n_tokens(c, i);
            r->segments[i].n_tokens = nt;
            r->segments[i].tokens = (whisper_bind_token*)calloc(nt > 0 ? (size_t)nt : 1, sizeof(whisper_bind_token));
            for (int j = 0; j < nt; ++j) {
                whisper_token_data td = s ? whisper_full_get_token_data_from_state(s, i, j) : whisper_full_get_token_data(c, i, j);
                r->segments[i].tokens[j].t0 = td.t0;
                r->segments[i].tokens[j].t1 = td.t1;
                r->segments[i].tokens[j].p  = td.p;
                // whisper_token_to_str(ctx, token) is state-independent; td.id came from token_data.
                const char* tt = whisper_token_to_str(c, td.id);
                r->segments[i].tokens[j].text = strdup(tt ? tt : "");
            }
        }
    }
    return r;
}
extern "C" void whisper_bind_free_result(whisper_bind_result* r) {
    if (!r) return;
    for (int i = 0; i < r->n_segments; ++i) {
        free((void*)r->segments[i].text);
        if (r->segments[i].tokens) {
            for (int j = 0; j < r->segments[i].n_tokens; ++j) free((void*)r->segments[i].tokens[j].text);
            free(r->segments[i].tokens);
        }
    }
    free(r->segments);
    free(r);
}

extern "C" int         whisper_bind_lang_id(const char* lang) { return whisper_lang_id(lang); }
extern "C" const char* whisper_bind_lang_str(int id)          { return whisper_lang_str(id); }
extern "C" int         whisper_bind_lang_max_id(void)         { return whisper_lang_max_id(); }
