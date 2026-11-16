// ai: codec.h over liblizard/src/any.h. The web decodes blind with gamma 1 (sim/ob.mjs LizardAny.decode), and so does
// ai: this.
#include "codec.h"
#include "any.h"
#include "internal.h"
#include <stdlib.h>
#include <string.h>

struct cpu_dec { lizard_any_t any; ob_result_t res; int block_bytes; };

cpu_dec_t *cpu_dec_new(int nmax) {
  cpu_dec_t *d = calloc(1, sizeof *d);
  if (!d) return NULL;
  d->any.which = -1;
  d->block_bytes = lizard_any_init(&d->any, nmax);
  if (d->block_bytes < 0) { free(d); return NULL; }
  return d;
}
void cpu_dec_free(cpu_dec_t *d) {
  if (!d) return;
  lizard_any_free(&d->any);
  free(d);
}
void cpu_dec_set_axis(float x, float y) { lizard_set_axis(x, y); }
int cpu_dec_top(const cpu_dec_t *d) { return lizard_any_top(&d->any); }
int cpu_dec_block_bytes(const cpu_dec_t *d) { return d->block_bytes; }

int cpu_dec_frame(cpu_dec_t *d, const uint8_t *img, int iw, int ih, int held, uint8_t *blocks, uint8_t *ok, cpu_frame_t *out) {
  memset(&d->res, 0, sizeof d->res);
  const int got = lizard_any_frame(&d->any, img, iw, ih, 1.0f, blocks, ok, held, &d->res);
  if (out) {
    const ob_fmt_t *fm = lizard_any_word(&d->any);
    out->found = d->res.found; out->ring = d->any.which; out->n = d->any.n; out->held = d->any.held;
    out->word = fm != 0; out->version = fm ? fm->version : 0; out->fps = fm ? fm->fps : 0;
    out->kind = fm ? fm->kind : 0; out->pos = fm ? fm->pos : 0;
    memcpy(out->ext, fm ? fm->ext : (const uint8_t[2]){ 0, 0 }, 2);
    memcpy(out->sym, fm ? fm->sym : (const uint8_t[3]){ 0, 0, 0 }, 3);
    out->total = d->any.last ? d->any.last->blocks : 0;
    memcpy(out->quad, d->res.quad, sizeof out->quad);
    out->pilot_blocks = lizard_any_pilot(&d->any, out->pilot_r, out->pilot_sd);
  }
  return got;
}
struct cpu_hold { ob_car_rx_t car; int version; };
cpu_hold_t *cpu_hold_new(void) { return calloc(1, sizeof(cpu_hold_t)); }
void cpu_hold_free(cpu_hold_t *h) { free(h); }
void cpu_hold_reset(cpu_hold_t *h) { ob_car_reset(&h->car); h->version = 0; }
int cpu_hold_version(const cpu_hold_t *h) { return h->version; }
int cpu_hold_take(cpu_hold_t *h, int kind, int version, int pos, const uint8_t sym[3]) {
  if (kind == OB_FMT_STATIC) {
    if (version >= OB_FMT_VERSION_MIN && version <= OB_FMT_VERSION_MAX) h->version = version;
  } else if (kind == OB_FMT_ANIM) {
    const int k = ob_car_add(&h->car, pos, sym), v = k ? h->car.hdr[0] : 0;
    if (v >= OB_FMT_VERSION_MIN && v <= OB_FMT_VERSION_MAX) h->version = v;
  }
  return h->version;
}
void cpu_dec_block_stats(const cpu_dec_t *d, const int8_t **its, const float **est) {
  *its = 0; *est = 0;
  if (d->any.last) lizard_block_stats(d->any.last, its, est);
}
// ai: lizard_phy.c's test hook, defined here for every build, the wasm module's included (src/wasm.c arms it through
// ai: ob_dbg_hash): armed (nonzero) before a frame, it comes back as a hash of the floats and soft values behind that
// ai: frame's decode. One global, so it is armed from one thread only (the check tool's single-thread runs); the pool
// ai: never arms it.
uint32_t ob_debug_hash;
void cpu_hash_arm(void) { ob_debug_hash = 1; }
uint32_t cpu_hash_take(void) { const uint32_t v = ob_debug_hash; ob_debug_hash = 0; return v; }
// ai: codec.h's stages to internal.h's slots
static const int PROF_OF[CPU_PROF_N] = {PROF_BINARIZE, PROF_MESH, PROF_FF_POINTS, PROF_FF_TURN, PROF_FF_FIT, PROF_FF_SCORE, PROF_FF_MARKS, PROF_FF_TRACK,
                                        PROF_F_SAMPLE, PROF_F_DETREND, PROF_F_FFT, PROF_F_LLR, PROF_F_LDPC};
double cpu_prof_ms(int stage) { return stage >= 0 && stage < CPU_PROF_N ? ob_prof_ms[PROF_OF[stage]] : 0; }
void cpu_prof_reset(void) { for (int i = 0; i < PROF_N; i++) ob_prof_ms[i] = 0; }
int cpu_simd(void) {
#ifdef OB_SIMD
  return 1;
#else
  return 0;
#endif
}
