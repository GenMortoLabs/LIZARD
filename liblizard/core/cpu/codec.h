// ai: The C decoder as the CPU path takes it: liblizard/src's blind receiver (any.h) behind a
// ai: C++-safe face, one object a thread. A frame in (tight 8-bit luma), its verified blocks out.
#ifndef CPU_CODEC_H
#define CPU_CODEC_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct cpu_dec cpu_dec_t;

// ai: What a frame's decode found. ring: the ring that registered (an index into src/lizard_phy.h LIZARD_RING), -1 none. n: the picture it was finished
// ai: at, 0 not finished. held: the frame (its word animated) was finished at the caller's held configuration. word: the
// ai: frame's own word read, kind 0 static (version = sub-channels / 16, count its 7/8, 2/3 and 1/2 blocks) or 1 animated (a carousel symbol, src/fmt.h: id, sym its bytes; version 0). total: the blocks
// ai: of the layout it was finished at (ok[] and blocks[] hold that many; cpu_dec_top the most). quad: the symbol's corners in the image, TL
// ai: TR BR BL. pilot_*: what the block tails read (liblizard/src/any.h lizard_any_pilot: r of the even blocks and of
// ai: the odd, their standard errors, the blocks read, 0 none).
typedef struct {
  int found, ring, n, held, word, version, total;
  float quad[8];
  int pilot_blocks;
  float pilot_r[2], pilot_sd[2];
  int kind, id, count[3];
  uint8_t sym[3];
} cpu_frame_t;

// ai: The held configuration, the one state a receiver carries from frame to frame (src/any.h), for its animated frames:
// ai: a static word's configuration (src/fmt.h ob_fmt_held: version and counts as one int), or that of the header the
// ai: animated words' carousel assembled (src/fmt.h ob_car_rx_t, ob_car_config). take: a frame's word (kind 0 static,
// ai: 1 animated; none, the hold as it was) and returns the configuration held after it, 0 none. fps: the display rate
// ai: the last assembled header states (0 none: a static word states none). One a receiver; reset for a new stream.
typedef struct cpu_hold cpu_hold_t;
cpu_hold_t *cpu_hold_new(void);
void cpu_hold_free(cpu_hold_t *h);
void cpu_hold_reset(cpu_hold_t *h);
int cpu_hold_take(cpu_hold_t *h, int kind, int version, const int count[3], int id, const uint8_t sym[3]);
int cpu_hold_held(const cpu_hold_t *h);
int cpu_hold_fps(const cpu_hold_t *h);

// ai: nmax: the largest picture decoded (1536 for every format; smaller saves memory). NULL where the codec refuses.
cpu_dec_t *cpu_dec_new(int nmax);
void cpu_dec_free(cpu_dec_t *d);
// ai: The most blocks a frame can carry (the size of ok[], and of blocks[] in cpu_dec_block_bytes each).
int cpu_dec_top(const cpu_dec_t *d);
int cpu_dec_block_bytes(const cpu_dec_t *d);
// ai: One frame: img is iw x ih luma, rows iw apart, alive for the call. held: the configuration the caller holds
// ai: (cpu_hold_take), 0 none: a frame whose word is animated is decoded at it; a frame whose own word does not read is
// ai: not decoded (the word before the frame, 2026-11-18).
// ai: Returns the blocks that verified; block b is good where ok[b] is 1.
int cpu_dec_frame(cpu_dec_t *d, const uint8_t *img, int iw, int ih, int held, uint8_t *blocks, uint8_t *ok, cpu_frame_t *out);
// ai: Where the camera's optical axis falls in the next image this thread decodes (NaN: its centre), the lens
// ai: model's (src/lizard_phy.h lizard_set_axis); it stays for the thread's later frames until set again.
void cpu_dec_set_axis(float x, float y);
// ai: The last frame's per-block LDPC iterations and information estimates (out->total of each), for a check.
void cpu_dec_block_stats(const cpu_dec_t *d, const int8_t **its, const float **est);
// ai: The codec's own digest of a frame's decode: arm before cpu_dec_frame, take after (a hash of the floats and soft
// ai: values behind the decode where the frame was finished, else 1). From one thread at a time only.
void cpu_hash_arm(void);
uint32_t cpu_hash_take(void);
// ai: The codec's stage timers for the calling thread (internal.h PROF_*), ms since the last reset, by stage: acquire's
// ai: parts, then the picture's (sample, detrend, transform, LLR, LDPC).
enum { CPU_PROF_BINARIZE, CPU_PROF_MESH, CPU_PROF_POINTS, CPU_PROF_TURN, CPU_PROF_FIT, CPU_PROF_SCORE, CPU_PROF_MARKS, CPU_PROF_TRACK,
       CPU_PROF_SAMPLE, CPU_PROF_DETREND, CPU_PROF_FFT, CPU_PROF_LLR, CPU_PROF_LDPC, CPU_PROF_N };
double cpu_prof_ms(int stage);
void cpu_prof_reset(void);
// ai: 1 where the codec's vector paths are compiled in (simd.h), else 0.
int cpu_simd(void);

#ifdef __cplusplus
}
#endif
#endif
