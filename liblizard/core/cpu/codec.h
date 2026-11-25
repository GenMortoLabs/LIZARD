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
// ai: at, 0 not finished. word: the frame's own word read (version = sub-channels / 16, count its 6/7, 3/4, 2/3 and 1/2 blocks; fps
// ai: and delay the sender's clock, src/fmt.h). total: the blocks
// ai: of the layout it was finished at (ok[] and blocks[] hold that many; cpu_dec_top the most). quad: the symbol's corners in the image, TL
// ai: TR BR BL. pilot_*: what the pilots read (liblizard/src/any.h lizard_any_pilot: r of the even sub-channels and of
// ai: the odd, their standard errors, the sub-channels read, 0 none).
typedef struct {
  int found, ring, n, word, version, total;
  float quad[8];
  int pilot_subch;
  float pilot_r[2], pilot_sd[2];
  int count[4];
  int fps, delay;
} cpu_frame_t;

// ai: nmax: the largest picture decoded (1536 for every format; smaller saves memory). NULL where the codec refuses.
cpu_dec_t *cpu_dec_new(int nmax);
void cpu_dec_free(cpu_dec_t *d);
// ai: The most blocks a frame can carry (the size of ok[], and of blocks[] in cpu_dec_block_bytes each).
int cpu_dec_top(const cpu_dec_t *d);
int cpu_dec_block_bytes(const cpu_dec_t *d);
// ai: One frame: img is iw x ih luma, rows iw apart, alive for the call. A frame whose own word does not read is not
// ai: decoded (the word before the frame, 2026-11-18); nothing carries from one frame to the next.
// ai: Returns the blocks that verified; block b is good where ok[b] is 1.
int cpu_dec_frame(cpu_dec_t *d, const uint8_t *img, int iw, int ih, uint8_t *blocks, uint8_t *ok, cpu_frame_t *out);
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
enum { CPU_PROF_GREY, CPU_PROF_MESH, CPU_PROF_TRACE, CPU_PROF_LENS, CPU_PROF_SCORE, CPU_PROF_TRACK,
       CPU_PROF_SAMPLE, CPU_PROF_DETREND, CPU_PROF_FFT, CPU_PROF_LLR, CPU_PROF_LDPC, CPU_PROF_N };
double cpu_prof_ms(int stage);
void cpu_prof_reset(void);
// ai: 1 where the codec's vector paths are compiled in (simd.h), else 0.
int cpu_simd(void);

#ifdef __cplusplus
}
#endif
#endif
