// src/rq.c against the raptorq crate, and its decoder under loss.
//   vectors <file>...   each file a block and the crate's symbols for a fixed ESI list (scripts/exp/raptorq_lab
//                       `dump`): every symbol equal, byte for byte
//   loss <bytes> <keep> <trials> [seed]  trials of a receiver from a first ESI uniform below 2^18, ESIs in order (round
//                       at 2^18), each kept with probability keep, fed until whole: the bytes equal the block's, the
//                       count past K a histogram
//   time <bytes> <n>    the encoder's solve, and a decode from K + 32 of a random start with a quarter lost, ms
// Build: gcc -O2 -ffp-contract=off -Iliblizard/src -o liblizard/build/rq_test liblizard/test/rq_test.c liblizard/src/rq.c
#include "rq.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ai: T the raptorq crate's vectors' symbol size (rq.c takes any); IDS the transfer's symbol field, 2^24 (src/xfer.h)
enum { T = 472, IDS = 1 << 24 };

static uint32_t mul32(uint32_t *a) {
  *a += 0x6d2b79f5u;
  uint32_t t = *a;
  t = (t ^ (t >> 15)) * (t | 1);
  t ^= t + (t ^ (t >> 7)) * (t | 61);
  return t ^ (t >> 14);
}
static double unit(uint32_t *a) { return mul32(a) / 4294967296.0; }
static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static uint32_t u32le(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }

static int vectors(int argc, char **argv) {
  int bad = 0;
  for (int f = 0; f < argc; f++) {
    FILE *fp = fopen(argv[f], "rb");
    if (!fp) { fprintf(stderr, "%s: cannot open\n", argv[f]); return 2; }
    fseek(fp, 0, SEEK_END);
    const long n = ftell(fp);
    rewind(fp);
    uint8_t *b = malloc((size_t)n);
    if (fread(b, 1, (size_t)n, fp) != (size_t)n) { fprintf(stderr, "%s: short\n", argv[f]); return 2; }
    fclose(fp);
    const uint32_t bytes = u32le(b), count = u32le(b + 4);
    const uint8_t *data = b + 8, *s = data + bytes;
    const double t0 = now_ms();
    rq_enc_t *e = rq_enc_new(data, bytes, T);
    const double t1 = now_ms();
    if (!e) { printf("%s: no encoder\n", argv[f]); bad++; free(b); continue; }
    uint8_t out[T];
    int miss = 0;
    for (uint32_t k = 0; k < count; k++, s += 4 + T) {
      rq_enc_symbol(e, u32le(s), out);
      if (memcmp(out, s + 4, T)) {
        if (miss < 4) printf("  ESI %u differs\n", u32le(s));
        miss++;
      }
    }
    printf("%s: %u B (K %u), %u symbols, %d differ (encoder %.1f ms)\n", argv[f], bytes, rq_symbols(bytes, T), count, miss, t1 - t0);
    bad += miss > 0;
    rq_enc_free(e), free(b);
  }
  printf("%s\n", bad ? "FAIL" : "PASS");
  return bad != 0;
}

static int loss(uint32_t bytes, double keep, int trials, uint32_t seed) {
  uint32_t r = seed * 2654435761u + bytes;
  uint8_t *data = malloc(bytes), *back = malloc(bytes), *syms = malloc((size_t)IDS * T);
  for (uint32_t i = 0; i < bytes; i++) data[i] = (uint8_t)mul32(&r);
  rq_enc_t *e = rq_enc_new(data, bytes, T);
  if (!e) { printf("no encoder\n"); return 2; }
  for (uint32_t id = 0; id < IDS; id++) rq_enc_symbol(e, id, syms + (size_t)id * T);
  rq_enc_free(e);
  const uint32_t K = rq_symbols(bytes, T);
  int hist[4] = {0}, worst = 0, wrong = 0;
  double ms = 0;
  for (int t = 0; t < trials; t++) {
    rq_dec_t *d = rq_dec_new(bytes, T, IDS);
    uint32_t id = (uint32_t)(unit(&r) * IDS), kept = 0;
    const double t0 = now_ms();
    for (;;) {
      if (keep >= 1 || unit(&r) < keep) {
        kept++;
        const int s = rq_dec_add(d, id, syms + (size_t)id * T);
        if (s < 0) { printf("decoder error\n"); return 2; }
        if (s == 1) break;
      }
      id = (id + 1) % IDS;
    }
    ms += now_ms() - t0;
    const int ex = (int)(kept - K);
    hist[ex > 2 ? 3 : ex]++, worst = ex > worst ? ex : worst;
    if (rq_dec_recover(d, back) || memcmp(back, data, bytes)) wrong++;
    rq_dec_free(d);
  }
  printf("loss: %u B (K %u), keep %g, %d trials: at K %d, K + 1 %d, K + 2 %d, past %d (the most K + %d); wrong bytes %d; %.1f ms a decode\n",
         bytes, K, keep, trials, hist[0], hist[1], hist[2], hist[3], worst, wrong, ms / trials);
  free(data), free(back), free(syms);
  return wrong != 0;
}

static int timing(uint32_t bytes, int n) {
  uint32_t r = 99;
  uint8_t *data = malloc(bytes), *back = malloc(bytes);
  for (uint32_t i = 0; i < bytes; i++) data[i] = (uint8_t)mul32(&r);
  double enc = 0, dec = 0;
  rq_enc_t *e = NULL;
  for (int i = 0; i < n; i++) {
    const double t0 = now_ms();
    rq_enc_t *x = rq_enc_new(data, bytes, T);
    enc += now_ms() - t0;
    if (e) rq_enc_free(e);
    e = x;
  }
  const uint32_t K = rq_symbols(bytes, T);
  uint8_t *s = malloc((size_t)(K + 32) * T);
  uint32_t *ids = malloc((size_t)(K + 32) * sizeof *ids);
  int ok = 1;
  for (int i = 0; i < n; i++) {
    uint32_t id = (uint32_t)(unit(&r) * IDS);
    for (uint32_t k = 0; k < K + 32;) {
      if (unit(&r) < 0.75) ids[k] = id, rq_enc_symbol(e, id, s + (size_t)k * T), k++;
      id = (id + 1) % IDS;
    }
    const double t0 = now_ms();
    rq_dec_t *d = rq_dec_new(bytes, T, IDS);
    for (uint32_t k = 0; k < K + 32; k++)
      if (rq_dec_add(d, ids[k], s + (size_t)k * T) == 1) break;
    rq_dec_recover(d, back);
    dec += now_ms() - t0;
    ok &= !memcmp(back, data, bytes);
    rq_dec_free(d);
  }
  printf("time: %u B (K %u): encoder %.1f ms, decode %.1f ms (%d each)%s\n", bytes, K, enc / n, dec / n, n, ok ? "" : " WRONG BYTES");
  rq_enc_free(e), free(data), free(back), free(s), free(ids);
  return !ok;
}

int main(int argc, char **argv) {
  if (argc >= 3 && !strcmp(argv[1], "vectors")) return vectors(argc - 2, argv + 2);
  if (argc >= 5 && !strcmp(argv[1], "loss")) return loss((uint32_t)atol(argv[2]), atof(argv[3]), atoi(argv[4]), argc > 5 ? (uint32_t)atol(argv[5]) : 1);
  if (argc >= 4 && !strcmp(argv[1], "time")) return timing((uint32_t)atol(argv[2]), atoi(argv[3]));
  fprintf(stderr, "usage: vectors <file>... | loss <bytes> <keep> <trials> [seed] | time <bytes> <n>\n");
  return 2;
}
