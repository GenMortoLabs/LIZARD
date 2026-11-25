// The format word (src/fmt.c) at each ring's code: RS(16,6), RS(32,6), RS(64,6) over every word cell of the 64, 128
// and 256 ring (RS(n, 5) until 2026-11-22). Lengths from the layout, the configuration's index against every
// configuration counted, the SPEC test vectors, round trips through the soft values, whole sides lost, errors with and
// without the least confident bytes erased, the acceptance checks (another magic refused: 0xB3, its complement; an index
// past the last), how often garbage is accepted. ai: the animated word's and the header carousel's tests went
// 2026-11-19 with them (archive/animated-header-2026-11-19/).
//   gcc -O2 -Wall -Wextra -Isrc -o build/fmt_test test/fmt_test.c src/fmt.c src/rs.c src/layout.c src/ldpc.c -lm && build/fmt_test
//   build/fmt_test N      N garbage reads a length through ob_fmt_decode (default 20000, about 10 s; 2000000 for
//                         counts worth reading, about 10 minutes)
#include "../src/fmt.h"
#include "../src/layout.h"
#include "../src/rs.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t st = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (uint32_t)(st >> 32); }
static float uni(void) { return (float)(rnd() >> 8) / 16777216.0f; }
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

// The three rings: B band cells a side, modules a side S = 2B + 30 (src/lizard_phy.h), band reserve 15
// (OB_THIN_RESERVE); W word cells a side, RS(BS, 6). Any ring carries any version (SPEC 5.2).
enum { NR = 3, NV = OB_FMT_VERSION_MAX - OB_FMT_VERSION_MIN + 1 };
static const int RB[NR] = { 64, 128, 256 }, WS[NR] = { 32, 64, 128 }, BS[NR] = { 16, 32, 64 };
// ai: a version a word may state, at random
static int any_version(void) { return OB_FMT_VERSION_MIN + (int)(rnd() % NV); }
// ai: counts that make a frame of version v, at random: each tier's blocks uniform in what the frame still holds after
// ai: the tiers before it, starting at a tier at random (src/fmt.c ob_block_slots, OB_FMT_SLOTS data slots a version)
static void any_counts(int v, int count[OB_TIERS]) {
  do {
    long left = (long)v * OB_FMT_SLOTS;
    const int s = (int)(rnd() % OB_TIERS);
    for (int j = 0; j < OB_TIERS; j++) {
      const int i = (s + j) % OB_TIERS, sl = ob_block_slots(OB_TIER_RATE[i]);
      const long most = left / sl;
      count[i] = (int)(rnd() % (uint32_t)(most + 1));
      left -= (long)count[i] * sl;
    }
  } while (!ob_fmt_layout(v, count));
}

// SPEC.md 5.2's words, a the ring's index (test/vectors.mjs holds the wasm build to the same words); `fmt_test
// vectors` prints them. A change here is a change to the format. ai: (2026-11-22, the index and the clock: the default
// senders' counts at 32, 64, 36, 4 and 16, a layout set at 67; a still, the clock's ends; version 1 at 30 a second,
// delay 2, since 2026-11-22; made again 2026-11-23, the clock enumerated as 7,201 values, a still's one; and again that
// day, every layout of the four counts, the 40 bits the index alone: the default senders' the 6/7 on the rest, the last
// row LIZARD-1072's most blocks, 152 6/7)
static const struct { int a; ob_fmt_t f; const char *hex; } VEC[] = {
  { 0, { .version = 32, .count = { 29, 17, 8, 8 }, .fps = 60, .delay = 1 }, "4c063a383d78d40782800b0278f682d9" },
  { 0, { .version = 64, .count = { 56, 33, 17, 17 }, .fps = 60, .delay = 1 }, "4cb1d6740c644793200d56dba803ca22" },
  { 0, { .version = 36, .count = { 30, 19, 10, 10 }, .fps = 60, .delay = 4 }, "4c0ae6c99ca5f564b4c913f656f83dca" },
  { 0, { .version = 4, .count = { 3, 2, 1, 1 }, .fps = 60, .delay = 1 }, "4c00002bfa7b97864bc6c74b97f63aad" },
  { 1, { .version = 4, .count = { 3, 2, 1, 1 }, .fps = 60, .delay = 1 }, "4c00002bfa7be499e4bfef0ace7ca40d729764ce8dcc297d97228002d0ad032d" },
  { 2, { .version = 16, .count = { 15, 8, 4, 4 }, .fps = 30, .delay = 2 }, "4c003df80ea2154994885d3fb06778874fd0afdabe0a5968957866ee53f71745a8008d56bb4022fde941fd0e01638cc6fcd56dad53fe8c28fa709a437c711d54" },
  { 0, { .version = 67, .count = { 90, 0, 0, 30 }, .fps = 240, .delay = 30 }, "4ce04b3aa62067c2e58f7b025f172a8f" },
  { 0, { .version = 1, .count = { 1, 1, 0, 0 }, .fps = 30, .delay = 2 }, "4c000000900de6f08b482b52049b997b" },
  { 1, { .version = 67, .count = { 152, 0, 0, 0 }, .fps = 0, .delay = 1 }, "4ce0c82eedd5fa520494b962de0eae552d16c5c9c33c062b7d1fad22fcb2c23e" },
};

// Soft values as the reader lays them out, q[side * cells + i]: +mag for a dark cell, -mag for a light one.
static void soften(const uint8_t *cw, int bytes, int cells, float mag, float *q) {
  for (int s = 0; s < 4; s++) for (int i = 0; i < cells; i++) q[s * cells + i] = ob_fmt_bit(cw, bytes, s, i) ? mag : -mag;
}
static float *cell_of(float *q, int cells, int k, int b) { return q + (k & 3) * cells + 8 * (k >> 2) + b; }
// Byte k read wrong with confidence mag (the word's own bits flipped), or faded to its local level, which is what a
// side under glare reads as (src/acquire.c ob_thin_read_fmt).
static void flip_byte(float *q, int cells, int k, float mag) { for (int b = 0; b < 8; b++) { float *c = cell_of(q, cells, k, b); *c = *c > 0 ? -mag : mag; } }
static void fade_side(float *q, int cells, int s) { for (int i = 0; i < cells; i++) q[s * cells + i] = (uni() - 0.5f) * 0.04f; }

// 1 decoded right, 0 nothing, -1 a wrong word accepted.
static int same(const ob_fmt_t *a, const ob_fmt_t *b) {
  return a->version == b->version && !memcmp(a->count, b->count, sizeof a->count) && a->fps == b->fps && a->delay == b->delay;
}
static int read_word(const float *q, int cells, int bytes, const ob_fmt_t *want) {
  ob_fmt_t f;
  if (ob_fmt_decode(q, cells, bytes, &f)) return 0;
  return same(&f, want) ? 1 : -1;
}
// ai: a word at random: any version and counts that make a frame, any clock
static ob_fmt_t any_word(void) {
  ob_fmt_t f = { .version = any_version(), .fps = (int)(rnd() % (OB_FMT_FPS_MAX + 1)), .delay = 1 + (int)(rnd() % OB_FMT_DELAY_MAX) };
  if (!f.fps) f.delay = 1;   // ai: (a still states no delay, and reads back 1)
  any_counts(f.version, f.count);
  return f;
}
// ai: a word's codeword from its six data bytes as given (x the 40 bits after the magic), for words the encoder refuses
static void raw_word(uint8_t *c, int magic, uint64_t x, int nroots) {
  memset(c, 0, OB_FMT_BYTES_MAX);
  c[0] = (uint8_t)magic;
  for (int i = 0; i < 5; i++) c[1 + i] = (uint8_t)(x >> (32 - 8 * i));
  rs_encode(c, OB_FMT_DATA, nroots);
}
// ai: a word's 40 bits from a configuration's index as given and a clock: the index alone (src/fmt.c ob_fmt_pack)
static uint64_t bits_of(long index, int fps, int delay) {
  return (uint64_t)index * OB_FMT_CLOCKS + (fps ? 1 + (uint64_t)(fps - 1) * OB_FMT_DELAY_MAX + (uint64_t)(delay - 1) : 0);
}
static void hexof(const uint8_t *b, int n, char *hex) { for (int k = 0; k < n; k++) sprintf(hex + 2 * k, "%02x", b[k]); }

static double binom(int n, int k) { double r = 1; for (int i = 0; i < k; i++) r = r * (n - i) / (i + 1); return r; }
// A received word uniform at random passes a bounded-distance decode of RS(n, 6) with f erasures when its other n - f
// bytes lie within t = (nroots - f) / 2 of a codeword of the punctured code: V(n - f, t) / 256^(nroots - f).
static double p_pass(int n, int f) {
  const int nroots = n - OB_FMT_DATA, t = (nroots - f) / 2;
  double v = 0;
  for (int i = 0; i <= t; i++) v += binom(n - f, i) * pow(255, i);
  return v / pow(256, nroots - f);
}

int main(int argc, char **argv) {
  float q[4 * 160];
  uint8_t cw[OB_FMT_BYTES_MAX + 1];
  char hex[2 * OB_FMT_BYTES_MAX + 1];
  if (argc > 1 && !strcmp(argv[1], "vectors")) {
    for (size_t v = 0; v < sizeof VEC / sizeof VEC[0]; v++) {
      if (ob_fmt_encode(&VEC[v].f, cw, BS[VEC[v].a])) { printf("vector %zu: refused\n", v); continue; }
      hexof(cw, BS[VEC[v].a], hex);
      printf("ring %d: %s\n", RB[VEC[v].a], hex);
    }
    return 0;
  }
  const long NG = argc > 1 ? atol(argv[1]) : 20000;

  // Lengths from the layout, as a receiver takes them from the module count it registered.
  const int r = OB_THIN_RESERVE;
  for (int a = 0; a < NR; a++) {
    const int S = 2 * RB[a] + 30;
    CHECK(ob_fmt_side_cells(S, r) == WS[a] && ob_fmt_bytes(S, r) == BS[a], "ring %d: W %d, %d bytes", RB[a], ob_fmt_side_cells(S, r), ob_fmt_bytes(S, r));
  }
  CHECK(ob_fmt_bytes(154, r) == 0 && ob_fmt_side_cells(154, r) == 0, "a side too short for the shortest word");
  CHECK(ob_fmt_bytes(700, r) == OB_FMT_BYTES_MAX, "a band past the longest word");

  // The vectors, and the encoder's range.
  for (size_t v = 0; v < sizeof VEC / sizeof VEC[0]; v++) {
    const int a = VEC[v].a;
    CHECK(ob_fmt_encode(&VEC[v].f, cw, BS[a]) == 0, "encode vector %zu", v);
    hexof(cw, BS[a], hex);
    CHECK(!strcmp(hex, VEC[v].hex), "vector %zu ring %d: %s", v, RB[a], hex);
    soften(cw, BS[a], WS[a], 0.4f, q);
    CHECK(read_word(q, WS[a], BS[a], &VEC[v].f) == 1, "vector %zu read back", v);
  }
  { const ob_fmt_t f = { .version = 4, .count = { 3, 2, 1, 1 }, .fps = 60, .delay = 1 }, v3 = { .version = 3, .count = { 3, 2, 1, 1 }, .fps = 60, .delay = 1 };
    const ob_fmt_t v67 = { .version = 67, .count = { 60, 18, 18, 18 }, .fps = 60, .delay = 1 }, v68 = { .version = 68, .count = { 60, 18, 18, 18 }, .fps = 60, .delay = 1 };
    const ob_fmt_t full = { .version = 4, .count = { 10, 0, 0, 0 }, .fps = 60, .delay = 1 }, neg = { .version = 4, .count = { 0, 0, -1, 2 }, .fps = 60, .delay = 1 };
    const ob_fmt_t big = { .version = 67, .count = { 153, 0, 0, 0 }, .fps = 60, .delay = 1 }, most = { .version = 67, .count = { 152, 0, 0, 0 }, .fps = 60, .delay = 1 };
    const ob_fmt_t none = { .version = 1, .count = { 0, 0, 0, 0 }, .fps = 60, .delay = 1 }, half = { .version = 1, .count = { 0, 0, 0, 1 }, .fps = 60, .delay = 1 };
    CHECK(ob_fmt_encode(&f, cw, 16) == 0 && ob_fmt_encode(&f, cw, 15) < 0 && ob_fmt_encode(&f, cw, OB_FMT_BYTES_MAX + 1) < 0, "encode lengths");
    // ai: the clock: 0 (a still) to 240 pictures a second, a delay of 1 to 30
    { ob_fmt_t g = f;
      g.fps = 0; CHECK(ob_fmt_encode(&g, cw, 16) == 0, "a still");
      { // ai: a still states no delay: any delay paints the same word, read back delay 1
        uint8_t c1[OB_FMT_BYTES_MAX], c7[OB_FMT_BYTES_MAX]; ob_fmt_t s1 = g, s7 = g, r; float qs[4 * OB_FMT_SIDE_CELLS];
        s1.delay = 1; s7.delay = 7;
        CHECK(ob_fmt_encode(&s1, c1, 16) == 0 && ob_fmt_encode(&s7, c7, 16) == 0 && !memcmp(c1, c7, 16), "a still's delay not stated");
        soften(c7, 16, OB_FMT_SIDE_CELLS, 0.4f, qs);
        CHECK(ob_fmt_decode(qs, OB_FMT_SIDE_CELLS, 16, &r) == 0 && r.fps == 0 && r.delay == 1, "a still reads back fps 0, delay 1"); }
      g.fps = 240; g.delay = 30; CHECK(ob_fmt_encode(&g, cw, 16) == 0, "fps 240, delay 30");
      g.fps = 241; CHECK(ob_fmt_encode(&g, cw, 16) < 0, "fps 241 refused");
      g.fps = -1; CHECK(ob_fmt_encode(&g, cw, 16) < 0, "fps -1 refused");
      g.fps = 60; g.delay = 31; CHECK(ob_fmt_encode(&g, cw, 16) < 0, "delay 31 refused");
      g.delay = 0; CHECK(ob_fmt_encode(&g, cw, 16) < 0, "delay 0 refused"); }
    CHECK(ob_fmt_encode(&v67, cw, 16) == 0 && ob_fmt_encode(&v3, cw, 16) < 0 && ob_fmt_encode(&v68, cw, 16) < 0, "encode version 67, refuse 68 and counts version 3 cannot hold");
    // ai: any counts that fit, the rest padding: ten 6/7 blocks pass LIZARD-64's 40,704 data slots (44,800), 153 version
    // ai: 67's, a count below 0 is none; 152 6/7 blocks the most version 67 holds, a frame of no blocks and one of a lone
    // ai: 1/2 block (7,680 of version 1's 10,176) frames
    CHECK(ob_fmt_encode(&full, cw, 16) < 0 && ob_fmt_encode(&big, cw, 16) < 0 && ob_fmt_encode(&neg, cw, 16) < 0, "encode counts that are no frame");
    CHECK(ob_fmt_encode(&most, cw, 16) == 0 && ob_fmt_encode(&none, cw, 16) == 0 && ob_fmt_encode(&half, cw, 16) == 0, "152 6/7 blocks at version 67, no blocks, a lone 1/2 block");
    { const int two[OB_TIERS] = { 0, 0, 1, 1 }, exact[OB_TIERS] = { 2, 0, 0, 0 };
      CHECK(!ob_fmt_layout(1, two), "version 1 refuses a 2/3 and a 1/2 block (13,440 slots)");
      CHECK(ob_fmt_layout(1, exact), "version 1 holds two 6/7 blocks (8,960 slots)"); }
    // ai: the default senders' at 656 sub-channels (36, 21, 11, 11, 576 slots padded)
    const int c41[OB_TIERS] = { 36, 21, 11, 11 };
    CHECK(ob_fmt_layout(41, c41) && ob_fmt_layout(4, f.count), "the default senders' layouts at versions 41 and 4"); }

  // ai: The index: every layout counted in index order (the version, then the 6/7, 3/4, 2/3 and 1/2 counts ascending,
  // ai: src/fmt.c), by the slots themselves (ob_block_slots, not the tables' units): the 1/2 counts of each (a, b, t) at
  // ai: once, the count OB_FMT_CONFIGS; each version's first layout, every 997th (a, b, t)'s first and last and every
  // ai: version's last taken both ways (ob_fmt_index, ob_fmt_config); an index past the last refused.
  { long n = 0, both = 0, firsts = 0, triples = 0;
    const long s7 = ob_block_slots(OB_TIER_RATE[OB_TIER_67]), s3 = ob_block_slots(OB_TIER_RATE[OB_TIER_34]), s2 = ob_block_slots(OB_TIER_RATE[OB_TIER_23]),
               s1 = ob_block_slots(OB_TIER_RATE[OB_TIER_12]);
    for (int v = OB_FMT_VERSION_MIN; v <= OB_FMT_VERSION_MAX; v++) {
      const long all = (long)v * OB_FMT_SLOTS;
      int first = 1;
      for (int x = 0; x * s7 <= all; x++) for (int y = 0; x * s7 + y * s3 <= all; y++) for (int z = 0; x * s7 + y * s3 + z * s2 <= all; z++) {
        const long k = (all - x * s7 - y * s3 - z * s2) / s1 + 1;
        if (first || triples % 997 == 0) {
          for (int e = 0; e < 2; e++) {
            const int kk[OB_TIERS] = { x, y, z, e ? (int)k - 1 : 0 };
            const long want = n + (e ? k - 1 : 0);
            int gv = 0, gk[OB_TIERS] = { -1, -1, -1, -1 };
            const int ok = ob_fmt_index(v, kk) == want && !ob_fmt_config(want, &gv, gk) && gv == v && !memcmp(gk, kk, sizeof kk);
            CHECK(ok, "index %ld: version %d counts %d %d %d %d", want, v, kk[0], kk[1], kk[2], kk[3]);
            both += ok;
          }
          firsts += first;
          first = 0;
        }
        n += k; triples++;
      }
      // ai: the version's last layout: its largest 6/7 count alone
      const int kl[OB_TIERS] = { (int)(all / s7), 0, 0, 0 };
      CHECK(ob_fmt_index(v, kl) == n - 1, "version %d's last layout at %ld", v, n - 1);
    }
    CHECK(n == OB_FMT_CONFIGS, "configurations %ld, OB_FMT_CONFIGS %d", n, (int)OB_FMT_CONFIGS);
    { int gv, gk[OB_TIERS]; CHECK(ob_fmt_config(OB_FMT_CONFIGS, &gv, gk) < 0 && ob_fmt_config(-1, &gv, gk) < 0, "an index past the last refused"); }
    { const int k[OB_TIERS] = { 10, 0, 0, 0 }; CHECK(ob_fmt_index(4, k) < 0, "no index for counts that are no frame"); }
    printf("configurations: %ld (2^%.2f) over %ld (6/7, 3/4, 2/3) triples, %ld taken both ways, %ld versions' first\n", n, log2((double)n), triples, both, firsts); }

  for (int a = 0; a < NR; a++) {
    const int W = WS[a], B = BS[a], nroots = B - OB_FMT_DATA, t = nroots / 2;
    // ai: The erasure steps (src/fmt.c): 1, 3, ... on an even nroots (10, 26, 58), to nroots - 1, one root left in every
    // ai: ring (three until 2026-11-23). Weak bytes past it are errors over the roots it leaves, so reach is the most weak
    // ai: bytes a word reads with.
    const int first = 2 - ((nroots - 1) & 1), last = nroots - 1, reach = last + (nroots - last) / 2;
    int ok = 0, wrong = 0, tries = 0;
    printf("RS(%d,6), ring %d, %d word cells a side, %d bytes a side, %d roots\n", B, RB[a], W, B / 4, nroots);
    // Round trips, with cell noise.
    for (int it = 0; it < 500; it++) {
      const ob_fmt_t f = any_word();
      ob_fmt_encode(&f, cw, B);
      soften(cw, B, W, 0.3f, q);
      for (int i = 0; i < 4 * W; i++) q[i] += (uni() - 0.5f) * 0.2f;
      const int r = read_word(q, W, B, &f);
      ok += r == 1; wrong += r < 0; tries++;
    }
    CHECK(ok == tries && !wrong, "round trips %d of %d, %d wrong", ok, tries, wrong);
    // Whole sides lost: 1, 2 (adjacent and opposite) and 3. A side is B / 4 bytes, erasures once faded.
    for (int lost = 1; lost <= 3; lost++) {
      int got = 0, bad = 0, n = 0;
      for (int m = 0; m < 16; m++) {
        if (__builtin_popcount(m) != lost) continue;
        for (int it = 0; it < 50; it++) {
          const ob_fmt_t f = any_word();
          ob_fmt_encode(&f, cw, B);
          soften(cw, B, W, 0.4f, q);
          for (int s = 0; s < 4; s++) if (m >> s & 1) fade_side(q, W, s);
          const int r = read_word(q, W, B, &f);
          got += r == 1; bad += r < 0; n++;
        }
      }
      // ai: lost * B / 4 faded bytes, read while they are within reach.
      const int need = lost * B / 4, in = need <= reach;
      printf("  %d side%s lost, %2d bytes: read %3d of %3d, wrong %d%s\n", lost, lost > 1 ? "s" : "", need, got, n, bad, in ? "" : "  (past the steps)");
      CHECK(!bad, "%d sides lost: %d wrong", lost, bad);
      CHECK(in ? got == n : got < n / 50 + 1, "%d sides lost: read %d of %d, reach %d", lost, got, n, reach);
    }
    // ai: Errors. The error bytes are made the most confident (0.45 against 0.4), so no erasure step lands on one: t
    // ai: read, t + 1 does not. Then errors in the least confident bytes (0.05), which the steps erase: reach of them
    // ai: read, and nroots - 3 with one confident error; reach + 1 do not.
    struct { int conf, weak, expect; } E[] = { { t, 0, 1 }, { t + 1, 0, 0 }, { 0, reach, 1 }, { 1, nroots - 3, 1 }, { 0, reach + 1, 0 } };
    for (size_t e = 0; e < sizeof E / sizeof E[0]; e++) {
      int got = 0, bad = 0;
      for (int it = 0; it < 200; it++) {
        const ob_fmt_t f = any_word();
        int pos[OB_FMT_BYTES_MAX], used[OB_FMT_BYTES_MAX] = { 0 };
        ob_fmt_encode(&f, cw, B);
        soften(cw, B, W, 0.4f, q);
        for (int i = 0; i < E[e].conf + E[e].weak; i++) { int p; do p = (int)(rnd() % B); while (used[p]); used[p] = 1; pos[i] = p; }
        for (int i = 0; i < E[e].conf + E[e].weak; i++) flip_byte(q, W, pos[i], i < E[e].conf ? 0.45f : 0.05f);
        const int r = read_word(q, W, B, &f);
        got += r == 1; bad += r < 0;
      }
      printf("  %2d confident errors, %2d weak: read %3d of 200, wrong %d\n", E[e].conf, E[e].weak, got, bad);
      CHECK(E[e].expect ? got == 200 && !bad : !got, "%d confident, %d weak: read %d, wrong %d", E[e].conf, E[e].weak, got, bad);
    }
    // Acceptance: a codeword RS takes that is not a word.
    { uint8_t c[OB_FMT_BYTES_MAX];
      const int k8[OB_TIERS] = { 8, 2, 2, 1 };
      const long i8 = ob_fmt_index(8, k8);
      raw_word(c, 0x4d, bits_of(i8, 60, 1), nroots); soften(c, B, W, 0.4f, q);
      { ob_fmt_t g; CHECK(ob_fmt_decode(q, W, B, &g) < 0, "magic 0x4d"); }
      raw_word(c, 0xb2, bits_of(i8, 60, 1), nroots); soften(c, B, W, 0.4f, q);
      { ob_fmt_t g; CHECK(ob_fmt_decode(q, W, B, &g) < 0, "magic 0xb2"); }
      // ai: an index past the last: the first, and the 40 bits' largest
      static const uint64_t PAST[] = { (uint64_t)OB_FMT_CONFIGS * OB_FMT_CLOCKS, ((uint64_t)1 << 40) - 1 };
      for (int k = 0; k < 2; k++) {
        raw_word(c, OB_FMT_MAGIC, PAST[k], nroots); soften(c, B, W, 0.4f, q);
        ob_fmt_t g;
        CHECK(ob_fmt_decode(q, W, B, &g) < 0, "index %llu", (unsigned long long)PAST[k]);
      }
      { const ob_fmt_t w = { .version = 67, .count = { 152, 0, 0, 0 }, .fps = 60, .delay = 1 };
        raw_word(c, OB_FMT_MAGIC, bits_of(OB_FMT_CONFIGS - 1, 60, 1), nroots); soften(c, B, W, 0.4f, q); CHECK(read_word(q, W, B, &w) == 1, "the last index: version 67, 152 6/7 blocks"); }
      { const ob_fmt_t w = { .version = 1, .count = { 0, 0, 0, 0 }, .fps = 0, .delay = 1 };
        raw_word(c, OB_FMT_MAGIC, bits_of(0, 0, 1), nroots); soften(c, B, W, 0.4f, q); CHECK(read_word(q, W, B, &w) == 1, "index 0: version 1, no blocks, a still"); }
      { const ob_fmt_t w = { .version = 4, .count = { 0, 0, 0, 4 }, .fps = 240, .delay = 30 };
        raw_word(c, OB_FMT_MAGIC, bits_of(166 + 4, 240, 30), nroots); soften(c, B, W, 0.4f, q); CHECK(read_word(q, W, B, &w) == 1, "version 4 (166 layouts of versions 1 to 3 before it), four 1/2 blocks, fps 240, delay 30"); }
      { const ob_fmt_t w = { .version = 67, .count = { 152, 0, 0, 0 }, .fps = 240, .delay = 30 };
        raw_word(c, OB_FMT_MAGIC, (uint64_t)OB_FMT_CONFIGS * OB_FMT_CLOCKS - 1, nroots); soften(c, B, W, 0.4f, q); CHECK(read_word(q, W, B, &w) == 1, "the last word"); }
      // ai: another format's magic (0xB3, the complement of 0x4C): refused, a valid word's 40 bits after it too
      raw_word(c, 0xb3, bits_of(i8, 60, 1), nroots); soften(c, B, W, 0.4f, q);
      { ob_fmt_t g; CHECK(ob_fmt_decode(q, W, B, &g) < 0, "magic 0xb3 with a word's bits"); }
    }

    // ai: Garbage: soft values uniform in -0.5 .. 0.5, what a wrong grid reads. Exact rates first: RS passing a step,
    // ai: the steps together, then with the magic and its check (a false codeword's data bytes are uniform: 1/256 for the
    // ai: magic, then the share of the 40 bits after it that are a word: an index under OB_FMT_CONFIGS x OB_FMT_CLOCKS and
    // ai: its range, `fits`). Then measured through ob_fmt_decode, with the rate the steps to nroots - 3 had beside it.
    const int other = nroots - 3;
    double steps = p_pass(B, 0), alt = p_pass(B, 0);
    for (int f = first; f <= last; f += 2) steps += p_pass(B, f);
    for (int f = first; f <= other; f += 2) alt += p_pass(B, f);
    const double fits = (double)OB_FMT_CONFIGS * OB_FMT_CLOCKS / 1099511627776.0;   // ai: the words' share of the 2^40 values
    const double mv = steps * fits / 256.0;
    long acc = 0;
    for (long it = 0; it < NG; it++) {
      for (int i = 0; i < 4 * W; i++) q[i] = uni() - 0.5f;
      ob_fmt_t f;
      acc += ob_fmt_decode(q, W, B, &f) == 0;
    }
    printf("  garbage, exact: errors only %.1e, steps %d to %d %.1e, the magic and its check %.1e (steps to %d: %.1e)\n",
           p_pass(B, 0), first, last, steps, mv, other, alt * fits / 256.0);
    printf("  garbage, measured over %ld reads: %ld accepted (expected %.2f)\n", NG, acc, mv * NG);
    CHECK(acc <= 10 + 4 * mv * NG, "garbage accepted %ld of %ld", acc, NG);
  }
  printf(fails ? "%d FAILED\n" : "all passed\n", fails);
  return fails != 0;
}
