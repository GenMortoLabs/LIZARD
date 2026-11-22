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
// ai: the tiers before it and one 3/4 block (src/fmt.c ob_block_slots, OB_FMT_SLOTS data slots a version)
static void any_counts(int v, int count[3]) {
  const int SL[3] = { ob_block_slots(OB_TIER_RATE[OB_TIER_67]), ob_block_slots(OB_TIER_RATE[OB_TIER_23]), ob_block_slots(OB_TIER_RATE[OB_TIER_12]) };
  do {
    long left = (long)v * OB_FMT_SLOTS - ob_block_slots(OB_TIER_RATE[OB_TIER_34]);
    for (int i = 0; i < 3; i++) {
      long most = left / SL[i];
      if (most > (i ? OB_FMT_COUNT_MAX : OB_FMT_COUNT67_MAX)) most = i ? OB_FMT_COUNT_MAX : OB_FMT_COUNT67_MAX;
      count[i] = (int)(rnd() % (uint32_t)(most + 1));
      left -= (long)count[i] * SL[i];
    }
  } while (!ob_fmt_layout(v, count, 0));
}

// SPEC.md 5.2's words, a the ring's index (test/vectors.mjs holds the wasm build to the same words); `fmt_test
// vectors` prints them. A change here is a change to the format. ai: (2026-11-22, the index and the clock: the default
// senders' counts at 32, 64, 36, 4 and 16, the most 6/7 blocks at 67; a still, the clock's ends)
static const struct { int a; ob_fmt_t f; const char *hex; } VEC[] = {
  { 0, { .version = 32, .count = { 28, 8, 8 }, .fps = 60, .delay = 1 }, "4c0d8f52f5d8e3acab9b60939a0619b8" },
  { 0, { .version = 64, .count = { 57, 17, 17 }, .fps = 60, .delay = 1 }, "4cd08c9abc0cf6350d2fffc59bbc62a4" },
  { 0, { .version = 36, .count = { 32, 10, 10 }, .fps = 60, .delay = 4 }, "4c158a6daefb354d31456e238ebdee77" },
  { 0, { .version = 4, .count = { 4, 1, 1 }, .fps = 60, .delay = 1 }, "4c0000dbc183ba98d76e1dbcb5fdb314" },
  { 1, { .version = 4, .count = { 4, 1, 1 }, .fps = 60, .delay = 1 }, "4c0000dbc183841dd462708cab75202552c99e25077748d21528cf2bb4118ee5" },
  { 2, { .version = 16, .count = { 14, 4, 4 }, .fps = 30, .delay = 2 }, "4c00ea01a67467de15e5d52ef86fbabe493fb716324852613e6e4cf216c2a64cfc6cc26e4df853d7950d959e13beef4576813758e4f3abb9fdecbdce1d79cbd3" },
  { 0, { .version = 67, .count = { 90, 0, 30 }, .fps = 240, .delay = 30 }, "4cfc4a101d3258e791006e11e7ffad21" },
  { 1, { .version = 67, .count = { 151, 0, 0 }, .fps = 0, .delay = 1 }, "4cfd4f2dd79dcff6bf932889d78a07c9c2e45e85092334f2be406a74304f94a4" },
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
// ai: src/fmt.c crc5's twin: x^5 + x^2 + 1 over the 35-bit index, most significant first, from 0
static uint32_t crc5(uint64_t c) { uint64_t r = c << 5; for (int i = 39; i >= 5; i--) if (r >> i & 1) r ^= (uint64_t)0x25 << (i - 5); return (uint32_t)(r & 31); }
// ai: a word's 40 bits from a configuration's index as given and a clock, its CRC-5 right (bad: flipped in its last bit)
static uint64_t bits_of(long index, int fps, int delay, int bad) {
  const uint64_t x = ((uint64_t)index * (OB_FMT_FPS_MAX + 1) + (uint64_t)fps) * OB_FMT_DELAY_MAX + (uint64_t)(delay - 1);
  return x << 5 | (crc5(x) ^ (uint32_t)(bad != 0));
}
// ai: a word's 40 bits from a whole index as given (past the last ones too), its CRC-5 right
static uint64_t bits_raw(uint64_t x) { return x << 5 | crc5(x); }
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
  { const ob_fmt_t f = { .version = 4, .count = { 4, 1, 1 }, .fps = 60, .delay = 1 }, v3 = { .version = 3, .count = { 4, 1, 1 }, .fps = 60, .delay = 1 };
    const ob_fmt_t v67 = { .version = 67, .count = { 60, 18, 18 }, .fps = 60, .delay = 1 }, v68 = { .version = 68, .count = { 60, 18, 18 }, .fps = 60, .delay = 1 };
    const ob_fmt_t full = { .version = 4, .count = { 9, 0, 0 }, .fps = 60, .delay = 1 }, one34 = { .version = 4, .count = { 0, 0, 4 }, .fps = 60, .delay = 1 };
    const ob_fmt_t big = { .version = 67, .count = { 152, 0, 0 }, .fps = 60, .delay = 1 };
    const ob_fmt_t big23 = { .version = 67, .count = { 0, 128, 0 }, .fps = 60, .delay = 1 }, most = { .version = 67, .count = { 151, 0, 0 }, .fps = 60, .delay = 1 };
    CHECK(ob_fmt_encode(&f, cw, 16) == 0 && ob_fmt_encode(&f, cw, 15) < 0 && ob_fmt_encode(&f, cw, OB_FMT_BYTES_MAX + 1) < 0, "encode lengths");
    // ai: the clock: 0 (a still) to 240 pictures a second, a delay of 1 to 30
    { ob_fmt_t g = f;
      g.fps = 0; CHECK(ob_fmt_encode(&g, cw, 16) == 0, "a still");
      g.fps = 240; g.delay = 30; CHECK(ob_fmt_encode(&g, cw, 16) == 0, "fps 240, delay 30");
      g.fps = 241; CHECK(ob_fmt_encode(&g, cw, 16) < 0, "fps 241 refused");
      g.fps = -1; CHECK(ob_fmt_encode(&g, cw, 16) < 0, "fps -1 refused");
      g.fps = 60; g.delay = 31; CHECK(ob_fmt_encode(&g, cw, 16) < 0, "delay 31 refused");
      g.delay = 0; CHECK(ob_fmt_encode(&g, cw, 16) < 0, "delay 0 refused"); }
    CHECK(ob_fmt_encode(&v67, cw, 16) == 0 && ob_fmt_encode(&v3, cw, 16) < 0 && ob_fmt_encode(&v68, cw, 16) < 0, "encode versions 4 to 67");
    // ai: no room for one 3/4 block (nine 6/7 blocks fill LIZARD-64's 40,704 data slots past 35,584; 152 version 67's),
    // ai: a 2/3 count past 127; 151 6/7 blocks the most version 67 holds. Four 1/2 blocks and the one 3/4 they leave are a
    // ai: frame: the pilots are apart from the blocks, so no layout needs a 6/7 or a second 3/4 block for them
    CHECK(ob_fmt_encode(&full, cw, 16) < 0 && ob_fmt_encode(&big, cw, 16) < 0 && ob_fmt_encode(&big23, cw, 16) < 0, "encode counts that are no frame");
    { int m34 = 0; CHECK(ob_fmt_encode(&one34, cw, 16) == 0 && ob_fmt_layout(4, one34.count, &m34) && m34 == 1, "four 1/2 blocks and one 3/4 at version 4"); }
    { int m34 = 0; CHECK(ob_fmt_encode(&most, cw, 16) == 0 && ob_fmt_layout(67, most.count, &m34) && m34 == 1, "151 6/7 blocks and one 3/4 at version 67"); }
    // ai: the layout: the 3/4 filling, and the default senders' at 656 sub-channels (36, 21, 11, 11)
    int m = 0;
    const int c41[3] = { 36, 11, 11 };
    CHECK(ob_fmt_layout(41, c41, &m) && m == 21, "layout at version 41: 3/4 x %d", m);
    CHECK(ob_fmt_layout(4, f.count, &m) && m == 1, "layout at version 4: 3/4 x %d", m); }

  // ai: The index: every configuration counted in index order (the version, then the 6/7, 2/3 and 1/2 counts ascending,
  // ai: src/fmt.c), the count OB_FMT_CONFIGS, the first of each version at its OB_FMT_BEFORE, and every 997th taken
  // ai: both ways (ob_fmt_index, ob_fmt_config); an index past the last refused.
  { long n = 0, both = 0, firsts = 0;
    for (int v = OB_FMT_VERSION_MIN; v <= OB_FMT_VERSION_MAX; v++) {
      int first = 1;
      for (int x = 0; x <= OB_FMT_COUNT67_MAX; x++) for (int y = 0; y <= OB_FMT_COUNT_MAX; y++) for (int z = 0; z <= OB_FMT_COUNT_MAX; z++) {
        const int k[3] = { x, y, z };
        if (!ob_fmt_layout(v, k, 0)) continue;
        if (first || n % 997 == 0) {
          int gv = 0, gk[3] = { -1, -1, -1 };
          const int ok = ob_fmt_index(v, k) == n && !ob_fmt_config(n, &gv, gk) && gv == v && !memcmp(gk, k, sizeof k);
          CHECK(ok, "index %ld: version %d counts %d %d %d", n, v, x, y, z);
          both += ok; firsts += first;
          first = 0;
        }
        n++;
      }
    }
    CHECK(n == OB_FMT_CONFIGS, "configurations %ld, OB_FMT_CONFIGS %d", n, (int)OB_FMT_CONFIGS);
    { int gv, gk[3]; CHECK(ob_fmt_config(OB_FMT_CONFIGS, &gv, gk) < 0 && ob_fmt_config(-1, &gv, gk) < 0, "an index past the last refused"); }
    { const int k[3] = { 9, 0, 0 }; CHECK(ob_fmt_index(4, k) < 0, "no index for counts that are no frame"); }
    printf("configurations: %ld (2^%.2f), %ld taken both ways, %ld versions' first\n", n, log2((double)n), both, firsts); }

  for (int a = 0; a < NR; a++) {
    const int W = WS[a], B = BS[a], nroots = B - OB_FMT_DATA, t = nroots / 2;
    // ai: The erasure steps (src/fmt.c): 1, 3, ... on an even nroots (10, 26, 58), to nroots - 3, three roots left in every
    // ai: ring. Weak bytes past it are errors over the roots it leaves, so reach is the most weak bytes a word reads with.
    const int first = 2 - ((nroots - 3) & 1), last = nroots - 3, reach = last + (nroots - last) / 2;
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
      const int k8[3] = { 8, 2, 2 };
      const long i8 = ob_fmt_index(8, k8);
      raw_word(c, 0x4d, bits_of(i8, 60, 1, 0), nroots); soften(c, B, W, 0.4f, q);
      { ob_fmt_t g; CHECK(ob_fmt_decode(q, W, B, &g) < 0, "magic 0x4d"); }
      raw_word(c, 0xb2, bits_of(i8, 60, 1, 0), nroots); soften(c, B, W, 0.4f, q);
      { ob_fmt_t g; CHECK(ob_fmt_decode(q, W, B, &g) < 0, "magic 0xb2"); }
      // ai: an index past the last: the first, and the 35 bits' largest
      static const uint64_t PAST[] = { (uint64_t)OB_FMT_CONFIGS * OB_FMT_CLOCKS, ((uint64_t)1 << 35) - 1 };
      for (int k = 0; k < 2; k++) {
        raw_word(c, OB_FMT_MAGIC, bits_raw(PAST[k]), nroots); soften(c, B, W, 0.4f, q);
        ob_fmt_t g;
        CHECK(ob_fmt_decode(q, W, B, &g) < 0, "index %llu", (unsigned long long)PAST[k]);
      }
      { const ob_fmt_t w = { .version = 67, .count = { 151, 0, 0 }, .fps = 60, .delay = 1 };
        raw_word(c, OB_FMT_MAGIC, bits_of(OB_FMT_CONFIGS - 1, 60, 1, 0), nroots); soften(c, B, W, 0.4f, q); CHECK(read_word(q, W, B, &w) == 1, "the last index: version 67, 151 6/7 blocks");
        raw_word(c, OB_FMT_MAGIC, bits_of(OB_FMT_CONFIGS - 1, 60, 1, 1), nroots); soften(c, B, W, 0.4f, q);
        ob_fmt_t g; CHECK(ob_fmt_decode(q, W, B, &g) < 0, "a CRC-5 one bit off"); }
      { const ob_fmt_t w = { .version = 4, .count = { 0, 0, 0 }, .fps = 0, .delay = 1 };
        raw_word(c, OB_FMT_MAGIC, bits_of(0, 0, 1, 0), nroots); soften(c, B, W, 0.4f, q); CHECK(read_word(q, W, B, &w) == 1, "index 0: version 4, 3/4 blocks alone, a still"); }
      { const ob_fmt_t w = { .version = 4, .count = { 0, 0, 4 }, .fps = 240, .delay = 30 };
        raw_word(c, OB_FMT_MAGIC, bits_of(4, 240, 30, 0), nroots); soften(c, B, W, 0.4f, q); CHECK(read_word(q, W, B, &w) == 1, "version 4, four 1/2 blocks and one 3/4, fps 240, delay 30"); }
      { const ob_fmt_t w = { .version = 67, .count = { 151, 0, 0 }, .fps = 240, .delay = 30 };
        raw_word(c, OB_FMT_MAGIC, bits_raw((uint64_t)OB_FMT_CONFIGS * OB_FMT_CLOCKS - 1), nroots); soften(c, B, W, 0.4f, q); CHECK(read_word(q, W, B, &w) == 1, "the last word"); }
      // ai: another format's magic (0xB3, the complement of 0x4C): refused, a valid word's 40 bits after it too
      raw_word(c, 0xb3, bits_of(i8, 60, 1, 0), nroots); soften(c, B, W, 0.4f, q);
      { ob_fmt_t g; CHECK(ob_fmt_decode(q, W, B, &g) < 0, "magic 0xb3 with a word's bits"); }
    }

    // ai: Garbage: soft values uniform in -0.5 .. 0.5, what a wrong grid reads. Exact rates first: RS passing a step,
    // ai: the steps together, then with the magic and its check (a false codeword's data bytes are uniform: 1/256 for the
    // ai: magic, then the share of the 40 bits after it that are a word: an index under OB_FMT_CONFIGS x OB_FMT_CLOCKS and
    // ai: its CRC-5, `fits`). Then measured through ob_fmt_decode, with the rate the steps to nroots - 1 would have beside it.
    const int other = nroots - 1;
    double steps = p_pass(B, 0), alt = p_pass(B, 0);
    for (int f = first; f <= last; f += 2) steps += p_pass(B, f);
    for (int f = first; f <= other; f += 2) alt += p_pass(B, f);
    const double fits = (double)OB_FMT_CONFIGS * OB_FMT_CLOCKS / 1099511627776.0;   // ai: one CRC-5 a word, of the 2^40 values
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
