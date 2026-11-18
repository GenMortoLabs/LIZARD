// The format word (src/fmt.c) at each ring's code: RS(16,5), RS(32,5), RS(64,5) over every word cell of the 64, 128
// and 256 ring, static and animated. Lengths from the layout, the SPEC test vectors, round trips through the soft
// values, whole sides lost, errors with and without the least confident bytes erased, the acceptance checks, how often
// garbage is accepted; then the header carousel: its sizes and vectors, assembly from any k symbols, wrong symbols
// corrected or outvoted, and garbage symbols never giving a header.
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
// (OB_THIN_RESERVE); W word cells a side, RS(BS, 5). Any ring carries any version (SPEC 5.2).
enum { NR = 3, NV = OB_FMT_VERSION_MAX - OB_FMT_VERSION_MIN + 1 };
static const int RB[NR] = { 64, 128, 256 }, WS[NR] = { 32, 64, 128 }, BS[NR] = { 16, 32, 64 };
// ai: a version a word may state, at random
static int any_version(void) { return OB_FMT_VERSION_MIN + (int)(rnd() % NV); }
// ai: counts that make a frame of version v, at random: each tier's blocks uniform in what the frame still holds after
// ai: the tiers before it and one 3/4 block, drawn again until two blocks carry pilots
static void any_counts(int v, int count[3]) {
  static const int SL[3] = { 4496, 5760, 7680 };
  do {
    long left = (long)v * OB_FMT_SLOTS - 5152;
    for (int i = 0; i < 3; i++) {
      long most = left / SL[i];
      if (most > OB_FMT_COUNT_MAX) most = OB_FMT_COUNT_MAX;
      count[i] = (int)(rnd() % (uint32_t)(most + 1));
      left -= (long)count[i] * SL[i];
    }
  } while (!ob_fmt_layout(v, count, 0));
}

// SPEC.md 5.2's words, a the ring's index (test/vectors.mjs holds the wasm build to the same words); `fmt_test
// vectors` prints them. A change here is a change to the format. ai: (2026-11-18, the static word's 32 bits: version,
// rate code and the three counts; the default senders' counts at 32, 64, 36, 4, 16 and 67, and others)
static const struct { int a; ob_fmt_t f; const char *hex; } VEC[] = {
  { 0, { .version = 32, .fps = 24, .count = { 28, 9, 9 } }, "4c72070489ff4bafb5169a22ec41140d" },
  { 0, { .version = 64, .fps = 60, .count = { 57, 17, 17 } }, "4cf3ee4891d77e66de79fa905a10fa3a" },
  { 0, { .version = 36, .fps = 30, .count = { 32, 10, 10 } }, "4c8288050ac88b24ade31b1087c3d7f8" },
  { 0, { .version = 4, .fps = 1, .count = { 4, 1, 1 } }, "4c000100819b51c3e0640815e05d2fce" },
  { 1, { .version = 4, .fps = 60, .count = { 4, 1, 1 } }, "4c03e10081ce315f0ebd05dea9666935945bda86d54607f35921028a9ed3b28f" },
  { 2, { .version = 16, .fps = 50, .count = { 14, 4, 4 } }, "4c338382042554f915f150f736cec3512dd8f38105e282e246f1c402266203d91e5098fc29dbbfbb33347b78ad3d6c6ab77be7bce86da5112cf6d865e3d6d35d" },
  { 0, { .version = 67, .fps = 45, .count = { 90, 0, 30 } }, "4cff56801e0cace093089fcd3d70cd72" },
  { 0, { .kind = OB_FMT_ANIM, .id = 0, .sym = { 0x4c, 0x97, 0xe9 } }, "b3004c97e9e1cfce9d46a576420bd1f1" },
  { 1, { .kind = OB_FMT_ANIM, .id = 254, .sym = { 1, 2, 3 } }, "b3fe0102037b7cb2a17921eed7b08d874b477f2cd0f8d419b67a407300ff063e" },
};
// ai: The carousel's vector (SPEC 5.5): the default header of version 41 (656 sub-channels), rate 60, the default
// ai: senders' counts 36, 11, 11 (ob_car_content: the static word's five bytes), k 3, its nine bytes and the symbols at
// ai: ids 0, 1, 2, 3, 100 and 254; and the same at 240 a second, its rate record making k 5.
static const ob_fmt_t CAR_CFG = { .version = 41, .fps = 60, .count = { 36, 11, 11 } };
static const char *CAR_HDR = "4c97e9058b8384a0af";
static const struct { int id; const char *hex; } CAR_SYM[] = { { 0, "4c97e9" }, { 1, "058b83" }, { 2, "84a0af" }, { 3, "f4546b" }, { 100, "39cfec" }, { 254, "ad32c6" } };

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
  if (a->kind != b->kind) return 0;
  if (a->kind == OB_FMT_ANIM) return a->id == b->id && !memcmp(a->sym, b->sym, 3);
  return a->version == b->version && a->fps == b->fps && !memcmp(a->count, b->count, sizeof a->count);
}
static int read_word(const float *q, int cells, int bytes, const ob_fmt_t *want) {
  ob_fmt_t f;
  if (ob_fmt_decode(q, cells, bytes, &f)) return 0;
  return same(&f, want) ? 1 : -1;
}
// ai: a word at random: half static (any version, rate and counts that make a frame), half animated
static ob_fmt_t any_word(void) {
  ob_fmt_t f = { .version = 0 };
  if (rnd() & 1) { f.kind = OB_FMT_ANIM; f.id = (int)(rnd() % (OB_FMT_ID_MAX + 1)); for (int j = 0; j < 3; j++) f.sym[j] = (uint8_t)rnd(); }
  else { f.version = any_version(); f.fps = OB_FMT_RATE[rnd() % OB_FMT_RATES]; any_counts(f.version, f.count); }
  return f;
}
// ai: a static word's codeword from its five data bytes as given (x the 32 bits after the magic), for words the encoder
// ai: refuses
static void raw_word(uint8_t *c, int magic, uint32_t x, int nroots) {
  memset(c, 0, OB_FMT_BYTES_MAX);
  c[0] = (uint8_t)magic;
  for (int i = 0; i < 4; i++) c[1 + i] = (uint8_t)(x >> (24 - 8 * i));
  rs_encode(c, OB_FMT_DATA, nroots);
}
static uint32_t bits_of(int version, int rc, int a, int t, int c) { return (uint32_t)(version - OB_FMT_VERSION_MIN) << 26 | (uint32_t)rc << 21 | (uint32_t)a << 14 | (uint32_t)t << 7 | (uint32_t)c; }
static void hexof(const uint8_t *b, int n, char *hex) { for (int k = 0; k < n; k++) sprintf(hex + 2 * k, "%02x", b[k]); }

static double binom(int n, int k) { double r = 1; for (int i = 0; i < k; i++) r = r * (n - i) / (i + 1); return r; }
// A received word uniform at random passes a bounded-distance decode of RS(n, 5) with f erasures when its other n - f
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
  char hex[2 * 3 * OB_CAR_N + 1];
  if (argc > 1 && !strcmp(argv[1], "vectors")) {
    for (size_t v = 0; v < sizeof VEC / sizeof VEC[0]; v++) {
      if (ob_fmt_encode(&VEC[v].f, cw, BS[VEC[v].a])) { printf("vector %zu: refused\n", v); continue; }
      hexof(cw, BS[VEC[v].a], hex);
      printf("ring %d kind %d: %s\n", RB[VEC[v].a], VEC[v].f.kind, hex);
    }
    uint8_t hdr[3 * OB_CAR_N], sym[3], content[OB_CAR_CONTENT_MAX];
    const int len = ob_car_content(&CAR_CFG, content), k = ob_car_header(content, len, hdr);
    hexof(hdr, 3 * k, hex);
    printf("carousel k %d header %s\n", k, hex);
    for (size_t i = 0; i < sizeof CAR_SYM / sizeof CAR_SYM[0]; i++) { ob_car_symbol(hdr, k, CAR_SYM[i].id, sym); hexof(sym, 3, hex); printf("  symbol %d %s\n", CAR_SYM[i].id, hex); }
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
  { const ob_fmt_t f = { .version = 4, .fps = 60, .count = { 4, 1, 1 } }, v3 = { .version = 3, .fps = 60, .count = { 4, 1, 1 } };
    const ob_fmt_t v67 = { .version = 67, .fps = 60, .count = { 60, 18, 18 } }, v68 = { .version = 68, .fps = 60, .count = { 60, 18, 18 } };
    const ob_fmt_t f0 = { .version = 4, .fps = 0, .count = { 4, 1, 1 } }, f11 = { .version = 4, .fps = 11, .count = { 4, 1, 1 } }, f61 = { .version = 4, .fps = 61, .count = { 4, 1, 1 } };
    const ob_fmt_t full = { .version = 4, .fps = 60, .count = { 9, 0, 0 } }, pil = { .version = 4, .fps = 60, .count = { 0, 0, 4 } }, big = { .version = 67, .fps = 60, .count = { 128, 0, 0 } };
    const ob_fmt_t p0 = { .kind = OB_FMT_ANIM, .id = 0 }, p254 = { .kind = OB_FMT_ANIM, .id = 254 }, p255 = { .kind = OB_FMT_ANIM, .id = 255 }, k2 = { .version = 4, .fps = 60, .count = { 4, 1, 1 }, .kind = 2 };
    CHECK(ob_fmt_encode(&f, cw, 16) == 0 && ob_fmt_encode(&f, cw, 15) < 0 && ob_fmt_encode(&f, cw, OB_FMT_BYTES_MAX + 1) < 0, "encode lengths");
    CHECK(ob_fmt_encode(&v67, cw, 16) == 0 && ob_fmt_encode(&v3, cw, 16) < 0 && ob_fmt_encode(&v68, cw, 16) < 0, "encode versions 4 to 67");
    CHECK(ob_fmt_encode(&f0, cw, 16) < 0 && ob_fmt_encode(&f11, cw, 16) < 0 && ob_fmt_encode(&f61, cw, 16) < 0, "encode rates off the table");
    // ai: no room for one 3/4 block (nine 7/8 blocks fill LIZARD-64's 40,960 slots past 36,808), one block with pilots
    // ai: (four 1/2 blocks leave one 3/4), a count past 127
    CHECK(ob_fmt_encode(&full, cw, 16) < 0 && ob_fmt_encode(&pil, cw, 16) < 0 && ob_fmt_encode(&big, cw, 16) < 0, "encode counts that are no frame");
    CHECK(ob_fmt_encode(&p0, cw, 16) == 0 && ob_fmt_encode(&p254, cw, 16) == 0 && ob_fmt_encode(&p255, cw, 16) < 0 && ob_fmt_encode(&k2, cw, 16) < 0, "encode ids and kinds");
    // ai: the layout: the 3/4 filling, and the default senders' at 656 sub-channels (36, 21, 11, 11)
    int m = 0;
    CHECK(ob_fmt_layout(41, CAR_CFG.count, &m) && m == 21, "layout at version 41: 3/4 x %d", m);
    CHECK(ob_fmt_layout(4, f.count, &m) && m == 1, "layout at version 4: 3/4 x %d", m);
    CHECK(ob_fmt_rate_code(60) == OB_FMT_RATES - 1 && ob_fmt_rate_code(1) == 0 && ob_fmt_rate_code(24) == 16 && ob_fmt_rate_code(11) < 0, "rate codes");
    // ai: the held configuration round trip, and 0 none
    ob_fmt_t g;
    CHECK(ob_fmt_held(&v67) > 0 && ob_fmt_unheld(ob_fmt_held(&v67), &g) && g.version == 67 && !memcmp(g.count, v67.count, sizeof g.count), "held round trip");
    CHECK(ob_fmt_held(&p0) == 0 && !ob_fmt_unheld(0, &g) && !ob_fmt_unheld(-1, &g), "held none"); }

  for (int a = 0; a < NR; a++) {
    const int W = WS[a], B = BS[a], nroots = B - OB_FMT_DATA, t = nroots / 2;
    // ai: The erasure steps' last (src/fmt.c): nroots - 3, three roots left in every ring. Weak bytes past it are errors
    // ai: over the roots it leaves, so reach is the most weak bytes a word reads with.
    const int last = nroots - 3, reach = last + (nroots - last) / 2;
    int ok = 0, wrong = 0, tries = 0;
    printf("RS(%d,5), ring %d, %d word cells a side, %d bytes a side, %d roots\n", B, RB[a], W, B / 4, nroots);
    // Round trips, with cell noise, static and animated.
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
      raw_word(c, 0x4d, bits_of(8, 31, 8, 2, 2), nroots); soften(c, B, W, 0.4f, q);
      { ob_fmt_t g; CHECK(ob_fmt_decode(q, W, B, &g) < 0, "magic 0x4d"); }
      raw_word(c, 0xb2, bits_of(8, 31, 8, 2, 2), nroots); soften(c, B, W, 0.4f, q);
      { ob_fmt_t g; CHECK(ob_fmt_decode(q, W, B, &g) < 0, "magic 0xb2"); }
      // ai: counts that are no frame: no room for a 3/4 block, one block with pilots, every count 127 at version 4
      static const int NOT_A_FRAME[][4] = { { 4, 9, 0, 0 }, { 4, 0, 0, 4 }, { 4, 127, 127, 127 }, { 67, 127, 127, 127 } };
      for (int k = 0; k < 4; k++) {
        raw_word(c, OB_FMT_MAGIC, bits_of(NOT_A_FRAME[k][0], 31, NOT_A_FRAME[k][1], NOT_A_FRAME[k][2], NOT_A_FRAME[k][3]), nroots); soften(c, B, W, 0.4f, q);
        ob_fmt_t g;
        CHECK(ob_fmt_decode(q, W, B, &g) < 0, "version %d counts %d %d %d", NOT_A_FRAME[k][0], NOT_A_FRAME[k][1], NOT_A_FRAME[k][2], NOT_A_FRAME[k][3]);
      }
      { const ob_fmt_t w = { .version = 67, .fps = 60, .count = { 127, 0, 0 } };
        raw_word(c, OB_FMT_MAGIC, bits_of(67, 31, 127, 0, 0), nroots); soften(c, B, W, 0.4f, q); CHECK(read_word(q, W, B, &w) == 1, "version 67, 127 7/8 blocks"); }
      { const ob_fmt_t w = { .version = 4, .fps = 1, .count = { 0, 0, 0 } };
        raw_word(c, OB_FMT_MAGIC, bits_of(4, 0, 0, 0, 0), nroots); soften(c, B, W, 0.4f, q); CHECK(read_word(q, W, B, &w) == 1, "version 4, 3/4 blocks alone"); }
      c[0] = 0xb3; c[1] = 255; memset(c + 2, 0, 3); rs_encode(c, OB_FMT_DATA, nroots); soften(c, B, W, 0.4f, q);
      { ob_fmt_t g; CHECK(ob_fmt_decode(q, W, B, &g) < 0, "id 255"); }
      c[1] = 254; c[2] = 7; c[3] = 8; c[4] = 9; rs_encode(c, OB_FMT_DATA, nroots); soften(c, B, W, 0.4f, q);
      { const ob_fmt_t w = { .kind = OB_FMT_ANIM, .id = 254, .sym = { 7, 8, 9 } }; CHECK(read_word(q, W, B, &w) == 1, "id 254"); }
    }

    // ai: Garbage: soft values uniform in -0.5 .. 0.5, what a wrong grid reads. Exact rates first: RS passing a step,
    // ai: the steps together, then with a magic and its check (a false codeword's data bytes are uniform: 1/256 for each
    // ai: magic, then the share of the static word's 32 bits whose counts make a frame, `fits`, and 255/256 for an
    // ai: animated word's id). Then measured through ob_fmt_decode, with the rate the steps to nroots - 1 would have
    // ai: beside it.
    const int other = nroots - 1;
    double steps = p_pass(B, 0), alt = p_pass(B, 0);
    for (int f = 2; f <= last; f += 2) steps += p_pass(B, f);
    for (int f = 2; f <= other; f += 2) alt += p_pass(B, f);
    static double fits = -1;
    if (fits < 0) {
      long long ok = 0;
      for (int v = OB_FMT_VERSION_MIN; v <= OB_FMT_VERSION_MAX; v++)
        for (int x = 0; x < 128; x++) for (int y = 0; y < 128; y++) for (int z = 0; z < 128; z++) { const int k[3] = { x, y, z }; ok += ob_fmt_layout(v, k, 0); }
      fits = (double)ok / ((double)NV * 128 * 128 * 128);
      printf("static words whose counts make a frame: %.4f of the 32 bits' values\n", fits);
    }
    const double mv = steps * (fits + (OB_FMT_ID_MAX + 1) / 256.0) / 256.0;
    long acc = 0;
    for (long it = 0; it < NG; it++) {
      for (int i = 0; i < 4 * W; i++) q[i] = uni() - 0.5f;
      ob_fmt_t f;
      acc += ob_fmt_decode(q, W, B, &f) == 0;
    }
    printf("  garbage, exact: errors only %.1e, steps to %d %.1e, a magic and its check %.1e, static alone %.1e (steps to %d: %.1e)\n",
           p_pass(B, 0), last, steps, mv, steps * fits / 256.0, other, alt * (fits + (OB_FMT_ID_MAX + 1) / 256.0) / 256.0);
    printf("  garbage, measured over %ld reads: %ld accepted (expected %.2f)\n", NG, acc, mv * NG);
    CHECK(acc <= 10 + 4 * mv * NG, "garbage accepted %ld of %ld", acc, NG);
  }
  // The carousel. Sizes: k the least that holds the content and its CRC, at least 2; 761 bytes the most.
  { uint8_t hdr[3 * OB_CAR_N], content[3 * OB_CAR_N] = { 0 }, sym[3];
    CHECK(ob_car_header(content, 0, hdr) == 2 && ob_car_header(content, 2, hdr) == 2 && ob_car_header(content, 3, hdr) == 3, "header sizes");
    CHECK(ob_car_header(content, 761, hdr) == 255 && ob_car_header(content, 762, hdr) < 0, "the largest header");
    CHECK(ob_car_symbol(hdr, 2, 255, sym) < 0 && ob_car_symbol(hdr, 1, 0, sym) < 0, "symbol range");
    // ai: the vector: the default header, the static word's five bytes, k 3, read back as its configuration
    int len = ob_car_content(&CAR_CFG, content), k = ob_car_header(content, len, hdr);
    hexof(hdr, 3 * k, hex);
    CHECK(len == 5 && k == 3 && !strcmp(hex, CAR_HDR), "carousel header k %d %s", k, hex);
    for (size_t i = 0; i < sizeof CAR_SYM / sizeof CAR_SYM[0]; i++) { ob_car_symbol(hdr, k, CAR_SYM[i].id, sym); hexof(sym, 3, hex); CHECK(!strcmp(hex, CAR_SYM[i].hex), "carousel symbol %d %s", CAR_SYM[i].id, hex); }
    ob_fmt_t g;
    CHECK(ob_car_config(hdr, k, &g) && g.version == 41 && g.fps == 60 && !memcmp(g.count, CAR_CFG.count, sizeof g.count), "carousel configuration");
    // ai: a rate off the table: its record, the five bytes stating the nearest table rate (240: 60), k 5, read back exact
    { ob_fmt_t r = CAR_CFG; r.fps = 240;
      len = ob_car_content(&r, content); k = ob_car_header(content, len, hdr);
      CHECK(len == 9 && k == 5 && ob_car_config(hdr, k, &g) && g.fps == 240 && g.version == 41, "carousel at 240 a second: k %d, fps %d", k, g.fps);
      r.fps = 0; CHECK(ob_car_content(&r, content) < 0, "carousel at 0 a second refused");
      // ai: a record past the content refused; an unknown one skipped
      memset(content, 0, sizeof content); ob_fmt_pack(&CAR_CFG, content); content[5] = 9; content[6] = 40;
      k = ob_car_header(content, 9, hdr); CHECK(!ob_car_config(hdr, k, &g), "a record past the content");
      content[6] = 2; k = ob_car_header(content, 9, hdr); CHECK(ob_car_config(hdr, k, &g) && g.fps == 60, "an unknown record skipped"); }
    // ai: any k symbols, in any order, give the header, and fewer give none; then e wrong symbols among k + 2e, read once
    // ai: each, corrected; a wrong read outvoted by two right ones; garbage symbols never give one
    static ob_car_rx_t rx;
    int early = 0, wrongk = 0, got = 0, fixed = 0, outvoted = 0, trials = 0;
    for (int it = 0; it < 120; it++) {
      const int len = 4 + (int)(rnd() % 60), e = it % 3;
      for (int i = 0; i < len; i++) content[i] = (uint8_t)rnd();
      const int kk = ob_car_header(content, len, hdr);
      int order[OB_CAR_N];
      for (int p = 0; p < OB_CAR_N; p++) order[p] = p;
      for (int p = OB_CAR_N - 1; p > 0; p--) { const int j = (int)(rnd() % (uint32_t)(p + 1)), t = order[p]; order[p] = order[j]; order[j] = t; }
      ob_car_reset(&rx);
      int at = 0;
      for (int i = 0; i < kk + 2 * e && i < OB_CAR_N; i++) {
        ob_car_symbol(hdr, kk, order[i], sym);
        if (i < e) sym[rnd() % 3] ^= (uint8_t)(1 + rnd() % 255);
        const int r = ob_car_add(&rx, order[i], sym);
        if (r && !at) at = i + 1;
      }
      trials++;
      if (at && at < kk) early++;
      if (rx.k && rx.k != kk) wrongk++;
      if (rx.k == kk && !memcmp(rx.hdr, hdr, (size_t)(3 * kk))) { got++; fixed += e > 0; }
      // ai: the wrong symbols read twice more, right: the votes flip and the header stands as before
      for (int i = 0; i < e; i++) for (int t = 0; t < 2; t++) { ob_car_symbol(hdr, kk, order[i], sym); ob_car_add(&rx, order[i], sym); }
      outvoted += rx.k == kk && !memcmp(rx.hdr, hdr, (size_t)(3 * kk));
    }
    printf("carousel: %d of %d headers assembled (%d with wrong symbols among them), %d after the wrong ones were outvoted; %d early, %d at another k\n", got, trials, fixed, outvoted, early, wrongk);
    CHECK(got == trials && outvoted == trials && !early && !wrongk, "carousel assembly");
    long false_headers = 0;
    for (int it = 0; it < 120; it++) {
      ob_car_reset(&rx);
      const int m = 2 + (int)(rnd() % 40);
      for (int i = 0; i < m; i++) { for (int j = 0; j < 3; j++) sym[j] = (uint8_t)rnd(); ob_car_add(&rx, (int)(rnd() % OB_CAR_N), sym); }
      false_headers += rx.k != 0;
    }
    printf("carousel garbage: %ld headers from 120 sets of 2 to 41 random symbols\n", false_headers);
    CHECK(!false_headers, "carousel garbage"); }
  printf(fails ? "%d FAILED\n" : "all passed\n", fails);
  return fails != 0;
}
