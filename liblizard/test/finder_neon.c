// ai: The finder (src/acquire.c, 2026-11-24) on synthetic captures, every exact output hashed: the grey pyramid, the
// ai: grey marks' dense pass (dense16) and peaks (peaks16) at each level searched, the candidates as find_frame takes them
// ai: (gray_cands: the re-measure and the ranking), and the traced border's quads (trace_quads), on codes of the three
// ai: rings painted through a homography with a little perspective, sharp and blurred, over noise, at sizes whose rows
// ai: and columns end ragged. The vector forms (arm64's NEON, x86-64's SSE4.1 and the wasm's, through simd.h) are the
// ai: scalar's integers lane for lane and the rest is the same C, so a build with -DOB_SCALAR prints the same hash (the
// ai: hash at the scalar twins' value, FINDER_HASH: the test fails where it differs). Then each capture through
// ai: find_frame, found and by which path (the track reads' own floats on arm64 and x86-64: counted, not hashed).
// ai: test/neon_kernels.sh builds it both ways on a phone and compares. It includes acquire.c and links the rest of the
// ai: codec:
// ai:   cc -O2 -ffp-contract=off -Isrc test/finder_neon.c src/{layout,fmt,rs,ldpc,fft,lizard_phy,any}.c -lm
#include "acquire.c"
#include <stdio.h>
uint32_t ob_debug_hash;
static uint64_t hs = 1469598103934665603ull;
// ai: the hash as the scalar twins give it (2026-11-24)
#define FINDER_HASH 0x4f13f77f6f5fa9fdull
static void mix(const void *p, size_t n) { const uint8_t *b = p; for (size_t i = 0; i < n; i++) { hs ^= b[i]; hs *= 1099511628211ull; } }
static uint32_t rs = 7;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static float frand(void) { return (float)(rnd() & 0xffff) / 65536.0f; }
int main(void) {
  static const int SIZE[6][2] = { { 1280, 960 }, { 1277, 957 }, { 640, 480 }, { 1920, 1080 }, { 803, 1203 }, { 1500, 1001 } };
  static const int sides[3] = { 158, 286, 542 };
  long ncand = 0, nquad = 0, found = 0, bymarks = 0;
  double tgrey = 0, ttrace = 0, tfind = 0;
  for (int t = 0; t < 24; t++) {
    const int iw = SIZE[t % 6][0], ih = SIZE[t % 6][1];
    uint8_t *px = malloc((size_t)iw * ih), *tmp = malloc((size_t)iw * ih);
    ob_layout_t L;
    ob_layout_init(&L, sides[t % 3]);
    // ai: the code over a fill of 0.55 to 0.9, turned up to 20 degrees, with its corners moved
    const float cx = iw * (0.45f + 0.1f * frand()), cy = ih * (0.45f + 0.1f * frand());
    const float s = 0.5f * (iw < ih ? iw : ih) * (0.55f + 0.35f * frand()), an = (frand() - 0.5f) * 0.7f, ca = cosf(an), sa = sinf(an);
    float quad[8];
    static const float CX[4] = { -1, 1, 1, -1 }, CY[4] = { -1, -1, 1, 1 };
    for (int k = 0; k < 4; k++) {
      const float x = CX[k] * s + (frand() - 0.5f) * 0.1f * s, y = CY[k] * s + (frand() - 0.5f) * 0.1f * s;
      quad[2 * k] = cx + ca * x - sa * y; quad[2 * k + 1] = cy + sa * x + ca * y;
    }
    float src[8];
    corner_coords(&L, src);
    homo_t Hi;
    homography(quad, src, &Hi);
    const int grain = 1 + (int)(rnd() % 6);
    for (int y = 0; y < ih; y++) for (int x = 0; x < iw; x++) {
      float mx, my;
      project(&Hi, x + 0.5f, y + 0.5f, &mx, &my);
      const int ix = (int)floorf(mx), iy = (int)floorf(my);
      int v = (int)((((x / grain) * 2654435761u) ^ ((y / grain) * 2246822519u)) >> 24);   // a background of blocks
      if (ix >= 0 && iy >= 0 && ix < L.w && iy < L.h) {
        const int k = L.kind[iy * L.w + ix];
        v = k != CELL_DATA ? ((k & 3) == CELL_DARK ? 40 : 215) + (int)(rnd() % 21) - 10 : (int)(rnd() & 255);
      }
      px[y * iw + x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
    // ai: every third capture blurred (a 3 x 3 box, twice), which closes the marks' gaps for the merged form
    for (int pass = 0; t % 3 == 2 && pass < 2; pass++) {
      memcpy(tmp, px, (size_t)iw * ih);
      for (int y = 1; y < ih - 1; y++) for (int x = 1; x < iw - 1; x++) {
        int a = 0;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) a += tmp[(y + dy) * iw + x + dx];
        px[y * iw + x] = (uint8_t)((a + 4) / 9);
      }
    }
    fctx_t f = { NULL, { px }, iw, ih, { 0 } };
    int lv[5];
    const int nlv = gray_levels(iw, ih, lv);
    for (int il = 0; il < nlv; il++) {
      int li = 0;
      while ((1 << li) < lv[il]) li++;
      const uint8_t *g = fctx_level(&f, li);
      const int w = iw >> li, h = ih >> li, pw = ((w + 7) & ~7) + 16;
      mix(g, (size_t)w * h);
      uint16_t *R = malloc((size_t)w * h * sizeof(uint16_t)), *ring = calloc((size_t)2 * 2 * GRING * pw + (size_t)2 * 2 * pw + (size_t)8 * w, sizeof(uint16_t));
      const gside_t sd[2] = { gside(3), gside(4) };
      dense16(g, w, h, sd, 2, R, ring, ring + (size_t)2 * 2 * GRING * pw, pw);
      mix(R, (size_t)w * h * sizeof(uint16_t));
      int xs[GRAY_MAX], ys[GRAY_MAX];
      uint16_t pr[GRAY_MAX];
      const int np = peaks16(R, w, h, GRAY_MAX, xs, ys, pr, ring + (size_t)2 * 2 * GRING * pw + (size_t)2 * 2 * pw);
      mix(xs, np * sizeof(int)); mix(ys, np * sizeof(int)); mix(pr, np * sizeof(uint16_t));
      free(R); free(ring);
      cand_t c[GRAY_MAX];
      float rc[GRAY_MAX];
      const double t0 = ob_now_ms();
      const int n = gray_cands(&f, lv[il], c, rc);
      tgrey += ob_now_ms() - t0;
      mix(c, n * sizeof(cand_t)); mix(rc, n * sizeof(float));
      ncand += n;
    }
    tquad_t tq[TR_MAXQ];
    const double t1 = ob_now_ms();
    const int nq = trace_quads(&f, 30, tq, TR_MAXQ);
    ttrace += ob_now_ms() - t1;
    mix(tq, nq * sizeof(tquad_t));
    nquad += nq;
    fctx_done(&f);
    const ob_layout_t *Ls[1] = { &L };
    image_t im;
    ob_image_init(&im, px, iw, ih, 1);
    float q[8], sc;
    int orient = -1, which = 0, seeded = 0;
    lensm_t seed;
    const double t2 = ob_now_ms();
    if (find_frame(Ls, 1, &im, NULL, q, &orient, &sc, &which, &seed, &seeded)) { found++; bymarks += ob_finder_info[4] == 0; }
    if (getenv("FINDER_EACH")) printf("capture %2d ring %d %dx%d%s: score %.3f orient %d path %.0f quads %d\n", t, L.w, iw, ih, t % 3 == 2 ? " blurred" : "", sc, orient, ob_finder_info[4], nq);
    tfind += ob_now_ms() - t2;
    free(px); free(tmp);
  }
  const int ok = hs == FINDER_HASH;
  printf("finder %016llx%s: %ld candidates, %ld traced quads; find_frame found %ld of 24 (%ld by the marks)\n",
         (unsigned long long)hs, ok ? "" : " (DIFFERS from FINDER_HASH)", ncand, nquad, found, bymarks);
  printf("time ms: grey %.1f, trace %.1f, find_frame %.1f\n", tgrey, ttrace, tfind);
  return ok ? 0 : 1;
}
