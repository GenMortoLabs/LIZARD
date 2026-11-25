// Registration: find the border (its corner marks, or its ring traced), fix orientation by the timing track, hang a mesh
// on the border's nodes. Nothing is thresholded: the finder reads the grey picture (2026-11-24; the border's search
// binarized the capture before).
#include "internal.h"
#include "fmt.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ai: Off wasm the clock is the wall's: clock() is the whole process's CPU time, which a receiver on several threads
// ai: cannot time a stage by.
#ifdef __EMSCRIPTEN__
double ob_now_ms(void) { return 1000.0 * clock() / CLOCKS_PER_SEC; }
#else
double ob_now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return 1e3 * (double)t.tv_sec + 1e-6 * (double)t.tv_nsec; }
#endif
OB_TLS double ob_prof_ms[PROF_N];
// The quad the finder settled on, BEFORE the corner nodes refine it. res->quad is the refined one, and handing
// that back to ob_acquire_quad would refine it a second time; this is what a finder running elsewhere has to
// reproduce and what it has to be compared against.
// What the thin frame's finder settled on, for a finder elsewhere to be held to: the final
// score and orient, then which form and which mark quad won, the source of the final quad (0 marks, 1 lines,
// -1 none), whether the marks alone settled it, the mark stage's best, and form 0's quad count.
OB_TLS float ob_finder_info[8];
// ai: The lens's axis in the next image this thread decodes (lizard_set_axis), image px; NaN for the image's centre.
static OB_TLS float ob_axis[2] = { NAN, NAN };
void lizard_set_axis(float x, float y) { ob_axis[0] = x; ob_axis[1] = y; }
// ai: The anchors (2026-10-26; part of every symbol since 2026-10-27, as the pilots are; SPEC 6.13,
// ai: research/build/anchors/PLAN.md): a k x k grid of squares inside the picture, by the ring: 3 x 3 of 4 modules a
// ai: side in the 64 ring, 5 x 5 of 6 in the 128, 7 x 7 of 8 in the 256 (since 2026-11-09; 7 x 7 of span / 64 modules
// ai: in every ring before: 2, 4, 8), square (i, j) on the modules from round(OB_MARGIN + span (i + 0.5) / k - a / 2)
// ai: on, each a light middle (Chebyshev distance under a quarter of the half-side), a dark ring (to 0.6 of it) and a
// ai: light rim. lizard_encode paints them over the picture. acquire_core finds them through the corners' homography
// ai: first, three modules wide, in rounds through the lens model the ones found so far give (anchor_register), starts
// ai: every border node's search from that model plus its measured neighbours' residuals from it, and fills the
// ai: interior from the border and the anchors (border_fill_anchors) where it found ob_anchor_min or more (12; 8 of the
// ai: 64 ring's 9); lizard_finish erases them always.
//
// ai: the samples erased round an anchor's square (lizard_phy.c): two, measured on the published lenses' captures (one left
// ai: the blurred rim in where the map sits a few tenths of a sample off: 26 of 192 frames lost blocks; three read as two)
// ai: the anchors the last acquisition counted (anchor_register's, read out by the wasm's lizard_anchors_found for the
// ai: lab's checks), -1 before the thread's first registration
OB_TLS int ob_anch_found = -1;
// ai: the ring by its span (128, 256, 512 modules: the 64, 128, 256 rings)
int ob_anchor_k(const ob_layout_t *L) { const int span = L->w - 2 * OB_MARGIN; return span <= 128 ? 3 : span <= 256 ? 5 : OB_ANCH_K; }
int ob_anchor_side(const ob_layout_t *L) { const int span = L->w - 2 * OB_MARGIN; return span <= 128 ? 4 : span <= 256 ? 6 : 8; }
int ob_anchor_min(const ob_layout_t *L) { const int k = ob_anchor_k(L); return k * k - 1 < OB_ANCH_MIN ? k * k - 1 : OB_ANCH_MIN; }
int ob_anchor_m0(const ob_layout_t *L, int i) {
  const int span = L->w - 2 * OB_MARGIN, a = ob_anchor_side(L);
  return (int)floor(OB_MARGIN + (double)span * (i + 0.5) / ob_anchor_k(L) - a / 2.0 + 0.5);
}
// ai: the lens model the anchors give (border_fill_anchors' fit), evaluated at an image point: defined further on
struct lensm;
static void lensm_at(const struct lensm *m, double u, double v, float *dx, float *dy);

void ob_image_init(image_t *im, const uint8_t *px, int w, int h, float gamma) {
  im->px = px; im->w = w; im->h = h;
  for (int i = 0; i < 256; i++) im->lut[i] = gamma == 1.0f ? i / 255.0f : powf(i / 255.0f, gamma);
  im->lin = gamma == 1.0f;
}

// ---------------------------------------------------------------- finder search

#include "simd.h"

// RGBA to luma, (77 r + 150 g + 29 b + 128) >> 8. The weights sum to 256, so the sum fits 16 bits
// and sixteen pixels go through as two lanes of eight.
void ob_luma(const uint8_t *rgba, int n, uint8_t *out) {
  int i = 0;
#ifdef OB_SIMD
  const v128_t cr = wasm_i16x8_splat(77), cg = wasm_i16x8_splat(150), cb = wasm_i16x8_splat(29), half = wasm_i16x8_splat(128);
  for (; i + 16 <= n; i += 16) {
    const v128_t p0 = wasm_v128_load(rgba + 4 * i), p1 = wasm_v128_load(rgba + 4 * i + 16), p2 = wasm_v128_load(rgba + 4 * i + 32), p3 = wasm_v128_load(rgba + 4 * i + 48);
    // Two rounds of byte shuffles pull each channel of sixteen pixels into one vector.
    const v128_t r01 = wasm_i8x16_shuffle(p0, p1, 0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29), b01 = wasm_i8x16_shuffle(p0, p1, 2, 6, 10, 14, 18, 22, 26, 30, 0, 0, 0, 0, 0, 0, 0, 0);
    const v128_t r23 = wasm_i8x16_shuffle(p2, p3, 0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29), b23 = wasm_i8x16_shuffle(p2, p3, 2, 6, 10, 14, 18, 22, 26, 30, 0, 0, 0, 0, 0, 0, 0, 0);
    const v128_t r = wasm_i8x16_shuffle(r01, r23, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23), g = wasm_i8x16_shuffle(r01, r23, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
    const v128_t b = wasm_i8x16_shuffle(b01, b23, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
    #define LUM8(ext) wasm_u16x8_shr(wasm_i16x8_add(wasm_i16x8_add(wasm_i16x8_mul(ext(r), cr), wasm_i16x8_mul(ext(g), cg)), wasm_i16x8_add(wasm_i16x8_mul(ext(b), cb), half)), 8)
    wasm_v128_store(out + i, wasm_u8x16_narrow_i16x8(LUM8(wasm_u16x8_extend_low_u8x16), LUM8(wasm_u16x8_extend_high_u8x16)));
    #undef LUM8
  }
#endif
  for (; i < n; i++) out[i] = (uint8_t)((77 * rgba[4 * i] + 150 * rgba[4 * i + 1] + 29 * rgba[4 * i + 2] + 128) >> 8);
}

// ai: an acquisition's buffer slot of n bytes (internal.h ob_fbufs_t): on arm64 and x86-64 the workspace's, grown where it
// ai: is short (its old bytes not kept), and frame_buf_done leaves it there; with no workspace, or elsewhere, a fresh
// ai: allocation frame_buf_done frees
#if defined(OB_NEON) || defined(OB_X86)
static void *frame_buf(ob_fbufs_t *fb, int slot, size_t n) {
  if (!fb) return malloc(n);
  if (fb->cap[slot] < n) { free(fb->p[slot]); fb->p[slot] = malloc(n); fb->cap[slot] = fb->p[slot] ? n : 0; }
  return fb->p[slot];
}
static void frame_buf_done(ob_fbufs_t *fb, void *p) { if (!fb) free(p); }
#else
#define frame_buf(fb, slot, n) ((void)(fb), malloc(n))
#define frame_buf_done(fb, p) ((void)(fb), free(p))
#endif
void ob_fbufs_free(ob_fbufs_t *fb) { for (int k = 0; k < OB_FB_SLOTS; k++) { free(fb->p[k]); fb->p[k] = NULL; fb->cap[k] = 0; } }
// ai: The node templates read a blurred image (box_blur) whenever the box has a radius, r = floor(mpx / 2) pixels,
// ai: so from a module of 2 px (2026-09-29: the threshold of 2.0 px this replaced was exactly r >= 1, the same
// ai: bytes). The fifteen recorded phone runs (scripts/exp/capture_check.mjs) had tried later starts: every threshold from 2
// ai: to 3 took 31005 blocks to 44200 where 2 took them to 45477, one run the difference (7677 to 8953); on the
// ai: simulator 2 and 3 the same under 2 px and within a few percent either way between them (LIZARD-128 at 45
// ai: degrees 2843 to 3166 B).
// Why the node templates read a blurred image when a module is several pixels across (border_blur). A template
// reads one point a module, at its centre, and that point only moves when an edge reaches it: at 8 px a module it
// sits 4 px from every edge, the correlation is flat over most of a module and the node cannot be placed to better
// than a large part of one. Measured on LIZARD-16 at 94 modules, 1080 fill 0.7: corners 0.76 px RMS off the truth
// against 0.16 on the 286-module grid, and half the blocks gone at 30 degrees. Under a box as wide as a module the
// centre point IS the module's mean, which moves linearly with the offset, so the peak is sharp again at any size.
//
// One rectangle [x0, x1) x [y0, y1) of the blurred image written into out, reading real pixels past its edges,
// so a set of rectangles blurs exactly as the whole image would inside them. Rows then columns by running sums,
// the image's own edges clamped, rounded back to bytes. tmp holds (x1 - x0) * (y1 - y0 + 2r) and acc (x1 - x0).
static void box_blur_rect(const uint8_t *src, uint8_t *out, int w, int h, int r, int x0, int y0, int x1, int y1, uint16_t *tmp, int *acc) {
  const int rw = x1 - x0, rows = y1 - y0 + 2 * r, kk = (2 * r + 1) * (2 * r + 1);
#ifdef OB_SIMD
  // ai: the same integer sums a vector at a time, so the same bytes. A row's sums directly, 2r + 1 widened byte loads per
  // ai: eight columns where no read is clamped (the clamped columns by the scalar sum); the column sums in 16-bit lanes
  // ai: (at most 255 (2r + 1)^2, 57,375 at r 7, and a sum less its leaving row before the entering one is added); the
  // ai: rounded division by kk a multiply and a shift, exact for sums under 2^32 / kk^2 (85,000 and more at r 7)
  if (r <= 7) {
    // the columns whose vector reads x0 + x - r .. x0 + x + r + 15 all lie in the row: x from xa while x <= xz
    const int xa = r - x0 > 0 ? r - x0 : 0, xz = w - 16 - r - x0;
    for (int j = 0; j < rows; j++) {
      const int sy = y0 - r + j;
      const uint8_t *srow = src + (size_t)(sy < 0 ? 0 : sy >= h ? h - 1 : sy) * w;
      uint16_t *t = tmp + (size_t)j * rw;
      int x = 0;
      for (; x < rw && x < xa; x++) { int a = 0; for (int i = x0 + x - r; i <= x0 + x + r; i++) a += srow[i < 0 ? 0 : i >= w ? w - 1 : i]; t[x] = (uint16_t)a; }
      for (; x + 8 <= rw && x <= xz; x += 8) {
        v128_t a = wasm_i16x8_splat(0);
        for (int sft = -r; sft <= r; sft++) a = wasm_i16x8_add(a, wasm_u16x8_extend_low_u8x16(wasm_v128_load(srow + x0 + x + sft)));
        wasm_v128_store(t + x, a);
      }
      for (; x < rw; x++) { int a = 0; for (int i = x0 + x - r; i <= x0 + x + r; i++) a += srow[i < 0 ? 0 : i >= w ? w - 1 : i]; t[x] = (uint16_t)a; }
    }
    uint16_t *col = (uint16_t *)acc;   // rw ints hold rw 16-bit sums twice over
    int x = 0;
    for (; x + 8 <= rw; x += 8) {
      v128_t c = wasm_i16x8_splat(0);
      for (int j = 0; j < 2 * r + 1; j++) c = wasm_i16x8_add(c, wasm_v128_load(tmp + (size_t)j * rw + x));
      wasm_v128_store(col + x, c);
    }
    for (; x < rw; x++) { int c = 0; for (int j = 0; j < 2 * r + 1; j++) c += tmp[(size_t)j * rw + x]; col[x] = (uint16_t)c; }
    const uint64_t M = ((uint64_t)1 << 32) / (uint64_t)kk + 1;
    const unsigned half = (unsigned)kk / 2;
    for (int y = y0; y < y1; y++) {
      uint8_t *o = out + (size_t)y * w + x0;
      for (int xx = 0; xx < rw; xx++) o[xx] = (uint8_t)(((uint64_t)(col[xx] + half) * M) >> 32);
      if (y + 1 == y1) break;
      const uint16_t *ta = tmp + (size_t)(y - y0 + 2 * r + 1) * rw, *ts = tmp + (size_t)(y - y0) * rw;
      int xx = 0;
      for (; xx + 8 <= rw; xx += 8) wasm_v128_store(col + xx, wasm_i16x8_add(wasm_i16x8_sub(wasm_v128_load(col + xx), wasm_v128_load(ts + xx)), wasm_v128_load(ta + xx)));
      for (; xx < rw; xx++) col[xx] = (uint16_t)(col[xx] - ts[xx] + ta[xx]);
    }
    return;
  }
#endif
  for (int j = 0; j < rows; j++) {
    const int sy = y0 - r + j;
    const uint8_t *srow = src + (size_t)(sy < 0 ? 0 : sy >= h ? h - 1 : sy) * w;
    uint16_t *t = tmp + (size_t)j * rw;
    int a = 0;
    for (int i = x0 - r; i <= x0 + r; i++) a += srow[i < 0 ? 0 : i >= w ? w - 1 : i];
    for (int x = 0; x < rw; x++) {
      t[x] = (uint16_t)a;
      const int add = x0 + x + r + 1, sub = x0 + x - r;
      a += srow[add >= w ? w - 1 : add] - srow[sub < 0 ? 0 : sub];
    }
  }
  for (int x = 0; x < rw; x++) acc[x] = 0;
  for (int j = 0; j < 2 * r + 1; j++) { const uint16_t *t = tmp + (size_t)j * rw; for (int x = 0; x < rw; x++) acc[x] += t[x]; }
  for (int y = y0; y < y1; y++) {
    uint8_t *o = out + (size_t)y * w + x0;
    for (int x = 0; x < rw; x++) o[x] = (uint8_t)((acc[x] + kk / 2) / kk);
    if (y + 1 == y1) break;
    const uint16_t *ta = tmp + (size_t)(y - y0 + 2 * r + 1) * rw, *ts = tmp + (size_t)(y - y0) * rw;
    for (int x = 0; x < rw; x++) acc[x] += ta[x] - ts[x];
  }
}

#ifdef OB_NEON
// ai: arm64's box_blur_rect for r <= 7 and rectangles 16 or more wide: the same integer sums, so the same bytes, with no
// ai: rectangle's worth of row sums: each row's sums are made as the column sums take them in, into a ring of 2r + 2
// ai: rows (ring: (2r + 2) rw), and a row's output and its column sums' step are one pass, from one copy of the column
// ai: sums into the other (col: 2 rw, so the last step's columns, back over ones already done, give the same values).
// ai: A row's sums sixteen columns a step from its rw + 2r pixels, through buf (rw + 2r bytes) only where a read is
// ai: clamped; the last step back over columns already summed.
static inline __attribute__((always_inline)) void msh_row_sums(const uint8_t *src, int w, int h, int r, int x0, int rw, int sy, uint16_t *t, uint8_t *buf) {
  const int lo = x0 - r, n = rw + 2 * r;
  const uint8_t *srow = src + (size_t)(sy < 0 ? 0 : sy >= h ? h - 1 : sy) * w, *p = srow + lo;
  if (lo < 0 || lo + n > w) {
    const int a0 = lo < 0 ? -lo : 0, a1 = (lo + n <= w ? lo + n : w) - lo;
    memset(buf, srow[0], (size_t)a0); memcpy(buf + a0, srow + lo + a0, (size_t)(a1 - a0)); memset(buf + a1, srow[w - 1], (size_t)(n - a1));
    p = buf;
  }
  for (int x = 0;; x += 16) {
    if (x + 16 > rw) x = rw - 16;
    uint16x8_t s0 = vdupq_n_u16(0), s1 = s0;
    for (int sft = 0; sft <= 2 * r; sft++) { const uint8x16_t b = vld1q_u8(p + x + sft); s0 = vaddw_u8(s0, vget_low_u8(b)); s1 = vaddw_high_u8(s1, b); }
    vst1q_u16(t + x, s0); vst1q_u16(t + x + 8, s1);
    if (x + 16 >= rw) break;
  }
}
static void msh_blur_rect(const uint8_t *src, uint8_t *out, int w, int h, int r, int x0, int y0, int x1, int y1, uint16_t *ring, uint16_t *col, uint8_t *buf) {
  const int rw = x1 - x0, R = 2 * r + 2, kk = (2 * r + 1) * (2 * r + 1);
  for (int j = 0; j <= 2 * r; j++) msh_row_sums(src, w, h, r, x0, rw, y0 - r + j, ring + (size_t)j * rw, buf);
  uint16_t *ca = col, *cb = col + rw;
  for (int x = 0;; x += 8) {
    if (x + 8 > rw) x = rw - 8;
    uint16x8_t c = vld1q_u16(ring + x);
    for (int j = 1; j <= 2 * r; j++) c = vaddq_u16(c, vld1q_u16(ring + (size_t)j * rw + x));
    vst1q_u16(ca + x, c);
    if (x + 8 >= rw) break;
  }
  // ai: the rounded division by kk: (c + kk / 2) M >> 32 in 64-bit lanes (M under 2^32), its high half (a 16-bit
  // ai: multiply's high half and a shift, exact at every radius, measured no faster)
  const uint64_t M = ((uint64_t)1 << 32) / (uint64_t)kk + 1;
  const uint32x4_t vh = vdupq_n_u32((uint32_t)kk / 2), vm = vdupq_n_u32((uint32_t)M);
  for (int y = y0; y < y1; y++) {
    uint8_t *o = out + (size_t)y * w + x0;
    const int last = y + 1 == y1, ja = y - y0 + 2 * r + 1;
    uint16_t *ta = ring + (size_t)(ja % R) * rw;
    const uint16_t *ts = ring + (size_t)((y - y0) % R) * rw;
    if (!last) msh_row_sums(src, w, h, r, x0, rw, y0 - r + ja, ta, buf);
    for (int x = 0;; x += 8) {
      if (x + 8 > rw) x = rw - 8;
      const uint16x8_t c = vld1q_u16(ca + x);
      const uint32x4_t lo = vaddw_u16(vh, vget_low_u16(c)), hi = vaddw_high_u16(vh, c);
      const uint32x4_t qlo = vcombine_u32(vshrn_n_u64(vmull_u32(vget_low_u32(lo), vget_low_u32(vm)), 32), vshrn_n_u64(vmull_high_u32(lo, vm), 32));
      const uint32x4_t qhi = vcombine_u32(vshrn_n_u64(vmull_u32(vget_low_u32(hi), vget_low_u32(vm)), 32), vshrn_n_u64(vmull_high_u32(hi, vm), 32));
      vst1_u8(o + x, vmovn_u16(vcombine_u16(vmovn_u32(qlo), vmovn_u32(qhi))));
      if (!last) vst1q_u16(cb + x, vaddq_u16(vsubq_u16(c, vld1q_u16(ts + x)), vld1q_u16(ta + x)));
      if (x + 8 >= rw) break;
    }
    uint16_t *sw = ca; ca = cb; cb = sw;
  }
}
#elif defined(OB_X86)
// ai: x86-64's msh_blur_rect, arm64's above in SSE4.1: the same ring of row sums and the same integer sums, so the same
// ai: bytes. A row's sums sixteen columns a step (bytes widened by PMOVZXBW and PUNPCKHBW); the rounded division by kk
// ai: as (c + kk / 2) M >> 32, the 64-bit products' high halves by PMULUDQ on the even and the odd lanes, packed back
// ai: to bytes (every quotient at most 255).
static inline __attribute__((always_inline)) void msh_row_sums(const uint8_t *src, int w, int h, int r, int x0, int rw, int sy, uint16_t *t, uint8_t *buf) {
  const int lo = x0 - r, n = rw + 2 * r;
  const uint8_t *srow = src + (size_t)(sy < 0 ? 0 : sy >= h ? h - 1 : sy) * w, *p = srow + lo;
  if (lo < 0 || lo + n > w) {
    const int a0 = lo < 0 ? -lo : 0, a1 = (lo + n <= w ? lo + n : w) - lo;
    memset(buf, srow[0], (size_t)a0); memcpy(buf + a0, srow + lo + a0, (size_t)(a1 - a0)); memset(buf + a1, srow[w - 1], (size_t)(n - a1));
    p = buf;
  }
  const __m128i z = _mm_setzero_si128();
  for (int x = 0;; x += 16) {
    if (x + 16 > rw) x = rw - 16;
    __m128i s0 = z, s1 = z;
    for (int sft = 0; sft <= 2 * r; sft++) {
      const __m128i b = _mm_loadu_si128((const __m128i *)(p + x + sft));
      s0 = _mm_add_epi16(s0, _mm_cvtepu8_epi16(b)); s1 = _mm_add_epi16(s1, _mm_unpackhi_epi8(b, z));
    }
    _mm_storeu_si128((__m128i *)(t + x), s0); _mm_storeu_si128((__m128i *)(t + x + 8), s1);
    if (x + 16 >= rw) break;
  }
}
static void msh_blur_rect(const uint8_t *src, uint8_t *out, int w, int h, int r, int x0, int y0, int x1, int y1, uint16_t *ring, uint16_t *col, uint8_t *buf) {
  const int rw = x1 - x0, R = 2 * r + 2, kk = (2 * r + 1) * (2 * r + 1);
  for (int j = 0; j <= 2 * r; j++) msh_row_sums(src, w, h, r, x0, rw, y0 - r + j, ring + (size_t)j * rw, buf);
  uint16_t *ca = col, *cb = col + rw;
  for (int x = 0;; x += 8) {
    if (x + 8 > rw) x = rw - 8;
    __m128i c = _mm_loadu_si128((const __m128i *)(ring + x));
    for (int j = 1; j <= 2 * r; j++) c = _mm_add_epi16(c, _mm_loadu_si128((const __m128i *)(ring + (size_t)j * rw + x)));
    _mm_storeu_si128((__m128i *)(ca + x), c);
    if (x + 8 >= rw) break;
  }
  const uint64_t M = ((uint64_t)1 << 32) / (uint64_t)kk + 1;
  const __m128i vh = _mm_set1_epi32(kk / 2), vm = _mm_set1_epi32((int)(uint32_t)M), z = _mm_setzero_si128();
#define MSH_DIVK(v) _mm_blend_epi16(_mm_srli_epi64(_mm_mul_epu32((v), vm), 32), _mm_mul_epu32(_mm_srli_epi64((v), 32), vm), 0xCC)
  for (int y = y0; y < y1; y++) {
    uint8_t *o = out + (size_t)y * w + x0;
    const int last = y + 1 == y1, ja = y - y0 + 2 * r + 1;
    uint16_t *ta = ring + (size_t)(ja % R) * rw;
    const uint16_t *ts = ring + (size_t)((y - y0) % R) * rw;
    if (!last) msh_row_sums(src, w, h, r, x0, rw, y0 - r + ja, ta, buf);
    for (int x = 0;; x += 8) {
      if (x + 8 > rw) x = rw - 8;
      const __m128i c = _mm_loadu_si128((const __m128i *)(ca + x));
      const __m128i lo = _mm_add_epi32(vh, _mm_cvtepu16_epi32(c)), hi = _mm_add_epi32(vh, _mm_unpackhi_epi16(c, z));
      const __m128i q = _mm_packus_epi32(MSH_DIVK(lo), MSH_DIVK(hi));
      _mm_storel_epi64((__m128i *)(o + x), _mm_packus_epi16(q, q));
      if (!last) _mm_storeu_si128((__m128i *)(cb + x), _mm_add_epi16(_mm_sub_epi16(c, _mm_loadu_si128((const __m128i *)(ts + x))), _mm_loadu_si128((const __m128i *)(ta + x))));
      if (x + 8 >= rw) break;
    }
    uint16_t *sw = ca; ca = cb; cb = sw;
  }
#undef MSH_DIVK
}
#endif

typedef struct { float x, y, unit; int hits; } cand_t;   // ai: a corner mark candidate: centre, module size (px), support

static float cross2(const cand_t *a, const cand_t *b, const cand_t *c) { return (b->x - a->x) * (c->y - a->y) - (b->y - a->y) * (c->x - a->x); }

// ---------------------------------------------------------------- geometry

static int solve8(double A[8][9]) {
  for (int c = 0; c < 8; c++) {
    int p = c;
    for (int r = c + 1; r < 8; r++) if (fabs(A[r][c]) > fabs(A[p][c])) p = r;
    if (fabs(A[p][c]) < 1e-12) return 0;
    for (int k = 0; k < 9; k++) { double t = A[c][k]; A[c][k] = A[p][k]; A[p][k] = t; }
    for (int r = 0; r < 8; r++) {
      if (r == c) continue;
      double f = A[r][c] / A[c][c];
#ifdef OB_NEON
      // ai: arm64: two columns a step in double pairs, each the multiply then the subtraction as below (no fused
      // ai: multiply-add, so the same doubles); the finder solves eight of these a quad it tries (1.4 M instructions a
      // ai: frame fewer on the blurred cell sim-lp-15)
      int k = c;
      if ((9 - c) & 1) { A[r][k] -= f * A[c][k]; k++; }
      const float64x2_t vf = vdupq_n_f64(f);
      for (; k < 9; k += 2) vst1q_f64(&A[r][k], vsubq_f64(vld1q_f64(&A[r][k]), vmulq_f64(vf, vld1q_f64(&A[c][k]))));
#elif defined(OB_X86)
      // ai: x86-64: arm64's double pairs in SSE2's, the multiply then the subtraction (the same doubles)
      int k = c;
      if ((9 - c) & 1) { A[r][k] -= f * A[c][k]; k++; }
      const __m128d vf = _mm_set1_pd(f);
      for (; k < 9; k += 2) _mm_storeu_pd(&A[r][k], _mm_sub_pd(_mm_loadu_pd(&A[r][k]), _mm_mul_pd(vf, _mm_loadu_pd(&A[c][k]))));
#else
      for (int k = c; k < 9; k++) A[r][k] -= f * A[c][k];
#endif
    }
  }
  return 1;
}

// src (module coordinates) to dst (image px), four point pairs.
static int homography(const float *src, const float *dst, homo_t *H) {
  double A[8][9];
  for (int i = 0; i < 4; i++) {
    double x = src[2 * i], y = src[2 * i + 1], u = dst[2 * i], v = dst[2 * i + 1];
    double r0[9] = { x, y, 1, 0, 0, 0, -u * x, -u * y, u }, r1[9] = { 0, 0, 0, x, y, 1, -v * x, -v * y, v };
    memcpy(A[2 * i], r0, sizeof r0); memcpy(A[2 * i + 1], r1, sizeof r1);
  }
  if (!solve8(A)) return 0;
  for (int i = 0; i < 8; i++) H->h[i] = (float)(A[i][8] / A[i][i]);
  H->h[8] = 1;
  return 1;
}

static void corner_coords(const ob_layout_t *L, float *src) {
  float x0 = L->node_x[0], x1 = L->node_x[L->nx - 1], y0 = L->node_y[0], y1 = L->node_y[L->ny - 1];
  float s[8] = { x0, y0, x1, y0, x1, y1, x0, y1 };
  memcpy(src, s, sizeof s);
}

#ifdef OB_NEON
// ai: arm64's bilinear read in integers at gamma 1 (image_t lin), eight points a call (lin_pts, then lin_get): the
// ai: decode's picture sampler's (sample_grid_lin, which describes it) and the registration mesh's (msh_read8). A point
// ai: comes as sample()'s coordinates less the half pixel, x a 512th of a pixel on (its fraction's top byte is then the
// ai: rounded one); the word reads stay inside the frame only through lin_pts' integer clamps to w - 2 and h - 2.

// ai: a point's word: row y's at r0 + off and row y + 1's at r1 + off, row y's low half into row y + 1's (a b c d), the
// ai: offset the low 32 bits of off (uxtw: no AND) or all of it. In asm: in C the compiler narrows the loads to halves
// ai: and adds an address for each, four instructions more a point. The frame is not written while it is read, so the
// ai: asm takes no memory operand.
static inline __attribute__((always_inline)) uint64_t lin_word_lo(const uint8_t *r0, const uint8_t *r1, uint64_t off) {
  uint64_t t, m;
  __asm__("ldr %w0, [%2, %w4, uxtw]\n\tldr %w1, [%3, %w4, uxtw]\n\tbfxil %w1, %w0, #0, #16" : "=&r"(t), "=&r"(m) : "r"(r0), "r"(r1), "r"(off));
  return m;
}
static inline __attribute__((always_inline)) uint64_t lin_word(const uint8_t *r0, const uint8_t *r1, uint64_t off) {
  uint64_t t, m;
  __asm__("ldr %w0, [%2, %4]\n\tldr %w1, [%3, %4]\n\tbfxil %w1, %w0, #0, #16" : "=&r"(t), "=&r"(m) : "r"(r0), "r"(r1), "r"(off));
  return m;
}
// ai: eight points' offsets (two vectors), fx twice (bytes), fy (Q15)
typedef struct { uint32x4_t i0, i1; uint8x16_t fx; int16x8_t fy; } lin_pts_t;
typedef struct { uint16x8_t wm, hm; uint8x16_t kab, kcd, kf; int16x8_t kc; } lin_k_t;   // kc: w, 257 (lanes 0, 1)
// ai: a frame's constants: the clamps, the TBL indices (a | b, c | d, the fraction bytes), w and 257
static inline lin_k_t lin_k(int w, int h) {
  return (lin_k_t){ vdupq_n_u16((uint16_t)(w - 2)), vdupq_n_u16((uint16_t)(h - 2)),
                    { 0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29 }, { 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31 },
                    { 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29 }, { (int16_t)w, 257, 0, 0, 0, 0, 0, 0 } };
}
static inline __attribute__((always_inline)) lin_pts_t lin_pts(const lin_k_t *k, float32x4_t x0, float32x4_t y0, float32x4_t x1, float32x4_t y1) {
  const uint32x4_t xq0 = vcvtq_n_u32_f32(x0, 16), xq1 = vcvtq_n_u32_f32(x1, 16), yq0 = vcvtq_n_u32_f32(y0, 16), yq1 = vcvtq_n_u32_f32(y1, 16);
  const uint16x8_t xi = vminq_u16(vuzp2q_u16(vreinterpretq_u16_u32(xq0), vreinterpretq_u16_u32(xq1)), k->wm);
  const uint16x8_t yi = vminq_u16(vuzp2q_u16(vreinterpretq_u16_u32(yq0), vreinterpretq_u16_u32(yq1)), k->hm);
  const uint16x8_t kw = vreinterpretq_u16_s16(k->kc);
  const uint8x16x2_t xq = { { vreinterpretq_u8_u32(xq0), vreinterpretq_u8_u32(xq1) } };
  return (lin_pts_t){ vmlal_laneq_u16(vmovl_u16(vget_low_u16(xi)), vget_low_u16(yi), kw, 0), vmlal_high_laneq_u16(vmovl_high_u16(xi), yi, kw, 0),
                      vqtbl2q_u8(xq, k->kf), vreinterpretq_s16_u16(vshrq_n_u16(vuzp1q_u16(vreinterpretq_u16_u32(yq0), vreinterpretq_u16_u32(yq1)), 1)) };
}
// ai: eight points' grey, 128 times over (0 to 32640, -1 where the blend's halves round so), on 16-bit lanes
static inline __attribute__((always_inline)) int16x8_t lin_grey(const uint8_t *r0, const uint8_t *r1, const lin_k_t *k, lin_pts_t p) {
  const uint64_t p01 = vgetq_lane_u64(vreinterpretq_u64_u32(p.i0), 0), p23 = vgetq_lane_u64(vreinterpretq_u64_u32(p.i0), 1);
  const uint64_t p45 = vgetq_lane_u64(vreinterpretq_u64_u32(p.i1), 0), p67 = vgetq_lane_u64(vreinterpretq_u64_u32(p.i1), 1);
  const uint8x16x2_t m = { { vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(lin_word_lo(r0, r1, p01) | lin_word(r0, r1, p01 >> 32) << 32), vcreate_u64(lin_word_lo(r0, r1, p23) | lin_word(r0, r1, p23 >> 32) << 32))),
                             vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(lin_word_lo(r0, r1, p45) | lin_word(r0, r1, p45 >> 32) << 32), vcreate_u64(lin_word_lo(r0, r1, p67) | lin_word(r0, r1, p67 >> 32) << 32))) } };
  const uint8x16_t ab = vqtbl2q_u8(m, k->kab), cd = vqtbl2q_u8(m, k->kcd);   // a | b, c | d: eight points each
  const uint8x8_t fx = vget_low_u8(p.fx);
  const uint16x8_t top = vmlsl_u8(vmlal_high_u8(vshll_n_u8(vget_low_u8(ab), 8), ab, p.fx), vget_low_u8(ab), fx);   // modulo 2^16 on the way
  const uint16x8_t bot = vmlsl_u8(vmlal_high_u8(vshll_n_u8(vget_low_u8(cd), 8), cd, p.fx), vget_low_u8(cd), fx);
  // ai: top / 2 + (bot - top) / 2 fy: 0 to 32640, or -1 where top is 1 and bot 0 (the 16-bit lane read as signed)
  return vreinterpretq_s16_u16(vsraq_n_u16(vreinterpretq_u16_s16(vqrdmulhq_s16(vreinterpretq_s16_u16(vhsubq_u16(bot, top)), p.fy)), top, 1));
}
// ai: the grey as floats: 257 r by SMULL, by 2^-23 in the conversion
static inline __attribute__((always_inline)) float32x4x2_t lin_get(const uint8_t *r0, const uint8_t *r1, const lin_k_t *k, lin_pts_t p) {
  const int16x8_t r = lin_grey(r0, r1, k, p);
  return (float32x4x2_t){ { vcvtq_n_f32_s32(vmull_laneq_s16(vget_low_s16(r), k->kc, 1), 23), vcvtq_n_f32_s32(vmull_high_laneq_s16(r, k->kc, 1), 23) } };
}
static inline __attribute__((always_inline)) void lin_read(const uint8_t *r0, const uint8_t *r1, const lin_k_t *k, lin_pts_t p, float *o) {
  const float32x4x2_t g = lin_get(r0, r1, k, p);
  vst1q_f32(o, g.val[0]); vst1q_f32(o + 4, g.val[1]);
}
#elif defined(OB_X86)
// ai: x86-64's bilinear read in integers at gamma 1, arm64's above in SSE4.1 (lin_pts, then lin_get), for the same
// ai: callers and on the same coordinates (sample()'s less the half pixel, x a 512th of a pixel on); from the same
// ai: coordinates the same integers and floats as arm64's. 16.16 by CVTTPS2DQ after a MAXPS with 0 (below 0, and NaN,
// ai: pixel 0, as FCVTZU) and a hold under 2^15 pixels (the signed conversion's range); the integer parts packed to
// ai: 16-bit lanes and clamped there to w - 2 and h - 2 (so every read stays in the frame), the offsets y w + x from
// ai: the 16-bit products' two halves; fx the fraction's top byte, fy its top 15 bits. A point's two pixel pairs read
// ai: as 16-bit words (row y's at the pixel, row y + 1's a row on), four points' to a 64-bit general register, so a
// ai: vector holds eight points' a | b (or c | d) a lane; each row's pair blended as 256 a + (b - a) fx (PMULLW, modulo
// ai: 2^16 on the way), the rows' halved difference (arm64's UHSUB: PAVGW with the complement, less 2^15) times fy by
// ai: PMULHRSW (SQRDMULH's rounding, fy never negative), the grey 128 times over; to floats by one conversion and a
// ai: multiply by 257 / 2^23 (exact).
typedef struct { __m128 val[2]; } lin_f8_t;
typedef struct { __m128i i0, i1, fx, fy; } lin_pts_t;   // ai: eight points' offsets (two vectors), fx and fy (16-bit lanes)
typedef struct { __m128i wm, hm, kw; } lin_k_t;   // ai: the clamps, and w on 16-bit lanes
static inline lin_k_t lin_k(int w, int h) { return (lin_k_t){ _mm_set1_epi16((short)(w - 2)), _mm_set1_epi16((short)(h - 2)), _mm_set1_epi16((short)w) }; }
static inline __attribute__((always_inline)) __m128i lin_q16(__m128 v) {
  return _mm_cvttps_epi32(_mm_mul_ps(_mm_min_ps(_mm_max_ps(v, _mm_setzero_ps()), _mm_set1_ps(32767.0f)), _mm_set1_ps(65536.0f)));
}
static inline __attribute__((always_inline)) lin_pts_t lin_pts(const lin_k_t *k, __m128 x0, __m128 y0, __m128 x1, __m128 y1) {
  const __m128i xq0 = lin_q16(x0), xq1 = lin_q16(x1), yq0 = lin_q16(y0), yq1 = lin_q16(y1), z = _mm_setzero_si128();
  const __m128i xi = _mm_min_epu16(_mm_packus_epi32(_mm_srli_epi32(xq0, 16), _mm_srli_epi32(xq1, 16)), k->wm);
  const __m128i yi = _mm_min_epu16(_mm_packus_epi32(_mm_srli_epi32(yq0, 16), _mm_srli_epi32(yq1, 16)), k->hm);
  const __m128i ml = _mm_mullo_epi16(yi, k->kw), mh = _mm_mulhi_epu16(yi, k->kw), b8 = _mm_set1_epi32(0xff), f16 = _mm_set1_epi32(0xffff);
  return (lin_pts_t){ _mm_add_epi32(_mm_unpacklo_epi16(ml, mh), _mm_unpacklo_epi16(xi, z)), _mm_add_epi32(_mm_unpackhi_epi16(ml, mh), _mm_unpackhi_epi16(xi, z)),
                      _mm_packus_epi32(_mm_and_si128(_mm_srli_epi32(xq0, 8), b8), _mm_and_si128(_mm_srli_epi32(xq1, 8), b8)),
                      _mm_srli_epi16(_mm_packus_epi32(_mm_and_si128(yq0, f16), _mm_and_si128(yq1, f16)), 1) };
}
static inline __attribute__((always_inline)) uint64_t lin_u16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
// ai: four points' pairs at r + offset, the offsets two to a 64-bit word
static inline __attribute__((always_inline)) uint64_t lin_pairs(const uint8_t *r, uint64_t p01, uint64_t p23) {
  return lin_u16(r + (uint32_t)p01) | lin_u16(r + (p01 >> 32)) << 16 | lin_u16(r + (uint32_t)p23) << 32 | lin_u16(r + (p23 >> 32)) << 48;
}
// ai: eight points' grey, 128 times over (0 to 32640, -1 where the blend's halves round so), on 16-bit lanes; r1 is
// ai: arm64's (the frame less two bytes a row on), row y + 1's pair two bytes past it
static inline __attribute__((always_inline)) __m128i lin_grey(const uint8_t *r0, const uint8_t *r1, const lin_k_t *k, lin_pts_t p) {
  (void)k;
  const uint64_t p01 = (uint64_t)_mm_cvtsi128_si64(p.i0), p23 = (uint64_t)_mm_extract_epi64(p.i0, 1);
  const uint64_t p45 = (uint64_t)_mm_cvtsi128_si64(p.i1), p67 = (uint64_t)_mm_extract_epi64(p.i1, 1);
  const __m128i ab = _mm_set_epi64x((long long)lin_pairs(r0, p45, p67), (long long)lin_pairs(r0, p01, p23));
  const __m128i cd = _mm_set_epi64x((long long)lin_pairs(r1 + 2, p45, p67), (long long)lin_pairs(r1 + 2, p01, p23));
  const __m128i m8 = _mm_set1_epi16(0xff);
  const __m128i a = _mm_and_si128(ab, m8), c = _mm_and_si128(cd, m8);
  const __m128i top = _mm_add_epi16(_mm_slli_epi16(ab, 8), _mm_mullo_epi16(_mm_sub_epi16(_mm_srli_epi16(ab, 8), a), p.fx));
  const __m128i bot = _mm_add_epi16(_mm_slli_epi16(cd, 8), _mm_mullo_epi16(_mm_sub_epi16(_mm_srli_epi16(cd, 8), c), p.fx));
  const __m128i hd = _mm_xor_si128(_mm_avg_epu16(bot, _mm_xor_si128(top, _mm_set1_epi16(-1))), _mm_set1_epi16((short)0x8000));
  return _mm_add_epi16(_mm_mulhrs_epi16(hd, p.fy), _mm_srli_epi16(top, 1));
}
static inline __attribute__((always_inline)) lin_f8_t lin_get(const uint8_t *r0, const uint8_t *r1, const lin_k_t *k, lin_pts_t p) {
  const __m128i r = lin_grey(r0, r1, k, p);
  const __m128 s = _mm_set1_ps(257.0f / 8388608.0f);
  return (lin_f8_t){ { _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(r)), s), _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(r, r), 16)), s) } };
}
static inline __attribute__((always_inline)) void lin_read(const uint8_t *r0, const uint8_t *r1, const lin_k_t *k, lin_pts_t p, float *o) {
  const lin_f8_t g = lin_get(r0, r1, k, p);
  _mm_storeu_ps(o, g.val[0]); _mm_storeu_ps(o + 4, g.val[1]);
}
#ifdef OB_V3_ON
// ai: x86-64-v3's read, sixteen points a call (2026-11-05): lin_pts and lin_grey's arithmetic on sixteen lanes, the pixel
// ai: pairs by gathers (VPGATHERDD: a pair's word read as the dword it starts, row y + 1's from two bytes before, lin_grey's
// ai: r1, its top half kept: in bounds through the clamps, as there), so from the same coordinates the same values as
// ai: the v2 read's. The decode's sampler and the registration mesh take it.
typedef struct { __m256i i0, i1, fx, fy; } lin16_pts_t;   // ai: offsets of points {0-3, 8-11} and {4-7, 12-15}; fx, fy in point order
typedef struct { __m256i wm, hm, kw; } lin16_k_t;
#define OB_V3_IN static inline __attribute__((always_inline, target("arch=x86-64-v3")))
OB_V3_IN __m256i lin16_q16(__m256 v) {
  return _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(v, _mm256_setzero_ps()), _mm256_set1_ps(32767.0f)), _mm256_set1_ps(65536.0f)));
}
// ai: two eight-lane vectors' 32-bit lanes to sixteen 16-bit lanes in point order (PACKUSDW works a 128-bit half at a time)
OB_V3_IN __m256i lin16_pack(__m256i a, __m256i b) { return _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0xD8); }
OB_V3_IN lin16_pts_t lin16_pts(const lin16_k_t *k, __m256 x0, __m256 y0, __m256 x1, __m256 y1) {
  const __m256i xq0 = lin16_q16(x0), xq1 = lin16_q16(x1), yq0 = lin16_q16(y0), yq1 = lin16_q16(y1), z = _mm256_setzero_si256();
  const __m256i xi = _mm256_min_epu16(lin16_pack(_mm256_srli_epi32(xq0, 16), _mm256_srli_epi32(xq1, 16)), k->wm);
  const __m256i yi = _mm256_min_epu16(lin16_pack(_mm256_srli_epi32(yq0, 16), _mm256_srli_epi32(yq1, 16)), k->hm);
  const __m256i ml = _mm256_mullo_epi16(yi, k->kw), mh = _mm256_mulhi_epu16(yi, k->kw), b8 = _mm256_set1_epi32(0xff), f16 = _mm256_set1_epi32(0xffff);
  return (lin16_pts_t){ _mm256_add_epi32(_mm256_unpacklo_epi16(ml, mh), _mm256_unpacklo_epi16(xi, z)), _mm256_add_epi32(_mm256_unpackhi_epi16(ml, mh), _mm256_unpackhi_epi16(xi, z)),
                        lin16_pack(_mm256_and_si256(_mm256_srli_epi32(xq0, 8), b8), _mm256_and_si256(_mm256_srli_epi32(xq1, 8), b8)),
                        _mm256_srli_epi16(lin16_pack(_mm256_and_si256(yq0, f16), _mm256_and_si256(yq1, f16)), 1) };
}
// ai: sixteen points' grey, lin_grey's arithmetic on sixteen lanes (r1 the frame less two bytes a row on)
OB_V3_IN __m256i lin16_grey(const uint8_t *r0, const uint8_t *r1, lin16_pts_t p) {
  const __m256i m16 = _mm256_set1_epi32(0xffff), m8 = _mm256_set1_epi16(0xff);
  const __m256i g0 = _mm256_i32gather_epi32((const int *)(const void *)r0, p.i0, 1), g1 = _mm256_i32gather_epi32((const int *)(const void *)r0, p.i1, 1);
  const __m256i h0 = _mm256_i32gather_epi32((const int *)(const void *)r1, p.i0, 1), h1 = _mm256_i32gather_epi32((const int *)(const void *)r1, p.i1, 1);
  const __m256i ab = _mm256_packus_epi32(_mm256_and_si256(g0, m16), _mm256_and_si256(g1, m16));
  const __m256i cd = _mm256_packus_epi32(_mm256_srli_epi32(h0, 16), _mm256_srli_epi32(h1, 16));
  const __m256i a = _mm256_and_si256(ab, m8), c = _mm256_and_si256(cd, m8);
  const __m256i top = _mm256_add_epi16(_mm256_slli_epi16(ab, 8), _mm256_mullo_epi16(_mm256_sub_epi16(_mm256_srli_epi16(ab, 8), a), p.fx));
  const __m256i bot = _mm256_add_epi16(_mm256_slli_epi16(cd, 8), _mm256_mullo_epi16(_mm256_sub_epi16(_mm256_srli_epi16(cd, 8), c), p.fx));
  const __m256i hd = _mm256_xor_si256(_mm256_avg_epu16(bot, _mm256_xor_si256(top, _mm256_set1_epi16(-1))), _mm256_set1_epi16((short)0x8000));
  return _mm256_add_epi16(_mm256_mulhrs_epi16(hd, p.fy), _mm256_srli_epi16(top, 1));
}
OB_V3_IN void lin16_read(const uint8_t *r0, const uint8_t *r1, lin16_pts_t p, float *o) {
  const __m256i r = lin16_grey(r0, r1, p);
  const __m256 sc = _mm256_set1_ps(257.0f / 8388608.0f);
  _mm256_storeu_ps(o, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(r))), sc));
  _mm256_storeu_ps(o + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(r, 1))), sc));
}
#endif
#endif

// ---------------------------------------------------------------- registration mesh

// A node's search judges a position only where the capture holds most of its template. A cell whose sample point (u, v),
// in pixels as sample() takes it, falls outside the capture (u under 0.5 or over w - 0.5, likewise v, or NaN) reads the
// edge's pixels repeated (sample()'s clamp): a little way past the edge they carry on the code's border and its quiet
// light, but they are no measurement, and a template mostly outside correlates with the repeated edge alone. So a
// position with fewer than half its template's cells inside is no candidate (NODE_NONE, under any correlation, and no
// point of the parabola either), and a node none of whose positions is judged keeps the place the neighbours and the
// model give it, as one judged under 0.35 does. Half: the correlation is then mostly of the capture's own pixels.
// ai: The case (run-2026-11-02T00-18-12-969Z frame 287, 2026-11-02): a corner searched again from the anchors' model
// ai: 55 px left of a 1920 x 1920 capture, every cell reading column 0 repeated; its correlations along x differed by
// ai: 1e-5, and a sampler rounding its own way took a spurious 0.55 three modules off where the exact one's path ended
// ai: under 0.35 (78 blocks to 2). Measured on the 11 S26 replays and 5 cells against 977469a: this rule +129 blocks
// ai: (+128 on that replay, +1 on run-2026-11-01T22-40-56-684Z), every other run the same; the cells outside masked out
// ai: of the sums as well lost 16 with half the template judged and 384 with 65 cells (a side node's whole template)
// ai: judged: the masked rim and quiet zone just past the edge moved corners of codes filling the frame by 0.05 px.
static const float NODE_NONE = -2.0f;
#if !defined(OB_NEON) && !defined(OB_X86)   // ai: arm64's and x86-64's is msh_in4r, four points at a time on its read's coordinates
static inline int in_capture(const image_t *im, float u, float v) { return u >= 0.5f && u <= im->w - 0.5f && v >= 0.5f && v <= im->h - 0.5f; }
#endif
// Whether every point a stage of a node's search reads is inside the capture with a pixel to spare: the template's cell
// middles (rect r about (cx0, cy0)) moved by offsets in [ax, bx] x [ay, by] modules (the coarse grid's +-1.5, a climb's
// move one step about its centre), through H, then by (pdx, pdy). w positive at the region's four corners holds it on
// the region (w is linear), so the region's image is the convex quad of the corners' images. Inside, every position is
// judged and the stage counts no cells: the same floats as with no rule.
static int node_inside(const image_t *im, const homo_t *H, int cx0, int cy0, const int *r, float pdx, float pdy, float ax, float bx, float ay, float by) {
  const float x0 = cx0 + r[0] + 0.5f + ax, x1 = cx0 + r[1] - 0.5f + bx, y0 = cy0 + r[2] + 0.5f + ay, y1 = cy0 + r[3] - 0.5f + by;
  const float *h = H->h;
  for (int c = 0; c < 4; c++) {
    const float x = c & 1 ? x1 : x0, y = c & 2 ? y1 : y0;
    float u, v;
    if (!(h[6] * x + h[7] * y + 1 > 0)) return 0;
    project(H, x, y, &u, &v);
    if (!(u + pdx >= 1.5f && u + pdx <= im->w - 1.5f && v + pdy >= 1.5f && v + pdy <= im->h - 1.5f)) return 0;
  }
  return 1;
}

#ifdef OB_NEON
// ai: arm64's registration mesh (refine_node, anchor_find) in NEON's own instructions and its own values: each form
// ai: says what it computes and where its floats part from the scalar twins' (node_ncc, the plain loops), which are the
// ai: statement of what each stage computes; verified blocks are its test. msh_read8 is sample() at eight points whose
// ai: coordinates come less (MSH_OX, MSH_OY), the half pixel and the x fraction's rounding (lin_pts), which every caller
// ai: folds into the offset it adds to its points anyway: at gamma 1 the integer read (lin_get: weights on bytes, the
// ai: grey 128 times over on 16-bit lanes, one conversion), otherwise sample4 through the table at the points put back.
// ai: It is sample()'s blend with x's fraction a rounded 256th of a pixel and the value within 2e-5 of the grey range
// ai: (at random points in and past the frame, within 0.0018 of the range of sample()'s, test/mesh_neon.c), past the
// ai: frame's edges too: a point beyond the last column or row is first held a fraction short of it (xm, ym: FMIN), as
// ai: sample() clamps it, where the integer clamps alone (lin_pts') would blend the last two pixels by the point's own
// ai: fraction (it moved corners of a code filling the frame up to 2.4 px and lost three frames of 79 blocks on
// ai: run-2026-11-02T00-15-17-920Z); below 0 FCVTZU holds pixel 0 as sample() does. msh_in4r is in_capture on the
// ai: read's coordinates: 1/512 <= x <= w - 1 + 1/512, 0 <= y <= h - 1 (xl, xh, yh; a NaN fails each compare).
typedef struct { lin_k_t k; float32x4_t xm, ym, xl, xh, yh; } msh_k_t;
static const float MSH_OX = 0.5f - 1.0f / 512, MSH_OY = 0.5f;
static inline msh_k_t msh_k(const image_t *im) {
  return (msh_k_t){ lin_k(im->w, im->h), vdupq_n_f32((float)(im->w - 2) + 0.998f), vdupq_n_f32((float)(im->h - 2) + 0.9999f),
                    vdupq_n_f32(0.5f - MSH_OX), vdupq_n_f32(im->w - 0.5f - MSH_OX), vdupq_n_f32(im->h - 0.5f - MSH_OY) };
}
static inline __attribute__((always_inline)) float32x4x2_t msh_read8(const image_t *im, const msh_k_t *K, float32x4_t x0, float32x4_t y0, float32x4_t x1, float32x4_t y1) {
  if (!im->lin) {
    const float32x4_t ox = vdupq_n_f32(MSH_OX), oy = vdupq_n_f32(MSH_OY);
    return (float32x4x2_t){ { vreinterpretq_f32_s32(sample4(im, vreinterpretq_s32_f32(vaddq_f32(x0, ox)), vreinterpretq_s32_f32(vaddq_f32(y0, oy)))),
                              vreinterpretq_f32_s32(sample4(im, vreinterpretq_s32_f32(vaddq_f32(x1, ox)), vreinterpretq_s32_f32(vaddq_f32(y1, oy)))) } };
  }
  return lin_get(im->px, im->px + im->w - 2, &K->k, lin_pts(&K->k, vminq_f32(x0, K->xm), vminq_f32(y0, K->ym), vminq_f32(x1, K->xm), vminq_f32(y1, K->ym)));
}
// ai: msh_read8's points as 12-bit greys, 0 to MSH_Q (the anchors' window, summed in integers): at gamma 1 the
// ai: integer read's 128 times the grey over 8, else the table's float times MSH_Q, rounded
enum { MSH_Q = 4080 };
static inline __attribute__((always_inline)) uint16x8_t msh_read8q(const image_t *im, const msh_k_t *K, float32x4_t x0, float32x4_t y0, float32x4_t x1, float32x4_t y1) {
  if (!im->lin) {
    const float32x4x2_t v = msh_read8(im, K, x0, y0, x1, y1);
    const float32x4_t q = vdupq_n_f32((float)MSH_Q);
    return vcombine_u16(vmovn_u32(vcvtnq_u32_f32(vmulq_f32(v.val[0], q))), vmovn_u32(vcvtnq_u32_f32(vmulq_f32(v.val[1], q))));
  }
  const int16x8_t r = lin_grey(im->px, im->px + im->w - 2, &K->k, lin_pts(&K->k, vminq_f32(x0, K->xm), vminq_f32(y0, K->ym), vminq_f32(x1, K->xm), vminq_f32(y1, K->ym)));
  return vshrq_n_u16(vreinterpretq_u16_s16(vmaxq_s16(r, vdupq_n_s16(0))), 3);
}
static inline __attribute__((always_inline)) uint32x4_t msh_in4r(const msh_k_t *K, float32x4_t x, float32x4_t y) {
  return vandq_u32(vandq_u32(vcleq_f32(K->xl, x), vcleq_f32(x, K->xh)), vandq_u32(vcgeq_f32(y, vdupq_n_f32(0)), vcleq_f32(y, K->yh)));
}
// ai: The homography at four points from its lines' values there (nu, nv, nw: h0 x + h1 y + h2, h3 x + h4 y + h5,
// ai: h6 x + h7 y + 1), which each caller makes from its points' lattice (a row's terms stepped along it, a cell's
// ai: tabled once a node): one reciprocal of nw for both coordinates where project() divides each, the offset (qx, qy)
// ai: added in the same fused multiply-add. (The reciprocal by FRECPE and two Newton steps measured 4 to 9% slower.)
static inline __attribute__((always_inline)) void msh_proj(float32x4_t nu, float32x4_t nv, float32x4_t nw, float32x4_t qx, float32x4_t qy, float32x4_t *x, float32x4_t *y) {
  const float32x4_t r = vdivq_f32(vdupq_n_f32(1), nw);
  *x = vfmaq_f32(qx, nu, r); *y = vfmaq_f32(qy, nv, r);
}
// ai: a lane's sums, the squares fused (tk is +-1, so its product is exact and the fused add is node_ncc's add); near
// ai: the capture's edge its count of cells inside the capture
static inline __attribute__((always_inline)) void msh_sums(float32x4_t sv, float32x4_t tk, float32x4_t *s, float32x4_t *q, float32x4_t *t) {
  *s = vaddq_f32(*s, sv); *q = vfmaq_f32(*q, sv, sv); *t = vfmaq_f32(*t, tk, sv);
}
static inline __attribute__((always_inline)) void msh_count(const msh_k_t *K, float32x4_t x, float32x4_t y, float32x4_t *n) {
  *n = vaddq_f32(*n, vbslq_f32(msh_in4r(K, x, y), vdupq_n_f32(1), vdupq_n_f32(0)));
}
// ai: node_ncc at G (4 or 8) offsets (ox[l], oy[l]) at once, a lane each: a lane places, samples and sums the node's
// ai: kept cells with node_ncc's operations; st and vt are the template's sums over those cells (refine_node's, the
// ai: same). Only the cells node_ncc keeps are read. A kept cell k comes as its sign tv[k] and its middle's three lines
// ai: (cu, cv, cw: refine_node's table), so a point's lines are the cell's plus the offset's (du, dv, dw, a lane each);
// ai: (qx, qy): the neighbours' offset (pdx, pdy) less (MSH_OX, MSH_OY). Eight points a read: at G 2 a cell's eight offsets, at G 1
// ai: two cells' four, the second cell's sums kept apart and added at the end (an odd last cell read twice, once summed).
// ai: edge (a constant of each call): the move's positions may read past the capture (node_inside), so each lane also
// ai: counts its cells inside it and is no candidate under nmin of them.
static inline __attribute__((always_inline)) void msh_ncc_g(const image_t *im, const homo_t *H, int cells, const float *cu, const float *cv, const float *cw, const float *tv, float st, float vt, float qx, float qy, const float *ox, const float *oy, float *out, const int G, const int edge, int nmin) {
  const msh_k_t K = msh_k(im);
  const float *h = H->h;
  const float32x4_t px = vdupq_n_f32(qx), py = vdupq_n_f32(qy), z = vdupq_n_f32(0);
  float32x4_t du[2], dv[2], dw[2];
  for (int g = 0; g < 2; g++) {
    const float32x4_t a = vld1q_f32(ox + 4 * g), b = vld1q_f32(oy + 4 * g);
    du[g] = vfmaq_n_f32(vmulq_n_f32(a, h[0]), b, h[1]); dv[g] = vfmaq_n_f32(vmulq_n_f32(a, h[3]), b, h[4]); dw[g] = vfmaq_n_f32(vmulq_n_f32(a, h[6]), b, h[7]);
  }
#define MSH_CELL(k_, g_, X, Y) msh_proj(vaddq_f32(vdupq_n_f32(cu[k_]), du[g_]), vaddq_f32(vdupq_n_f32(cv[k_]), dv[g_]), vaddq_f32(vdupq_n_f32(cw[k_]), dw[g_]), px, py, &X, &Y)
  float32x4_t s[2] = { z, z }, q[2] = { z, z }, t[2] = { z, z }, nn[2] = { z, z }, xa, ya, xb, yb;
  int k = 0;
  if (G == 2) {
    for (; k < cells; k++) {
      MSH_CELL(k, 0, xa, ya); MSH_CELL(k, 1, xb, yb);
      const float32x4x2_t sv = msh_read8(im, &K, xa, ya, xb, yb);
      const float32x4_t tk = vdupq_n_f32(tv[k]);
      if (edge) { msh_count(&K, xa, ya, &nn[0]); msh_count(&K, xb, yb, &nn[1]); }
      msh_sums(sv.val[0], tk, &s[0], &q[0], &t[0]); msh_sums(sv.val[1], tk, &s[1], &q[1], &t[1]);
    }
  } else {
    for (; k + 1 < cells; k += 2) {
      MSH_CELL(k, 0, xa, ya); MSH_CELL(k + 1, 0, xb, yb);
      const float32x4x2_t sv = msh_read8(im, &K, xa, ya, xb, yb);
      const float32x4_t ta = vdupq_n_f32(tv[k]), tb = vdupq_n_f32(tv[k + 1]);
      if (edge) { msh_count(&K, xa, ya, &nn[0]); msh_count(&K, xb, yb, &nn[1]); }
      msh_sums(sv.val[0], ta, &s[0], &q[0], &t[0]); msh_sums(sv.val[1], tb, &s[1], &q[1], &t[1]);
    }
    if (k < cells) {
      MSH_CELL(k, 0, xa, ya);
      const float32x4_t ta = vdupq_n_f32(tv[k]);
      if (edge) msh_count(&K, xa, ya, &nn[0]);
      msh_sums(msh_read8(im, &K, xa, ya, xa, ya).val[0], ta, &s[0], &q[0], &t[0]);
    }
    s[0] = vaddq_f32(s[0], s[1]); q[0] = vaddq_f32(q[0], q[1]); t[0] = vaddq_f32(t[0], t[1]); nn[0] = vaddq_f32(nn[0], nn[1]);
  }
#undef MSH_CELL
  const float32x4_t n = vdupq_n_f32((float)cells), vst = vdupq_n_f32(st), vvt = vdupq_n_f32(vt);
  for (int g = 0; g < G; g++) {
    const float32x4_t vs = vsubq_f32(q[g], vdivq_f32(vmulq_f32(s[g], s[g]), n));
    const float32x4_t c = vdivq_f32(vsubq_f32(t[g], vdivq_f32(vmulq_f32(vst, s[g]), n)), vsqrtq_f32(vmulq_f32(vvt, vs)));
    float32x4_t r = vt <= 0 ? z : vbslq_f32(vcleq_f32(vs, vdupq_n_f32(1e-9f)), z, c);
    if (edge) r = vbslq_f32(vcltq_f32(nn[g], vdupq_n_f32((float)nmin)), vdupq_n_f32(NODE_NONE), r);
    vst1q_f32(out + 4 * g, r);
  }
}
static void msh_ncc(const image_t *im, const homo_t *H, int cells, const float *cu, const float *cv, const float *cw, const float *tv, float st, float vt, float qx, float qy, const float *ox, const float *oy, float *out, int lanes, int edge, int nmin) {
  if (edge) {
    if (lanes > 4) msh_ncc_g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 2, 1, nmin);
    else msh_ncc_g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 1, 1, nmin);
  } else if (lanes > 4) msh_ncc_g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 2, 0, nmin);
  else msh_ncc_g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 1, 0, nmin);
}
#elif defined(OB_X86)
// ai: x86-64's registration mesh, arm64's above in SSE4.1, form for form (msh_read8 on the integer read, msh_read8q,
// ai: msh_in4r, msh_proj, msh_ncc's lanes, refine_node's coarse grid in whole vectors): the same coordinates' offsets
// ai: (MSH_OX, MSH_OY) and the same hold short of the frame's last column and row (MINPS with the hold first, so a NaN
// ai: passes to lin_pts' MAXPS, pixel 0), with no fused multiply-adds: its own values (from the same coordinates
// ai: arm64's reads), held within 1/250 of sample() by test/mesh_neon.c; verified blocks are its test.
typedef struct { lin_k_t k; __m128 xm, ym, xl, xh, yh; } msh_k_t;
static const float MSH_OX = 0.5f - 1.0f / 512, MSH_OY = 0.5f;
static inline msh_k_t msh_k(const image_t *im) {
  return (msh_k_t){ lin_k(im->w, im->h), _mm_set1_ps((float)(im->w - 2) + 0.998f), _mm_set1_ps((float)(im->h - 2) + 0.9999f),
                    _mm_set1_ps(0.5f - MSH_OX), _mm_set1_ps(im->w - 0.5f - MSH_OX), _mm_set1_ps(im->h - 0.5f - MSH_OY) };
}
#define MSH_MA(a, b, c) _mm_add_ps((a), _mm_mul_ps((b), (c)))
static inline __attribute__((always_inline)) lin_f8_t msh_read8(const image_t *im, const msh_k_t *K, __m128 x0, __m128 y0, __m128 x1, __m128 y1) {
  if (!im->lin) {
    const __m128 ox = _mm_set1_ps(MSH_OX), oy = _mm_set1_ps(MSH_OY);
    return (lin_f8_t){ { _mm_castsi128_ps(sample4(im, _mm_castps_si128(_mm_add_ps(x0, ox)), _mm_castps_si128(_mm_add_ps(y0, oy)))),
                         _mm_castsi128_ps(sample4(im, _mm_castps_si128(_mm_add_ps(x1, ox)), _mm_castps_si128(_mm_add_ps(y1, oy)))) } };
  }
  return lin_get(im->px, im->px + im->w - 2, &K->k, lin_pts(&K->k, _mm_min_ps(K->xm, x0), _mm_min_ps(K->ym, y0), _mm_min_ps(K->xm, x1), _mm_min_ps(K->ym, y1)));
}
// ai: msh_read8's points as 12-bit greys, 0 to MSH_Q (the anchors' window, summed in integers): at gamma 1 the
// ai: integer read's 128 times the grey over 8, else the table's float times MSH_Q, rounded to even (CVTPS2DQ; a NaN's
// ai: INT_MIN and anything below 0 packed to 0, as FCVTNU gives)
enum { MSH_Q = 4080 };
static inline __attribute__((always_inline)) __m128i msh_read8q(const image_t *im, const msh_k_t *K, __m128 x0, __m128 y0, __m128 x1, __m128 y1) {
  if (!im->lin) {
    const lin_f8_t v = msh_read8(im, K, x0, y0, x1, y1);
    const __m128 q = _mm_set1_ps((float)MSH_Q);
    return _mm_packus_epi32(_mm_cvtps_epi32(_mm_mul_ps(v.val[0], q)), _mm_cvtps_epi32(_mm_mul_ps(v.val[1], q)));
  }
  const __m128i r = lin_grey(im->px, im->px + im->w - 2, &K->k, lin_pts(&K->k, _mm_min_ps(K->xm, x0), _mm_min_ps(K->ym, y0), _mm_min_ps(K->xm, x1), _mm_min_ps(K->ym, y1)));
  return _mm_srli_epi16(_mm_max_epi16(r, _mm_setzero_si128()), 3);
}
static inline __attribute__((always_inline)) __m128 msh_in4r(const msh_k_t *K, __m128 x, __m128 y) {
  return _mm_and_ps(_mm_and_ps(_mm_cmple_ps(K->xl, x), _mm_cmple_ps(x, K->xh)), _mm_and_ps(_mm_cmpge_ps(y, _mm_setzero_ps()), _mm_cmple_ps(y, K->yh)));
}
static inline __attribute__((always_inline)) void msh_proj(__m128 nu, __m128 nv, __m128 nw, __m128 qx, __m128 qy, __m128 *x, __m128 *y) {
  const __m128 r = _mm_div_ps(_mm_set1_ps(1), nw);
  *x = MSH_MA(qx, nu, r); *y = MSH_MA(qy, nv, r);
}
static inline __attribute__((always_inline)) void msh_sums(__m128 sv, __m128 tk, __m128 *s, __m128 *q, __m128 *t) {
  *s = _mm_add_ps(*s, sv); *q = MSH_MA(*q, sv, sv); *t = MSH_MA(*t, tk, sv);
}
static inline __attribute__((always_inline)) void msh_count(const msh_k_t *K, __m128 x, __m128 y, __m128 *n) {
  *n = _mm_add_ps(*n, _mm_and_ps(msh_in4r(K, x, y), _mm_set1_ps(1)));
}
static inline __attribute__((always_inline)) void msh_ncc_g(const image_t *im, const homo_t *H, int cells, const float *cu, const float *cv, const float *cw, const float *tv, float st, float vt, float qx, float qy, const float *ox, const float *oy, float *out, const int G, const int edge, int nmin) {
  const msh_k_t K = msh_k(im);
  const float *h = H->h;
  const __m128 px = _mm_set1_ps(qx), py = _mm_set1_ps(qy), z = _mm_setzero_ps();
  __m128 du[2], dv[2], dw[2];
  for (int g = 0; g < 2; g++) {
    const __m128 a = _mm_loadu_ps(ox + 4 * g), b = _mm_loadu_ps(oy + 4 * g);
    du[g] = MSH_MA(_mm_mul_ps(a, _mm_set1_ps(h[0])), b, _mm_set1_ps(h[1])); dv[g] = MSH_MA(_mm_mul_ps(a, _mm_set1_ps(h[3])), b, _mm_set1_ps(h[4]));
    dw[g] = MSH_MA(_mm_mul_ps(a, _mm_set1_ps(h[6])), b, _mm_set1_ps(h[7]));
  }
#define MSH_CELL(k_, g_, X, Y) msh_proj(_mm_add_ps(_mm_set1_ps(cu[k_]), du[g_]), _mm_add_ps(_mm_set1_ps(cv[k_]), dv[g_]), _mm_add_ps(_mm_set1_ps(cw[k_]), dw[g_]), px, py, &X, &Y)
  __m128 s[2] = { z, z }, q[2] = { z, z }, t[2] = { z, z }, nn[2] = { z, z }, xa, ya, xb, yb;
  int k = 0;
  if (G == 2) {
    for (; k < cells; k++) {
      MSH_CELL(k, 0, xa, ya); MSH_CELL(k, 1, xb, yb);
      const lin_f8_t sv = msh_read8(im, &K, xa, ya, xb, yb);
      const __m128 tk = _mm_set1_ps(tv[k]);
      if (edge) { msh_count(&K, xa, ya, &nn[0]); msh_count(&K, xb, yb, &nn[1]); }
      msh_sums(sv.val[0], tk, &s[0], &q[0], &t[0]); msh_sums(sv.val[1], tk, &s[1], &q[1], &t[1]);
    }
  } else {
    for (; k + 1 < cells; k += 2) {
      MSH_CELL(k, 0, xa, ya); MSH_CELL(k + 1, 0, xb, yb);
      const lin_f8_t sv = msh_read8(im, &K, xa, ya, xb, yb);
      const __m128 ta = _mm_set1_ps(tv[k]), tb = _mm_set1_ps(tv[k + 1]);
      if (edge) { msh_count(&K, xa, ya, &nn[0]); msh_count(&K, xb, yb, &nn[1]); }
      msh_sums(sv.val[0], ta, &s[0], &q[0], &t[0]); msh_sums(sv.val[1], tb, &s[1], &q[1], &t[1]);
    }
    if (k < cells) {
      MSH_CELL(k, 0, xa, ya);
      const __m128 ta = _mm_set1_ps(tv[k]);
      if (edge) msh_count(&K, xa, ya, &nn[0]);
      msh_sums(msh_read8(im, &K, xa, ya, xa, ya).val[0], ta, &s[0], &q[0], &t[0]);
    }
    s[0] = _mm_add_ps(s[0], s[1]); q[0] = _mm_add_ps(q[0], q[1]); t[0] = _mm_add_ps(t[0], t[1]); nn[0] = _mm_add_ps(nn[0], nn[1]);
  }
#undef MSH_CELL
  const __m128 n = _mm_set1_ps((float)cells), vst = _mm_set1_ps(st), vvt = _mm_set1_ps(vt);
  for (int g = 0; g < G; g++) {
    const __m128 vs = _mm_sub_ps(q[g], _mm_div_ps(_mm_mul_ps(s[g], s[g]), n));
    const __m128 c = _mm_div_ps(_mm_sub_ps(t[g], _mm_div_ps(_mm_mul_ps(vst, s[g]), n)), _mm_sqrt_ps(_mm_mul_ps(vvt, vs)));
    __m128 r = vt <= 0 ? z : _mm_andnot_ps(_mm_cmple_ps(vs, _mm_set1_ps(1e-9f)), c);
    if (edge) r = _mm_blendv_ps(r, _mm_set1_ps(NODE_NONE), _mm_cmplt_ps(nn[g], _mm_set1_ps((float)nmin)));
    _mm_storeu_ps(out + 4 * g, r);
  }
}
#ifdef OB_V3_ON
// ai: x86-64-v3 (2026-11-05): msh_ncc_g at gamma 1 on eight lanes, two cells a sixteen-point read (lin16), each lane's
// ai: sums in the v2 form's order: at G 2 a lane an offset and a cell a step, at G 1 the v2 form's even and odd cells'
// ai: sums as the two halves (an odd last cell into the first half alone), added at the end; so bit for bit the v2
// ai: form's (test/mesh_neon.c's x86 hash)
typedef struct { __m256 lo, hi; } lin16_f_t;
OB_V3_IN lin16_f_t lin16_get(const uint8_t *r0, const uint8_t *r1, lin16_pts_t p) {
  const __m256i r = lin16_grey(r0, r1, p);
  const __m256 sc = _mm256_set1_ps(257.0f / 8388608.0f);
  return (lin16_f_t){ _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(r))), sc),
                      _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(r, 1))), sc) };
}
OB_V3_IN void msh_ncc_v3g(const image_t *im, const homo_t *H, int cells, const float *cu, const float *cv, const float *cw, const float *tv, float st, float vt, float qx, float qy, const float *ox, const float *oy, float *out, const int G, const int edge, int nmin) {
  const float *h = H->h;
  const lin16_k_t K = { _mm256_set1_epi16((short)(im->w - 2)), _mm256_set1_epi16((short)(im->h - 2)), _mm256_set1_epi16((short)im->w) };
  const __m256 xm = _mm256_set1_ps((float)(im->w - 2) + 0.998f), ym = _mm256_set1_ps((float)(im->h - 2) + 0.9999f);
  const __m256 xl = _mm256_set1_ps(0.5f - MSH_OX), xh = _mm256_set1_ps(im->w - 0.5f - MSH_OX), yh = _mm256_set1_ps(im->h - 0.5f - MSH_OY);
  const __m256 px = _mm256_set1_ps(qx), py = _mm256_set1_ps(qy), z = _mm256_setzero_ps(), one = _mm256_set1_ps(1);
  const uint8_t *r0 = im->px, *r1 = im->px + im->w - 2;
#define V3_MA(a, b, c) _mm256_add_ps((a), _mm256_mul_ps((b), (c)))
  __m256 a8, b8;
  if (G == 2) { a8 = _mm256_loadu_ps(ox); b8 = _mm256_loadu_ps(oy); }
  else { const __m128 a = _mm_loadu_ps(ox), b = _mm_loadu_ps(oy); a8 = _mm256_set_m128(a, a); b8 = _mm256_set_m128(b, b); }
  const __m256 du = V3_MA(_mm256_mul_ps(a8, _mm256_set1_ps(h[0])), b8, _mm256_set1_ps(h[1])), dv = V3_MA(_mm256_mul_ps(a8, _mm256_set1_ps(h[3])), b8, _mm256_set1_ps(h[4]));
  const __m256 dw = V3_MA(_mm256_mul_ps(a8, _mm256_set1_ps(h[6])), b8, _mm256_set1_ps(h[7]));
  // ai: a vector's points: G 2 one cell's eight offsets; G 1 cells K0 (lanes 0-3) and K1 (4-7), four offsets each
#define V3_CELL(K0, K1, X, Y) do { \
    const __m256 cu_ = _mm256_set_m128(_mm_set1_ps(cu[K1]), _mm_set1_ps(cu[K0])), cv_ = _mm256_set_m128(_mm_set1_ps(cv[K1]), _mm_set1_ps(cv[K0])), cw_ = _mm256_set_m128(_mm_set1_ps(cw[K1]), _mm_set1_ps(cw[K0])); \
    const __m256 r_ = _mm256_div_ps(one, _mm256_add_ps(cw_, dw)); \
    X = V3_MA(px, _mm256_add_ps(cu_, du), r_); Y = V3_MA(py, _mm256_add_ps(cv_, dv), r_); } while (0)
#define V3_TK(K0, K1) _mm256_set_m128(_mm_set1_ps(tv[K1]), _mm_set1_ps(tv[K0]))
#define V3_IN(X, Y) _mm256_and_ps(_mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(xl, X, _CMP_LE_OQ), _mm256_cmp_ps(X, xh, _CMP_LE_OQ)), \
                      _mm256_and_ps(_mm256_cmp_ps(Y, z, _CMP_GE_OQ), _mm256_cmp_ps(Y, yh, _CMP_LE_OQ))), one)
#define V3_SUMS(SV, TK) do { s = _mm256_add_ps(s, SV); q = V3_MA(q, SV, SV); t = V3_MA(t, TK, SV); } while (0)
#define V3_READ(XA, YA, XB, YB) lin16_get(r0, r1, lin16_pts(&K, _mm256_min_ps(xm, XA), _mm256_min_ps(ym, YA), _mm256_min_ps(xm, XB), _mm256_min_ps(ym, YB)))
  __m256 s = z, q = z, t = z, nn = z, xa, ya, xb, yb;
  int k = 0;
  if (G == 2) {
    for (; k + 1 < cells; k += 2) {
      V3_CELL(k, k, xa, ya); V3_CELL(k + 1, k + 1, xb, yb);
      const lin16_f_t v = V3_READ(xa, ya, xb, yb);
      if (edge) { nn = _mm256_add_ps(nn, V3_IN(xa, ya)); nn = _mm256_add_ps(nn, V3_IN(xb, yb)); }
      V3_SUMS(v.lo, V3_TK(k, k)); V3_SUMS(v.hi, V3_TK(k + 1, k + 1));
    }
    if (k < cells) {
      V3_CELL(k, k, xa, ya);
      const lin16_f_t v = V3_READ(xa, ya, xa, ya);
      if (edge) nn = _mm256_add_ps(nn, V3_IN(xa, ya));
      V3_SUMS(v.lo, V3_TK(k, k));
    }
  } else {
    for (; k + 3 < cells; k += 4) {
      V3_CELL(k, k + 1, xa, ya); V3_CELL(k + 2, k + 3, xb, yb);
      const lin16_f_t v = V3_READ(xa, ya, xb, yb);
      if (edge) { nn = _mm256_add_ps(nn, V3_IN(xa, ya)); nn = _mm256_add_ps(nn, V3_IN(xb, yb)); }
      V3_SUMS(v.lo, V3_TK(k, k + 1)); V3_SUMS(v.hi, V3_TK(k + 2, k + 3));
    }
    if (k + 1 < cells) {
      V3_CELL(k, k + 1, xa, ya);
      const lin16_f_t v = V3_READ(xa, ya, xa, ya);
      if (edge) nn = _mm256_add_ps(nn, V3_IN(xa, ya));
      V3_SUMS(v.lo, V3_TK(k, k + 1));
      k += 2;
    }
    if (k < cells) {   // ai: the odd last cell into the first half alone, the second kept exactly
      V3_CELL(k, k, xa, ya);
      const lin16_f_t v = V3_READ(xa, ya, xa, ya);
      const __m256 tk = V3_TK(k, k);
      const __m256 s1 = _mm256_add_ps(s, v.lo), q1 = V3_MA(q, v.lo, v.lo), t1 = V3_MA(t, tk, v.lo);
      s = _mm256_blend_ps(s, s1, 0x0f); q = _mm256_blend_ps(q, q1, 0x0f); t = _mm256_blend_ps(t, t1, 0x0f);
      if (edge) nn = _mm256_blend_ps(nn, _mm256_add_ps(nn, V3_IN(xa, ya)), 0x0f);
    }
  }
#undef V3_CELL
#undef V3_TK
#undef V3_IN
#undef V3_SUMS
#undef V3_READ
#undef V3_MA
  if (G == 2) {
    const __m256 n = _mm256_set1_ps((float)cells), vst = _mm256_set1_ps(st), vvt = _mm256_set1_ps(vt);
    const __m256 vs = _mm256_sub_ps(q, _mm256_div_ps(_mm256_mul_ps(s, s), n));
    const __m256 c = _mm256_div_ps(_mm256_sub_ps(t, _mm256_div_ps(_mm256_mul_ps(vst, s), n)), _mm256_sqrt_ps(_mm256_mul_ps(vvt, vs)));
    __m256 r = vt <= 0 ? z : _mm256_andnot_ps(_mm256_cmp_ps(vs, _mm256_set1_ps(1e-9f), _CMP_LE_OQ), c);
    if (edge) r = _mm256_blendv_ps(r, _mm256_set1_ps(NODE_NONE), _mm256_cmp_ps(nn, _mm256_set1_ps((float)nmin), _CMP_LT_OQ));
    _mm256_storeu_ps(out, r);
    return;
  }
  // ai: G 1: the halves added (the v2 form's s[0] + s[1]), then its four lanes' arithmetic
  const __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1)), q4 = _mm_add_ps(_mm256_castps256_ps128(q), _mm256_extractf128_ps(q, 1));
  const __m128 t4 = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1)), n4 = _mm_add_ps(_mm256_castps256_ps128(nn), _mm256_extractf128_ps(nn, 1));
  const __m128 n = _mm_set1_ps((float)cells), vst = _mm_set1_ps(st), vvt = _mm_set1_ps(vt), z4 = _mm_setzero_ps();
  const __m128 vs = _mm_sub_ps(q4, _mm_div_ps(_mm_mul_ps(s4, s4), n));
  const __m128 c = _mm_div_ps(_mm_sub_ps(t4, _mm_div_ps(_mm_mul_ps(vst, s4), n)), _mm_sqrt_ps(_mm_mul_ps(vvt, vs)));
  __m128 r = vt <= 0 ? z4 : _mm_andnot_ps(_mm_cmple_ps(vs, _mm_set1_ps(1e-9f)), c);
  if (edge) r = _mm_blendv_ps(r, _mm_set1_ps(NODE_NONE), _mm_cmplt_ps(n4, _mm_set1_ps((float)nmin)));
  _mm_storeu_ps(out, r);
}
OB_V3 static void msh_ncc_v3(const image_t *im, const homo_t *H, int cells, const float *cu, const float *cv, const float *cw, const float *tv, float st, float vt, float qx, float qy, const float *ox, const float *oy, float *out, int lanes, int edge, int nmin) {
  if (edge) {
    if (lanes > 4) msh_ncc_v3g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 2, 1, nmin);
    else msh_ncc_v3g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 1, 1, nmin);
  } else if (lanes > 4) msh_ncc_v3g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 2, 0, nmin);
  else msh_ncc_v3g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 1, 0, nmin);
}
#endif
static void msh_ncc(const image_t *im, const homo_t *H, int cells, const float *cu, const float *cv, const float *cw, const float *tv, float st, float vt, float qx, float qy, const float *ox, const float *oy, float *out, int lanes, int edge, int nmin) {
#ifdef OB_V3_ON
  if (im->lin && ob_v3()) { msh_ncc_v3(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, lanes, edge, nmin); return; }
#endif
  if (edge) {
    if (lanes > 4) msh_ncc_g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 2, 1, nmin);
    else msh_ncc_g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 1, 1, nmin);
  } else if (lanes > 4) msh_ncc_g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 2, 0, nmin);
  else msh_ncc_g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 1, 0, nmin);
}
#endif

#if !defined(OB_NEON) && !defined(OB_X86)   // ai: arm64's and x86-64's is msh_ncc's lanes
// Normalized correlation of a node's known cells against the image, with the node shifted by
// (ox, oy) modules from where the current estimate puts it.
// r: the patch as cell offsets from (cx0, cy0), x from r[0] to r[1] and y from r[2] to r[3], ends excluded.
// Every cell is placed through the homography itself, then moved by what the neighbours say (pdx, pdy). A local
// affine map at the node was used until 2026-09-23 and is right only while the template is small in pixels: a
// corner template is 12 modules, 31 px on a 286-module symbol at 1080 fill 0.7 and 96 px on a 94-module one, and
// at 30 degrees this camera's perspective changes scale by about 6% across 96 px, which misplaces the template's
// far cells by over a pixel. Measured: the refined corners of LIZARD-16 at 94 modules came out 1.7 px off the
// truth where the unrefined mark quad was 0.6.
// edge: the position may read past the capture (node_inside), so each cell's point is placed again (project, the same
// floats as the read's) and those inside counted; nmin: the cells inside a judged position needs.
static float node_ncc(const ob_layout_t *L, const image_t *im, int cx0, int cy0, const int *r, const homo_t *H, float pdx, float pdy, float ox, float oy, int edge, int nmin) {
  float st = 0, ss = 0, stt = 0, sss = 0, sts = 0;
  int n = 0, nin = 0, lo = r[0], hi = r[1], ylo = r[2], yhi = r[3], side = hi - lo;
  // Image reads first, four to a vector along each row of the patch; the sums then run over them
  // in the original order, so the correlation comes out the same float.
  float S[OB_FB * OB_FB];
  for (int dy = ylo; dy < yhi; dy++) {
    const float my = cy0 + dy + 0.5f + oy;
    int dx = lo;
#ifdef OB_SIMD
    const v128_t vy = wasm_f32x4_splat(my);
    for (; dx + 4 <= hi; dx += 4) {
      int cx = cx0 + dx;
      const v128_t mx = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_make((float)cx, (float)(cx + 1), (float)(cx + 2), (float)(cx + 3)), wasm_f32x4_splat(0.5f)), wasm_f32x4_splat(ox));
      v128_t u, v;
      project4(H, mx, vy, &u, &v);
      wasm_v128_store(S + (dy - ylo) * side + dx - lo, sample4(im, wasm_f32x4_add(u, wasm_f32x4_splat(pdx)), wasm_f32x4_add(v, wasm_f32x4_splat(pdy))));
    }
#endif
    for (; dx < hi; dx++) { const float mx = cx0 + dx + 0.5f + ox; float u, v; project(H, mx, my, &u, &v); S[(dy - ylo) * side + dx - lo] = sample(im, u + pdx, v + pdy); }
  }
  for (int dy = ylo; dy < yhi; dy++) for (int dx = lo; dx < hi; dx++) {
    int kind = L->kind[(cy0 + dy) * L->w + cx0 + dx];
    if (kind == CELL_DATA) continue;   // the picture inside the border: not the border's to know
    if (kind & CELL_WORD) continue;               // nor the format word, whose bits this end did not choose (layout.h)
    if (edge) { float u, v; project(H, cx0 + dx + 0.5f + ox, cy0 + dy + 0.5f + oy, &u, &v); nin += in_capture(im, u + pdx, v + pdy); }
    float t = (kind & 3) == CELL_DARK ? -1.0f : 1.0f, s = S[(dy - ylo) * side + dx - lo];
    st += t; ss += s; stt += t * t; sss += s * s; sts += t * s; n++;
  }
  if (edge && nin < nmin) return NODE_NONE;
  float vt = stt - st * st / n, vs = sss - ss * ss / n;
  if (vt <= 0 || vs <= 1e-9f) return 0;
  return (sts - st * ss / n) / sqrtf(vt * vs);
}
#endif


static void refine_node(const ob_layout_t *L, const image_t *im, const homo_t *H, int i, int j, node_t *nodes, const struct lensm *lm) {
  const int cx0 = (int)L->node_x[i], cy0 = (int)L->node_y[j];
  const float mcx = cx0 + 0.5f, mcy = cy0 + 0.5f;
  // A thin frame's template is as long as the border lets it be: thirteen modules along, the border's
  // five across, and an L of both at a corner. Small symbols need the area; 5 x 5 is 8 px there.
  int rect[4];
  {
    int ex = (j == 0 || j == L->ny - 1) ? 6 : 2, ey = (i == 0 || i == L->nx - 1) ? 6 : 2;
    // A corner that carries a mark is a different thing to match. Six modules either way stays INSIDE a 12-module
    // mark, so the template is a dark block with one module of light rim, and once blur has closed the mark's gap
    // there is nothing in it to lock to: measured at 2 px of defocus, refining the corners took a marked symbol
    // from 6 blocks of 96 to 2 where it takes an unmarked one from 19 to 35. Reaching a module past the mark's
    // inner edge puts the square's own edges in the template, which is the strongest feature the symbol has.
    if (ex == 6 && ey == 6) { ex = L->corner - 1 < OB_FB - 3 ? L->corner - 1 : OB_FB - 3; ey = ex; }
    rect[0] = -ex < -cx0 ? -cx0 : -ex; rect[1] = ex + 1 > L->w - cx0 ? L->w - cx0 : ex + 1;
    rect[2] = -ey < -cy0 ? -cy0 : -ey; rect[3] = ey + 1 > L->h - cy0 ? L->h - cy0 : ey + 1;
  }
  // Start from what the already-measured neighbours say the local error is.
  float pdx = 0, pdy = 0;
  int pn = 0;
  for (int dj = -1; dj <= 1; dj++) for (int di = -1; di <= 1; di++) {
    int ii = i + di, jj = j + dj;
    if (ii < 0 || jj < 0 || ii >= L->nx || jj >= L->ny || !nodes[jj * L->nx + ii].done) continue;
    float rx = nodes[jj * L->nx + ii].dx, ry = nodes[jj * L->nx + ii].dy;
    // ai: with the anchors' model, the neighbour's residual from it (the model carries the lens, the neighbours the rest)
    if (lm) { float u, v, mx, my; project(H, L->node_x[ii], L->node_y[jj], &u, &v); lensm_at(lm, u, v, &mx, &my); rx -= mx; ry -= my; }
    pdx += rx; pdy += ry; pn++;
  }
  if (pn) { pdx /= pn; pdy /= pn; }
  if (lm) { float u, v, mx, my; project(H, L->node_x[i], L->node_y[j], &u, &v); lensm_at(lm, u, v, &mx, &my); pdx += mx; pdy += my; }
  // ai: near the capture's edge a position judged where most of its template is inside it (the rule above); GM marks
  // ai: the coarse grid's points inside
  const int edge = !node_inside(im, H, cx0, cy0, rect, pdx, pdy, -1.5f, 1.5f, -1.5f, 1.5f);
  float px, py, ax, ay, bx, by;
  project(H, mcx, mcy, &px, &py); project(H, mcx + 1, mcy, &ax, &ay); project(H, mcx, mcy + 1, &bx, &by);
  // The node's local Jacobian, only to turn the offset found in modules into pixels at the end.
  float jxx = ax - px, jxy = ay - py, jyx = bx - px, jyy = by - py;
  // The coarse search, 7 x 7 offsets of half a module. A shift of a whole module puts every cell where its
  // neighbour was, so the 49 correlations read the same image points many times over. They are read once, on a
  // half-module grid, and each correlation picks its cells out of it: the same floats summed in the same order
  // as node_ncc would, for about a seventh of the reads.
  enum { GMAX = 2 * OB_FB + 5 };
  float G[GMAX * GMAX + 8], GM[GMAX * GMAX + 8], tv[OB_FB * OB_FB];   // + 8: arm64's last row written eight floats at a time; the eighth lane of its sums reads one float past the grid
  int at[OB_FB * OB_FB], cells = 0;
  const int lo = rect[0], ylo = rect[2], side = rect[1] - rect[0], tall = rect[3] - rect[2], GW = 2 * side + 5, GH = 2 * tall + 5;
  // Placed through the homography, as node_ncc does and for the same reason.
#ifdef OB_NEON
  // ai: a row eight points a read: its last read runs on into the next row's first floats, which that row writes
  // ai: again (the last row's into G's spare floats, zeroed below); near the capture's edge GM marks the points inside
  const msh_k_t K = msh_k(im);
  const float32x4_t qx = vdupq_n_f32(pdx - MSH_OX), qy = vdupq_n_f32(pdy - MSH_OY), z = vdupq_n_f32(0), one = vdupq_n_f32(1);
  const float *hh = H->h;
  for (int gy = 0; gy < GH; gy++) {
    const float my = (float)(2 * (cy0 + ylo) + gy - 2) * 0.5f;
    const float32x4_t ru = vdupq_n_f32(hh[1] * my + hh[2]), rv = vdupq_n_f32(hh[4] * my + hh[5]), rw = vdupq_n_f32(hh[7] * my + 1), lane = { 0, 1, 2, 3 };
    for (int gx = 0; gx < GW; gx += 8) {
      const float32x4_t ma = vmulq_f32(vaddq_f32(vdupq_n_f32((float)(2 * (cx0 + lo) + gx - 2)), lane), vdupq_n_f32(0.5f)), mb = vaddq_f32(ma, vdupq_n_f32(2));
      float32x4_t xa, ya, xb, yb;
      msh_proj(vfmaq_n_f32(ru, ma, hh[0]), vfmaq_n_f32(rv, ma, hh[3]), vfmaq_n_f32(rw, ma, hh[6]), qx, qy, &xa, &ya);
      msh_proj(vfmaq_n_f32(ru, mb, hh[0]), vfmaq_n_f32(rv, mb, hh[3]), vfmaq_n_f32(rw, mb, hh[6]), qx, qy, &xb, &yb);
      const float32x4x2_t sv = msh_read8(im, &K, xa, ya, xb, yb);
      if (edge) { vst1q_f32(GM + gy * GW + gx, vbslq_f32(msh_in4r(&K, xa, ya), one, z)); vst1q_f32(GM + gy * GW + gx + 4, vbslq_f32(msh_in4r(&K, xb, yb), one, z)); }
      vst1q_f32(G + gy * GW + gx, sv.val[0]); vst1q_f32(G + gy * GW + gx + 4, sv.val[1]);
    }
  }
#elif defined(OB_X86)
  // ai: arm64's form above: a row eight points a read, its last read running on into the next row's first floats
  const msh_k_t K = msh_k(im);
  const __m128 qx = _mm_set1_ps(pdx - MSH_OX), qy = _mm_set1_ps(pdy - MSH_OY), one = _mm_set1_ps(1), lane = _mm_setr_ps(0, 1, 2, 3);
  const __m128 h0 = _mm_set1_ps(H->h[0]), h3 = _mm_set1_ps(H->h[3]), h6 = _mm_set1_ps(H->h[6]);
  for (int gy = 0; gy < GH; gy++) {
    const float my = (float)(2 * (cy0 + ylo) + gy - 2) * 0.5f;
    const __m128 ru = _mm_set1_ps(H->h[1] * my + H->h[2]), rv = _mm_set1_ps(H->h[4] * my + H->h[5]), rw = _mm_set1_ps(H->h[7] * my + 1);
    for (int gx = 0; gx < GW; gx += 8) {
      const __m128 ma = _mm_mul_ps(_mm_add_ps(_mm_set1_ps((float)(2 * (cx0 + lo) + gx - 2)), lane), _mm_set1_ps(0.5f)), mb = _mm_add_ps(ma, _mm_set1_ps(2));
      __m128 xa, ya, xb, yb;
      msh_proj(MSH_MA(ru, ma, h0), MSH_MA(rv, ma, h3), MSH_MA(rw, ma, h6), qx, qy, &xa, &ya);
      msh_proj(MSH_MA(ru, mb, h0), MSH_MA(rv, mb, h3), MSH_MA(rw, mb, h6), qx, qy, &xb, &yb);
      const lin_f8_t sv = msh_read8(im, &K, xa, ya, xb, yb);
      if (edge) { _mm_storeu_ps(GM + gy * GW + gx, _mm_and_ps(msh_in4r(&K, xa, ya), one)); _mm_storeu_ps(GM + gy * GW + gx + 4, _mm_and_ps(msh_in4r(&K, xb, yb), one)); }
      _mm_storeu_ps(G + gy * GW + gx, sv.val[0]); _mm_storeu_ps(G + gy * GW + gx + 4, sv.val[1]);
    }
  }
#else
  for (int gy = 0; gy < GH; gy++) {
    const float my = (float)(2 * (cy0 + ylo) + gy - 2) * 0.5f;
    int gx = 0;
#ifdef OB_SIMD
    const v128_t vy = wasm_f32x4_splat(my);
    for (; gx + 4 <= GW; gx += 4) {
      const int k = 2 * (cx0 + lo) + gx - 2;
      const v128_t mx = wasm_f32x4_mul(wasm_f32x4_make((float)k, (float)(k + 1), (float)(k + 2), (float)(k + 3)), wasm_f32x4_splat(0.5f));
      v128_t u, v;
      project4(H, mx, vy, &u, &v);
      const v128_t x = wasm_f32x4_add(u, wasm_f32x4_splat(pdx)), y = wasm_f32x4_add(v, wasm_f32x4_splat(pdy)), lo5 = wasm_f32x4_splat(0.5f);
      wasm_v128_store(G + gy * GW + gx, sample4(im, x, y));
      if (edge) {
        const v128_t in = wasm_v128_and(wasm_v128_and(wasm_f32x4_le(lo5, x), wasm_f32x4_le(x, wasm_f32x4_splat(im->w - 0.5f))), wasm_v128_and(wasm_f32x4_le(lo5, y), wasm_f32x4_le(y, wasm_f32x4_splat(im->h - 0.5f))));
        wasm_v128_store(GM + gy * GW + gx, wasm_v128_and(wasm_f32x4_splat(1), in));
      }
    }
#endif
    for (; gx < GW; gx++) {
      const float mx = (float)(2 * (cx0 + lo) + gx - 2) * 0.5f;
      float u, v;
      project(H, mx, my, &u, &v);
      G[gy * GW + gx] = sample(im, u + pdx, v + pdy); GM[gy * GW + gx] = !edge || in_capture(im, u + pdx, v + pdy) ? 1.0f : 0;
    }
  }
#endif
  float st = 0, stt = 0;
#if defined(OB_NEON) || defined(OB_X86)
  float cu[OB_FB * OB_FB], cv[OB_FB * OB_FB], cw[OB_FB * OB_FB];   // ai: a kept cell's middle (node_ncc's cx0 + dx + 0.5f) through H's three lines
#endif
  for (int dy = 0; dy < tall; dy++) for (int dx = 0; dx < side; dx++) {
    int kind = L->kind[(cy0 + ylo + dy) * L->w + cx0 + lo + dx];
    if (kind == CELL_DATA) continue;   // the picture inside the border: not the border's to know
    if (kind & CELL_WORD) continue;               // nor the format word, whose bits this end did not choose (layout.h)
    tv[cells] = (kind & 3) == CELL_DARK ? -1.0f : 1.0f; at[cells] = (2 * dy + 3) * GW + 2 * dx + 3;
#if defined(OB_NEON) || defined(OB_X86)
    const float mx = cx0 + lo + dx + 0.5f, my = cy0 + ylo + dy + 0.5f;
    cu[cells] = H->h[0] * mx + H->h[1] * my + H->h[2]; cv[cells] = H->h[3] * mx + H->h[4] * my + H->h[5]; cw[cells] = H->h[6] * mx + H->h[7] * my + 1;
#endif
    st += tv[cells]; stt += tv[cells] * tv[cells]; cells++;
  }
  const float vt = stt - st * st / cells;
  const int nmin = (cells + 1) / 2;
  float best = NODE_NONE, box = 0, boy = 0;
  for (int k = 0; k < 4; k++) G[GW * GH + k] = GM[GW * GH + k] = 0;
  for (int b = -3; b <= 3; b++) {
    // The seven offsets a of one b sit side by side in the grid, so their sums run as lanes: each lane adds the
    // same cells in the same order as a loop of its own would. Near the capture's edge a lane's cells inside the
    // capture are counted too (SN, from GM), in a pass of their own.
    float SS[8], SQ[8], ST[8], SN[8];
#ifdef OB_NEON
    // ai: msh_sums' lanes over the grid
    const float *g0 = G + b * GW - 3, *m0 = GM + b * GW - 3;
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, q0 = s0, q1 = s0, t0 = s0, t1 = s0, n0 = s0, n1 = s0;
    for (int k = 0; k < cells; k++) {
      const float32x4_t tk = vdupq_n_f32(tv[k]);
      msh_sums(vld1q_f32(g0 + at[k]), tk, &s0, &q0, &t0); msh_sums(vld1q_f32(g0 + at[k] + 4), tk, &s1, &q1, &t1);
    }
    if (edge) for (int k = 0; k < cells; k++) { n0 = vaddq_f32(n0, vld1q_f32(m0 + at[k])); n1 = vaddq_f32(n1, vld1q_f32(m0 + at[k] + 4)); }
    vst1q_f32(SS, s0); vst1q_f32(SS + 4, s1); vst1q_f32(SQ, q0); vst1q_f32(SQ + 4, q1); vst1q_f32(ST, t0); vst1q_f32(ST + 4, t1); vst1q_f32(SN, n0); vst1q_f32(SN + 4, n1);
#elif defined(OB_SIMD)
    const float *g0 = G + b * GW - 3, *m0 = GM + b * GW - 3;
    v128_t s0 = wasm_f32x4_splat(0), s1 = s0, q0 = s0, q1 = s0, t0 = s0, t1 = s0, n0 = s0, n1 = s0;
    for (int k = 0; k < cells; k++) {
      const v128_t v0 = wasm_v128_load(g0 + at[k]), v1 = wasm_v128_load(g0 + at[k] + 4), t = wasm_f32x4_splat(tv[k]);
      s0 = wasm_f32x4_add(s0, v0); s1 = wasm_f32x4_add(s1, v1); q0 = wasm_f32x4_add(q0, wasm_f32x4_mul(v0, v0)); q1 = wasm_f32x4_add(q1, wasm_f32x4_mul(v1, v1));
      t0 = wasm_f32x4_add(t0, wasm_f32x4_mul(t, v0)); t1 = wasm_f32x4_add(t1, wasm_f32x4_mul(t, v1));
    }
    if (edge) for (int k = 0; k < cells; k++) { n0 = wasm_f32x4_add(n0, wasm_v128_load(m0 + at[k])); n1 = wasm_f32x4_add(n1, wasm_v128_load(m0 + at[k] + 4)); }
    wasm_v128_store(SS, s0); wasm_v128_store(SS + 4, s1); wasm_v128_store(SQ, q0); wasm_v128_store(SQ + 4, q1); wasm_v128_store(ST, t0); wasm_v128_store(ST + 4, t1);
    wasm_v128_store(SN, n0); wasm_v128_store(SN + 4, n1);
#else
    for (int a = -3; a <= 3; a++) {
      const float *g0 = G + b * GW + a, *m0 = GM + b * GW + a;
      float ss = 0, sss = 0, sts = 0, sn = 0;
      for (int k = 0; k < cells; k++) { const float v = g0[at[k]]; ss += v; sss += v * v; sts += tv[k] * v; }
      if (edge) for (int k = 0; k < cells; k++) sn += m0[at[k]];
      SS[a + 3] = ss; SQ[a + 3] = sss; ST[a + 3] = sts; SN[a + 3] = sn;
    }
#endif
    for (int a = -3; a <= 3; a++) {
      const float ss = SS[a + 3], vs = SQ[a + 3] - ss * ss / cells;
      const float sc = edge && SN[a + 3] < nmin ? NODE_NONE : vt <= 0 || vs <= 1e-9f ? 0 : (ST[a + 3] - st * ss / cells) / sqrtf(vt * vs);
      if (sc > best) { best = sc; box = a * 0.5f; boy = b * 0.5f; }
    }
  }
  // Hill-climb at a quarter module, then an eighth, then a parabola through the peak. A correlation already
  // known is not taken again: the centre always is, and after a move so are the cells the old grid shares.
  float g[3][3], step = 0.25f;
  int have[3][3] = { { 0 } }, ma = 1, mb = 1;   // (ma, mb): the cell the last step moved to, not yet made the centre
  g[1][1] = best; have[1][1] = 1;
  for (int pass = 0; pass < 2; pass++, step *= 0.5f) {
    for (int move = 0; move < 8; move++) {
      if (ma != 1 || mb != 1) {
        float ng[3][3]; int nh[3][3];
        for (int b2 = 0; b2 < 3; b2++) for (int a2 = 0; a2 < 3; a2++) { int sb = b2 + mb - 1, sa = a2 + ma - 1, in = sb >= 0 && sb < 3 && sa >= 0 && sa < 3; nh[b2][a2] = in && have[sb][sa]; ng[b2][a2] = in ? g[sb][sa] : 0; }
        memcpy(g, ng, sizeof g); memcpy(have, nh, sizeof have); ma = mb = 1;
      }
      // ai: whether this move's positions may read past the capture (the climb's edge)
      const int ce = !node_inside(im, H, cx0, cy0, rect, pdx, pdy, box - step, box + step, boy - step, boy + step);
#if defined(OB_NEON) || defined(OB_X86)
      {
        // ai: the correlations this move lacks, a lane each (lanes past them repeat the first and are dropped)
        float ox[8], oy[8], got[8];
        int cell[8], m = 0;
        for (int b2 = -1; b2 <= 1; b2++) for (int a2 = -1; a2 <= 1; a2++) if (!have[b2 + 1][a2 + 1]) { ox[m] = box + a2 * step; oy[m] = boy + b2 * step; cell[m++] = (b2 + 1) * 3 + a2 + 1; }
        if (m) {
          for (int l = m; l < 8; l++) { ox[l] = ox[0]; oy[l] = oy[0]; }
          msh_ncc(im, H, cells, cu, cv, cw, tv, st, vt, pdx - MSH_OX, pdy - MSH_OY, ox, oy, got, m, ce, nmin);
          for (int l = 0; l < m; l++) { g[cell[l] / 3][cell[l] % 3] = got[l]; have[cell[l] / 3][cell[l] % 3] = 1; }
        }
      }
#else
      for (int b2 = -1; b2 <= 1; b2++) for (int a2 = -1; a2 <= 1; a2++) if (!have[b2 + 1][a2 + 1]) { g[b2 + 1][a2 + 1] = node_ncc(L, im, cx0, cy0, rect, H, pdx, pdy, box + a2 * step, boy + b2 * step, ce, nmin); have[b2 + 1][a2 + 1] = 1; }
#endif
      int ba = 1, bb = 1;
      for (int b2 = 0; b2 < 3; b2++) for (int a2 = 0; a2 < 3; a2++) if (g[b2][a2] > g[bb][ba]) { ba = a2; bb = b2; }
      if (ba == 1 && bb == 1) break;
      box += (ba - 1) * step; boy += (bb - 1) * step; ma = ba; mb = bb;
    }
    if (pass == 0) {
      // The finer pass starts from the centre alone. If the moves ran out, that is the cell last moved to.
      float c = g[mb][ma];
      memset(have, 0, sizeof have); g[1][1] = c; have[1][1] = 1; ma = mb = 1;
    }
  }
  best = g[1][1];
  float dxx = g[1][0] - 2 * g[1][1] + g[1][2], dyy = g[0][1] - 2 * g[1][1] + g[2][1];
  if (dxx < -1e-6f && g[1][0] > NODE_NONE && g[1][2] > NODE_NONE) box += 0.125f * 0.5f * (g[1][0] - g[1][2]) / dxx;
  if (dyy < -1e-6f && g[0][1] > NODE_NONE && g[2][1] > NODE_NONE) boy += 0.125f * 0.5f * (g[0][1] - g[2][1]) / dyy;
  node_t *nd = &nodes[j * L->nx + i];
  nd->done = 1; nd->score = best;
  if (best < 0.35f) { nd->dx = pdx; nd->dy = pdy; return; }
  nd->dx = pdx + box * jxx + boy * jyx;
  nd->dy = pdy + box * jxy + boy * jyy;
}

// The mesh: the interior of a code, computed from its border and its anchors. Averaging neighbours inward is wrong for
// the two errors that matter: a lens's radial distortion, once the corners are pinned, is largest along the edges and
// zero in the middle, which no average of edge values reproduces, and rolling-shutter jello is a sideways shift that
// depends on the image row alone. So the border's residuals and the anchors' are fitted with a model of the lens and of
// what the homography missed (below), per frame, nothing calibrated, and what the fit leaves at the anchors is carried
// between them by a cubic mesh (border_fill_anchors). Under the anchors' minimum none of them is used and the same fit
// over the border alone is the interior.
// ai: The lens about its axis (2026-10-17; STATUS "The lens about its axis"). What the border nodes and the anchors say
// ai: H missed, fitted as the radial terms k1 to kK about the camera's optical axis, k |q|^(2k) q (the radial part of
// ai: Android's lens form, linearised about H), beside the first-order change of the homography itself (an affine map,
// ai: a shift and the two perspective terms -(g . q) q, which H's four corners, distorted points, leave), q a point's
// ai: place about the axis over the farthest image corner's distance from it: a lens about an axis off the code's middle
// ai: (a 2:1 half's is the middle of its inner edge) bent the main camera's picture 2.3 samples off in the middle of the
// ai: code, where one term about the image's centre left it. K (0 to 3) is the one the evidence pays for, by AICc
// ai: (lensm_fit), each frame on its own; the anchors sample the interior's radii, so no cap inside a 1:1 crop. The
// ai: centre: the axis the caller gives (lizard_set_axis), else the image's centre, for the anchors' first fits; the final
// ai: fit (border_fill_anchors) fits the centre itself (lens_centre).
enum { LENS_TERMS = 3, LENS_P = 8 + LENS_TERMS };
// ai: In the 128 and 256 rings. The 64 ring's 3 x 3 grid takes 8 of its 9 (ob_anchor_min), so a fit over the anchors alone
// ai: (and the corners H passes through: anchor_register's, anchor_model, which seed the border's search) takes no more
// ai: terms than anchors (lens_cap: the homography's change alone at 8, one radial term at 9); uncapped, the 64 ring's
// ai: clean paints at 1280 and 1536 lost their word to a border searched from a bent model (ring_pairs, 2026-11-07). The
// ai: final fit (border_fill_anchors) holds the border's nodes too and takes every term: capped there as well, the S26's
// ai: and an S25 ultrawide's lenses read 17% fewer blocks of the 64 ring filling a 1:1 crop and 2:1 halves than the build
// ai: before, uncapped 10 and 3.5% fewer (research/build/anchors-ring-2026-11-07).
_Static_assert((int)OB_ANCH_MIN >= (int)LENS_P, "the anchors the geometry takes are at least the lens model's terms (internal.h)");
static int lens_cap(int nf) { return nf >= LENS_P ? LENS_TERMS : nf > 8 ? nf - 8 : 0; }
static int solve_lens(double A[LENS_P][LENS_P + 1], int n) {
  for (int c = 0; c < n; c++) {
    int p = c;
    for (int r = c + 1; r < n; r++) if (fabs(A[r][c]) > fabs(A[p][c])) p = r;
    if (fabs(A[p][c]) < 1e-14) return 0;
    for (int k = 0; k <= LENS_P; k++) { double t = A[c][k]; A[c][k] = A[p][k]; A[p][k] = t; }
    for (int r = 0; r < n; r++) {
      if (r == c) continue;
      const double f = A[r][c] / A[c][c];
      for (int k = c; k <= LENS_P; k++) A[r][k] -= f * A[c][k];
    }
  }
  return 1;
}
// ai: a node's rows of the model at q = (x, y): x, y, the shift, the two perspective terms, then the K radial terms
static void lens_rows(double x, double y, int K, double *rx, double *ry) {
  const double ex[8] = { x, y, 0, 0, 1, 0, -x * x, -x * y }, ey[8] = { 0, 0, x, y, 0, 1, -x * y, -y * y };
  for (int a = 0; a < 8; a++) { rx[a] = ex[a]; ry[a] = ey[a]; }
  const double s = x * x + y * y;
  double sk = s;
  for (int k = 0; k < K; k++, sk *= s) { rx[8 + k] = sk * x; ry[8 + k] = sk * y; }
}
// ---------------------------------------------------------------- the anchors

// ai: The lens model over points: q a point's place about the axis over R, d its displacement from H in px; K = 0 to
// ai: LENS_TERMS radial terms by AICc, with no cap inside a 1:1 crop (the anchors sample the interior's radii). Robust (isb given): a border point further from the model than the larger of tau_floor
// ai: and three robust spreads (1.4826 MAD of the border's residuals) is dropped and the model refitted, three rounds.
struct lensm { int K; double t[LENS_P], ax, ay, R; };
typedef struct lensm lensm_t;
static void lensm_at(const struct lensm *m, double u, double v, float *dx, float *dy) {
  double rx[LENS_P], ry[LENS_P], fx = 0, fy = 0;
  lens_rows((u - m->ax) / m->R, (v - m->ay) / m->R, m->K, rx, ry);
  for (int a = 0; a < 8 + m->K; a++) { fx += m->t[a] * rx[a]; fy += m->t[a] * ry[a]; }
  *dx = (float)fx; *dy = (float)fy;
}
// ai: the axis (ob_axis, else the image's centre) and R, its distance to the farthest image corner
static void lensm_axis(int iw, int ih, lensm_t *m) {
  float x = ob_axis[0], y = ob_axis[1];
  if (!(x == x) || !(y == y)) { x = iw * 0.5f; y = ih * 0.5f; }
  m->ax = x; m->ay = y; m->R = 0;
  for (int c = 0; c < 4; c++) { const double d = hypot((c & 1 ? iw : 0) - (double)x, (c & 2 ? ih : 0) - (double)y); if (d > m->R) m->R = d; }
  if (!(m->R > 0)) m->R = 1;
}
static int lensm_fit(int np, const double *qx, const double *qy, const double *dx, const double *dy, const unsigned char *isb, unsigned char *drop, double tau_floor, int kmax, lensm_t *m) {
  double *res = isb ? malloc((size_t)np * sizeof(double)) : 0;
  int ok = 0;
  for (int round = 0; round < 4; round++) {
    double best = INFINITY, tb[LENS_P] = { 0 };
    int K = -1;
    // ai: The normal equations summed once, at every term: a model of fewer terms takes their leading block (lens_rows'
    // ai: first 8 + k entries are the same whatever follows, and an entry sums its products in the points' order), and
    // ai: an entry below the diagonal is its transpose's sum (the same products), so the upper triangle is summed and
    // ai: copied down. Each k's equations are the floats a sum of its own made; the block past p is never read (solve_lens
    // ai: works rows below p and each column on its own).
    double A3[LENS_P][LENS_P + 1], rx[LENS_P], ry[LENS_P];
    memset(A3, 0, sizeof A3);
    int n = 0;
    for (int i = 0; i < np; i++) {
      if (drop[i]) continue;
      n++;
      lens_rows(qx[i], qy[i], LENS_TERMS, rx, ry);
      for (int a = 0; a < LENS_P; a++) { for (int b = a; b < LENS_P; b++) A3[a][b] += rx[a] * rx[b] + ry[a] * ry[b]; A3[a][LENS_P] += rx[a] * dx[i] + ry[a] * dy[i]; }
    }
    for (int a = 1; a < LENS_P; a++) for (int b = 0; b < a; b++) A3[a][b] = A3[b][a];
    for (int k = 0; k <= kmax; k++) {
      const int p = 8 + k;
      double A[LENS_P][LENS_P + 1];
      memcpy(A, A3, sizeof A);
      for (int a = 0; a < p; a++) A[a][a] += 1e-12;
      if (2 * n - p - 1 <= 0 || !solve_lens(A, p)) continue;
      double t[LENS_P] = { 0 }, rss = 0;
      for (int a = 0; a < p; a++) t[a] = A[a][LENS_P] / A[a][a];
      for (int i = 0; i < np; i++) {
        if (drop[i]) continue;
        lens_rows(qx[i], qy[i], k, rx, ry);
        double fx = 0, fy = 0;
        for (int a = 0; a < p; a++) { fx += t[a] * rx[a]; fy += t[a] * ry[a]; }
        rss += (dx[i] - fx) * (dx[i] - fx) + (dy[i] - fy) * (dy[i] - fy);
      }
      const double nn = 2.0 * n, aicc = nn * log(rss / nn + 1e-30) + 2.0 * p + 2.0 * p * (p + 1) / (nn - p - 1);
      if (aicc < best) { best = aicc; K = k; memcpy(tb, t, sizeof tb); }
    }
    if (K < 0) break;
    m->K = K; memcpy(m->t, tb, sizeof tb); ok = 1;
    if (!isb || round == 3) break;
    int nb = 0, changed = 0;
    for (int i = 0; i < np; i++) {
      if (!isb[i]) continue;
      float fx, fy;
      lensm_at(m, qx[i] * m->R + m->ax, qy[i] * m->R + m->ay, &fx, &fy);
      res[nb++] = hypot(dx[i] - fx, dy[i] - fy);
    }
    if (!nb) break;
    double *srt = malloc((size_t)nb * sizeof(double));
    memcpy(srt, res, (size_t)nb * sizeof(double));
    const double tau = fmax(tau_floor, 3 * 1.4826 * ob_select(srt, nb, nb / 2));
    free(srt);
    for (int i = 0, b = 0; i < np; i++) {
      if (!isb[i]) continue;
      const unsigned char bad = res[b++] > tau;
      if (bad != drop[i]) { drop[i] = bad; changed++; }
    }
    if (!changed) break;
  }
  free(res);
  return ok;
}

// ai: where module point (X, Y) is in the image: through H, and the lens model about it when given
typedef struct { float X, Y, dx, dy, peak, u, v; int ok; } anchor_t;
static void anchor_map(const homo_t *H, const lensm_t *lm, float X, float Y, float *u, float *v) {
  project(H, X, Y, u, v);
  if (lm) { float dx, dy; lensm_at(lm, *u, *v, &dx, &dy); *u += dx; *v += dy; }
}

// ai: Every anchor not yet ok found as anchor_map places it: a window resampled on a grid of a / OB_ANCH_TS modules
// ai: about its place, the pattern's normalised correlation at every whole step within ANCH_REACH modules, a parabola each
// ai: way. An anchor's dx, dy is where its middle really is less where H puts it (px), peak its correlation, and it is
// ai: ok where the peak reaches ANCH_PEAK inside the window (a peak on the window's edge may stand short of one beyond
// ai: it). ANCH_PEAK from 771 captures of the published lenses searched three modules wide through H alone
// ai: (research/build/anchors/PLAN.md): every right find (35,293) peaked at 0.669 or more, 0.774 at its 0.1%; the
// ai: wrong ones (1,498, more than a module off) at 0.404 in the median, 0.584 at their 99%; 586 of 792 edge peaks
// ai: wrong. The count of anchors ok. (The pattern inverted, a dark middle and rim, took 20 wrong finds over 0.6 and
// ai: read the anchors four times less precisely in a 2:1 half with no lens, 0.060 modules against 0.014.)
// ai: ANCH_CORNER: how far, in modules, the finder's corner may sit from where the anchors' map puts it before it is
// ai: searched again (the corners the finder lost on the published lenses sat 4 to 63 modules off, the ones it held
// ai: within 1.2; the anchors' map, extrapolated past the last anchor to a corner, is a module or two off on a strong lens)
// ai: ANCH_KNOT_ROUNDS: the rounds a missing anchor's mesh knot is filled in from its found neighbours (border_fill_anchors)
enum { ANCH_ROUNDS = 6, ANCH_CORNER = 3, ANCH_REACH = 3, ANCH_KNOT_ROUNDS = 6 };
static const float ANCH_PEAK = 0.6f;
#ifdef OB_NEON
// ai: lensm_at's displacement added to four points (u, v), in floats: with x, y the point about the axis over R and s
// ai: their squared radius, lens_rows' sum regrouped as dx = t4 + t1 y + (t0 + c) x, dy = t5 + t2 x + (t3 + c) y, c =
// ai: s (t8 + s (t9 + s t10)) (Horner, K terms) - t6 x - t7 y, fused. x and y lie in [-1, 1] and the terms are pixels,
// ai: so the floats keep it within 1e-5 px of the doubles (lensm_at, the statement); the coefficients are made floats
// ai: once a window (msh_lens).
typedef struct { float a, b, iR, t[8], tk[LENS_TERMS]; int K; } msh_lens_t;   // a, b: -ax / R, -ay / R
static inline msh_lens_t msh_lens(const struct lensm *m) {
  msh_lens_t f = { (float)(-m->ax / m->R), (float)(-m->ay / m->R), (float)(1 / m->R), { 0 }, { 0 }, m->K };
  for (int a = 0; a < 8; a++) f.t[a] = (float)m->t[a];
  for (int k = 0; k < m->K; k++) f.tk[k] = (float)m->t[8 + k];
  return f;
}
static inline __attribute__((always_inline)) void msh_lens4(const msh_lens_t *f, float32x4_t *u, float32x4_t *v) {
  const float32x4_t x = vfmaq_n_f32(vdupq_n_f32(f->a), *u, f->iR), y = vfmaq_n_f32(vdupq_n_f32(f->b), *v, f->iR);
  const float32x4_t s = vfmaq_f32(vmulq_f32(x, x), y, y);
  float32x4_t p = vdupq_n_f32(0);
  for (int k = f->K - 1; k >= 0; k--) p = vmulq_f32(vaddq_f32(p, vdupq_n_f32(f->tk[k])), s);
  const float32x4_t c = vfmsq_n_f32(vfmsq_n_f32(p, x, f->t[6]), y, f->t[7]);
  *u = vaddq_f32(*u, vfmaq_f32(vfmaq_n_f32(vdupq_n_f32(f->t[4]), y, f->t[1]), vaddq_f32(c, vdupq_n_f32(f->t[0])), x));
  *v = vaddq_f32(*v, vfmaq_f32(vfmaq_n_f32(vdupq_n_f32(f->t[5]), x, f->t[2]), vaddq_f32(c, vdupq_n_f32(f->t[3])), y));
}
// ai: anchor_find's window, eight points a read (W is 28, 40 or 64: whole reads, then a half), each placed as
// ai: anchor_map places it, read as 12-bit greys (msh_read8q)
static void msh_anchor_window(const image_t *im, const homo_t *H, const struct lensm *lm, float x0, float y0, float h, int r, int W, uint16_t *win) {
  const msh_k_t K = msh_k(im);
  const msh_lens_t lf = lm ? msh_lens(lm) : (msh_lens_t){ 0 };
  const float *hh = H->h;
  const float32x4_t lane = { 0, 1, 2, 3 }, half = vdupq_n_f32(0.5f), vh = vdupq_n_f32(h), vx0 = vdupq_n_f32(x0), ox = vdupq_n_f32(MSH_OX), oy = vdupq_n_f32(MSH_OY), z = vdupq_n_f32(0);
#define MSH_WPT(wx_, u_, v_) do { const float32x4_t X_ = vaddq_f32(vx0, vmulq_f32(vaddq_f32(vaddq_f32(vdupq_n_f32((float)((wx_) - r)), lane), half), vh)); \
                                  msh_proj(vfmaq_n_f32(ru, X_, hh[0]), vfmaq_n_f32(rv, X_, hh[3]), vfmaq_n_f32(rw, X_, hh[6]), z, z, &u_, &v_); \
                                  if (lm) msh_lens4(&lf, &u_, &v_); u_ = vsubq_f32(u_, ox); v_ = vsubq_f32(v_, oy); } while (0)
  for (int wy = 0; wy < W; wy++) {
    const float Y = y0 + (wy - r + 0.5f) * h;
    const float32x4_t ru = vdupq_n_f32(hh[1] * Y + hh[2]), rv = vdupq_n_f32(hh[4] * Y + hh[5]), rw = vdupq_n_f32(hh[7] * Y + 1);
    int wx = 0;
    for (; wx + 8 <= W; wx += 8) {
      float32x4_t u0, v0, u1, v1;
      MSH_WPT(wx, u0, v0); MSH_WPT(wx + 4, u1, v1);
      vst1q_u16(win + wy * W + wx, msh_read8q(im, &K, u0, v0, u1, v1));
    }
    if (wx + 4 <= W) {
      float32x4_t u0, v0;
      MSH_WPT(wx, u0, v0);
      vst1_u16(win + wy * W + wx, vget_low_u16(msh_read8q(im, &K, u0, v0, u0, v0)));
      wx += 4;
    }
    for (; wx < W; wx++) {
      float u, v;
      anchor_map(H, lm, x0 + (wx - r + 0.5f) * h, y0 + (wy - r + 0.5f) * h, &u, &v);
      win[wy * W + wx] = (uint16_t)lrintf(sample(im, u, v) * MSH_Q);
    }
  }
#undef MSH_WPT
}
// ai: The window's integral images in 32-bit integers, of the greys and of their squares: I[(y + 1) W1 + x + 1] the
// ai: sum over rows to y and columns to x. Sums of integers in any order are the same sums, so a row's prefix runs four
// ai: lanes a vector (two shifted adds, the last lane carried on) over the running column sums. The squares' image wraps
// ai: past 2^32 on a large window, but every box anchor_find takes is at most OB_ANCH_TS^2 greys of at most MSH_Q (256 x
// ai: 4080^2 < 2^32), so its box sums, taken modulo 2^32, are exact.
_Static_assert((unsigned long long)OB_ANCH_TS * OB_ANCH_TS * MSH_Q * MSH_Q < (1ull << 32), "the anchors' box sums of squared greys fit 32 bits");
static void msh_integral(const uint16_t *win, int W, uint32_t *I1, uint32_t *I2) {
  const int W1 = W + 1;
  uint32_t c1[64], c2[64];   // ai: the column sums down to the current row (W <= 64, a multiple of 4)
  for (int x = 0; x < W; x++) c1[x] = c2[x] = 0;
  for (int x = 0; x < W1; x++) I1[x] = I2[x] = 0;
  const uint32x4_t z = vdupq_n_u32(0);
  for (int y = 0; y < W; y++) {
    uint32_t *i1 = I1 + (y + 1) * W1, *i2 = I2 + (y + 1) * W1;
    i1[0] = i2[0] = 0;
    uint32x4_t k1 = z, k2 = z;
    for (int x = 0; x < W; x += 4) {
      const uint16x4_t q = vld1_u16(win + y * W + x);
      uint32x4_t a = vaddw_u16(vld1q_u32(c1 + x), q), b = vmlal_u16(vld1q_u32(c2 + x), q, q);
      vst1q_u32(c1 + x, a); vst1q_u32(c2 + x, b);
      a = vaddq_u32(a, vextq_u32(z, a, 3)); a = vaddq_u32(a, vextq_u32(z, a, 2)); a = vaddq_u32(a, k1);
      b = vaddq_u32(b, vextq_u32(z, b, 3)); b = vaddq_u32(b, vextq_u32(z, b, 2)); b = vaddq_u32(b, k2);
      vst1q_u32(i1 + x + 1, a); vst1q_u32(i2 + x + 1, b);
      k1 = vdupq_laneq_u32(a, 3); k2 = vdupq_laneq_u32(b, 3);
    }
  }
  for (int k = 0; k < 4; k++) I1[W1 * W1 + k] = I2[W1 * W1 + k] = 0;   // ai: the spare entries the last vector of offsets reads
}
#elif defined(OB_X86)
// ai: x86-64's anchor_find forms, arm64's above in SSE4.1: msh_lens4 (lensm_at's displacement in floats, the same
// ai: regrouping, no fused multiply-adds), msh_anchor_window (eight points a read as 12-bit greys), msh_integral (the
// ai: same integer sums: the squares by PMULLD, a row's prefix by two shifted adds and the last lane carried on).
typedef struct { float a, b, iR, t[8], tk[LENS_TERMS]; int K; } msh_lens_t;   // a, b: -ax / R, -ay / R
static inline msh_lens_t msh_lens(const struct lensm *m) {
  msh_lens_t f = { (float)(-m->ax / m->R), (float)(-m->ay / m->R), (float)(1 / m->R), { 0 }, { 0 }, m->K };
  for (int a = 0; a < 8; a++) f.t[a] = (float)m->t[a];
  for (int k = 0; k < m->K; k++) f.tk[k] = (float)m->t[8 + k];
  return f;
}
static inline __attribute__((always_inline)) void msh_lens4(const msh_lens_t *f, __m128 *u, __m128 *v) {
  const __m128 iR = _mm_set1_ps(f->iR), x = MSH_MA(_mm_set1_ps(f->a), *u, iR), y = MSH_MA(_mm_set1_ps(f->b), *v, iR);
  const __m128 s = MSH_MA(_mm_mul_ps(x, x), y, y);
  __m128 p = _mm_setzero_ps();
  for (int k = f->K - 1; k >= 0; k--) p = _mm_mul_ps(_mm_add_ps(p, _mm_set1_ps(f->tk[k])), s);
  const __m128 c = _mm_sub_ps(_mm_sub_ps(p, _mm_mul_ps(x, _mm_set1_ps(f->t[6]))), _mm_mul_ps(y, _mm_set1_ps(f->t[7])));
  *u = _mm_add_ps(*u, MSH_MA(MSH_MA(_mm_set1_ps(f->t[4]), y, _mm_set1_ps(f->t[1])), _mm_add_ps(c, _mm_set1_ps(f->t[0])), x));
  *v = _mm_add_ps(*v, MSH_MA(MSH_MA(_mm_set1_ps(f->t[5]), x, _mm_set1_ps(f->t[2])), _mm_add_ps(c, _mm_set1_ps(f->t[3])), y));
}
static void msh_anchor_window(const image_t *im, const homo_t *H, const struct lensm *lm, float x0, float y0, float h, int r, int W, uint16_t *win) {
  const msh_k_t K = msh_k(im);
  const msh_lens_t lf = lm ? msh_lens(lm) : (msh_lens_t){ 0 };
  const float *hh = H->h;
  const __m128 lane = _mm_setr_ps(0, 1, 2, 3), half = _mm_set1_ps(0.5f), vh = _mm_set1_ps(h), vx0 = _mm_set1_ps(x0), ox = _mm_set1_ps(MSH_OX), oy = _mm_set1_ps(MSH_OY), z = _mm_setzero_ps();
  const __m128 h0 = _mm_set1_ps(hh[0]), h3 = _mm_set1_ps(hh[3]), h6 = _mm_set1_ps(hh[6]);
#define MSH_WPT(wx_, u_, v_) do { const __m128 X_ = _mm_add_ps(vx0, _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_set1_ps((float)((wx_) - r)), lane), half), vh)); \
                                  msh_proj(MSH_MA(ru, X_, h0), MSH_MA(rv, X_, h3), MSH_MA(rw, X_, h6), z, z, &u_, &v_); \
                                  if (lm) { msh_lens4(&lf, &u_, &v_); } u_ = _mm_sub_ps(u_, ox); v_ = _mm_sub_ps(v_, oy); } while (0)
  for (int wy = 0; wy < W; wy++) {
    const float Y = y0 + (wy - r + 0.5f) * h;
    const __m128 ru = _mm_set1_ps(hh[1] * Y + hh[2]), rv = _mm_set1_ps(hh[4] * Y + hh[5]), rw = _mm_set1_ps(hh[7] * Y + 1);
    int wx = 0;
    for (; wx + 8 <= W; wx += 8) {
      __m128 u0, v0, u1, v1;
      MSH_WPT(wx, u0, v0); MSH_WPT(wx + 4, u1, v1);
      _mm_storeu_si128((__m128i *)(win + wy * W + wx), msh_read8q(im, &K, u0, v0, u1, v1));
    }
    if (wx + 4 <= W) {
      __m128 u0, v0;
      MSH_WPT(wx, u0, v0);
      _mm_storel_epi64((__m128i *)(win + wy * W + wx), msh_read8q(im, &K, u0, v0, u0, v0));
      wx += 4;
    }
    for (; wx < W; wx++) {
      float u, v;
      anchor_map(H, lm, x0 + (wx - r + 0.5f) * h, y0 + (wy - r + 0.5f) * h, &u, &v);
      win[wy * W + wx] = (uint16_t)lrintf(sample(im, u, v) * MSH_Q);
    }
  }
#undef MSH_WPT
}
_Static_assert((unsigned long long)OB_ANCH_TS * OB_ANCH_TS * MSH_Q * MSH_Q < (1ull << 32), "the anchors' box sums of squared greys fit 32 bits");
static void msh_integral(const uint16_t *win, int W, uint32_t *I1, uint32_t *I2) {
  const int W1 = W + 1;
  uint32_t c1[64], c2[64];   // ai: the column sums down to the current row (W <= 64, a multiple of 4)
  for (int x = 0; x < W; x++) c1[x] = c2[x] = 0;
  for (int x = 0; x < W1; x++) I1[x] = I2[x] = 0;
  for (int y = 0; y < W; y++) {
    uint32_t *i1 = I1 + (y + 1) * W1, *i2 = I2 + (y + 1) * W1;
    i1[0] = i2[0] = 0;
    __m128i k1 = _mm_setzero_si128(), k2 = k1;
    for (int x = 0; x < W; x += 4) {
      const __m128i q = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(win + y * W + x)));
      __m128i a = _mm_add_epi32(_mm_loadu_si128((const __m128i *)(c1 + x)), q), b = _mm_add_epi32(_mm_loadu_si128((const __m128i *)(c2 + x)), _mm_mullo_epi32(q, q));
      _mm_storeu_si128((__m128i *)(c1 + x), a); _mm_storeu_si128((__m128i *)(c2 + x), b);
      a = _mm_add_epi32(a, _mm_slli_si128(a, 4)); a = _mm_add_epi32(a, _mm_slli_si128(a, 8)); a = _mm_add_epi32(a, k1);
      b = _mm_add_epi32(b, _mm_slli_si128(b, 4)); b = _mm_add_epi32(b, _mm_slli_si128(b, 8)); b = _mm_add_epi32(b, k2);
      _mm_storeu_si128((__m128i *)(i1 + x + 1), a); _mm_storeu_si128((__m128i *)(i2 + x + 1), b);
      k1 = _mm_shuffle_epi32(a, 0xff); k2 = _mm_shuffle_epi32(b, 0xff);
    }
  }
  for (int k = 0; k < 4; k++) I1[W1 * W1 + k] = I2[W1 * W1 + k] = 0;   // ai: the spare entries the last vector of offsets reads
}
#endif
static int anchor_find(const ob_layout_t *L, const image_t *im, const homo_t *H, const lensm_t *lm, anchor_t *A) {
  const int a = ob_anchor_side(L), TS = OB_ANCH_TS;
  if (a < 1) return 0;
  const float h = (float)a / TS;
  const int r = (int)ceilf((float)ANCH_REACH / h), W = TS + 2 * r, O = 2 * r + 1;
  float *C = malloc((size_t)(O * O + 4) * sizeof(float)), T[OB_ANCH_TS * OB_ANCH_TS];   // ai: + 4: arm64's last row of offsets four a vector
#if defined(OB_NEON) || defined(OB_X86)
  // ai: arm64's window in 12-bit greys and its integral images in integers (msh_integral; + 4: the last vector of
  // ai: offsets reads three entries past a row's end)
  uint16_t *win = malloc((size_t)W * W * sizeof(uint16_t));
  uint32_t *I1 = malloc(((size_t)(W + 1) * (W + 1) + 4) * sizeof(uint32_t)), *I2 = malloc(((size_t)(W + 1) * (W + 1) + 4) * sizeof(uint32_t));
#else
  float *win = malloc((size_t)W * W * sizeof(float));
  double *I1 = malloc((size_t)(W + 1) * (W + 1) * sizeof(double)), *I2 = malloc((size_t)(W + 1) * (W + 1) * sizeof(double));
#endif
  float tm = 0, tn = 0;
  for (int y = 0; y < TS; y++) for (int x = 0; x < TS; x++) {
    const float u = (x + 0.5f) * h - a * 0.5f, v = (y + 0.5f) * h - a * 0.5f, dc = fmaxf(fabsf(u), fabsf(v)) / (a * 0.5f);
    T[y * TS + x] = dc < 0.25f || dc >= 0.6f ? 1.0f : -1.0f;
    tm += T[y * TS + x];
  }
  tm /= (float)(TS * TS);
  for (int k = 0; k < TS * TS; k++) { T[k] -= tm; tn += T[k] * T[k]; }
  tn = sqrtf(tn);
  // ai: The pattern is two levels in concentric squares (dc is the larger of the two axes' distances, so dc under 0.25
  // ai: or 0.6 is a square, [i25, j25] or [i6, j6] along each axis), and the window's sum against it at an offset is
  // ai: 2 (S25 - S6) + S (1 - tm), S the sum over the template's box and S25, S6 over the squares': box sums of the
  // ai: window's integral images (in double, its squares' too; on arm64 in integers of its 12-bit greys, exact:
  // ai: msh_integral), a few reads an offset where the whole template was summed (16 x 16 a step; at three modules 169
  // ai: steps an anchor in the 256 ring, 289 in the 128, 625 in the 64; 625 and 2,401 in the 128 and 64 before 2026-11-09)
  int i25 = TS, j25 = -1, i6 = TS, j6 = -1;
  for (int x = 0; x < TS; x++) {
    const float d = fabsf((x + 0.5f) * h - a * 0.5f) / (a * 0.5f);
    if (d < 0.25f) { if (x < i25) i25 = x; j25 = x; }
    if (d < 0.6f) { if (x < i6) i6 = x; j6 = x; }
  }
  const int W1 = W + 1;
#define ABOX(I, x, y, s) ((I)[((y) + (s)) * W1 + (x) + (s)] - (I)[(y) * W1 + (x) + (s)] - (I)[((y) + (s)) * W1 + (x)] + (I)[(y) * W1 + (x)])
  int found = 0;
  const int K = ob_anchor_k(L);   // ai: the ring's grid; an anchor's index j OB_ANCH_K + i in every ring
  for (int j = 0; j < K; j++) for (int i = 0; i < K; i++) {
    anchor_t *an = &A[j * OB_ANCH_K + i];
    if (an->ok) { found++; continue; }
    const float x0 = (float)ob_anchor_m0(L, i), y0 = (float)ob_anchor_m0(L, j);
    an->X = x0 + a * 0.5f; an->Y = y0 + a * 0.5f; an->peak = -1; an->dx = an->dy = 0;
#ifdef OB_NEON
    msh_anchor_window(im, H, lm, x0, y0, h, r, W, win);
    msh_integral(win, W, I1, I2);
    float best = -2;
    int bx = 0, by = 0;
    {
      // ai: The correlations four offsets a vector from the integer box sums (exact): the template's sum against the
      // ai: window st = 2 (S25 - S6) + S (1 - tm) in floats from them; the window's variance n times over, n S2 - S^2
      // ai: (n = TS^2), exact in 64-bit lanes, so the correlation st / (sqrt(var) tn) is st / (sqrt(n S2 - S^2) tn / TS);
      // ai: a flat box (n S2 = S^2) reads 0, as the scalar's var <= 1e-12 does. A row's last vector runs past O into the
      // ai: next row's first offsets, which that row writes again (the last row's into C's four spare floats).
      const int l25 = j25 - i25 + 1, l6 = j6 - i6 + 1;
      const float32x4_t ktm = vdupq_n_f32(1 - tm), ktn = vdupq_n_f32(tn / (float)TS), z = vdupq_n_f32(0);
#define QBOX(I, x, y, s) vaddq_u32(vsubq_u32(vsubq_u32(vld1q_u32((I) + ((y) + (s)) * W1 + (x) + (s)), vld1q_u32((I) + (y) * W1 + (x) + (s))), vld1q_u32((I) + ((y) + (s)) * W1 + (x))), vld1q_u32((I) + (y) * W1 + (x)))
      for (int oy = -r; oy <= r; oy++) {
        const int py = r + oy;
        for (int ox = -r; ox <= r; ox += 4) {
          const int px = r + ox;
          const uint32x4_t s1 = QBOX(I1, px, py, TS), s2 = QBOX(I2, px, py, TS);
          const uint32x4_t s25 = j25 >= i25 ? QBOX(I1, px + i25, py + i25, l25) : vdupq_n_u32(0), s6 = j6 >= i6 ? QBOX(I1, px + i6, py + i6, l6) : vdupq_n_u32(0);
          const float32x4_t st = vfmaq_f32(vmulq_n_f32(vcvtq_f32_s32(vreinterpretq_s32_u32(vsubq_u32(s25, s6))), 2), vcvtq_f32_u32(s1), ktm);
          const uint64x2_t vl = vsubq_u64(vmull_n_u32(vget_low_u32(s2), TS * TS), vmull_u32(vget_low_u32(s1), vget_low_u32(s1)));
          const uint64x2_t vh = vsubq_u64(vmull_high_n_u32(s2, TS * TS), vmull_high_u32(s1, s1));
          const float32x4_t var = vcvt_high_f32_f64(vcvt_f32_f64(vcvtq_f64_u64(vl)), vcvtq_f64_u64(vh));
          const float32x4_t c = vdivq_f32(st, vmulq_f32(vsqrtq_f32(var), ktn));
          vst1q_f32(C + (oy + r) * O + ox + r, vbslq_f32(vcgtq_f32(var, z), c, z));
        }
      }
#undef QBOX
    }
    for (int oy = -r; oy <= r; oy++) for (int ox = -r; ox <= r; ox++) {
      const float c = C[(oy + r) * O + ox + r];
      if (c > best) { best = c; bx = ox; by = oy; }
    }
#elif defined(OB_X86)
    msh_anchor_window(im, H, lm, x0, y0, h, r, W, win);
    msh_integral(win, W, I1, I2);
    float best = -2;
    int bx = 0, by = 0;
    {
      // ai: arm64's correlations above in SSE4.1: the box sums in 32-bit integers; n S2 - S^2 in 64-bit lanes by
      // ai: PMULUDQ on the even and the odd lanes (exact, under 2^41), to doubles by the 2^52 bias (exact under 2^52)
      // ai: and to floats as FCVTN rounds, so the same variance from the same window; st without the fused add
      const int l25 = j25 - i25 + 1, l6 = j6 - i6 + 1;
      const __m128 ktm = _mm_set1_ps(1 - tm), ktn = _mm_set1_ps(tn / (float)TS), two = _mm_set1_ps(2), z = _mm_setzero_ps();
      const __m128i n2 = _mm_set1_epi32(TS * TS), bias = _mm_set1_epi64x(0x4330000000000000ll);
      const __m128d b52 = _mm_set1_pd(4503599627370496.0);
#define QLD(I, o) _mm_loadu_si128((const __m128i *)((I) + (o)))
#define QBOX(I, x, y, s) _mm_add_epi32(_mm_sub_epi32(_mm_sub_epi32(QLD(I, ((y) + (s)) * W1 + (x) + (s)), QLD(I, (y) * W1 + (x) + (s))), QLD(I, ((y) + (s)) * W1 + (x))), QLD(I, (y) * W1 + (x)))
#define QVAR(e) _mm_sub_pd(_mm_castsi128_pd(_mm_or_si128((e), bias)), b52)
      for (int oy = -r; oy <= r; oy++) {
        const int py = r + oy;
        for (int ox = -r; ox <= r; ox += 4) {
          const int px = r + ox;
          const __m128i s1 = QBOX(I1, px, py, TS), s2 = QBOX(I2, px, py, TS);
          const __m128i s25 = j25 >= i25 ? QBOX(I1, px + i25, py + i25, l25) : _mm_setzero_si128(), s6 = j6 >= i6 ? QBOX(I1, px + i6, py + i6, l6) : _mm_setzero_si128();
          const __m128 st = MSH_MA(_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(s25, s6)), two), _mm_cvtepi32_ps(s1), ktm);
          const __m128i s1o = _mm_srli_epi64(s1, 32);
          const __m128i ve = _mm_sub_epi64(_mm_mul_epu32(s2, n2), _mm_mul_epu32(s1, s1)), vo = _mm_sub_epi64(_mm_mul_epu32(_mm_srli_epi64(s2, 32), n2), _mm_mul_epu32(s1o, s1o));
          const __m128 var = _mm_unpacklo_ps(_mm_cvtpd_ps(QVAR(ve)), _mm_cvtpd_ps(QVAR(vo)));
          const __m128 c = _mm_div_ps(st, _mm_mul_ps(_mm_sqrt_ps(var), ktn));
          _mm_storeu_ps(C + (oy + r) * O + ox + r, _mm_and_ps(_mm_cmpgt_ps(var, z), c));
        }
      }
#undef QVAR
#undef QBOX
#undef QLD
    }
    for (int oy = -r; oy <= r; oy++) for (int ox = -r; ox <= r; ox++) {
      const float c = C[(oy + r) * O + ox + r];
      if (c > best) { best = c; bx = ox; by = oy; }
    }
#else
    for (int wy = 0; wy < W; wy++) for (int wx = 0; wx < W; wx++) {
      float u, v;
      anchor_map(H, lm, x0 + (wx - r + 0.5f) * h, y0 + (wy - r + 0.5f) * h, &u, &v);
      win[wy * W + wx] = sample(im, u, v);
    }
    for (int x = 0; x < W1; x++) I1[x] = I2[x] = 0;
    for (int wy = 0; wy < W; wy++) {
      double s = 0, q = 0;
      I1[(wy + 1) * W1] = I2[(wy + 1) * W1] = 0;
      for (int wx = 0; wx < W; wx++) {
        const double v = win[wy * W + wx];
        s += v; q += v * v;
        I1[(wy + 1) * W1 + wx + 1] = I1[wy * W1 + wx + 1] + s; I2[(wy + 1) * W1 + wx + 1] = I2[wy * W1 + wx + 1] + q;
      }
    }
    float best = -2;
    int bx = 0, by = 0;
    for (int oy = -r; oy <= r; oy++) for (int ox = -r; ox <= r; ox++) {
      const int px = r + ox, py = r + oy;
      const double s1 = ABOX(I1, px, py, TS), s2 = ABOX(I2, px, py, TS), var = s2 - s1 * s1 / (TS * TS);
      const double s25 = j25 >= i25 ? ABOX(I1, px + i25, py + i25, j25 - i25 + 1) : 0, s6 = j6 >= i6 ? ABOX(I1, px + i6, py + i6, j6 - i6 + 1) : 0;
      const double st = 2 * (s25 - s6) + s1 * (1 - (double)tm);
      const float c = var > 1e-12 ? (float)(st / (sqrt(var) * tn)) : 0;
      C[(oy + r) * O + ox + r] = c;
      if (c > best) { best = c; bx = ox; by = oy; }
    }
#endif
    float sx = 0, sy = 0;
    if (bx > -r && bx < r) { const float l = C[(by + r) * O + bx + r - 1], c = best, rr = C[(by + r) * O + bx + r + 1], d = l - 2 * c + rr; if (d < 0) sx = 0.5f * (l - rr) / d; }
    if (by > -r && by < r) { const float l = C[(by + r - 1) * O + bx + r], c = best, rr = C[(by + r + 1) * O + bx + r], d = l - 2 * c + rr; if (d < 0) sy = 0.5f * (l - rr) / d; }
    an->peak = best;
    float u, v, hu, hv;
    anchor_map(H, lm, an->X + (bx + sx) * h, an->Y + (by + sy) * h, &u, &v);
    project(H, an->X, an->Y, &hu, &hv);
    an->dx = u - hu; an->dy = v - hv; an->u = u; an->v = v;
    an->ok = best >= ANCH_PEAK && bx > -r && bx < r && by > -r && by < r;
    found += an->ok;
  }
  free(win); free(C); free(I1); free(I2);
#undef ABOX
  return found;
}

// ai: the lens model the anchors and the four corners say (the corners' displacement 0: H passes through them), for
// ai: the border's search to start from; 0 where it does not fit
static int anchor_model(const ob_layout_t *L, const homo_t *H, int iw, int ih, const anchor_t *A, lensm_t *m) {
  enum { NA = OB_ANCH_K * OB_ANCH_K };
  double qx[NA + 4], qy[NA + 4], dx[NA + 4], dy[NA + 4];
  unsigned char drop[NA + 4] = { 0 };
  int np = 0;
  lensm_axis(iw, ih, m);
  for (int c = 0; c < 4; c++) {
    float u, v;
    project(H, L->node_x[(c == 1 || c == 2) ? L->nx - 1 : 0], L->node_y[c >= 2 ? L->ny - 1 : 0], &u, &v);
    qx[np] = (u - m->ax) / m->R; qy[np] = (v - m->ay) / m->R; dx[np] = 0; dy[np] = 0; np++;
  }
  for (int k = 0; k < NA; k++) {
    if (!A[k].ok) continue;
    float u, v;
    project(H, A[k].X, A[k].Y, &u, &v);
    qx[np] = (u - m->ax) / m->R; qy[np] = (v - m->ay) / m->R; dx[np] = A[k].dx; dy[np] = A[k].dy; np++;
  }
  return lensm_fit(np, qx, qy, dx, dy, 0, drop, 0, lens_cap(np - 4), m);
}

// ai: A homography through the anchors found (module place to image place), least squares with h33 = 1 on
// ai: coordinates centred and scaled; 0 where they do not hold it (fewer than eight, or within three rows or columns
// ai: of the grid)
static int anchor_homo(const anchor_t *A, homo_t *H) {
  enum { NA = OB_ANCH_K * OB_ANCH_K };
  int n = 0, rows = 0, cols = 0, rm = 0, cm = 0;
  double mx = 0, my = 0, mu = 0, mv = 0;
  for (int k = 0; k < NA; k++) if (A[k].ok) { n++; mx += A[k].X; my += A[k].Y; mu += A[k].u; mv += A[k].v; rm |= 1 << (k / OB_ANCH_K); cm |= 1 << (k % OB_ANCH_K); }
  for (int b = 0; b < OB_ANCH_K; b++) { rows += (rm >> b) & 1; cols += (cm >> b) & 1; }
  if (n < 8 || rows < 3 || cols < 3) return 0;
  mx /= n; my /= n; mu /= n; mv /= n;
  double sx = 0, su = 0;
  for (int k = 0; k < NA; k++) if (A[k].ok) { sx += hypot(A[k].X - mx, A[k].Y - my); su += hypot(A[k].u - mu, A[k].v - mv); }
  sx = sx / n / 1.4142135623730951; su = su / n / 1.4142135623730951;
  if (!(sx > 0) || !(su > 0)) return 0;
  double M[LENS_P][LENS_P + 1];
  memset(M, 0, sizeof M);
  for (int k = 0; k < NA; k++) {
    if (!A[k].ok) continue;
    const double x = (A[k].X - mx) / sx, y = (A[k].Y - my) / sx, u = (A[k].u - mu) / su, v = (A[k].v - mv) / su;
    const double r0[8] = { x, y, 1, 0, 0, 0, -u * x, -u * y }, r1[8] = { 0, 0, 0, x, y, 1, -v * x, -v * y };
    for (int a = 0; a < 8; a++) { for (int b = 0; b < 8; b++) M[a][b] += r0[a] * r0[b] + r1[a] * r1[b]; M[a][LENS_P] += r0[a] * u + r1[a] * v; }
  }
  if (!solve_lens(M, 8)) return 0;
  double g[9];
  for (int a = 0; a < 8; a++) g[a] = M[a][LENS_P] / M[a][a];
  g[8] = 1;
  // ai: undo the scalings: H = Tu^-1 G Tx, Tx = [1/sx 0 -mx/sx; 0 1/sx -my/sx; 0 0 1], Tu^-1 = [su 0 mu; 0 su mv; 0 0 1]
  const double tx[9] = { 1 / sx, 0, -mx / sx, 0, 1 / sx, -my / sx, 0, 0, 1 }, ti[9] = { su, 0, mu, 0, su, mv, 0, 0, 1 };
  double gt[9], h[9];
  for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) { gt[3 * r + c] = 0; for (int k = 0; k < 3; k++) gt[3 * r + c] += g[3 * r + k] * tx[3 * k + c]; }
  for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) { h[3 * r + c] = 0; for (int k = 0; k < 3; k++) h[3 * r + c] += ti[3 * r + k] * gt[3 * k + c]; }
  if (!(fabs(h[8]) > 1e-12)) return 0;
  for (int a = 0; a < 9; a++) H->h[a] = (float)(h[a] / h[8]);
  return 1;
}

// ai: the lens model the anchors alone say, about H: their displacements from it (the corners left out, so a corner
// ai: the finder misplaced does not bend it), held to the border nodes' robust rule (lensm_fit: a reading further from
// ai: the model than half a module, mpx px, or three robust spreads is dropped and the model fitted again), the dropped
// ai: anchors no longer found; 0 where it does not fit
static int anchor_model_free(const homo_t *H, int iw, int ih, anchor_t *A, lensm_t *m, float mpx) {
  enum { NA = OB_ANCH_K * OB_ANCH_K };
  double qx[NA], qy[NA], dx[NA], dy[NA];
  unsigned char drop[NA] = { 0 }, isb[NA];
  int np = 0, at[NA];
  memset(isb, 1, sizeof isb);
  lensm_axis(iw, ih, m);
  for (int k = 0; k < NA; k++) {
    if (!A[k].ok) continue;
    float hu, hv;
    project(H, A[k].X, A[k].Y, &hu, &hv);
    A[k].dx = A[k].u - hu; A[k].dy = A[k].v - hv;
    qx[np] = (hu - m->ax) / m->R; qy[np] = (hv - m->ay) / m->R; dx[np] = A[k].dx; dy[np] = A[k].dy; at[np] = k; np++;
  }
  if (np < 4 || !lensm_fit(np, qx, qy, dx, dy, isb, drop, 0.5 * mpx, LENS_TERMS, m)) return 0;
  for (int i = 0; i < np; i++) if (drop[i]) A[at[i]].ok = 0;
  return 1;
}

// ai: The code registered from the inside out. The anchors searched three modules wide through the corners'
// ai: homography, then, round after round, those still missing through the homography the anchors found say and the
// ai: lens model they say about it (each round's readings held to the robust rule, anchor_model_free) (an anchor further out than the reach comes within it once the model has the anchors
// ai: nearer the middle). Then each corner judged by that map: one further than ANCH_CORNER modules from where it puts
// ai: the corner (its mark cut by the crop, or its sides bent past the line fit) searched again from where the model
// ai: over the anchors and the kept corners puts it, kept where its template reads 0.35 or more, else the model's place,
// ai: and H made again from the four. (Judging the corners by that joint model instead, dropping outliers round by
// ai: round, kept 82.4% of the exact map's blocks on the hard cells against this rule's 85.6%: a misplaced corner bends
// ai: it.) Then the anchors' model about H (anchor_model) for the border's search. 0 where fewer than OB_ANCH_MIN
// ai: anchors were found (a round's homography takes four, its lens model five: lensm_fit's 2 n - 9 > 0).
// ai: seed: the finder's lens as the first round's model (find_frame's, a traced quad's), NULL for none
static int anchor_register(const ob_layout_t *L, const image_t *im, const image_t *rim, homo_t *H, int iw, int ih, float mpx, float *quad, node_t *nodes, anchor_t *A, lensm_t *m, const lensm_t *seed) {
  enum { NA = OB_ANCH_K * OB_ANCH_K };
  homo_t Ha = *H;
  lensm_t mf;
  int have = 0, nf = 0;
  for (int k = 0; k < NA; k++) A[k].ok = 0;
  for (int round = 0; round < ANCH_ROUNDS; round++) {
    const int now = anchor_find(L, im, &Ha, have ? &mf : seed, A);
    if (now == nf || now < 4) break;
    nf = now;
    homo_t Hn;
    if (anchor_homo(A, &Hn)) Ha = Hn;
    if (!(have = anchor_model_free(&Ha, iw, ih, A, &mf, mpx))) break;
    nf = 0;
    for (int k = 0; k < NA; k++) nf += A[k].ok;
  }
  if (!have || nf < ob_anchor_min(L)) return 0;
  const int nx = L->nx, ny = L->ny;
  // ai: the anchors and the corners about the corners' H, the corners' outliers left out round by round
  double qx[NA + 4], qy[NA + 4], dx[NA + 4], dy[NA + 4];
  unsigned char drop[NA + 4] = { 0 };
  int np = 4;
  lensm_t mc;
  lensm_axis(iw, ih, &mc);
  for (int c = 0; c < 4; c++) {
    float u, v;
    project(H, L->node_x[(c == 1 || c == 2) ? nx - 1 : 0], L->node_y[c >= 2 ? ny - 1 : 0], &u, &v);
    qx[c] = (u - mc.ax) / mc.R; qy[c] = (v - mc.ay) / mc.R; dx[c] = dy[c] = 0;
  }
  for (int k = 0; k < NA; k++) {
    if (!A[k].ok) continue;
    float hu, hv;
    project(H, A[k].X, A[k].Y, &hu, &hv);
    qx[np] = (hu - mc.ax) / mc.R; qy[np] = (hv - mc.ay) / mc.R; dx[np] = A[k].u - hu; dy[np] = A[k].v - hv; np++;
  }
  // ai: each corner against the anchors' own map (Ha and mf: no corner holds it, so a misplaced one cannot bend it);
  // ai: the model about H fitted over the anchors and the corners it keeps, for the search of those it does not
  for (int c = 0; c < 4; c++) {
    float pu, pv, mu, mv;
    project(&Ha, L->node_x[(c == 1 || c == 2) ? nx - 1 : 0], L->node_y[c >= 2 ? ny - 1 : 0], &pu, &pv);
    lensm_at(&mf, pu, pv, &mu, &mv);
    drop[c] = hypotf(pu + mu - quad[2 * c], pv + mv - quad[2 * c + 1]) > ANCH_CORNER * mpx;
  }
  const int fit = lensm_fit(np, qx, qy, dx, dy, 0, drop, 0, lens_cap(nf), &mc);
  float src[8];
  corner_coords(L, src);
  int moved = 0;
  for (int c = 0; fit && c < 4; c++) {
    if (!drop[c]) continue;
    const int i = (c == 1 || c == 2) ? nx - 1 : 0, j = c >= 2 ? ny - 1 : 0;
    float hu, hv, mu, mv;
    project(H, L->node_x[i], L->node_y[j], &hu, &hv);
    lensm_at(&mc, hu, hv, &mu, &mv);
    node_t *nd = &nodes[j * nx + i];
    nd->done = 0;
    refine_node(L, rim, H, i, j, nodes, &mc);
    if (nd->score >= 0.35f) { quad[2 * c] = hu + nd->dx; quad[2 * c + 1] = hv + nd->dy; }
    else { quad[2 * c] = hu + mu; quad[2 * c + 1] = hv + mv; }
    moved++;
  }
  homo_t Hc;
  if (moved && homography(src, quad, &Hc)) {
    *H = Hc;
    for (int c = 0; c < 4; c++) { node_t *nd = &nodes[((c >= 2) ? ny - 1 : 0) * nx + ((c == 1 || c == 2) ? nx - 1 : 0)]; nd->dx = nd->dy = 0; nd->done = 1; }
  }
  for (int k = 0; k < NA; k++) if (A[k].ok) { float hu, hv; project(H, A[k].X, A[k].Y, &hu, &hv); A[k].dx = A[k].u - hu; A[k].dy = A[k].v - hv; }
  return anchor_model(L, H, iw, ih, A, m);
}

// ai: A cubic Hermite through knots t[0..n-1] at x, its slopes from the neighbouring knots (one-sided at the ends), as
// ai: weights: the value is sum w[a] v[b + a], a = 0..3, b = the cell's first knot less one (out-of-range weights 0)
static void anch_herm4(const double *t, int n, double x, int *b, double w[4]) {
  int i = 0;
  while (i < n - 2 && x >= t[i + 1]) i++;
  const double hc = t[i + 1] - t[i], s = (x - t[i]) / hc, s2 = s * s, s3 = s2 * s;
  const double h00 = 2 * s3 - 3 * s2 + 1, h01 = -2 * s3 + 3 * s2, A = (s3 - 2 * s2 + s) * hc, Bc = (s3 - s2) * hc;
  w[0] = w[3] = 0; w[1] = h00; w[2] = h01;
  // ai: the slope at knot i: (v[i+1] - v[i-1]) / (t[i+1] - t[i-1]), or (v[i+1] - v[i]) / hc at the first knot
  if (i > 0) { const double d = t[i + 1] - t[i - 1]; w[0] -= A / d; w[2] += A / d; } else { w[1] -= A / hc; w[2] += A / hc; }
  // ai: the slope at knot i + 1: (v[i+2] - v[i]) / (t[i+2] - t[i]), or (v[i+1] - v[i]) / hc at the last knot
  if (i + 2 < n) { const double d = t[i + 2] - t[i]; w[1] -= Bc / d; w[3] += Bc / d; } else { w[1] -= Bc / hc; w[2] += Bc / hc; }
  *b = i - 1;
}
// ai: The lens model's centre, fitted per frame (2026-11-10, STATUS "The optical centre fitted in the C's lens model"):
// ai: a parameter of border_fill_anchors' fit as its terms are, each frame on its own, so the decoder needs nothing from
// ai: the camera; the axis a receiver gives (lizard_set_axis) is only where the search starts. The model is linear in its
// ai: terms and the centre enters alone. Where no axis is given, a coarse stage first: the image's centre and a 9 x 9 grid
// ai: over the image and half an image round it, each scored by one solve of the full model (lens_score); then
// ai: Gauss-Newton from the best (lens_step), each step taken where the points' residuals truncated at half a module fall
// ai: (whole, a half, a quarter), the first steps exact for the first radial term, then the full model's, to half a px.
// ai: The centre leaves its start only where its two parameters pay by AIC. On the S26's real replays (blocks in the 64,
// ai: 128 and 256 rings) the axis given reads 254,881, 86,380, 150,428 (held at the axis 255,837, 86,394, 150,443),
// ai: withheld 254,595, 86,357, 150,446 (held at the image's centre 235,060, 85,594, 113,659); the lens set's 2:1 halves
// ai: withheld 99.6, 98.0, 96.5% of the axis given; the S26's prime core 0 to 1% a frame with the axis, without it +4% on
// ai: a 64-ring half and -19% on a 4K half (a misregistered picture costs the LDPC more).
// ai: the robust lens fit (lensm_fit) about the centre (cx, cy), R its farthest image corner, the points' places about it
// ai: in qx, qy and its drops in dr; returns the points' residuals truncated at tau, summed (INFINITY where unsolved)
static double lens_fit_at(int np, const double *pu, const double *pv, const double *dx, const double *dy, const unsigned char *isb, double tau, int iw, int ih, double cx, double cy, double *qx, double *qy, unsigned char *dr, lensm_t *m) {
  if (np <= 0) return INFINITY;
  memset(m, 0, sizeof *m);
  m->ax = cx; m->ay = cy;
  for (int c = 0; c < 4; c++) { const double d = hypot((c & 1 ? iw : 0) - cx, (c & 2 ? ih : 0) - cy); if (d > m->R) m->R = d; }
  for (int i = 0; i < np; i++) { qx[i] = (pu[i] - cx) / m->R; qy[i] = (pv[i] - cy) / m->R; }
  memset(dr, 0, (size_t)np);
  if (!lensm_fit(np, qx, qy, dx, dy, isb, dr, tau, LENS_TERMS, m)) return INFINITY;
  double cost = 0;
  for (int i = 0; i < np; i++) {
    float fx, fy;
    lensm_at(m, pu[i], pv[i], &fx, &fy);
    const double r2 = (dx[i] - fx) * (dx[i] - fx) + (dy[i] - fy) * (dy[i] - fy);
    cost += r2 < tau * tau ? r2 : tau * tau;
  }
  return cost;
}
// ai: the coarse stage's score at a centre: the model at its full K solved once (no AICc, no drop rounds) over every
// ai: other border point and every anchor, the start's drops d0 left out; those points' residuals truncated at tau
static double lens_score(int np, const double *pu, const double *pv, const double *dx, const double *dy, const unsigned char *isb, const unsigned char *d0, double tau, int iw, int ih, double cx, double cy) {
  double R = 0;
  for (int c = 0; c < 4; c++) { const double d = hypot((c & 1 ? iw : 0) - cx, (c & 2 ? ih : 0) - cy); if (d > R) R = d; }
  double A[LENS_P][LENS_P + 1], rx[LENS_P], ry[LENS_P], t[LENS_P];
  memset(A, 0, sizeof A);
  int n = 0;
  for (int i = 0; i < np; i++) {
    if (d0[i] || (isb[i] && (i & 1))) continue;
    n++;
    lens_rows((pu[i] - cx) / R, (pv[i] - cy) / R, LENS_TERMS, rx, ry);
    for (int a = 0; a < LENS_P; a++) { for (int b = a; b < LENS_P; b++) A[a][b] += rx[a] * rx[b] + ry[a] * ry[b]; A[a][LENS_P] += rx[a] * dx[i] + ry[a] * dy[i]; }
  }
  for (int a = 1; a < LENS_P; a++) for (int b = 0; b < a; b++) A[a][b] = A[b][a];
  for (int a = 0; a < LENS_P; a++) A[a][a] += 1e-12;
  if (2 * n - LENS_P - 1 <= 0 || !solve_lens(A, LENS_P)) return INFINITY;
  for (int a = 0; a < LENS_P; a++) t[a] = A[a][LENS_P] / A[a][a];
  double cost = 0;
  for (int i = 0; i < np; i++) {
    if (isb[i] && (i & 1)) continue;
    lens_rows((pu[i] - cx) / R, (pv[i] - cy) / R, LENS_TERMS, rx, ry);
    double fx = 0, fy = 0;
    for (int a = 0; a < LENS_P; a++) { fx += t[a] * rx[a]; fy += t[a] * ry[a]; }
    const double r2 = (dx[i] - fx) * (dx[i] - fx) + (dy[i] - fy) * (dy[i] - fy);
    cost += r2 < tau * tau ? r2 : tau * tau;
  }
  return cost;
}
// ai: One Gauss-Newton step on the centre. A radial term t s^n q about the centre moved by delta R (q, delta over R,
// ai: s = |q|^2) is, to first order, t (s^n q - s^n delta - 2n s^(n-1) q (q . delta)): the model's 8 + K terms and these
// ai: two columns at m's coefficients, solved together by least squares over m's kept points (dr); the step in px. k1 set
// ai: (one radial term, its coefficient from the solve): a centre moved by delta is exactly the first radial term about the
// ai: old centre, the perspective terms (-2 k1 q (q . delta)), an affine map and a shift, plus -k1 delta s, so two columns
// ai: (s, 0) and (0, s) whose coefficients e give delta = -e / k1 from any distance (global for the first term)
static int lens_step(int np, const double *qx, const double *qy, const double *dx, const double *dy, const unsigned char *dr, const lensm_t *m, int k1, double *sx, double *sy) {
  enum { P = LENS_P + 2 };
  const int K = k1 ? 1 : m->K, p = 8 + K + 2;
  if (K < 1) return 0;
  double A[P][P + 1], rx[P], ry[P];
  memset(A, 0, sizeof A);
  int n = 0;
  for (int i = 0; i < np; i++) {
    if (dr[i]) continue;
    n++;
    lens_rows(qx[i], qy[i], K, rx, ry);
    const double x = qx[i], y = qy[i], s = x * x + y * y;
    double jxx = 0, jxy = 0, jyy = 0, sn1 = 1;
    for (int k = 1; k <= K; k++, sn1 *= s) {
      const double t = m->t[8 + k - 1];
      jxx -= t * (s * sn1 + 2 * k * sn1 * x * x);
      jyy -= t * (s * sn1 + 2 * k * sn1 * y * y);
      jxy -= t * 2 * k * sn1 * x * y;
    }
    if (k1) { jxx = s; jxy = 0; jyy = s; }
    rx[8 + K] = jxx; ry[8 + K] = jxy; rx[9 + K] = jxy; ry[9 + K] = jyy;
    for (int a = 0; a < p; a++) { for (int b = 0; b < p; b++) A[a][b] += rx[a] * rx[b] + ry[a] * ry[b]; A[a][P] += rx[a] * dx[i] + ry[a] * dy[i]; }
  }
  if (2 * n <= p + 1) return 0;
  for (int a = 0; a < p; a++) A[a][a] += 1e-12;
  for (int c = 0; c < p; c++) {
    int pv = c;
    for (int r = c + 1; r < p; r++) if (fabs(A[r][c]) > fabs(A[pv][c])) pv = r;
    if (fabs(A[pv][c]) < 1e-14) return 0;
    for (int k = 0; k <= P; k++) { const double t = A[c][k]; A[c][k] = A[pv][k]; A[pv][k] = t; }
    for (int r = 0; r < p; r++) {
      if (r == c) continue;
      const double f = A[r][c] / A[c][c];
      for (int k = c; k <= P; k++) A[r][k] -= f * A[c][k];
    }
  }
  *sx = A[8 + K][P] / A[8 + K][8 + K] * m->R; *sy = A[9 + K][P] / A[9 + K][9 + K] * m->R;
  if (k1) { const double t1 = A[8][P] / A[8][8]; if (!(fabs(t1) > 1e-9)) return 0; *sx /= -t1; *sy /= -t1; }
  return isfinite(*sx) && isfinite(*sy);
}
// ai: the search: from (*ax, *ay), the coarse stage where the axis was not given, then Gauss-Newton; the centre found
// ai: written back where it pays by AIC over the 2 np residuals
static void lens_centre(int np, const double *pu, const double *pv, const double *dx, const double *dy, const unsigned char *isb, double tau, int iw, int ih, int given, double *ax, double *ay) {
  if (np <= 0) return;
  double *w = malloc((size_t)4 * np * sizeof(double)), *qx = w, *qy = w + np, *tx = w + 2 * np, *ty = w + 3 * np;
  unsigned char *dc = malloc((size_t)2 * np), *dr = dc + np;
  lensm_t cur, tr;
  const double ax0 = *ax, ay0 = *ay;
  double cost = lens_fit_at(np, pu, pv, dx, dy, isb, tau, iw, ih, ax0, ay0, qx, qy, dc, &cur);
  const double cost0 = cost;
  if (!given) {
    double cb = lens_score(np, pu, pv, dx, dy, isb, dc, tau, iw, ih, ax0, ay0), bx = ax0, by = ay0;
    for (int j = -4; j <= 4; j++) for (int i = -4; i <= 4; i++) {
      const double cx = iw * 0.5 + i * iw * 0.25, cy = ih * 0.5 + j * ih * 0.25;
      const double c = lens_score(np, pu, pv, dx, dy, isb, dc, tau, iw, ih, cx, cy);
      if (c < cb * 0.999) { cb = c; bx = cx; by = cy; }
    }
    if (bx != ax0 || by != ay0) {
      const double c = lens_fit_at(np, pu, pv, dx, dy, isb, tau, iw, ih, bx, by, tx, ty, dr, &tr);
      if (c < cost) { cost = c; cur = tr; memcpy(qx, tx, (size_t)np * sizeof(double)); memcpy(qy, ty, (size_t)np * sizeof(double)); memcpy(dc, dr, (size_t)np); }
    }
  }
  for (int it = 0, k1 = 1; it < 10 && cost < INFINITY; it++) {
    double sx, sy;
    if (!lens_step(np, qx, qy, dx, dy, dc, &cur, k1, &sx, &sy)) { if (k1) { k1 = 0; continue; } break; }
    const double lim = 0.5 * (iw > ih ? iw : ih), sl = hypot(sx, sy);
    if (sl > lim) { sx *= lim / sl; sy *= lim / sl; }
    double took = 0;
    for (double f = 1; f > 0.2; f *= 0.5) {
      const double c = lens_fit_at(np, pu, pv, dx, dy, isb, tau, iw, ih, cur.ax + f * sx, cur.ay + f * sy, tx, ty, dr, &tr);
      if (c < cost) {
        cost = c; cur = tr; took = f * hypot(sx, sy);
        memcpy(qx, tx, (size_t)np * sizeof(double)); memcpy(qy, ty, (size_t)np * sizeof(double)); memcpy(dc, dr, (size_t)np);
        break;
      }
    }
    // ai: the first term's steps until one is refused or moves under 2 px, then the full model's to half a px
    if (k1) { if (took < 2) k1 = 0; } else if (took < 0.5) break;
  }
  if (cost < cost0 * exp(-4.0 / (2.0 * np))) { *ax = cur.ax; *ay = cur.ay; }
  free(w); free(dc);
}
// ai: The interior from the border and the anchors: the lens model refitted over the border's trusted nodes and the
// ai: anchors found (robust: border nodes the anchors outvote, further than half a module or three robust spreads, are
// ai: dropped and take the model's displacement), the model at every interior node, then a cubic mesh of what the
// ai: anchors still see (0 at the lattice's ends, the anchors' readings at their centres). The border's nodes reach the
// ai: interior through the fit alone: a Coons patch of what the model left on the border carried each node's reading
// ai: error across the picture as a stripe, which the anchors see only where they cross it (2026-11-10, STATUS "The C's
// ai: geometry without the Coons patch": the real replays' leftover misregistration 0.13 to 0.39 samples rms with the
// ai: patch and the shrink below, 0.09 to 0.20 without; 2 to 54% more blocks in every ring). It runs on every frame:
// ai: with the anchors where anchor_register found OB_ANCH_MIN of them, else with none (the caller clears them), the
// ai: border's trusted nodes alone carrying the fit and no mesh. Under the minimum the border's own fit replaced a second
// ai: geometry (2026-11-11, STATUS "The C's border-only fallback and the GPU's interior, measured and left": H's
// ai: first-order change and a Coons patch of the border's residuals), the same frames found and more read (long range
// ai: in the 128 ring +1.2 to +17.9%, the rigor test's 256 ring +1.6%, the real replays equal). lm starts as the zero
// ai: model (H alone) for a solve rounding alone could fail.
static void border_fill_anchors(const ob_layout_t *L, const homo_t *H, int iw, int ih, node_t *nodes, const anchor_t *A, float mpx) {
  enum { NA = OB_ANCH_K * OB_ANCH_K };
  const int nx = L->nx, ny = L->ny, nn = nx * ny, cap = 2 * (nx + ny) + NA;
  double *qx = calloc((size_t)cap, sizeof(double)), *qy = calloc((size_t)cap, sizeof(double)), *dx = malloc((size_t)cap * sizeof(double)), *dy = malloc((size_t)cap * sizeof(double));
  unsigned char *isb = calloc((size_t)cap, 1), *drop = calloc((size_t)cap, 1);
  int *at = malloc((size_t)cap * sizeof(int)), np = 0;
  double *pu = malloc((size_t)cap * sizeof(double)), *pv = malloc((size_t)cap * sizeof(double));   // ai: the points in px
  const int K = ob_anchor_k(L);   // ai: the ring's grid (anchor j OB_ANCH_K + i)
  lensm_t lm = { 0 };
  lensm_axis(iw, ih, &lm);
  for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) {
    if (i && j && i < nx - 1 && j < ny - 1) continue;
    const int k = j * nx + i, corner = (i == 0 || i == nx - 1) && (j == 0 || j == ny - 1);
    if (!(nodes[k].done && (corner || (L->node_mark[k] && nodes[k].score >= 0.35f)))) continue;
    float u, v;
    project(H, L->node_x[i], L->node_y[j], &u, &v);
    pu[np] = u; pv[np] = v; dx[np] = nodes[k].dx; dy[np] = nodes[k].dy; isb[np] = 1; at[np] = k; np++;
  }
  for (int k = 0; k < NA; k++) {
    if (!A[k].ok) continue;
    float u, v;
    project(H, A[k].X, A[k].Y, &u, &v);
    pu[np] = u; pv[np] = v; dx[np] = A[k].dx; dy[np] = A[k].dy; at[np] = -1 - k; np++;
  }
  // ai: the centre fitted (lens_centre), from the axis given, else the image's centre
  double cx = lm.ax, cy = lm.ay;
  lens_centre(np, pu, pv, dx, dy, isb, 0.5 * mpx, iw, ih, ob_axis[0] == ob_axis[0] && ob_axis[1] == ob_axis[1], &cx, &cy);
  lm.ax = cx; lm.ay = cy; lm.R = 0;
  for (int c = 0; c < 4; c++) { const double d = hypot((c & 1 ? iw : 0) - cx, (c & 2 ? ih : 0) - cy); if (d > lm.R) lm.R = d; }
  for (int i = 0; i < np; i++) { qx[i] = (pu[i] - cx) / lm.R; qy[i] = (pv[i] - cy) / lm.R; }
  lensm_fit(np, qx, qy, dx, dy, isb, drop, 0.5 * mpx, LENS_TERMS, &lm);
  float *mx = malloc((size_t)nn * sizeof(float)), *my = malloc((size_t)nn * sizeof(float));
  for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) {
    float u, v;
    project(H, L->node_x[i], L->node_y[j], &u, &v);
    lensm_at(&lm, u, v, &mx[j * nx + i], &my[j * nx + i]);
  }
  // ai: a border node the anchors outvote reads the model's displacement
  for (int p = 0; p < np; p++) if (isb[p] && drop[p]) { nodes[at[p]].dx = mx[at[p]]; nodes[at[p]].dy = my[at[p]]; }
  for (int j = 1; j < ny - 1; j++) for (int i = 1; i < nx - 1; i++) {
    node_t *nd = &nodes[j * nx + i];
    if (L->node_mark[j * nx + i]) continue;
    nd->dx = mx[j * nx + i]; nd->dy = my[j * nx + i]; nd->done = 1; nd->score = 1;
  }
  // ai: what the anchors still see through that mesh, as the sampler blends it, carried inward by the knots
  const double X0 = L->node_x[0], Y0 = L->node_y[0], SX = L->node_x[nx - 1] - X0, SY = L->node_y[ny - 1] - Y0;
  double (*tv)[2] = malloc((size_t)cap * sizeof *tv);
  int *gi = malloc((size_t)cap * sizeof(int)), m = 0;
  for (int p = 0; p < np; p++) {
    if (isb[p]) continue;
    const anchor_t *an = &A[-1 - at[p]];
    const ob_reg_t r = { *H, nodes };
    float u, v, hu, hv;
    ob_map_point(L, &r, an->X, an->Y, &u, &v);
    project(H, an->X, an->Y, &hu, &hv);
    tv[m][0] = an->dx - (u - hu); tv[m][1] = an->dy - (v - hv); gi[m] = -1 - at[p]; m++;
  }
  // ai: Each reading is taken whole. A shrink of the readings towards 0 by the share of them that is signal (empirical
  // ai: Bayes, the noise from each reading against its grid neighbours') went with the Coons patch: on the 3 x 3 grid its
  // ai: neighbours do not measure noise (the shrink 0.14 to 0.24 on distorted captures), and with the patch gone it cost
  // ai: the 64 ring's replays 4.4% and the lens set 1.8 and 0.9% in the 64 and 128 rings, the 256 ring level.
  {
    // ai: The mesh (2026-10-26; a thin-plate spline before it, the same blocks over the published lenses for 2.6 to 8.2%
    // ai: less of a frame): knots on both axes at the lattice's ends, where the border holds the map (0), and the ring's
    // ai: k anchors' centres between; an inner knot is its anchor's reading, a missing one the mean of its found
    // ai: 4-neighbours (rounds until none is left, at most ANCH_KNOT_ROUNDS; none near: 0). The arrays hold the 7 x 7
    // ai: grid's 9 knots.
    enum { KM = OB_ANCH_K + 2 };
    const int KN = K + 2;
    double kx[KM], ky[KM], kv[KM][KM][2];
    unsigned char kn[KM][KM];
    memset(kv, 0, sizeof kv); memset(kn, 0, sizeof kn);
    const double ha = 0.5 * ob_anchor_side(L);
    kx[0] = ky[0] = 0; kx[KN - 1] = ky[KN - 1] = 1;
    for (int i = 0; i < K; i++) { kx[i + 1] = (ob_anchor_m0(L, i) + ha - X0) / SX; ky[i + 1] = (ob_anchor_m0(L, i) + ha - Y0) / SY; }
    for (int q = 0; q < m; q++) if (gi[q] >= 0) {
      const int i = gi[q] % OB_ANCH_K + 1, j = gi[q] / OB_ANCH_K + 1;
      kv[j][i][0] = tv[q][0]; kv[j][i][1] = tv[q][1]; kn[j][i] = 1;
    }
    for (int round = 0; round < ANCH_KNOT_ROUNDS; round++) {
      unsigned char now[KM][KM];
      memcpy(now, kn, sizeof now);
      int left = 0;
      for (int j = 1; j < KN - 1; j++) for (int i = 1; i < KN - 1; i++) {
        if (kn[j][i]) continue;
        const int nb[4][2] = { { i - 1, j }, { i + 1, j }, { i, j - 1 }, { i, j + 1 } };
        double s0 = 0, s1 = 0;
        int c = 0;
        for (int e = 0; e < 4; e++) {
          const int a = nb[e][0], b = nb[e][1];
          if (a < 1 || a > KN - 2 || b < 1 || b > KN - 2 || !kn[b][a]) continue;
          s0 += kv[b][a][0]; s1 += kv[b][a][1]; c++;
        }
        if (c) { kv[j][i][0] = s0 / c; kv[j][i][1] = s1 / c; now[j][i] = 1; } else left++;
      }
      memcpy(kn, now, sizeof kn);
      if (!left) break;
    }
    // ai: each axis a cubic Hermite through the knots, the slopes from the neighbouring knots (one-sided at the ends):
    // ai: four knots' weights an axis (anch_herm4), so a node is a 4 x 4 blend of knot values
    for (int j = 1; j < ny - 1; j++) for (int i = 1; i < nx - 1; i++) {
      if (L->node_mark[j * nx + i]) continue;
      int bx, by;
      double wx[4], wy[4];
      anch_herm4(kx, KN, (L->node_x[i] - X0) / SX, &bx, wx);
      anch_herm4(ky, KN, (L->node_y[j] - Y0) / SY, &by, wy);
      double d0 = 0, d1 = 0;
      for (int b = 0; b < 4; b++) for (int a = 0; a < 4; a++) {
        const int qx = bx + a, qy = by + b;
        if (qx < 0 || qx >= KN || qy < 0 || qy >= KN || (wx[a] == 0 && wy[b] == 0)) continue;
        d0 += wy[b] * wx[a] * kv[qy][qx][0]; d1 += wy[b] * wx[a] * kv[qy][qx][1];
      }
      nodes[j * nx + i].dx += (float)d0; nodes[j * nx + i].dy += (float)d1;
    }
  }
  free(tv); free(gi);
  free(mx); free(my); free(qx); free(qy); free(dx); free(dy); free(isb); free(drop); free(at); free(pu); free(pv);
}

// ---------------------------------------------------------------- thin frame: a side's curve
// ai: A traced side's curve (the finder's tr_fit): across = c (along - mid)^2 + a along + b, along x on the top and bottom
// ai: sides and y on the right and left
typedef struct { float a, b; int n; float c, mid; } fline_t;
static inline float line_at(const fline_t *l, float along) { float d = along - l->mid; return l->c * d * d + l->a * along + l->b; }

// ---------------------------------------------------------------- thin frame: the corner marks
//
// A LOCAL finder, in the sense QR's is: one place in the image, judged on its own evidence, with nothing to crowd out.
// ai: The marks are proposed from the grey picture (find_frame's gray_cands, 2026-11-24); the binarized scan before it
// ai: (mark_candidates, 2026-09-21 to 2026-11-24) read each mark's cross-section as a run-length ratio. What stays here
// ai: is what both share: the marks' geometry, and four candidates as a quad.

// The core's modules, and where its centre sits in layout coordinates. thin_corner paints the core from the end
// of the gap band to the mark's light rim, so it spans [OB_THIN_TRACK, S - 1) and its middle is the two averaged. A
// corner mark borrows the border's line and gap for two of its sides.
static inline int mark_core(const ob_layout_t *L) { return L->corner - 1 - OB_THIN_TRACK; }
static inline float mark_mid(const ob_layout_t *L) { return (OB_THIN_TRACK + L->corner - 1) / 2.0f; }

// Four marks chosen the way zxing chooses finder patterns: not by where they are but by whether they AGREE with
// each other (zxing's QR detector rejects a triple of finder patterns on size ratio, leg ratio, implied module count
// and angle). The widest convex quad picks picture texture here: the picture offers
// dozens of small things with a mark's cross-section, and four of them always span more of the image than the marks do.
//
// LIZARD can press the module-count test much harder than QR can, because the count is not being estimated: two
// adjacent marks are exactly L->w - 2 * mark_mid modules apart and a mark's own cross-section gives the module
// size, so a quad whose side over its module size is not that number is not this symbol.
// One form at a time. The two forms have different centres (the gapped core's middle is (OB_THIN_TRACK + S - 1) / 2 in
// from the corner, 8 at S = 12; the merged square's is S / 2), and blur is the same all over a capture, so four marks
// of one symbol are one form; mixing them would only let a picture blob stand in for a mark. `score` comes back as the quad's
// disagreement with the known spacing, smaller is better, so the caller can set the forms against each other.
//
// Which quad is RIGHT is not decided here. Spacing ranks them; the track read in find_frame is what believes one,
// the way zxing walks its pattern sets until one decodes. So the best few come back, not the best one.
// form: 1 the merged square, 2 the core (find_frame reads each grey candidate both ways, gray_form); 0, the gapped
// cross-section the binarized scan read until 2026-11-24, is centred as the core is.
static float mark_form_mid(const ob_layout_t *L, int form) { return form == 1 ? L->corner / 2.0f : mark_mid(L); }

// The mark geometry a finder outside the codec needs to be scale-free the way mark_quads is
// (gpu/tables.mjs, scripts/exp/fmt_word.mjs). Nothing in the decoder calls this. A blob detector measures a mark in PIXELS and has
// no module size of its own, so it has to get one the way this file does: divide the blob by the mark's known
// module count, then check that four of them sit the known number of modules apart. Both forms are given
// because a threshold that keeps the light gap open finds the CORE and one that closes it finds the whole
// square, and the two have different centres, so they imply different quads from the same four blobs.
// The last two are what an external quad is MEASURED AGAINST: acquire_core reads one through corner_coords,
// which spans the mesh's outermost nodes and not modules 0 to w. Seven modules of difference on a 286 module
// symbol, which is 2.5% of the quad and far more than the registration tolerates.
//   0 w  1 corner  2 core modules (gapped)  3 square modules (merged)  4 mid gapped  5 mid merged
//   6 node span x  7 node span y
void ob_mark_geom(const ob_layout_t *L, float *out) {
  out[0] = (float)L->w;
  out[1] = (float)L->corner;
  out[2] = (float)mark_core(L);
  out[3] = (float)(L->corner - 2);
  out[4] = mark_form_mid(L, 0);
  out[5] = mark_form_mid(L, 1);
  out[6] = L->nx > 1 ? L->node_x[L->nx - 1] - L->node_x[0] : (float)L->w;
  out[7] = L->ny > 1 ? L->node_y[L->ny - 1] - L->node_y[0] : (float)L->h;
}
enum { MARK_POOL = 24, MARK_QUADS = 4 };
typedef struct { cand_t q[4]; float score; } mquad_t;
// Four marks at the corners, as a quad. The pool is ranked by support TIMES size, not support alone: a mark is
// the biggest solid thing of its kind in the symbol, and a blurred picture is full of smaller blobs that a scan
// crosses nearly as often. Ranked by hits alone, a true mark seen on four rows (traced: the top right one at
// 1.5 px of defocus) lost its place in the pool to blobs two thirds its size seen on five.
//
// A TRIPLE was tried on 2026-09-21, QR's three corners with the fourth inferred as a parallelogram's. It reads
// its own rotation, which is elegant, and it costs the perspective: three points determine an affine map, and
// the term that makes a square photograph as a trapezium is exactly what the fourth measured point supplies.
// The edge marks were meant to pay that back, and they are found in the captures where the map was already
// right and missed in the ones where it is not. fill 0.3 fell from 2990 B on 16 of 16 frames to 352 B on 7.
// ai: The pool, and every pair's distance in module sizes, once a form: they hold for every layout a receiver looks
// ai: for (only the spacing the layout asks for, want, differs), so mark_quads takes them for each of the rings.
typedef struct { cand_t c[MARK_POOL]; int n; float mods[MARK_POOL][MARK_POOL]; } mpool_t;
static void mark_pool(const cand_t *all, int nall, mpool_t *P) {
  cand_t *c = P->c;
  int n = 0;
  #define RANK(k) ((k).hits * (k).unit)
  for (int i = 0; i < nall; i++) {
    int j;
    if (n < MARK_POOL) j = n++;
    else { if (RANK(c[MARK_POOL - 1]) >= RANK(all[i])) continue; j = MARK_POOL - 1; }
    while (j > 0 && RANK(c[j - 1]) < RANK(all[i])) { c[j] = c[j - 1]; j--; }
    c[j] = all[i];
  }
  #undef RANK
  P->n = n;
  for (int a = 0; a < n; a++) for (int b = a; b < n; b++) {
    const float dx = c[b].x - c[a].x, dy = c[b].y - c[a].y;
    P->mods[a][b] = sqrtf(dx * dx + dy * dy) / (0.5f * (c[a].unit + c[b].unit));
  }
}
static int mark_quads(const ob_layout_t *L, const mpool_t *P, int merged, mquad_t *out) {
  const float want = L->w - 2 * mark_form_mid(L, merged);
  const cand_t *c = P->c;
  const int n = P->n;
  int nq = 0;
  // Whether two candidates can be ADJACENT marks, and how far off the known spacing they are if so. Most pairs
  // cannot, so walking four-cycles of this table visits a few hundred quads where every four of the pool is
  // 10626, each of which used to be sorted round its centroid by atan2 before it could be refused.
  // rat: a side's length over the module size its two ends report, as a share of the known count, or -1 where
  // the two cannot be adjacent marks at all.
  float rat[MARK_POOL][MARK_POOL];
  // ai: adj[a]: bit b where rat[a][b] >= 0, so the walk below takes each next corner from a mask, in the same order.
  // ai: one[a]: bit b where a and b may be marks of one symbol, neither unit more than four times the other's (one
  // ai: symbol, one module size, within what perspective and a wide lens's stretch do to it; twice cost the lens set's
  // ai: 64 ring blocks, 2026-11-24): a quad needs all six of its pairs so, so each corner is taken only from the ones
  // ai: every corner before it allows, and a quad that would fail it is never walked
  uint32_t adj[MARK_POOL] = { 0 }, one[MARK_POOL] = { 0 };
  _Static_assert(MARK_POOL <= 32, "a corner's neighbours are the bits of one word");
  for (int a = 0; a < n; a++) for (int b = a; b < n; b++) {
    const float mods = P->mods[a][b];
    const int can = a != b && mods >= 0.65f * want && mods <= 1.5f * want;   // unit shrinks with the same tilt, so a squeezed side reads 2cos / (1 + cos): 0.65 is about 61 degrees
    rat[a][b] = rat[b][a] = can ? mods / want : -1;
    if (can) { adj[a] |= 1u << b; adj[b] |= 1u << a; }
    const float ua = c[a].unit, ub = c[b].unit;
    if (!(ua > 4 * ub) && !(ub > 4 * ua)) { one[a] |= 1u << b; one[b] |= 1u << a; }
  }
  // a is the cycle's lowest index and b < e, so each cycle is met once and not again backwards. No corner is its own
  // neighbour, so d != b and e != d hold of the masks.
  for (int a = 0; a < n; a++) for (uint32_t mb = adj[a] & one[a] & (~1u << a); mb; mb &= mb - 1) {
    const int b = __builtin_ctz(mb);
    for (uint32_t md = adj[b] & one[a] & one[b] & (~1u << a); md; md &= md - 1) {
      const int d = __builtin_ctz(md);
      for (uint32_t me = adj[d] & adj[a] & one[a] & one[b] & one[d] & (~1u << b); me; me &= me - 1) {
        const int e = __builtin_ctz(me);
        // The gapped form reads its module size from a ratio of line, gap and core, which a threshold that shaves
        // dark runs leaves alone, so its sides are scored on their distance from the count. The two square forms
        // read it from a dark square alone, and the binarizer shaves every one of them alike, by 10 to 25% on a
        // 6-module core of 11 to 20 px: four real marks then agree with each other and are off the count
        // TOGETHER (1.12 on every side, traced at 40 degrees of turn), where four picture blobs do not agree. So
        // for those the score is the sides' spread about their own mean plus half the mean's distance from the
        // count, which still keeps a wrong count out. Not for the gapped form: under perspective its four real
        // sides legitimately disagree, and blobs that happen to agree then outrank it (30 degrees of yaw).
        // ai: The score first: it reads the cycle's four sides, the same whichever way round the cycle is taken, so a
        // ai: quad that could not enter the best MARK_QUADS is refused before the module-size and convexity tests (the
        // ai: same quads kept; 0.17 to 0.44 M instructions a frame fewer on the S26 runs).
        const float r4[4] = { rat[a][b], rat[b][d], rat[d][e], rat[e][a] }, k = 0.25f * (r4[0] + r4[1] + r4[2] + r4[3]);
        const float score = merged ? fabsf(r4[0] - k) + fabsf(r4[1] - k) + fabsf(r4[2] - k) + fabsf(r4[3] - k) + 0.5f * fabsf(k - 1)
                                   : fabsf(r4[0] - 1) + fabsf(r4[1] - 1) + fabsf(r4[2] - 1) + fabsf(r4[3] - 1);
        if (nq == MARK_QUADS && out[MARK_QUADS - 1].score <= score) continue;
        int id[4] = { a, b, d, e };
        int pos = 0, neg = 0;
        for (int k = 0; k < 4; k++) { const float x = cross2(&c[id[k]], &c[id[(k + 1) & 3]], &c[id[(k + 2) & 3]]); pos += x > 0; neg += x < 0; }
        if (pos != 4 && neg != 4) continue;   // convex
        if (neg) { id[1] = e; id[3] = b; }
        int j;
        if (nq < MARK_QUADS) j = nq++;
        else { if (out[MARK_QUADS - 1].score <= score) continue; j = MARK_QUADS - 1; }
        while (j > 0 && out[j - 1].score > score) { out[j] = out[j - 1]; j--; }
        for (int k = 0; k < 4; k++) out[j].q[k] = c[id[k]];
        out[j].score = score;
      }
    }
  }
  // Each starts at the corner nearest the image's top left as seen from its own middle, which is what the
  // hypothesis numbering downstream was built on.
  for (int i = 0; i < nq; i++) {
    cand_t *q = out[i].q, r[4];
    float gx = 0, gy = 0, lo = 1e9f;
    int first = 0;
    for (int k = 0; k < 4; k++) { gx += q[k].x / 4; gy += q[k].y / 4; }
    for (int k = 0; k < 4; k++) { const float an = atan2f(q[k].y - gy, q[k].x - gx); if (an < lo) { lo = an; first = k; } }
    for (int k = 0; k < 4; k++) r[k] = q[(first + k) & 3];
    memcpy(q, r, sizeof r);
  }
  return nq;
}

// What a mesh built elsewhere has to be built against: the layout's node lattice, the per-module kind map and which
// nodes carry a mark. Read once a configuration. dims: nx, ny, w, h, corner.
int ob_test_mesh_tables(const ob_layout_t *L, int32_t *dims, float *nodeX, float *nodeY, uint8_t *kind, int32_t *mark) {
  dims[0] = L->nx; dims[1] = L->ny; dims[2] = L->w; dims[3] = L->h; dims[4] = L->corner;
  if (!nodeX) return 1;
  for (int i = 0; i < L->nx; i++) nodeX[i] = L->node_x[i];
  for (int j = 0; j < L->ny; j++) nodeY[j] = L->node_y[j];
  memcpy(kind, L->kind, (size_t)L->w * L->h);
  for (int k = 0; k < L->nx * L->ny; k++) mark[k] = L->node_mark[k];
  return 1;
}
// The mark centres of one form as quads, best spacing first, and the rectangle they make in layout modules.
static int mark_rect(const ob_layout_t *L, const mpool_t *P, int n, int iw, int ih, int merged, float *src, mquad_t *quads) {
  if (n < 4) return 0;
  const float m = mark_form_mid(L, merged), s4[8] = { m, m, L->w - m, m, L->w - m, L->h - m, m, L->h - m };
  memcpy(src, s4, sizeof s4);
  int nq = mark_quads(L, P, merged, quads), keep = 0;
  for (int i = 0; i < nq; i++) {
    const cand_t *q = quads[i].q;
    float x0 = q[0].x, x1 = x0, y0 = q[0].y, y1 = y0;
    for (int k = 1; k < 4; k++) { x0 = fminf(x0, q[k].x); x1 = fmaxf(x1, q[k].x); y0 = fminf(y0, q[k].y); y1 = fmaxf(y1, q[k].y); }
    if (x1 - x0 > 0.08f * iw && y1 - y0 > 0.08f * ih) quads[keep++] = quads[i];   // by extent: which corner a quad starts at turns with the symbol
  }
  return keep;
}

// Where a near-vertical side (x from y) meets a near-horizontal one (y from x). Both are nearly
// straight, so stepping from one to the other settles in a few rounds.
static void meet(const fline_t *v, const fline_t *hz, float *x, float *y) {
  *x = (v->a * hz->b + v->b) / (1 - v->a * hz->a);
  *y = hz->a * *x + hz->b;
  for (int it = 0; it < 4; it++) { *x = line_at(v, *y); *y = line_at(hz, *x); }
}

// The depth the traced border runs at (find_frame's passes): the inner edge of the dark ring while the one-module light
// gap inside it survives (depth 1 + OB_THIN_LINE), the picture's edge once line, track and ring have merged under blur
// (depth OB_THIN), or the ring's outer edge (depth 1). find_frame tries each and places its quad's corners there.

// ai: The finder's lens (2026-11-24): a traced quad's one radial term about the axis (find_frame's tr_lens), kr = k / R^2;
// ai: a point the homography places at p is read at a + (p - a) (1 + kr |p - a|^2). A traced quad's track and corners
// ai: are read through it; it took the place of the fitted sides' bow (thin_point and its forms, 2026-09 to 2026-11-24),
// ai: whose curves came from the binarized line path. A mark quad has none (on 0).
typedef struct { int on; float kr, ax, ay; } flens_t;
static inline void flens_bend(const flens_t *lk, float *u, float *v) {
  if (!lk || !lk->on) return;
  const float dx = *u - lk->ax, dy = *v - lk->ay, s = 1 + lk->kr * (dx * dx + dy * dy);
  *u = lk->ax + dx * s; *v = lk->ay + dy * s;
}

// Layout coordinates on side 0..3 (top, right, bottom, left) at module t along it and depth d from the edge.
static void side_xy(const ob_layout_t *L, int side, float t, float d, float *x, float *y) {
  *x = side == 0 || side == 2 ? t : side == 3 ? d : L->w - d;
  *y = side == 0 ? d : side == 2 ? L->h - d : t;
}

#ifdef OB_SIMD
// The modules at the middle of track cells j .. j + 3, taken through the macro one cell at a time rather than
// stepped from the first. It was stepped until 2026-09-21, when the band's runs made the spacing non-uniform and
// the vector build read the format word's cells as the track's: it registered off the scalar build's quad by up
// to 0.8 px and lost frames the scalar kept. The runs are gone, but a reader that asks the layout where a cell
// is cannot be wrong about it again.
#define TRACK_AT4(L, j) wasm_f32x4_make((float)((L)->reserve + OB_TRACK_AT(j)) + OB_CELL_MID, \
                                        (float)((L)->reserve + OB_TRACK_AT((j) + 1)) + OB_CELL_MID, \
                                        (float)((L)->reserve + OB_TRACK_AT((j) + 2)) + OB_CELL_MID, \
                                        (float)((L)->reserve + OB_TRACK_AT((j) + 3)) + OB_CELL_MID)
#if defined(OB_NEON) || defined(OB_X86)
// ai: arm64's track reads (thin_levels, thin_score, thin_read_both, each its own form), their own floats: a
// ai: side's map at one depth folded to u = (hu t + ku) / w, v = (hv t + kv) / w, w = hw t + kw in the module t along
// ai: it, fused, one division for both; the finder's lens where a traced quad has one; the read sample4r (internal.h)
// ai: at gamma 1, sample4 otherwise. Each side's map, lens and frame are taken into locals before its loop (through the
// ai: pointers, a store to the reading reloaded them every step). Every
// ai: layout's track cells a side, B / 2, are a multiple of 4: none of these loops has a tail. x86-64 has the same forms
// ai: in SSE4.1, each multiply-add a multiply and an add: its own floats again.
typedef struct { float hu, ku, hv, kv, hw, kw; } trk_map_t;
static inline trk_map_t trk_map(const ob_layout_t *L, const homo_t *H, int side, float d) {
  const float *h = H->h, f = side == 0 || side == 3 ? d : side == 2 ? L->h - d : L->w - d;
  const int a = side & 1;   // t is x on sides 0 and 2, y on 1 and 3
  return (trk_map_t){ h[a], h[1 - a] * f + h[2], h[3 + a], h[4 - a] * f + h[5], h[6 + a], h[7 - a] * f + 1 };
}
#ifdef OB_NEON
static inline __attribute__((always_inline)) float32x4_t trk_w(const trk_map_t *p, float32x4_t t) { return vfmaq_n_f32(vdupq_n_f32(p->kw), t, p->hw); }
static inline __attribute__((always_inline)) void trk_uv(const trk_map_t *p, float32x4_t t, float32x4_t r, float32x4_t *u, float32x4_t *v) {
  *u = vmulq_f32(vfmaq_n_f32(vdupq_n_f32(p->ku), t, p->hu), r);
  *v = vmulq_f32(vfmaq_n_f32(vdupq_n_f32(p->kv), t, p->hv), r);
}
#else
static inline __attribute__((always_inline)) __m128 trk_w(const trk_map_t *p, __m128 t) { return _mm_add_ps(_mm_set1_ps(p->kw), _mm_mul_ps(t, _mm_set1_ps(p->hw))); }
static inline __attribute__((always_inline)) void trk_uv(const trk_map_t *p, __m128 t, __m128 r, __m128 *u, __m128 *v) {
  *u = _mm_mul_ps(_mm_add_ps(_mm_set1_ps(p->ku), _mm_mul_ps(t, _mm_set1_ps(p->hu))), r);
  *v = _mm_mul_ps(_mm_add_ps(_mm_set1_ps(p->kv), _mm_mul_ps(t, _mm_set1_ps(p->hv))), r);
}
#endif
// a side as one depth reads it: its point's map, and the lens where its quad has one
typedef struct { trk_map_t pt; flens_t lk; } trk_side_t;
static inline trk_side_t trk_side(const ob_layout_t *L, const homo_t *H, int side, float d, const flens_t *lk) {
  trk_side_t s = { .pt = trk_map(L, H, side, d) };
  if (lk && lk->on) s.lk = *lk;
  return s;
}
// the frame as sample() reads it
typedef struct { const image_t *im; const uint8_t *px; int w, lin; float wm, hm; } trk_img_t;
static inline trk_img_t trk_img(const image_t *im) { return (trk_img_t){ im, im->px, im->w, im->lin, im->w - 1.001f, im->h - 1.001f }; }
#ifdef OB_NEON
static inline __attribute__((always_inline)) float32x4_t trk_px(const trk_img_t *g, float32x4_t u, float32x4_t v) {
  if (!g->lin) return vreinterpretq_f32_s32(sample4(g->im, vreinterpretq_s32_f32(u), vreinterpretq_s32_f32(v)));
  const float32x4_t half = vdupq_n_f32(0.5f), zero = vdupq_n_f32(0);
  const float32x4_t x = vminq_f32(vmaxq_f32(vsubq_f32(u, half), zero), vdupq_n_f32(g->wm));
  const float32x4_t y = vminq_f32(vmaxq_f32(vsubq_f32(v, half), zero), vdupq_n_f32(g->hm));
  return sample4r(g->px, g->w, x, y);
}
// ai: the lens's bend at four points (flens_bend's operations)
static inline __attribute__((always_inline)) void trk_lens(const flens_t *lk, float32x4_t *u, float32x4_t *v) {
  const float32x4_t ax = vdupq_n_f32(lk->ax), ay = vdupq_n_f32(lk->ay), dx = vsubq_f32(*u, ax), dy = vsubq_f32(*v, ay);
  const float32x4_t s = vaddq_f32(vdupq_n_f32(1), vmulq_n_f32(vaddq_f32(vmulq_f32(dx, dx), vmulq_f32(dy, dy)), lk->kr));
  *u = vaddq_f32(ax, vmulq_f32(dx, s)); *v = vaddq_f32(ay, vmulq_f32(dy, s));
}
// the side's grey at four points t
static inline __attribute__((always_inline)) float32x4_t trk_get(const trk_side_t *s, const trk_img_t *g, float32x4_t t) {
  float32x4_t u, v;
  trk_uv(&s->pt, t, vdivq_f32(vdupq_n_f32(1), trk_w(&s->pt, t)), &u, &v);
  if (s->lk.on) trk_lens(&s->lk, &u, &v);
  return trk_px(g, u, v);
}
// a row of thin_read_both's reading: the track cells at w0 + sg pos[i] (pos: each cell's module along the side)
static inline __attribute__((always_inline)) void trk_row(const trk_side_t *s, const trk_img_t *g, const float *pos, int npts, float32x4_t w0, float32x4_t sg, float *out) {
  for (int i = 0; i < npts; i += 4) vst1q_f32(out + i, trk_get(s, g, vfmaq_f32(w0, vld1q_f32(pos + i), sg)));
}
#else
// ai: x86-64: the same helpers in SSE4.1. A NaN coordinate reads pixel 0 at gamma 1 (MAXPS gives its second operand
// ai: back on a NaN, so the clamp makes it 0).
static inline __attribute__((always_inline)) __m128 trk_px(const trk_img_t *g, __m128 u, __m128 v) {
  if (!g->lin) return _mm_castsi128_ps(sample4(g->im, _mm_castps_si128(u), _mm_castps_si128(v)));
  const __m128 half = _mm_set1_ps(0.5f), zero = _mm_setzero_ps();
  const __m128 x = _mm_min_ps(_mm_max_ps(_mm_sub_ps(u, half), zero), _mm_set1_ps(g->wm));
  const __m128 y = _mm_min_ps(_mm_max_ps(_mm_sub_ps(v, half), zero), _mm_set1_ps(g->hm));
  return sample4r(g->px, g->w, x, y);
}
static inline __attribute__((always_inline)) void trk_lens(const flens_t *lk, __m128 *u, __m128 *v) {
  const __m128 ax = _mm_set1_ps(lk->ax), ay = _mm_set1_ps(lk->ay), dx = _mm_sub_ps(*u, ax), dy = _mm_sub_ps(*v, ay);
  const __m128 s = _mm_add_ps(_mm_set1_ps(1), _mm_mul_ps(_mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy)), _mm_set1_ps(lk->kr)));
  *u = _mm_add_ps(ax, _mm_mul_ps(dx, s)); *v = _mm_add_ps(ay, _mm_mul_ps(dy, s));
}
static inline __attribute__((always_inline)) __m128 trk_get(const trk_side_t *s, const trk_img_t *g, __m128 t) {
  __m128 u, v;
  trk_uv(&s->pt, t, _mm_div_ps(_mm_set1_ps(1), trk_w(&s->pt, t)), &u, &v);
  if (s->lk.on) trk_lens(&s->lk, &u, &v);
  return trk_px(g, u, v);
}
// ai: w0 + pos sg: the product exact (sg is 1 or -1), so one rounding, as arm64's fused form
static inline __attribute__((always_inline)) void trk_row(const trk_side_t *s, const trk_img_t *g, const float *pos, int npts, __m128 w0, __m128 sg, float *out) {
  for (int i = 0; i < npts; i += 4) _mm_storeu_ps(out + i, trk_get(s, g, _mm_add_ps(w0, _mm_mul_ps(_mm_loadu_ps(pos + i), sg))));
}
// ai: the four lanes' sum as arm64's FADDP pair takes it, (0 + 1) + (2 + 3)
static inline float trk_sum4(__m128 a) { const __m128 t = _mm_add_ps(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1))); return _mm_cvtss_f32(_mm_add_ss(t, _mm_movehl_ps(t, t))); }
#endif
// ai: the track's signs (ob_thin_track) as the sign bit a cell's -q takes, 0 where the cell is dark, side by side,
// ai: made once a thread; room for TRK_SIGNS cells a side, where the largest ring (the 256) has 128
enum { TRK_SIGNS = 256 };
static OB_TLS uint32_t trk_sign_tab[4][TRK_SIGNS];
static OB_TLS int trk_sign_made;
static const uint32_t *trk_signs(int side) {
  if (!trk_sign_made) {
    for (int sd = 0; sd < 4; sd++) for (int j = 0; j < TRK_SIGNS; j++) trk_sign_tab[sd][j] = ob_thin_track(sd, j) ? 0 : 0x80000000u;
    trk_sign_made = 1;
  }
  return trk_sign_tab[side];
}
#else
// The read for four points of one side, at module positions `along` and depth d, bent by the lens where lk has one:
// the same operations in the same order as the scalar's (project, flens_bend, sample), so the same floats.
static inline v128_t thin_sample4(const ob_layout_t *L, const image_t *im, const homo_t *H, const flens_t *lk, int side, v128_t along, float d) {
  const float *h = H->h;
  const v128_t one = wasm_f32x4_splat(1);
  v128_t x = along, y = along;
  if (side == 0 || side == 2) y = wasm_f32x4_splat(side == 0 ? d : L->h - d);
  else x = wasm_f32x4_splat(side == 3 ? d : L->w - d);
  const v128_t w_ = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_splat(h[6]), x), wasm_f32x4_mul(wasm_f32x4_splat(h[7]), y)), one);
  v128_t u = wasm_f32x4_div(wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_splat(h[0]), x), wasm_f32x4_mul(wasm_f32x4_splat(h[1]), y)), wasm_f32x4_splat(h[2])), w_);
  v128_t v = wasm_f32x4_div(wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_splat(h[3]), x), wasm_f32x4_mul(wasm_f32x4_splat(h[4]), y)), wasm_f32x4_splat(h[5])), w_);
  if (lk && lk->on) {
    const v128_t ax = wasm_f32x4_splat(lk->ax), ay = wasm_f32x4_splat(lk->ay), dx = wasm_f32x4_sub(u, ax), dy = wasm_f32x4_sub(v, ay);
    const v128_t sc = wasm_f32x4_add(one, wasm_f32x4_mul(wasm_f32x4_splat(lk->kr), wasm_f32x4_add(wasm_f32x4_mul(dx, dx), wasm_f32x4_mul(dy, dy))));
    u = wasm_f32x4_add(ax, wasm_f32x4_mul(dx, sc)); v = wasm_f32x4_add(ay, wasm_f32x4_mul(dy, sc));
  }
  return sample4(im, u, v);
}
#endif
#endif

// Grey levels of the solid line and of the light line inside it, and the threshold from the track pairs.
// The line and the gap are the same for every rotation and mirror image of a quad, so the contrast
// alone rejects a quad (a bezel is dark where the light line should be).
#ifdef OB_NEON
// ai: arm64's form: each side's three depths taken into locals once (trk_side), four steps a vector, a last vector's
// ai: lanes past the side's end read and dropped; the sums in the steps' order
static int thin_levels(const ob_layout_t *L, const image_t *im, const homo_t *H, float *ref, float *con) {
  const float dl = 1 + OB_THIN_LINE / 2.0f;
  const trk_img_t g = trk_img(im);
  float dark = 0, light = 0, trk = 0;
  int nref = 0;
  for (int side = 0; side < 4; side++) {
    const int end = (side & 1 ? L->h : L->w) - L->reserve;
    const trk_side_t sl = trk_side(L, H, side, dl, NULL), sp = trk_side(L, H, side, OB_GAP_MID, NULL), st = trk_side(L, H, side, OB_TRACK_MID, NULL);
    for (int t = L->reserve; t < end; t += 64) {
      const float32x4_t at = { t + 0.5f, t + 16 + 0.5f, t + 32 + 0.5f, t + 48 + 0.5f };
      // The pair near each of the four points, taken through the macro for the reason TRACK_AT4 gives.
      #define PAIR_AT(k, w) ((float)(L->reserve + OB_TRACK_AT(OB_PAIR_J(t + 16 * (k) - L->reserve) + (w))) + OB_CELL_MID)
      const float32x4_t ca = { PAIR_AT(0, 0), PAIR_AT(1, 0), PAIR_AT(2, 0), PAIR_AT(3, 0) }, cb = { PAIR_AT(0, 1), PAIR_AT(1, 1), PAIR_AT(2, 1), PAIR_AT(3, 1) };
      #undef PAIR_AT
      float D[4], G[4], A[4], B[4];
      vst1q_f32(D, trk_get(&sl, &g, at)); vst1q_f32(G, trk_get(&sp, &g, at));
      vst1q_f32(A, trk_get(&st, &g, ca)); vst1q_f32(B, trk_get(&st, &g, cb));
      for (int l = 0; l < 4 && t + 16 * l < end; l++, nref++) { dark += D[l]; light += G[l]; trk += A[l]; trk += B[l]; }
    }
  }
  *ref = 0.5f * trk / nref; *con = (light - dark) / nref;
  return *con >= 0.08f;
}
#elif defined(OB_X86)
// ai: x86-64: arm64's form (above), its own floats
static int thin_levels(const ob_layout_t *L, const image_t *im, const homo_t *H, float *ref, float *con) {
  const float dl = 1 + OB_THIN_LINE / 2.0f;
  const trk_img_t g = trk_img(im);
  float dark = 0, light = 0, trk = 0;
  int nref = 0;
  for (int side = 0; side < 4; side++) {
    const int end = (side & 1 ? L->h : L->w) - L->reserve;
    const trk_side_t sl = trk_side(L, H, side, dl, NULL), sp = trk_side(L, H, side, OB_GAP_MID, NULL), st = trk_side(L, H, side, OB_TRACK_MID, NULL);
    for (int t = L->reserve; t < end; t += 64) {
      const __m128 at = _mm_setr_ps(t + 0.5f, t + 16 + 0.5f, t + 32 + 0.5f, t + 48 + 0.5f);
      #define PAIR_AT(k, w) ((float)(L->reserve + OB_TRACK_AT(OB_PAIR_J(t + 16 * (k) - L->reserve) + (w))) + OB_CELL_MID)
      const __m128 ca = _mm_setr_ps(PAIR_AT(0, 0), PAIR_AT(1, 0), PAIR_AT(2, 0), PAIR_AT(3, 0)), cb = _mm_setr_ps(PAIR_AT(0, 1), PAIR_AT(1, 1), PAIR_AT(2, 1), PAIR_AT(3, 1));
      #undef PAIR_AT
      float D[4], G[4], A[4], B[4];
      _mm_storeu_ps(D, trk_get(&sl, &g, at)); _mm_storeu_ps(G, trk_get(&sp, &g, at));
      _mm_storeu_ps(A, trk_get(&st, &g, ca)); _mm_storeu_ps(B, trk_get(&st, &g, cb));
      for (int l = 0; l < 4 && t + 16 * l < end; l++, nref++) { dark += D[l]; light += G[l]; trk += A[l]; trk += B[l]; }
    }
  }
  *ref = 0.5f * trk / nref; *con = (light - dark) / nref;
  return *con >= 0.08f;
}
#else
static int thin_levels(const ob_layout_t *L, const image_t *im, const homo_t *H, float *ref, float *con) {
  float dark = 0, light = 0, trk = 0, u, v, x, y, dl = 1 + OB_THIN_LINE / 2.0f;
  int nref = 0;
  for (int side = 0; side < 4; side++) for (int t = L->reserve; t < (side & 1 ? L->h : L->w) - L->reserve; t += 16, nref++) {
#ifdef OB_SIMD
    if (t + 48 < (side & 1 ? L->h : L->w) - L->reserve) {
      const v128_t along = wasm_f32x4_make(t + 0.5f, t + 16 + 0.5f, t + 32 + 0.5f, t + 48 + 0.5f);
      // The pair near each of the four points, taken through the macro for the reason TRACK_AT4 gives.
      #define PAIR_AT(k, w) ((float)(L->reserve + OB_TRACK_AT(OB_PAIR_J(t + 16 * (k) - L->reserve) + (w))) + OB_CELL_MID)
      const v128_t ca = wasm_f32x4_make(PAIR_AT(0, 0), PAIR_AT(1, 0), PAIR_AT(2, 0), PAIR_AT(3, 0)),
                   cb = wasm_f32x4_make(PAIR_AT(0, 1), PAIR_AT(1, 1), PAIR_AT(2, 1), PAIR_AT(3, 1));
      #undef PAIR_AT
      float D[4], G[4], A[4], B[4];
      wasm_v128_store(D, thin_sample4(L, im, H, NULL, side, along, dl));
      wasm_v128_store(G, thin_sample4(L, im, H, NULL, side, along, OB_GAP_MID));
      wasm_v128_store(A, thin_sample4(L, im, H, NULL, side, ca, OB_TRACK_MID));
      wasm_v128_store(B, thin_sample4(L, im, H, NULL, side, cb, OB_TRACK_MID));
      for (int l = 0; l < 4; l++) { dark += D[l]; light += G[l]; trk += A[l]; trk += B[l]; }
      t += 48; nref += 3;
      continue;
    }
#endif
    side_xy(L, side, t + 0.5f, dl, &x, &y);
    project(H, x, y, &u, &v); dark += sample(im, u, v);
    side_xy(L, side, t + 0.5f, OB_GAP_MID, &x, &y);
    project(H, x, y, &u, &v); light += sample(im, u, v);
    for (int k = 0; k < 2; k++) {
      const float ct = L->reserve + OB_TRACK_AT(OB_PAIR_J(t - L->reserve) + k) + OB_CELL_MID;
      side_xy(L, side, ct, OB_TRACK_MID, &x, &y);
      project(H, x, y, &u, &v); trk += sample(im, u, v);
    }
  }
  // The threshold the TRACK is read against is the track's own. A pair of track cells is one dark and one light
  // by construction (Manchester, or the clock), so the mean of a pair is the right threshold wherever it is
  // measured and whatever the blur. The old one, halfway between the solid line and the light ring, is right for
  // features as thick as those two; a track cell is one module deep, and under blur a thin dark cell with white
  // on both sides floats above it. With the picture at depth 6 its grey happened to hold the level down; with a
  // wider border there is white there, and the track stopped reading at 1.5 px of defocus (3% against 70%).
  *ref = 0.5f * trk / nref; *con = (light - dark) / nref;
  return *con >= 0.08f;
}
#endif

// How well the coded track reads as the layout says, in -0.5 .. 0.5. It validates a quad and picks
// its rotation in one measurement.
// beat: the caller keeps a quad only if it scores above this, and most of the up to 81 line combinations are
// nowhere near. A point adds at most 0.5, so once the points left cannot lift the sum past beat the rest are not
// read and beat itself is returned. A quad that is read to the end gets the same sum in the same order as ever.
#ifdef OB_NEON
// ai: arm64's form: four cells a vector, the clamped levels (by the contrast's reciprocal) summed a lane each and the
// ai: stop tested on the lanes' total every eight cells: the sum its own floats, the stop as sound (the same margin).
static float thin_score(const ob_layout_t *L, const image_t *im, const homo_t *H, float ref, float con, float beat) {
  int n = 0, total = 0;
  for (int side = 0; side < 4; side++) total += ob_track_cells(side & 1 ? L->h : L->w, L->reserve);
  const float need = (beat - 0.001f) * total;
  const float32x4_t rf = vdupq_n_f32(ref), rc = vdupq_n_f32(1 / con), hi = vdupq_n_f32(0.5f), lo = vdupq_n_f32(-0.5f);
  const trk_img_t g = trk_img(im);
  float32x4_t acc = vdupq_n_f32(0);
  for (int side = 0; side < 4; side++) {
    const int nj = ob_track_cells(side & 1 ? L->h : L->w, L->reserve);
    const trk_side_t st = trk_side(L, H, side, OB_TRACK_MID, NULL);
    const uint32_t *fl = trk_signs(side);
    for (int j = 0; j < nj; j += 4) {
      const float32x4_t q = vmulq_f32(vsubq_f32(rf, trk_get(&st, &g, vreinterpretq_f32_s32(TRACK_AT4(L, j)))), rc);
      acc = vaddq_f32(acc, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vmaxq_f32(vminq_f32(q, hi), lo)), vld1q_u32(fl + j))));
      if (!((n += 4) & 7) && vaddvq_f32(acc) + 0.5f * (total - n) < need) return beat;
    }
  }
  return vaddvq_f32(acc) / n;
}
#elif defined(OB_X86)
// ai: x86-64: arm64's form (above), its own floats
static float thin_score(const ob_layout_t *L, const image_t *im, const homo_t *H, float ref, float con, float beat) {
  int n = 0, total = 0;
  for (int side = 0; side < 4; side++) total += ob_track_cells(side & 1 ? L->h : L->w, L->reserve);
  const float need = (beat - 0.001f) * total;
  const __m128 rf = _mm_set1_ps(ref), rc = _mm_set1_ps(1 / con), hi = _mm_set1_ps(0.5f), lo = _mm_set1_ps(-0.5f);
  const trk_img_t g = trk_img(im);
  __m128 acc = _mm_setzero_ps();
  for (int side = 0; side < 4; side++) {
    const int nj = ob_track_cells(side & 1 ? L->h : L->w, L->reserve);
    const trk_side_t st = trk_side(L, H, side, OB_TRACK_MID, NULL);
    const uint32_t *fl = trk_signs(side);
    for (int j = 0; j < nj; j += 4) {
      const __m128 q = _mm_mul_ps(_mm_sub_ps(rf, trk_get(&st, &g, _mm_castsi128_ps(TRACK_AT4(L, j)))), rc);
      acc = _mm_add_ps(acc, _mm_xor_ps(_mm_max_ps(_mm_min_ps(q, hi), lo), _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)(fl + j)))));
      if (!((n += 4) & 7) && trk_sum4(acc) + 0.5f * (total - n) < need) return beat;
    }
  }
  return trk_sum4(acc) / n;
}
#else
static float thin_score(const ob_layout_t *L, const image_t *im, const homo_t *H, float ref, float con, float beat) {
  float u, v, x, y, score = 0;
  int n = 0, total = 0;
  for (int side = 0; side < 4; side++) total += ob_track_cells(side & 1 ? L->h : L->w, L->reserve);
  const float need = (beat - 0.001f) * total;   // the margin is a thousand times the sum's rounding
  for (int side = 0; side < 4; side++) for (int j = 0, nj = ob_track_cells(side & 1 ? L->h : L->w, L->reserve); j < nj; j++) {
    const int t = L->reserve + OB_TRACK_AT(j);
#ifdef OB_SIMD
    if (j + 3 < nj) {
      float Q[4];
      // The lane positions come from the macro, one per cell, never from a stride off the first (TRACK_AT4).
      const v128_t smp = thin_sample4(L, im, H, NULL, side, TRACK_AT4(L, j), OB_TRACK_MID);
      const v128_t q4 = wasm_f32x4_div(wasm_f32x4_sub(wasm_f32x4_splat(ref), smp), wasm_f32x4_splat(con));
      wasm_v128_store(Q, wasm_f32x4_pmax(wasm_f32x4_pmin(q4, wasm_f32x4_splat(0.5f)), wasm_f32x4_splat(-0.5f)));
      for (int l = 0; l < 4; l++) {
        score += ob_thin_track(side, j + l) ? Q[l] : -Q[l];
        if (!(++n & 7) && score + 0.5f * (total - n) < need) return beat;
      }
      j += 3;
      continue;
    }
#endif
    side_xy(L, side, t + OB_CELL_MID, OB_TRACK_MID, &x, &y);
    project(H, x, y, &u, &v);
    float q = (ref - sample(im, u, v)) / con;
    if (q > 0.5f) q = 0.5f; else if (q < -0.5f) q = -0.5f;
    score += ob_thin_track(side, j) ? q : -q;
    if (!(++n & 7) && score + 0.5f * (total - n) < need) return beat;
  }
  return score / n;
}
#endif

// thin_levels and the track for all eight ways round a SQUARE quad from one read, taken the way up the quad is
// (layout side k on image side k) and in BOTH directions along each side. A turn or a flip of the layout square puts
// each of its sides on one of the quad's, running with that side's modules or against them. Against is not the same
// cells backwards: since OB_BAND_RUN 4 a side starts with a track run and ends with a word run, so a side turned end
// for end has its track where the word was. Read at W - (reserve + OB_TRACK_AT(i) + OB_CELL_MID), a cell is where the
// turned layout puts it.
//   s     s[(dir * 4 + side) * npts + i]: grey of track cell i of a layout side on image side `side`, dir 1 against.
//   pair  pair[dir * 4 + side]: the Manchester pairs thin_levels takes its threshold from, summed along the side and
//         read the same two ways, so a hypothesis's threshold is the one its own map would give it.
// The line and the gap are the same all along a side, so the contrast is read once and serves every hypothesis.
// Returns it, or 0 under thin_levels' floor; *nref: the steps the pairs were summed over.
#ifdef OB_NEON
// ai: arm64's form. A step's Manchester pair, track cells OB_PAIR_J(16 m) and the next of a side (m the step), are
// ai: cells the track's own read takes in any case, at the same points, so they are not read on their own: once the
// ai: contrast has passed, each side's pairs are summed from s in the steps' order. The line and the gap four steps a
// ai: vector, a last vector's lanes past the side's end read and dropped. The pairs were 256 of a quad's 896 reads at
// ai: the 128 ring.
static float thin_read_both(const ob_layout_t *L, const image_t *im, const homo_t *H, const flens_t *lk, float *s, float *pair, int npts, int *nref) {
  const int W = L->w, end = W - L->reserve;
  const float dl = 1 + OB_THIN_LINE / 2.0f;
  const trk_img_t g = trk_img(im);
  float dark = 0, light = 0;
  int nr = 0;
  for (int side = 0; side < 4; side++) {
    const trk_side_t sl = trk_side(L, H, side, dl, lk), sp = trk_side(L, H, side, OB_GAP_MID, lk);
    for (int t = L->reserve; t < end; t += 64) {
      const float32x4_t at = { t + 0.5f, t + 16 + 0.5f, t + 32 + 0.5f, t + 48 + 0.5f };
      float D[4], G[4];
      vst1q_f32(D, trk_get(&sl, &g, at));
      vst1q_f32(G, trk_get(&sp, &g, at));
      for (int l = 0; l < 4 && t + 16 * l < end; l++, nr++) { dark += D[l]; light += G[l]; }
    }
  }
  *nref = nr;
  const float con = (light - dark) / nr;
  if (con < 0.08f) return 0;
  const float *pos = s + 8 * npts;
  for (int side = 0; side < 4; side++) {
    const trk_side_t st = trk_side(L, H, side, OB_TRACK_MID, lk);
    for (int dir = 0; dir < 2; dir++) {
      // ai: against the side, the cell at W - ct: W + ct * -1, rounded once as the subtraction is
      float *out = s + (dir * 4 + side) * npts;
      const float32x4_t w0 = vdupq_n_f32(dir ? (float)W : 0), sg = vdupq_n_f32(dir ? -1.0f : 1.0f);
      trk_row(&st, &g, pos, npts, w0, sg, out);
      float sum = 0;
      for (int t = L->reserve; t < end; t += 16) { const int j = OB_PAIR_J(t - L->reserve); sum += out[j]; sum += out[j + 1]; }
      pair[dir * 4 + side] = sum;
    }
  }
  return con;
}
#elif defined(OB_X86)
// ai: x86-64: arm64's form (above), its own floats
static float thin_read_both(const ob_layout_t *L, const image_t *im, const homo_t *H, const flens_t *lk, float *s, float *pair, int npts, int *nref) {
  const int W = L->w, end = W - L->reserve;
  const float dl = 1 + OB_THIN_LINE / 2.0f;
  const trk_img_t g = trk_img(im);
  float dark = 0, light = 0;
  int nr = 0;
  for (int side = 0; side < 4; side++) {
    const trk_side_t sl = trk_side(L, H, side, dl, lk), sp = trk_side(L, H, side, OB_GAP_MID, lk);
    for (int t = L->reserve; t < end; t += 64) {
      const __m128 at = _mm_setr_ps(t + 0.5f, t + 16 + 0.5f, t + 32 + 0.5f, t + 48 + 0.5f);
      float D[4], G[4];
      _mm_storeu_ps(D, trk_get(&sl, &g, at));
      _mm_storeu_ps(G, trk_get(&sp, &g, at));
      for (int l = 0; l < 4 && t + 16 * l < end; l++, nr++) { dark += D[l]; light += G[l]; }
    }
  }
  *nref = nr;
  const float con = (light - dark) / nr;
  if (con < 0.08f) return 0;
  const float *pos = s + 8 * npts;
  for (int side = 0; side < 4; side++) {
    const trk_side_t st = trk_side(L, H, side, OB_TRACK_MID, lk);
    for (int dir = 0; dir < 2; dir++) {
      float *out = s + (dir * 4 + side) * npts;
      const __m128 w0 = _mm_set1_ps(dir ? (float)W : 0), sg = _mm_set1_ps(dir ? -1.0f : 1.0f);
      trk_row(&st, &g, pos, npts, w0, sg, out);
      float sum = 0;
      for (int t = L->reserve; t < end; t += 16) { const int j = OB_PAIR_J(t - L->reserve); sum += out[j]; sum += out[j + 1]; }
      pair[dir * 4 + side] = sum;
    }
  }
  return con;
}
#else
static float thin_read_both(const ob_layout_t *L, const image_t *im, const homo_t *H, const flens_t *lk, float *s, float *pair, int npts, int *nref) {
  const int W = L->w;
  const float dl = 1 + OB_THIN_LINE / 2.0f;
  float dark = 0, light = 0, u, v, x, y;
  int nr = 0;
  for (int side = 0; side < 4; side++) {
    float *fw = pair + side, *bw = pair + 4 + side;
    *fw = *bw = 0;
    for (int t = L->reserve; t < W - L->reserve; t += 16, nr++) {
#ifdef OB_SIMD
      if (t + 48 < W - L->reserve) {
        const v128_t along = wasm_f32x4_make(t + 0.5f, t + 16 + 0.5f, t + 32 + 0.5f, t + 48 + 0.5f), w4 = wasm_f32x4_splat((float)W);
        #define PAIR_AT(k, w) ((float)(L->reserve + OB_TRACK_AT(OB_PAIR_J(t + 16 * (k) - L->reserve) + (w))) + OB_CELL_MID)
        const v128_t ca = wasm_f32x4_make(PAIR_AT(0, 0), PAIR_AT(1, 0), PAIR_AT(2, 0), PAIR_AT(3, 0)),
                     cb = wasm_f32x4_make(PAIR_AT(0, 1), PAIR_AT(1, 1), PAIR_AT(2, 1), PAIR_AT(3, 1));
        #undef PAIR_AT
        float D[4], G[4], A[4], B[4], RA[4], RB[4];
        wasm_v128_store(D, thin_sample4(L, im, H, lk, side, along, dl));
        wasm_v128_store(G, thin_sample4(L, im, H, lk, side, along, OB_GAP_MID));
        wasm_v128_store(A, thin_sample4(L, im, H, lk, side, ca, OB_TRACK_MID));
        wasm_v128_store(B, thin_sample4(L, im, H, lk, side, cb, OB_TRACK_MID));
        wasm_v128_store(RA, thin_sample4(L, im, H, lk, side, wasm_f32x4_sub(w4, ca), OB_TRACK_MID));
        wasm_v128_store(RB, thin_sample4(L, im, H, lk, side, wasm_f32x4_sub(w4, cb), OB_TRACK_MID));
        for (int l = 0; l < 4; l++) { dark += D[l]; light += G[l]; *fw += A[l]; *fw += B[l]; *bw += RA[l]; *bw += RB[l]; }
        t += 48; nr += 3;
        continue;
      }
#endif
      side_xy(L, side, t + 0.5f, dl, &x, &y);
      project(H, x, y, &u, &v); flens_bend(lk, &u, &v); dark += sample(im, u, v);
      side_xy(L, side, t + 0.5f, OB_GAP_MID, &x, &y);
      project(H, x, y, &u, &v); flens_bend(lk, &u, &v); light += sample(im, u, v);
      for (int k = 0; k < 2; k++) {
        const float ct = (float)(L->reserve + OB_TRACK_AT(OB_PAIR_J(t - L->reserve) + k)) + OB_CELL_MID;
        for (int dir = 0; dir < 2; dir++) {
          const float at = dir ? W - ct : ct;
          side_xy(L, side, at, OB_TRACK_MID, &x, &y);
          project(H, x, y, &u, &v); flens_bend(lk, &u, &v);
          *(dir ? bw : fw) += sample(im, u, v);
        }
      }
    }
  }
  *nref = nr;
  const float con = (light - dark) / nr;
  if (con < 0.08f) return 0;
  for (int side = 0; side < 4; side++) for (int dir = 0; dir < 2; dir++) {
    float *out = s + (dir * 4 + side) * npts;
    for (int i = 0; i < npts; i++) {
#ifdef OB_SIMD
      if (i + 3 < npts) {
        const v128_t at = dir ? wasm_f32x4_sub(wasm_f32x4_splat((float)W), TRACK_AT4(L, i)) : TRACK_AT4(L, i);
        wasm_v128_store(out + i, thin_sample4(L, im, H, lk, side, at, OB_TRACK_MID));
        i += 3;
        continue;
      }
#endif
      const float ct = (float)(L->reserve + OB_TRACK_AT(i)) + OB_CELL_MID, at = dir ? W - ct : ct;
      side_xy(L, side, at, OB_TRACK_MID, &x, &y);
      project(H, x, y, &u, &v); flens_bend(lk, &u, &v);
      out[i] = sample(im, u, v);
    }
  }
  return con;
}
#endif

#ifdef OB_NEON
// ai: arm64: find_frame's hypothesis scores from thin_read_both's reading, four or eight at once (G vectors of four),
// ai: each a vector of four sums, cell i's in lane i mod 4, added up at the end; the level by the contrast's
// ai: reciprocal, the clamp as min then max, the sign as the sign bit. row[h][sd]: hypothesis h's row of s for layout
// ai: side sd; ref[h] its threshold.
static inline __attribute__((always_inline)) void trk_scores_g(const float *s, int npts, int (*row)[4], const float *ref, float con, float *sc, const int G) {
  const float32x4_t rc = vdupq_n_f32(1 / con), hi = vdupq_n_f32(0.5f), lo = vdupq_n_f32(-0.5f);
  float32x4_t acc[8], rf[8];
  for (int h = 0; h < 4 * G; h++) { acc[h] = vdupq_n_f32(0); rf[h] = vdupq_n_f32(ref[h]); }
  for (int sd = 0; sd < 4; sd++) {
    const float *p[8];
    for (int h = 0; h < 4 * G; h++) p[h] = s + row[h][sd] * npts;
    const uint32_t *fl = trk_signs(sd);
    for (int i = 0; i < npts; i += 4) {
      const uint32x4_t f = vld1q_u32(fl + i);
      for (int h = 0; h < 4 * G; h++) {
        const float32x4_t q = vmulq_f32(vsubq_f32(rf[h], vld1q_f32(p[h] + i)), rc);
        acc[h] = vaddq_f32(acc[h], vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vmaxq_f32(vminq_f32(q, hi), lo)), f)));
      }
    }
  }
  for (int h = 0; h < 4 * G; h++) sc[h] = vaddvq_f32(acc[h]) / (float)(4 * npts);
}
// ai: hypotheses h0 to h0 + 4 G - 1 into sc[h0 ...], each threshold as the scalar loop makes it; returns h0 + 4 G
static int trk_scores(const float *s, int npts, int (*row)[4], const float *pair, int nref, float con, int h0, int G, float *sc) {
  float ref[8];
  for (int h = 0; h < 4 * G; h++) {
    float trk = 0;
    for (int sd = 0; sd < 4; sd++) trk += pair[row[h0 + h][sd]];
    ref[h] = 0.5f * trk / nref;
  }
  if (G == 2) trk_scores_g(s, npts, row + h0, ref, con, sc + h0, 2);
  else trk_scores_g(s, npts, row + h0, ref, con, sc + h0, 1);
  return h0 + 4 * G;
}
#elif defined(OB_X86)
// ai: x86-64: arm64's trk_scores_g and trk_scores (above), its own floats
static inline __attribute__((always_inline)) void trk_scores_g(const float *s, int npts, int (*row)[4], const float *ref, float con, float *sc, const int G) {
  const __m128 rc = _mm_set1_ps(1 / con), hi = _mm_set1_ps(0.5f), lo = _mm_set1_ps(-0.5f);
  __m128 acc[8], rf[8];
  for (int h = 0; h < 4 * G; h++) { acc[h] = _mm_setzero_ps(); rf[h] = _mm_set1_ps(ref[h]); }
  for (int sd = 0; sd < 4; sd++) {
    const float *p[8];
    for (int h = 0; h < 4 * G; h++) p[h] = s + row[h][sd] * npts;
    const uint32_t *fl = trk_signs(sd);
    for (int i = 0; i < npts; i += 4) {
      const __m128 f = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)(fl + i)));
      for (int h = 0; h < 4 * G; h++) {
        const __m128 q = _mm_mul_ps(_mm_sub_ps(rf[h], _mm_loadu_ps(p[h] + i)), rc);
        acc[h] = _mm_add_ps(acc[h], _mm_xor_ps(_mm_max_ps(_mm_min_ps(q, hi), lo), f));
      }
    }
  }
  for (int h = 0; h < 4 * G; h++) sc[h] = trk_sum4(acc[h]) / (float)(4 * npts);
}
static int trk_scores(const float *s, int npts, int (*row)[4], const float *pair, int nref, float con, int h0, int G, float *sc) {
  float ref[8];
  for (int h = 0; h < 4 * G; h++) {
    float trk = 0;
    for (int sd = 0; sd < 4; sd++) trk += pair[row[h0 + h][sd]];
    ref[h] = 0.5f * trk / nref;
  }
  if (G == 2) trk_scores_g(s, npts, row + h0, ref, con, sc + h0, 2);
  else trk_scores_g(s, npts, row + h0, ref, con, sc + h0, 1);
  return h0 + 4 * G;
}
#endif

// The format word (fmt.h), read after the frame has registered and before the picture is sampled.
// It goes through the mesh rather than the bare homography, so a bowed or tilted border reads as
// well as the picture does, and it is the last thing the decoder can learn without already knowing
// the picture size.
//
// Only the contrast comes from the two references thin_levels uses, the solid line and the light ring
// inside it, on a coarse pass. The threshold starts at the local track level and two means over the word's
// own cells settle the rest.
// The format read's sample PLAN, emitted by the reader itself so the two cannot drift: every position it would
// sample, in the order it samples them, plus the indices its arithmetic needs. Positions are module coordinates
// and depend only on the layout, so this is read once a configuration and not once a frame; what changes with
// the frame is the map they go through, which is the GPU's business (gpu/tables.mjs carries this plan to it).
//
//   pts   2 floats a point. refs first (2 a reference, dark then light), then each side's track pairs
//         (2 a pair), then each side's word cells.
//   meta  nref, cells, 1, nmid, nword, then npair[4], midBase[4], wordBase[4]. Slot 2 counted the copies of the word
//         while there were copies; it stays so the layout of meta does not move.
//   widx  3 ints a word cell: lo, hi (absolute into the mid array) and which soft value it is
// Returns the total number of points, or 0 where there is no word to read.
int ob_thin_fmt_plan(const ob_layout_t *L, int cells, float *pts, int32_t *meta, int32_t *widx) {
  float x, y;
  int nref = 0, np = 0;
  if (cells < 1) return 0;
  for (int side = 0; side < 4; side++) for (int t = L->reserve; t < (side & 1 ? L->h : L->w) - L->reserve; t += 16, nref++) {
    side_xy(L, side, t + 0.5f, 1 + OB_THIN_LINE / 2.0f, &x, &y); pts[2 * np] = x; pts[2 * np + 1] = y; np++;
    side_xy(L, side, t + 0.5f, OB_GAP_MID, &x, &y); pts[2 * np] = x; pts[2 * np + 1] = y; np++;
  }
  const int REACH = 12 / (2 * OB_CELL), PAIRMOD = 4 * OB_CELL;
  if (!nref) return 0;
  int npair[4], midBase[4], wordBase[4], nmid = 0;
  for (int side = 0; side < 4; side++) {
    const int len = side & 1 ? L->h : L->w;
    npair[side] = ob_track_cells(len, L->reserve) / 2;
    if (npair[side] < 1) return 0;
    midBase[side] = nmid;
    for (int pr = 0; pr < npair[side]; pr++) for (int k = 0; k < 2; k++) {
      side_xy(L, side, L->reserve + OB_TRACK_AT(2 * pr + k) + OB_CELL_MID, OB_TRACK_MID, &x, &y);
      pts[2 * np] = x; pts[2 * np + 1] = y; np++;
    }
    nmid += npair[side];
  }
  int nword = 0;
  for (int side = 0; side < 4; side++) {
    wordBase[side] = nword;
    for (int i = 0; i < cells; i++) {
      const int off = OB_WORD_AT(i);
      side_xy(L, side, L->reserve + off + OB_CELL_MID, OB_FMT_MID, &x, &y);
      pts[2 * np] = x; pts[2 * np + 1] = y; np++;
      const int at = off / PAIRMOD, lo = at - REACH < 0 ? 0 : at - REACH, hi = at + REACH >= npair[side] ? npair[side] - 1 : at + REACH;
      widx[3 * nword] = midBase[side] + lo; widx[3 * nword + 1] = midBase[side] + hi;
      widx[3 * nword + 2] = side * cells + i;
      nword++;
    }
  }
  meta[0] = nref; meta[1] = cells; meta[2] = 1; meta[3] = nmid; meta[4] = nword;
  for (int k = 0; k < 4; k++) { meta[5 + k] = npair[k]; meta[9 + k] = midBase[k]; meta[13 + k] = wordBase[k]; }
  return np;
}

// The track read's own sample plan, in the reader's order, for a reader elsewhere. Layout
// coordinates: the caller projects them through its hypothesis and samples. thin_levels takes four points a
// step (the solid line, the light ring, the Manchester pair of track cells nearest the step); thin_score one
// point a track cell with a sign from the per-side hash. Neither takes the fitted-line bow: a mark quad has no
// fitted sides, and thin_point with cv == NULL is H alone. meta: steps, track cells, w, h, cells a side x4,
// reserve. Returns the point count.
int ob_thin_score_plan(const ob_layout_t *L, float *pts, int32_t *meta, int32_t *signs) {
  float x, y;
  int np = 0, nref = 0, total = 0;
  const float dl = 1 + OB_THIN_LINE / 2.0f;
  for (int side = 0; side < 4; side++) for (int t = L->reserve; t < (side & 1 ? L->h : L->w) - L->reserve; t += 16, nref++) {
    side_xy(L, side, t + 0.5f, dl, &x, &y); pts[2 * np] = x; pts[2 * np + 1] = y; np++;
    side_xy(L, side, t + 0.5f, OB_GAP_MID, &x, &y); pts[2 * np] = x; pts[2 * np + 1] = y; np++;
    for (int k = 0; k < 2; k++) {
      const float ct = L->reserve + OB_TRACK_AT(OB_PAIR_J(t - L->reserve) + k) + OB_CELL_MID;
      side_xy(L, side, ct, OB_TRACK_MID, &x, &y); pts[2 * np] = x; pts[2 * np + 1] = y; np++;
    }
  }
  for (int side = 0; side < 4; side++) {
    const int nj = ob_track_cells(side & 1 ? L->h : L->w, L->reserve);
    meta[4 + side] = nj;
    for (int j = 0; j < nj; j++) {
      const int t = L->reserve + OB_TRACK_AT(j);
      side_xy(L, side, t + OB_CELL_MID, OB_TRACK_MID, &x, &y); pts[2 * np] = x; pts[2 * np + 1] = y;
      signs[total] = ob_thin_track(side, j) ? 1 : -1;
      np++; total++;
    }
  }
  meta[0] = nref; meta[1] = total; meta[2] = L->w; meta[3] = L->h; meta[8] = L->reserve;
  return np;
}

// One (mark quad, hypothesis) through the track read, exactly as find_frame does it, to hold a device's score
// to. centres: the four mark centres as mark_quads orders them; hyp: rotation and mirror as
// find_frame numbers them; merged: which form's mark middle the map is solved from. out: levels ok, ref, con,
// the score read to the end (no early-out), then H (9) and the registration quad corner_coords projects to (8).
int ob_test_thin_score(const ob_layout_t *L, const uint8_t *img, int w, int h, const float *centres, int hyp, int merged, float *out) {
  image_t im;
  ob_image_init(&im, img, w, h, 1.0f);
  const float m = mark_form_mid(L, merged), msrc[8] = { m, m, L->w - m, m, L->w - m, L->h - m, m, L->h - m };
  const int rot = hyp & 3, mir = hyp >> 2;
  float d2[8], ref = 0, con = 0, nc[8];
  homo_t H;
  for (int k = 0; k < 21; k++) out[k] = 0;
  out[0] = -1;
  for (int k = 0; k < 4; k++) { const int from = (rot + (mir ? 4 - k : k)) & 3; d2[2 * k] = centres[2 * from]; d2[2 * k + 1] = centres[2 * from + 1]; }
  if (!homography(msrc, d2, &H)) return 0;
  for (int k = 0; k < 9; k++) out[4 + k] = H.h[k];
  corner_coords(L, nc);
  for (int k = 0; k < 4; k++) project(&H, nc[2 * k], nc[2 * k + 1], &out[13 + 2 * k], &out[14 + 2 * k]);
  const int lv = thin_levels(L, &im, &H, &ref, &con);
  out[0] = (float)lv; out[1] = ref; out[2] = con;
  if (!lv) return 1;
  out[3] = thin_score(L, &im, &H, ref, con, -1e9f);
  return 1;
}

int ob_thin_read_fmt(const ob_layout_t *L, const image_t *im, const ob_reg_t *reg, float *q, int cells) {
  float dark = 0, light = 0, x, y, u, v;
  int nref = 0;
  if (cells < 1) return 0;
  for (int side = 0; side < 4; side++) for (int t = L->reserve; t < (side & 1 ? L->h : L->w) - L->reserve; t += 16, nref++) {
    side_xy(L, side, t + 0.5f, 1 + OB_THIN_LINE / 2.0f, &x, &y);
    ob_map_point(L, reg, x, y, &u, &v); dark += sample(im, u, v);
    side_xy(L, side, t + 0.5f, OB_GAP_MID, &x, &y);
    ob_map_point(L, reg, x, y, &u, &v); light += sample(im, u, v);
  }
  if (!nref) return 0;
  const float con = (light - dark) / nref;
  if (con < 0.08f) return 0;   // the same floor thin_levels rejects a quad on
  // Each cell is read against the track beside it, not against one level for the whole band. The band's cells are
  // the track's size, and a Manchester pair of track cells averages to the right threshold at any blur (the reason
  // thin_levels gives), but only for the light it was read under: traced on a phone capture with glare over one
  // corner, the word's levels there sat above the global threshold's dark AND light, and RS lost the word on a
  // frame that had registered to within a pixel. Three pairs either side is 28 modules, far inside any glare.
  // Reach in MODULES, not in pairs: the window has to cover the same stretch of border whatever the cell size,
  // or a smaller cell averages over less light and the threshold gets noisier exactly where the cell is weakest.
  const int REACH = 12 / (2 * OB_CELL);
  // A Manchester pair is two TRACK cells, which are cells 2p and 2p + 1 of a run, so it spans four band cells.
  const int PAIRMOD = 4 * OB_CELL;
  float *mid = malloc(((size_t)(L->w > L->h ? L->w : L->h) / PAIRMOD + 2) * sizeof(float));
  if (!mid) return 0;
  for (int side = 0; side < 4; side++) {
    const int len = side & 1 ? L->h : L->w, npair = ob_track_cells(len, L->reserve) / 2;
    if (npair < 1) { free(mid); return 0; }
    for (int pr = 0; pr < npair; pr++) {
      float s = 0;
      for (int k = 0; k < 2; k++) { side_xy(L, side, L->reserve + OB_TRACK_AT(2 * pr + k) + OB_CELL_MID, OB_TRACK_MID, &x, &y); ob_map_point(L, reg, x, y, &u, &v); s += sample(im, u, v); }
      mid[pr] = 0.5f * s;
    }
    for (int i = 0; i < cells; i++) {
      const int off = OB_WORD_AT(i);
      side_xy(L, side, L->reserve + off + OB_CELL_MID, OB_FMT_MID, &x, &y);
      ob_map_point(L, reg, x, y, &u, &v);
      const int at = off / PAIRMOD, lo = at - REACH < 0 ? 0 : at - REACH, hi = at + REACH >= npair ? npair - 1 : at + REACH;
      float m = 0;
      for (int k = lo; k <= hi; k++) m += mid[k];
      q[side * cells + i] = sample(im, u, v) - m / (hi - lo + 1);
    }
  }
  free(mid);
  // What is left is the cell against its local track level, and the word's own two populations settle the rest.
  // Under blur its cells sit a little lighter than track cells do and the track's midpoint calls the faintest
  // dark ones light. A codeword is near enough half dark for two means to find its own levels; a blank band has
  // one population, keeps the track's level and reads as the nothing it is.
  float ref = 0;
  for (int it = 0; it < 4; it++) {
    float lo = 0, hi = 0;
    int nlo = 0;
    for (int i = 0; i < 4 * cells; i++) if (q[i] < ref) { lo += q[i]; nlo++; } else hi += q[i];
    if (nlo < cells / 2 || 4 * cells - nlo < cells / 2) break;
    ref = 0.5f * (lo / nlo + hi / (4 * cells - nlo));
  }
  for (int i = 0; i < 4 * cells; i++) { const float s = (ref - q[i]) / con; q[i] = s > 0.5f ? 0.5f : s < -0.5f ? -0.5f : s; }
  return 1;
}

// The image halved, each pixel the mean of four.
// ai: fb, slot: frame_buf's (the grey pyramid's, OB_FB_PYR1 to OB_FB_PYR4)
static uint8_t *shrink2(const uint8_t *src, int w, int h, ob_fbufs_t *fb, int slot) {
  const int sw = w / 2, sh = h / 2;
  uint8_t *out = frame_buf(fb, slot, (size_t)sw * sh);
  (void)slot;
  if (!out) return NULL;
  for (int y = 0; y < sh; y++) {
    const uint8_t *a = src + (size_t)2 * y * w, *b = a + w;
    uint8_t *o = out + (size_t)y * sw;
    int x = 0;
#ifdef OB_NEON
    // ai: arm64: sixteen a step, each 2 x 2's sum by pairwise adds (UADDLP, UADALP) and (s + 2) >> 2 by a rounding
    // ai: narrowing shift (RSHRN): the same integers
    for (; x + 16 <= sw; x += 16) {
      const uint16x8_t s0 = vpadalq_u8(vpaddlq_u8(vld1q_u8(a + 2 * x)), vld1q_u8(b + 2 * x));
      const uint16x8_t s1 = vpadalq_u8(vpaddlq_u8(vld1q_u8(a + 2 * x + 16)), vld1q_u8(b + 2 * x + 16));
      vst1q_u8(o + x, vcombine_u8(vrshrn_n_u16(s0, 2), vrshrn_n_u16(s1, 2)));
    }
#elif defined(OB_X86)
    // ai: x86-64: sixteen a step, each row's pixel pairs summed by PMADDUBSW against ones (510 at most), the two rows'
    // ai: added and (s + 2) >> 2 packed back to bytes: the same integers
    const __m128i one = _mm_set1_epi8(1), two = _mm_set1_epi16(2);
#define PAIRS(p) _mm_maddubs_epi16(_mm_loadu_si128((const __m128i *)(p)), one)
    for (; x + 16 <= sw; x += 16) {
      const __m128i s0 = _mm_add_epi16(PAIRS(a + 2 * x), PAIRS(b + 2 * x)), s1 = _mm_add_epi16(PAIRS(a + 2 * x + 16), PAIRS(b + 2 * x + 16));
      _mm_storeu_si128((__m128i *)(o + x), _mm_packus_epi16(_mm_srli_epi16(_mm_add_epi16(s0, two), 2), _mm_srli_epi16(_mm_add_epi16(s1, two), 2)));
    }
#undef PAIRS
#endif
    for (; x < sw; x++) o[x] = (uint8_t)((a[2 * x] + a[2 * x + 1] + b[2 * x] + b[2 * x + 1] + 2) >> 2);
  }
  return out;
}

// A screen is seen mirrored only through a mirror, so the mirror images (hypotheses 4 to 7) are a last resort, not
// tried once a rotation holds the best. A mirror image holding it does not stop them: that is a capture through a
// mirror, and its other quads are read the same way. 0.15 is 3.4 sd of a texture read (a sign test, sd 0.5 /
// sqrt(cells)) over the 64 ring's 128 track cells, the fewest any ring has: the same significance or more on every
// ring (2026-10-04: the 32 ring's 64 cells held at 0.212 until it left the format on 2026-10-25).
static inline int rotation_holds(float best, int orient) { return best >= 0.15f && orient < 4; }

// ---------------------------------------------------------------- the finder's grey marks
// ai: The corner marks proposed from the grey picture (2026-11-24, from the front half's rewrite lab: research/build/
// ai: rewrite-2026-11-23, STATUS "The new C finder in liblizard"), in place of the binarized mark scan. On a pyramid
// ai: level (the capture's 2 x 2 means, scale s), for each box side b (3 and 4 px of the level) the mean of the b x b
// ai: box against the mean of the frame 1 px wide round it: a dark square against a lighter surround, a corner mark's
// ai: core (6 modules, the gap round it light) or, under blur, its merged square (10 modules). A candidate where the
// ai: response is the largest over the sides at its place and a local maximum within 2 px; the strongest GRAY_MAX kept,
// ai: each re-measured one level finer (gray_refine_c2f) and re-ranked by its response there; read as both forms (the
// ai: merged square and the core) through mark_pool and mark_quads as the binarized scan's were.
// ai: The response in integers, on one scale for both sides: K (frame mean - box mean) / 255 with K = 720 (the sides'
// ai: 16 and 20 frame pixels and 9 and 16 box pixels all divide it): r = cf S_f - cb S_b, cf = K / (o^2 - b^2),
// ai: cb = K / b^2, S_f and S_b the frame's and the box's sums, shifted down 2 (45,900 at most) and held at 0 below (no
// ai: candidate takes a negative response). Row sums over b and o pixels in 16 bits, the columns' running sums in 16 bits
// ai: (255 o^2 < 65,536). The peaks: a pixel equal to its 5 x 5's maximum and no earlier pixel of the 5 x 5 equal to it,
// ai: in reading order. The vector forms (dense16, peaks16) are the scalar's integers lane for lane.
// ai: Measured and dropped (the lab's README): a side of 6 at the coarsest level, the grey quantized to 4 or 2 bits, a
// ai: float form, the re-measure carried down to the capture or searched 2s + 1 px each way there, 24 and 32 candidates.
enum { GRAY_MAX = 64, GRING = 8, GQ_NEAR = 48 };
typedef struct { int b, o, c, cf, cb; } gside_t;   // ai: o = b + 2 (a frame 1 px wide), c: the box's centre from its frame's corner
static gside_t gside(int b) { gside_t s = { b, b + 2, 1 + b / 2, 0, 0 }; s.cf = 720 / (s.o * s.o - b * b); s.cb = 720 / (b * b); return s; }
static inline unsigned g_r16(int sf, int sb, const gside_t *s) {
  const int q = (s->cf * sf - s->cb * sb) >> 2;
  return q <= 0 ? 0 : (unsigned)q;
}

// ai: the dense pass: R (w x h, 16 bits) the largest response over the sides at each box's centre, 0 where no box fits.
// ai: ring: per side GRING rows of the box's and the frame's row sums (pw a row); S: per side the columns' running sums.
static void dense16(const uint8_t *px, int w, int h, const gside_t *sd, int ns, uint16_t *R, uint16_t *ring, uint16_t *S, int pw) {
  for (int cy = 0; cy < h; cy++) {
    uint16_t *Rr = R + (size_t)cy * w;
    memset(Rr, 0, (size_t)w * sizeof(uint16_t));
    for (int si = 0; si < ns; si++) {
      const gside_t *s = sd + si;
      const int b = s->b, o = s->o, nx = w - o + 1, y = cy - s->c;
      if (y < 0 || y + o > h) continue;
      uint16_t *Hb = ring + (size_t)si * 2 * GRING * pw, *Ho = Hb + (size_t)GRING * pw;
      uint16_t *Sb = S + (size_t)si * 2 * pw, *So = Sb + pw;
      // ai: the rows entering: all o at y = 0, else the newest
      for (int r = y ? y + o - 1 : 0; r < y + o; r++) {
        const uint8_t *row = px + (size_t)r * w;
        uint16_t *hb = Hb + (size_t)(r % GRING) * pw, *ho = Ho + (size_t)(r % GRING) * pw;
#ifdef OB_SIMD
        // ai: eight a step, the last vector over the one before (the same sums again)
        for (int x = 0;; x += 8) {
          if (x + 8 > nx) x = nx - 8;
          v128_t vb = wasm_u16x8_load8x8(row + x + 1);
          for (int i = 1; i < b; i++) vb = wasm_i16x8_add(vb, wasm_u16x8_load8x8(row + x + 1 + i));
          const v128_t vo = wasm_i16x8_add(vb, wasm_i16x8_add(wasm_u16x8_load8x8(row + x), wasm_u16x8_load8x8(row + x + 1 + b)));
          wasm_v128_store(hb + x, vb); wasm_v128_store(ho + x, vo);
          if (x + 8 >= nx) break;
        }
#else
        unsigned vb = 0, vo = 0;
        for (int i = 0; i < b; i++) vb += row[1 + i];
        for (int i = 0; i < o; i++) vo += row[i];
        for (int x = 0;; x++) {
          hb[x] = (uint16_t)vb; ho[x] = (uint16_t)vo;
          if (x + 1 >= nx) break;
          vb += row[x + 1 + b] - row[x + 1]; vo += row[x + o] - row[x];
        }
#endif
      }
      // ai: the columns' running sums (to whole vectors: the lanes past nx are the ring's padding, never read back)
      const int nv = (nx + 7) & ~7;
      if (!y) {
        for (int x = 0; x < nv; x++) {
          unsigned vb = 0, vo = 0;
          for (int j = 0; j < b; j++) vb += Hb[(size_t)((1 + j) % GRING) * pw + x];
          for (int j = 0; j < o; j++) vo += Ho[(size_t)(j % GRING) * pw + x];
          Sb[x] = (uint16_t)vb; So[x] = (uint16_t)vo;
        }
      } else {
        const uint16_t *bi = Hb + (size_t)((y + b) % GRING) * pw, *bo = Hb + (size_t)(y % GRING) * pw;
        const uint16_t *oi = Ho + (size_t)((y + o - 1) % GRING) * pw, *oo = Ho + (size_t)((y - 1) % GRING) * pw;
#ifdef OB_SIMD
        for (int x = 0; x < nv; x += 8) {
          wasm_v128_store(Sb + x, wasm_i16x8_add(wasm_v128_load(Sb + x), wasm_i16x8_sub(wasm_v128_load(bi + x), wasm_v128_load(bo + x))));
          wasm_v128_store(So + x, wasm_i16x8_add(wasm_v128_load(So + x), wasm_i16x8_sub(wasm_v128_load(oi + x), wasm_v128_load(oo + x))));
        }
#else
        for (int x = 0; x < nv; x++) { Sb[x] = (uint16_t)(Sb[x] + bi[x] - bo[x]); So[x] = (uint16_t)(So[x] + oi[x] - oo[x]); }
#endif
      }
      uint16_t *Rc = Rr + s->c;
#ifdef OB_SIMD
      // ai: in 32 bits, cf S_f - cb S_b, shifted down 2, narrowed with negatives to 0; the last vector over the one before
      // ai: (the maximum is idempotent)
      const v128_t cf = wasm_i32x4_splat(s->cf), mcb = wasm_i32x4_splat(-s->cb);
      for (int x = 0;; x += 8) {
        if (x + 8 > nx) x = nx - 8;
        const v128_t vb = wasm_v128_load(Sb + x), vf = wasm_i16x8_sub(wasm_v128_load(So + x), vb);
        const v128_t lo = wasm_i32x4_add(wasm_i32x4_mul(wasm_u32x4_extend_low_u16x8(vf), cf), wasm_i32x4_mul(wasm_u32x4_extend_low_u16x8(vb), mcb));
        const v128_t hi = wasm_i32x4_add(wasm_i32x4_mul(wasm_u32x4_extend_high_u16x8(vf), cf), wasm_i32x4_mul(wasm_u32x4_extend_high_u16x8(vb), mcb));
        const v128_t r = wasm_u16x8_narrow_i32x4(wasm_i32x4_shr(lo, 2), wasm_i32x4_shr(hi, 2));
        wasm_v128_store(Rc + x, wasm_u16x8_max(wasm_v128_load(Rc + x), r));
        if (x + 8 >= nx) break;
      }
#else
      for (int x = 0; x < nx; x++) { const unsigned r = g_r16((int)(So[x] - Sb[x]), (int)Sb[x], s); if (r > Rc[x]) Rc[x] = (uint16_t)r; }
#endif
    }
  }
}

// ai: the peaks: the strongest maxc into (xs, ys, pr), strongest first, ties in reading order. The vector form: each
// ai: row's 5-wide maxima kept for five rows (hm, 8 rows of w), their maximum down the column the 5 x 5's; a pixel
// ai: equal to it and at least the list's entry bar goes to the test for an earlier equal pixel and into the list
static void peak_put(int maxc, int *n, int *xs, int *ys, uint16_t *pr, int x, int y, unsigned r) {
  int j = *n < maxc ? (*n)++ : maxc - 1;
  while (j > 0 && pr[j - 1] < r) { xs[j] = xs[j - 1]; ys[j] = ys[j - 1]; pr[j] = pr[j - 1]; j--; }
  xs[j] = x; ys[j] = y; pr[j] = (uint16_t)r;
}
static int peaks16(const uint16_t *R, int w, int h, int maxc, int *xs, int *ys, uint16_t *pr, uint16_t *hm) {
  int n = 0;
#ifdef OB_SIMD
  int made = 0;
  for (int y = 2; y < h - 2; y++) {
    for (; made <= y + 2; made++) {
      const uint16_t *Rr = R + (size_t)made * w;
      uint16_t *o = hm + (size_t)(made & 7) * w;
      for (int x = 2;; x += 8) {
        if (x + 8 > w - 2) x = w - 2 - 8;
        v128_t m = wasm_v128_load(Rr + x - 2);
        for (int d = -1; d <= 2; d++) m = wasm_u16x8_max(m, wasm_v128_load(Rr + x + d));
        wasm_v128_store(o + x, m);
        if (x + 8 >= w - 2) break;
      }
    }
    const uint16_t *Rr = R + (size_t)y * w;
    int done = 2;
    for (int x = 2;; x += 8) {
      if (x + 8 > w - 2) x = w - 2 - 8;
      const unsigned bar = n == maxc ? pr[maxc - 1] + 1u : 1u;
      if (bar > 65535) break;
      v128_t m = wasm_v128_load(hm + (size_t)((y - 2) & 7) * w + x);
      for (int d = -1; d <= 2; d++) m = wasm_u16x8_max(m, wasm_v128_load(hm + (size_t)((y + d) & 7) * w + x));
      const v128_t v = wasm_v128_load(Rr + x), ge = wasm_i16x8_eq(wasm_u16x8_max(v, wasm_i16x8_splat((int16_t)bar)), v);
      unsigned mk = (unsigned)wasm_i8x16_bitmask(wasm_v128_and(wasm_i16x8_eq(v, m), ge)) & 0x5555u;
      if (x < done) mk &= ~((1u << (2 * (done - x))) - 1u);   // ai: the last vector's lanes the one before took
      while (mk) {
        const int xx = x + (__builtin_ctz(mk) >> 1);
        mk &= mk - 1;
        const int q = y * w + xx;
        const unsigned r = R[q];
        if (n == maxc && r <= pr[maxc - 1]) continue;
        int peak = 1;
        for (int dy = -2; dy <= 0 && peak; dy++) for (int dx = -2; dx <= 2 && peak; dx++) {
          if (dy == 0 && dx >= 0) break;
          if (R[q + dy * w + dx] == r) peak = 0;
        }
        if (peak) peak_put(maxc, &n, xs, ys, pr, xx, y, r);
      }
      done = x + 8;
      if (done >= w - 2) break;
    }
  }
#else
  (void)hm;
  for (int y = 2; y < h - 2; y++) for (int x = 2; x < w - 2; x++) {
    const int q = y * w + x;
    const unsigned r = R[q];
    if (!r || (n == maxc && r <= pr[maxc - 1])) continue;
    int peak = 1;
    for (int dy = -2; dy <= 2 && peak; dy++) for (int dx = -2; dx <= 2 && peak; dx++) {
      if (!dx && !dy) continue;
      const unsigned o2 = R[q + dy * w + dx];
      if (o2 > r || (o2 == r && (dy < 0 || (dy == 0 && dx < 0)))) peak = 0;
    }
    if (peak) peak_put(maxc, &n, xs, ys, pr, x, y, r);
  }
#endif
  return n;
}

// ai: The re-measure's sums: a local integral image of a level over a window, the response of the box of side B px at
// ai: top-left (X, Y) against the frame B / 3 wide round it ((frame mean - box mean) / 255), -1e9 where it leaves the
// ai: level or the window
typedef struct { uint32_t *I; size_t cap; int x0, y0, x1, y1, w, h; } gwin_t;
static int gwin_build(gwin_t *g, const uint8_t *img, int w, int h, double cx, double cy, int hw) {
  int x0 = (int)floor(cx) - hw, y0 = (int)floor(cy) - hw, x1 = (int)ceil(cx) + hw, y1 = (int)ceil(cy) + hw;
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 > w) x1 = w;
  if (y1 > h) y1 = h;
  if (x1 <= x0 || y1 <= y0) return 0;
  const int lw = x1 - x0, lh = y1 - y0;
  if ((size_t)(lw + 1) * (lh + 1) > g->cap) {
    free(g->I); g->cap = (size_t)(lw + 1) * (lh + 1); g->I = malloc(g->cap * sizeof(uint32_t));
    if (!g->I) { g->cap = 0; return 0; }
  }
  uint32_t *I = g->I;
  for (int x = 0; x <= lw; x++) I[x] = 0;
  for (int y = 0; y < lh; y++) {
    uint32_t row = 0;
    const uint8_t *p = img + (size_t)(y0 + y) * w + x0;
    uint32_t *o = I + (size_t)(y + 1) * (lw + 1), *u = o - (lw + 1);
    o[0] = 0;
    int x = 0;
#ifdef OB_SIMD
    // ai: eight a step: the eight bytes' running sums in 16-bit lanes (three shifted adds, 2,040 at most), widened, plus
    // ai: the row's sum before them (carry, every lane) and the row above: the scalar's integers
    const v128_t z = wasm_i32x4_splat(0);
    v128_t carry = z;
    for (; x + 8 <= lw; x += 8) {
      v128_t v = wasm_u16x8_load8x8(p + x);
      v = wasm_i16x8_add(v, wasm_i8x16_shuffle(z, v, 0, 1, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29));
      v = wasm_i16x8_add(v, wasm_i8x16_shuffle(z, v, 0, 1, 2, 3, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27));
      v = wasm_i16x8_add(v, wasm_i8x16_shuffle(z, v, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23));
      const v128_t rl = wasm_i32x4_add(wasm_u32x4_extend_low_u16x8(v), carry), rh = wasm_i32x4_add(wasm_u32x4_extend_high_u16x8(v), carry);
      wasm_v128_store(o + x + 1, wasm_i32x4_add(rl, wasm_v128_load(u + x + 1)));
      wasm_v128_store(o + x + 5, wasm_i32x4_add(rh, wasm_v128_load(u + x + 5)));
      carry = wasm_i8x16_shuffle(rh, rh, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15);
    }
    row = (uint32_t)wasm_i32x4_extract_lane(carry, 0);
#endif
    for (; x < lw; x++) { row += p[x]; o[x + 1] = u[x + 1] + row; }
  }
  g->x0 = x0; g->y0 = y0; g->x1 = x1; g->y1 = y1; g->w = w; g->h = h;
  return 1;
}
static double gray_resp(const gwin_t *g, int X, int Y, int B) {
  const int Wd = B / 3 > 1 ? B / 3 : 1, O = B + 2 * Wd, W1 = g->x1 - g->x0 + 1;
  if (X - Wd < 0 || Y - Wd < 0 || X - Wd + O > g->w || Y - Wd + O > g->h || B < 2) return -1e9;
  if (X - Wd < g->x0 || Y - Wd < g->y0 || X - Wd + O > g->x1 || Y - Wd + O > g->y1) return -1e9;
  X -= g->x0; Y -= g->y0;
  const uint32_t *I0 = g->I;
  const double ai = (double)B * B, ao = (double)O * O - ai;
  const double sin = (double)I0[(Y + B) * W1 + X + B] - I0[Y * W1 + X + B] - I0[(Y + B) * W1 + X] + I0[Y * W1 + X];
  const double sout = (double)I0[(Y - Wd + O) * W1 + X - Wd + O] - I0[(Y - Wd) * W1 + X - Wd + O] - I0[(Y - Wd + O) * W1 + X - Wd] + I0[(Y - Wd) * W1 + X - Wd];
  return ((sout - sin) / ao - sin / ai) / 255.0;
}
// ai: the vertex's offset from c's place, held within a pixel: where c is not the largest of the three the parabola
// ai: extrapolates, by tens of pixels where it is nearly flat
static double parab(double l, double c, double r) {
  const double d = l - 2 * c + r, v = l > -1e8 && r > -1e8 && d < 0 ? 0.5 * (l - r) / d : 0;
  return v > 1 ? 1 : v < -1 ? -1 : v;
}
// ai: the fractions at one level (scale 1 there): the centre a pixel each way, then the side a pixel either way, each
// ai: with its parabola, then the centre again; unit comes back the side found, rc the response
static void gray_refine(const gwin_t *g, cand_t *c, float *rc) {
  int B = (int)lroundf(c->unit);
  double cx = c->x, cy = c->y, best = -1e9;
  for (int round = 0; round < 2; round++) {
    const int x0 = (int)lround(cx - B / 2.0), y0 = (int)lround(cy - B / 2.0);
    int bx = x0, by = y0;
    best = -1e9;
    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
      const double r = gray_resp(g, x0 + dx, y0 + dy, B);
      if (r > best) { best = r; bx = x0 + dx; by = y0 + dy; }
    }
    if (best <= -1e8) break;
    cx = bx + B / 2.0 + parab(gray_resp(g, bx - 1, by, B), best, gray_resp(g, bx + 1, by, B));
    cy = by + B / 2.0 + parab(gray_resp(g, bx, by - 1, B), best, gray_resp(g, bx, by + 1, B));
    if (round) break;
    int bb = B;
    double bs = -1e9, rs[3];
    const int lo = B - 1 > 3 ? B - 1 : 3, hi = B + 1;
    for (int q = lo; q <= hi; q++) {
      const double r = gray_resp(g, (int)lround(cx - q / 2.0), (int)lround(cy - q / 2.0), q);
      rs[q - lo] = r;
      if (r > bs) { bs = r; bb = q; }
    }
    B = bb;
    c->unit = (float)(bb + (bb > lo && bb < hi ? parab(rs[bb - 1 - lo], bs, rs[bb + 1 - lo]) : 0));
  }
  if (best <= -1e8) return;
  c->x = (float)cx; c->y = (float)cy; *rc = (float)best;
}
// ai: the centre and the side climbed together on one level: the best of the 27 neighbours (the centre a pixel each way
// ai: and the side a pixel either way at once) until the middle is best, three moves at most. Each alone sat in a local
// ai: best a pixel or two off a mark's middle where growing the box alone read worse (2026-11-23: a 64-ring corner mark
// ai: of 43 px, between the levels' 32 and 48, r 0.30 there and 0.52 at its middle and side)
static void gray_climb(const gwin_t *g, double *cx, double *cy, int *B) {
  for (int it = 0; it < 3; it++) {
    double best = gray_resp(g, (int)lround(*cx - *B / 2.0), (int)lround(*cy - *B / 2.0), *B), bx = *cx, by = *cy;
    int bb = *B;
    for (int db = -1; db <= 1; db++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
      const int q = *B + db;
      if ((!db && !dx && !dy) || q < 3) continue;
      const double x = *cx + dx, y = *cy + dy, r = gray_resp(g, (int)lround(x - q / 2.0), (int)lround(y - q / 2.0), q);
      if (r > best) { best = r; bx = x; by = y; bb = q; }
    }
    if (bb == *B && bx == *cx && by == *cy) break;
    *cx = bx; *cy = by; *B = bb;
  }
}
// ai: The re-measure, coarse to fine: a candidate found at level L (scale 2^L) measured again one level finer (lv[L - 1];
// ai: lv[0] the capture): its centre and side doubled, climbed (gray_climb), then gray_refine there for the fractions,
// ai: back in the capture's pixels; a candidate found on the capture measured there
static void gray_refine_c2f(gwin_t *g, const uint8_t *const *lv, int w, int h, cand_t *c, float *rc, int n, int scale) {
  int L = 0;
  while ((1 << L) < scale) L++;
  const int ls = L > 0 ? L - 1 : 0;
  for (int k = 0; k < n; k++) {
    double cx = c[k].x / scale, cy = c[k].y / scale;
    int B = (int)lroundf(c[k].unit / scale);
    if (L > 0) { cx *= 2; cy *= 2; B *= 2; }
    const int wl = w >> ls, hl = h >> ls, Bm = B + 3, Wm = Bm / 3 > 1 ? Bm / 3 : 1, hw = (Bm + 2 * Wm) / 2 + 3 + 10;
    if (!gwin_build(g, lv[ls], wl, hl, cx, cy, hw)) continue;
    gray_climb(g, &cx, &cy, &B);
    cand_t t = c[k];
    t.x = (float)cx; t.y = (float)cy; t.unit = (float)B;
    float r = rc[k];
    gray_refine(g, &t, &r);
    t.x *= (float)(1 << ls); t.y *= (float)(1 << ls); t.unit *= (float)(1 << ls);
    c[k] = t; rc[k] = r;
  }
}

// ai: the scales the grey marks search, coarse to fine: from the first whose boxes (3 and 4 px of the level) hold the
// ai: largest core a capture this size can show (the fewest modules a side, the 64 ring's 158, filling its shorter side:
// ai: a core of 6 side / 158 px), down to 2 (the capture's 2 x 2 means; a candidate found there is measured on the
// ai: capture itself); a level under 64 px a side left out. Scale 1 searched as well cost the empty frames 0.8 ms and
// ai: read 71 more blocks of 31,658 (2026-11-23).
static int gray_levels(int w, int h, int *lv) {
  const int side = w < h ? w : h;
  int smax = 1, k = 0;
  while (smax < 16 && 6 * smax * 158 < 6 * side) smax *= 2;
  for (int s = smax; s >= 2; s /= 2) if (w / s >= 64 && h / s >= 64) lv[k++] = s;
  return k;
}

// ai: The finder's workspace, a frame's: the grey pyramid (pyr[l], the capture's 2 x 2 means l times; pyr[0] the capture)
// ai: and the grey stage's buffers, in frame_buf's slots
typedef struct { ob_fbufs_t *fb; const uint8_t *pyr[5]; int w, h; gwin_t win; } fctx_t;
static const uint8_t *fctx_level(fctx_t *f, int l) {
  for (int k = 1; k <= l; k++)
    if (!f->pyr[k] && !(f->pyr[k] = shrink2(f->pyr[k - 1], f->w >> (k - 1), f->h >> (k - 1), f->fb, OB_FB_PYR1 + k - 1))) return NULL;
  return f->pyr[l];
}
static void fctx_done(fctx_t *f) {
  for (int k = 1; k < 5; k++) if (f->pyr[k]) frame_buf_done(f->fb, (void *)f->pyr[k]);
  free(f->win.I);
}
// ai: the candidates at scale s (a level of the pyramid): the dense pass, the peaks, each peak's side (the first side,
// ai: 3 then 4, whose response there is the peak's: the dense pass keeps the first on a tie) and its place (a level
// ai: pixel i spans [i s, (i + 1) s) of the capture; for an even side the box's middle is a pixel boundary), then the
// ai: re-measure and the ranking by the response measured; unit the box side in capture px, rc the response
static int gray_cands(fctx_t *f, int scale, cand_t *c, float *rc) {
  int li = 0;
  while ((1 << li) < scale) li++;
  const uint8_t *px = fctx_level(f, li);
  const int w = f->w >> li, h = f->h >> li;
  if (!px || w < 64 || h < 64) return 0;
  const int pw = ((w + 7) & ~7) + 16;
  uint16_t *R = frame_buf(f->fb, OB_FB_GREY, (size_t)w * h * sizeof(uint16_t));
  uint16_t *ring = frame_buf(f->fb, OB_FB_GRING, ((size_t)2 * 2 * GRING * pw + (size_t)2 * 2 * pw + (size_t)8 * w) * sizeof(uint16_t));
  if (!R || !ring) { frame_buf_done(f->fb, R); frame_buf_done(f->fb, ring); return 0; }
  uint16_t *S = ring + (size_t)2 * 2 * GRING * pw, *hm = S + (size_t)2 * 2 * pw;
  memset(ring, 0, ((size_t)2 * 2 * GRING * pw + (size_t)2 * 2 * pw) * sizeof(uint16_t));
  const gside_t sd[2] = { gside(3), gside(4) };
  dense16(px, w, h, sd, 2, R, ring, S, pw);
  int xs[GRAY_MAX], ys[GRAY_MAX];
  uint16_t pr[GRAY_MAX];
  const int n = peaks16(R, w, h, GRAY_MAX, xs, ys, pr, hm);
  const float norm = 4.0f / (720.0f * 255.0f);
  for (int j = 0; j < n; j++) {
    int b = 3;
    for (int si = 0; si < 2; si++) {
      const gside_t *s = sd + si;
      const int X = xs[j] - s->c, Y = ys[j] - s->c;
      if (X < 0 || Y < 0 || X + s->o > w || Y + s->o > h) continue;
      int sb = 0, so = 0;
      for (int yy = 0; yy < s->o; yy++) for (int xx = 0; xx < s->o; xx++) {
        const int v = px[(size_t)(Y + yy) * w + X + xx];
        so += v;
        if (yy >= 1 && yy < 1 + s->b && xx >= 1 && xx < 1 + s->b) sb += v;
      }
      if (g_r16(so - sb, sb, s) == pr[j]) { b = s->b; break; }
    }
    c[j].x = (float)((xs[j] - b / 2 + b / 2.0) * scale); c[j].y = (float)((ys[j] - b / 2 + b / 2.0) * scale);
    c[j].unit = (float)(b * scale); c[j].hits = 0; rc[j] = pr[j] * norm;
  }
  frame_buf_done(f->fb, R); frame_buf_done(f->fb, ring);
  gray_refine_c2f(&f->win, f->pyr, f->w, f->h, c, rc, n, scale);
  for (int a = 1; a < n; a++) {
    const cand_t cc = c[a];
    const float rr = rc[a];
    int b2 = a - 1;
    while (b2 >= 0 && rc[b2] < rr) { c[b2 + 1] = c[b2]; rc[b2 + 1] = rc[b2]; b2--; }
    c[b2 + 1] = cc; rc[b2 + 1] = rr;
  }
  return n;
}
// ai: the candidates read as a form, 1 the merged square (10 modules), 2 the core (6), their support the response over
// ai: the module size, so that mark_pool's rank (support times size) is the response
static void gray_form(const cand_t *c, const float *rc, int n, int form, cand_t *out) {
  for (int k = 0; k < n; k++) {
    out[k] = c[k];
    out[k].unit = c[k].unit * (form == 1 ? 1.0f / 10.0f : 1.0f / 6.0f);
    out[k].hits = (int)(rc[k] * 1e5f / (out[k].unit > 0 ? out[k].unit : 1));
  }
}

// ai: The track's contrast gate over the cells within GQ_NEAR modules of a side's either end (thin_levels' samples
// ai: there): where the four marks' homography still holds the border when it bows between them (the 256 ring: within
// ai: 0.4 modules for 35 modules from a corner, 1.8 at mid-side, 2026-11-23); a quad's score is still thin_score's,
// ai: the whole side
static inline int gq_near_end(float t, int len) { return t <= GQ_NEAR || len - t <= GQ_NEAR; }
static int thin_levels_near(const ob_layout_t *L, const image_t *im, const homo_t *H, float *ref, float *con) {
  float dark = 0, light = 0, trk = 0, u, v, x, y, dl = 1 + OB_THIN_LINE / 2.0f;
  int nref = 0;
  for (int side = 0; side < 4; side++) {
    const int len = side & 1 ? L->h : L->w;
    for (int t = L->reserve; t < len - L->reserve; t += 16) {
      if (!gq_near_end(t + 0.5f, len)) continue;
      side_xy(L, side, t + 0.5f, dl, &x, &y); project(H, x, y, &u, &v); dark += sample(im, u, v);
      side_xy(L, side, t + 0.5f, OB_GAP_MID, &x, &y); project(H, x, y, &u, &v); light += sample(im, u, v);
      for (int k = 0; k < 2; k++) {
        const float ct = L->reserve + OB_TRACK_AT(OB_PAIR_J(t - L->reserve) + k) + OB_CELL_MID;
        side_xy(L, side, ct, OB_TRACK_MID, &x, &y); project(H, x, y, &u, &v); trk += sample(im, u, v);
      }
      nref++;
    }
  }
  if (!nref) return 0;
  *ref = 0.5f * trk / nref; *con = (light - dark) / nref;
  return *con >= 0.08f;
}

// ---------------------------------------------------------------- the finder's traced border
// ai: The border found by tracing its ring (2026-11-24, from the rewrite lab; the ring one closed loop through the marks
// ai: since 2026-11-23): contours of f = grey - T, T the local mean (the capture's 2 x 2 means three times, 8 px a cell,
// ai: read bilinearly), followed cell by cell by marching squares, so no binarized picture is made. Each cell's code (its
// ai: corners' signs, top-left 1, top-right 2, bottom-right 4, bottom-left 8) is made a tile of TR_TILE x TR_TILE cells
// ai: at a time the first time a contour enters it, and a step is a byte read and a table step (TR_EXIT). Traces start
// ai: where a row scanned every h / 12 (a symbol at least TR_MIN of the picture tall is crossed by them) crosses zero
// ai: with at least the contrast floor across the pixel pair, and follow the iso-line until it closes or leaves the
// ai: picture (then the other way from the start). A contour the size of a symbol (its box TR_MIN of the picture each
// ai: way) becomes a quad by its straight runs (tr_segs): a closed one, or an open one with both ends at the picture's
// ai: edge (a symbol the picture cuts). The mean at 8 px rather than 16: at steep angles the far side's line, under a
// ai: pixel, read above a 16-px mean that took in the dark data (2026-11-23, the 256 ring at 50 degrees).
enum { TR_TILE = 32, TR_MAXP = 1 << 15, TR_MAXQ = 64, TR_LEVEL = 3, TR_ROWS = 12 };
// ai: the least share of the picture each way a traced symbol takes (smaller symbols are the marks' to find). TR_ROWS,
// ai: TR_MIN, the contrast floor's 0.35 and the quads' spacing (0.65) and size (4x) from a linear scan of the finder's
// ai: constants on the closed-ring sets and the lens set (2026-11-24): the blocks level, the finder 10 to 15% faster
static const float TR_MIN = 0.15f;
typedef struct { fline_t in[4]; } tquad_t;   // ai: the image's top, right, bottom and left curves
typedef struct {
  const uint8_t *img; int w, h;
  const uint8_t *lv; int lw, lh, ls;   // ai: the mean's level (2^ls px a cell)
  const int32_t *xi; const float *ax;  // ai: a column's two level samples and the weight between them
  const int32_t *yi; const float *ay;  // ai: and a row's
  const float *Ly; int lw1;            // ai: every row's level rows blended (h x lw1)
  uint8_t *cc, *tdone; int tw;         // ai: the cells' codes; a tile's codes made this frame
  uint8_t *vis;                        // ai: a bit a cell, its contour followed
  int32_t step[64], dir[4];            // ai: the chase's steps (tr_steps): by code << 2 | entry edge, and by exit edge
} tctx_t;
static signed char TR_EXIT[16][4];   // ai: the edge a code leaves by from the one it was entered across; -1 none, -2 a saddle
static void tr_exit_init(void) {
  static int done = 0;   // ai: the table's values never change: a race writes the same bytes
  if (done) return;
  for (int code = 0; code < 16; code++) {
    const int sa = code & 1, sb = code >> 1 & 1, sc = code >> 2 & 1, sd = code >> 3 & 1;
    const int cross[4] = { sa != sb, sb != sc, sd != sc, sa != sd }, nc = cross[0] + cross[1] + cross[2] + cross[3];
    for (int e = 0; e < 4; e++) {
      if (!cross[e]) { TR_EXIT[code][e] = -1; continue; }
      if (nc == 4) { TR_EXIT[code][e] = -2; continue; }
      for (int o = 0; o < 4; o++) if (o != e && cross[o]) TR_EXIT[code][e] = (signed char)o;
    }
  }
  done = 1;
}
// ai: the mean at a pixel, bilinear on the level, clamped at its outer samples (each column's and row's sample and weight
// ai: tabled once a frame: xi, ax, yi, ay)
static inline float tr_T(const tctx_t *c, int x, int y) {
  const float ax = c->ax[x], ay = c->ay[y];
  const uint8_t *r0 = c->lv + (size_t)c->yi[y] * c->lw + c->xi[x], *r1 = r0 + c->lw;
  return (1 - ay) * ((1 - ax) * r0[0] + ax * r0[1]) + ay * ((1 - ax) * r1[0] + ax * r1[1]);
}
static inline float tr_f(const tctx_t *c, int x, int y) { return (float)c->img[(size_t)y * c->w + x] - tr_T(c, x, y); }
// ai: row y's level rows blended into one (Ly, lw + 1 values: the local mean down the column at that row)
static void tr_ly(const tctx_t *c, int y, float *Ly) {
  const float ay = c->ay[y];
  const uint8_t *r0 = c->lv + (size_t)c->yi[y] * c->lw, *r1 = r0 + c->lw;
  int j = 0;
#ifdef OB_SIMD
  // ai: sixteen a step, the bytes widened to floats exactly: the scalar's floats
  const v128_t wa = wasm_f32x4_splat(1 - ay), wb = wasm_f32x4_splat(ay);
  for (; j + 16 <= c->lw; j += 16) {
    const v128_t p0 = wasm_v128_load(r0 + j), p1 = wasm_v128_load(r1 + j);
    const v128_t l0 = wasm_u16x8_extend_low_u8x16(p0), h0 = wasm_u16x8_extend_high_u8x16(p0), l1 = wasm_u16x8_extend_low_u8x16(p1), h1 = wasm_u16x8_extend_high_u8x16(p1);
    const v128_t g0[4] = { wasm_u32x4_extend_low_u16x8(l0), wasm_u32x4_extend_high_u16x8(l0), wasm_u32x4_extend_low_u16x8(h0), wasm_u32x4_extend_high_u16x8(h0) };
    const v128_t g1[4] = { wasm_u32x4_extend_low_u16x8(l1), wasm_u32x4_extend_high_u16x8(l1), wasm_u32x4_extend_low_u16x8(h1), wasm_u32x4_extend_high_u16x8(h1) };
    for (int g = 0; g < 4; g++)
      wasm_v128_store(Ly + j + 4 * g, wasm_f32x4_add(wasm_f32x4_mul(wa, wasm_f32x4_convert_i32x4(g0[g])), wasm_f32x4_mul(wb, wasm_f32x4_convert_i32x4(g1[g]))));
  }
#endif
  for (; j < c->lw; j++) Ly[j] = (1 - ay) * r0[j] + ay * r1[j];
  Ly[c->lw] = Ly[c->lw - 1];
}
// ai: a tile's codes: its rows' signs from grey and the blended mean (the edgelets), a sign a byte as a mask (0 or 0xff),
// ai: then each cell's code from its four corners' masks (1, 2, 4, 8 kept of each), so the code is ANDs and ORs. The
// ai: vector form (2026-11-24): the signs 16 pixels a step, each aligned four sharing the level's pair of samples at the
// ai: 1/8 level (the mean a + ax (b - a) with one a and b: the same floats as the scalar's), the compares narrowed to
// ai: bytes; the codes 16 cells a step. The ends (the clamped columns, a picture's last tile) by the scalar form.
static inline uint8_t tr_sign(const tctx_t *c, const float *Ly, const uint8_t *row, int x) {
  const float a = Ly[c->xi[x]], b = Ly[c->xi[x] + 1];
  return (float)row[x] > a + c->ax[x] * (b - a) ? 0xff : 0;
}
static void tr_tile(tctx_t *c, int tx, int ty) {
  const int w = c->w, h = c->h, x0 = tx * TR_TILE, y0 = ty * TR_TILE;
  const int x1 = x0 + TR_TILE + 1 < w ? x0 + TR_TILE + 1 : w, y1 = y0 + TR_TILE + 1 < h ? y0 + TR_TILE + 1 : h, nw = x1 - x0;
  uint8_t sg[2][TR_TILE + 16], *sp = sg[0], *sn = sg[1];
  for (int y = y0; y < y1; y++) {
    const float *Ly = c->Ly + (size_t)y * c->lw1;
    const uint8_t *row = c->img + (size_t)y * w;
    int x = x0;
#ifdef OB_SIMD
    const v128_t ones = wasm_i32x4_splat(-1);
    for (; x + 16 <= x1; x += 16) {
      int same = 1;
      for (int g = 0; g < 4; g++) same &= c->xi[x + 4 * g] == c->xi[x + 4 * g + 3];
      if (!same) { for (int k = 0; k < 16; k++) sn[x + k - x0] = tr_sign(c, Ly, row, x + k); continue; }
      const v128_t px = wasm_v128_load(row + x), lo = wasm_u16x8_extend_low_u8x16(px), hi = wasm_u16x8_extend_high_u8x16(px);
      const v128_t gv[4] = { wasm_u32x4_extend_low_u16x8(lo), wasm_u32x4_extend_high_u16x8(lo), wasm_u32x4_extend_low_u16x8(hi), wasm_u32x4_extend_high_u16x8(hi) };
      v128_t gt[4];
      for (int g = 0; g < 4; g++) {
        const int j = c->xi[x + 4 * g];
        const float a = Ly[j], d = Ly[j + 1] - a;
        const v128_t T = wasm_f32x4_add(wasm_f32x4_splat(a), wasm_f32x4_mul(wasm_v128_load(c->ax + x + 4 * g), wasm_f32x4_splat(d)));
        gt[g] = wasm_v128_xor(wasm_f32x4_le(wasm_f32x4_convert_i32x4(gv[g]), T), ones);   // ai: grey > the mean, as !(<=): no NaN here
      }
      wasm_v128_store(sn + (x - x0), wasm_i8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(gt[0], gt[1]), wasm_i16x8_narrow_i32x4(gt[2], gt[3])));
    }
#endif
    for (; x < x1; x++) sn[x - x0] = tr_sign(c, Ly, row, x);
    if (y > y0) {
      uint8_t *out = c->cc + (size_t)(y - 1) * w + x0;
      int k = 0;
#ifdef OB_SIMD
      const v128_t m1 = wasm_u8x16_splat(1), m2 = wasm_u8x16_splat(2), m4 = wasm_u8x16_splat(4), m8 = wasm_u8x16_splat(8);
      for (; k + 16 < nw; k += 16)
        wasm_v128_store(out + k, wasm_v128_or(wasm_v128_or(wasm_v128_and(wasm_v128_load(sp + k), m1), wasm_v128_and(wasm_v128_load(sp + k + 1), m2)),
                                              wasm_v128_or(wasm_v128_and(wasm_v128_load(sn + k + 1), m4), wasm_v128_and(wasm_v128_load(sn + k), m8))));
#endif
      for (; k + 1 < nw; k++) out[k] = (uint8_t)((sp[k] & 1) | (sp[k + 1] & 2) | (sn[k + 1] & 4) | (sn[k] & 8));
    }
    uint8_t *t = sp; sp = sn; sn = t;
  }
  c->tdone[(size_t)ty * c->tw + tx] = 1;
}
static const int TR_DX[4] = { 0, 1, 0, -1 }, TR_DY[4] = { -1, 0, 1, 0 };
// ai: The chase's step table, a frame's (its offsets the row length's): for a cell's code and its entry edge, the next
// ai: cell's offset in cells times 16, the exit edge times 4 and the next cell's entry edge (the exit's opposite), so a
// ai: step is the code's load and this one; TR_STOP where the iso-line does not cross the cell, TR_SADDLE where the
// ai: centre decides (dir[exit] then). The walk the table-free form's, cell for cell.
enum { TR_STOP = INT32_MIN, TR_SADDLE = INT32_MIN + 1 };
static void tr_steps(tctx_t *c) {
  for (int e = 0; e < 4; e++) c->dir[e] = (TR_DY[e] * c->w + TR_DX[e]) * 16 + e * 4 + ((e + 2) & 3);
  for (int code = 0; code < 16; code++) for (int ein = 0; ein < 4; ein++) {
    const int e = TR_EXIT[code][ein];
    c->step[code << 2 | ein] = e == -1 ? TR_STOP : e == -2 ? TR_SADDLE : c->dir[e];
  }
}
// ai: the chase from cell (x, y) entered across ein (0 top, 1 right, 2 bottom, 3 left): each step's cell (xy, y << 16 | x)
// ai: and exit edge (eo), the box; 1 where it closed, 0 where it left the picture or ran out of room. The cell's index
// ai: steps by the table (tr_steps), so a step waits on two loads; x and y follow beside it for the edge, tiles and box.
static int tr_chase(tctx_t *c, int x, int y, int ein, uint32_t *xy, uint8_t *eo, int *n, int max, int *bx) {
  const int x0 = x, y0 = y, e0 = ein, w = c->w, tw = c->tw;
  const unsigned wm = (unsigned)(w - 1), hm = (unsigned)(c->h - 1);
  const uint8_t *cc = c->cc, *tdone = c->tdone;
  uint8_t *vis = c->vis;
  int k = *n, b0 = bx[0], b1 = bx[1], b2 = bx[2], b3 = bx[3], closed = 0;
  ptrdiff_t i = (ptrdiff_t)y * w + x;
  for (int steps = 0; steps < max; steps++) {
    if ((unsigned)x >= wm || (unsigned)y >= hm) break;
    if (!tdone[(size_t)((unsigned)y / TR_TILE) * tw + (unsigned)x / TR_TILE]) tr_tile(c, x / TR_TILE, y / TR_TILE);
    int32_t st = c->step[cc[i] << 2 | ein];
    if (st == TR_STOP) break;
    if (st == TR_SADDLE) {   // ai: a saddle: the centre's sign says which corners join
      const float a = tr_f(c, x, y), b = tr_f(c, x + 1, y), cv = tr_f(c, x + 1, y + 1), d = tr_f(c, x, y + 1);
      static const int PAIR_AC[4] = { 1, 0, 3, 2 }, PAIR_BD[4] = { 3, 2, 1, 0 };
      st = c->dir[((a + b + cv + d) > 0) == (a > 0) ? PAIR_AC[ein] : PAIR_BD[ein]];
    }
    const int eout = st >> 2 & 3;
    vis[i >> 3] |= (uint8_t)(1u << (i & 7));
    if (k < max) { xy[k] = (uint32_t)y << 16 | (uint32_t)x; eo[k] = (uint8_t)eout; k++; }
    b0 = x < b0 ? x : b0; b1 = x > b1 ? x : b1; b2 = y < b2 ? y : b2; b3 = y > b3 ? y : b3;
    i += st >> 4; x += TR_DX[eout]; y += TR_DY[eout]; ein = st & 3;
    if (x == x0 && y == y0 && ein == e0) { closed = 1; break; }
  }
  *n = k; bx[0] = b0; bx[1] = b1; bx[2] = b2; bx[3] = b3;
  return closed;
}
// ai: a chased contour's crossing points (image coordinates, x and y apart), from f at each exit edge's two corners
static void tr_pts(const tctx_t *c, const uint32_t *xy, const uint8_t *eo, int n, float *qx, float *qy) {
  for (int k = 0; k < n; k++) {
    const int x = (int)(xy[k] & 0xffff), y = (int)(xy[k] >> 16);
    float px, py;
    switch (eo[k]) {
      case 0: { const float a = tr_f(c, x, y), b = tr_f(c, x + 1, y); px = x + a / (a - b); py = (float)y; break; }
      case 1: { const float b = tr_f(c, x + 1, y), d = tr_f(c, x + 1, y + 1); px = (float)(x + 1); py = y + b / (b - d); break; }
      case 2: { const float d = tr_f(c, x, y + 1), e = tr_f(c, x + 1, y + 1); px = x + d / (d - e); py = (float)(y + 1); break; }
      default: { const float a = tr_f(c, x, y), d = tr_f(c, x, y + 1); px = (float)x; py = y + a / (a - d); break; }
    }
    qx[k] = px + 0.5f; qy[k] = py + 0.5f;
  }
}
// ai: the length of (x, y) as glibc's hypotf makes it, from double and rounded once to float: one value on every target
// ai: (bionic's hypotf is an ulp away in about 7% of pairs) and the same in the vector form below
static inline float tr_hyp(float x, float y) { return (float)sqrt((double)x * x + (double)y * y); }
#if defined(OB_NEON)
static inline v128_t tr_hyp4(v128_t x, v128_t y) {
  const float32x4_t fx = vreinterpretq_f32_s32(x), fy = vreinterpretq_f32_s32(y);
  const float64x2_t xl = vcvt_f64_f32(vget_low_f32(fx)), xh = vcvt_high_f64_f32(fx), yl = vcvt_f64_f32(vget_low_f32(fy)), yh = vcvt_high_f64_f32(fy);
  const float64x2_t sl = vsqrtq_f64(vaddq_f64(vmulq_f64(xl, xl), vmulq_f64(yl, yl))), sh = vsqrtq_f64(vaddq_f64(vmulq_f64(xh, xh), vmulq_f64(yh, yh)));
  return vreinterpretq_s32_f32(vcvt_high_f32_f64(vcvt_f32_f64(sl), sh));
}
#define TR_HYP4 1
#elif defined(OB_X86)
static inline v128_t tr_hyp4(v128_t x, v128_t y) {
  const __m128 fx = _mm_castsi128_ps(x), fy = _mm_castsi128_ps(y);
  const __m128d xl = _mm_cvtps_pd(fx), xh = _mm_cvtps_pd(_mm_movehl_ps(fx, fx)), yl = _mm_cvtps_pd(fy), yh = _mm_cvtps_pd(_mm_movehl_ps(fy, fy));
  const __m128d sl = _mm_sqrt_pd(_mm_add_pd(_mm_mul_pd(xl, xl), _mm_mul_pd(yl, yl))), sh = _mm_sqrt_pd(_mm_add_pd(_mm_mul_pd(xh, xh), _mm_mul_pd(yh, yh)));
  return _mm_castps_si128(_mm_movelh_ps(_mm_cvtpd_ps(sl), _mm_cvtpd_ps(sh)));
}
#define TR_HYP4 1
#endif
// ai: The first point of [i0, i1) farthest from the line through (ax, ay) along (ex, ey) of length en, or from the point
// ai: (ax, ay) itself where en < 1 (no chord); its distance in *dm (-1 and -1 back for none: a NaN distance never wins).
// ai: The vector form takes four points a step, each lane its own first maximum (a strict compare, so a lane keeps its
// ai: earliest), then the largest of the four and the earliest place it stands: the scalar's choice exactly.
static int tr_far(const float *qx, const float *qy, int i0, int i1, float ax, float ay, float ex, float ey, float en, float *dm) {
  float bd = -1;
  int at = -1, i = i0;
  const int chord = en >= 1;
#ifdef OB_SIMD
#ifdef TR_HYP4
  const int vec = 1;
#else
  const int vec = chord;
#endif
  if (vec && i + 4 <= i1) {
    const v128_t vax = wasm_f32x4_splat(ax), vay = wasm_f32x4_splat(ay), vex = wasm_f32x4_splat(ex), vey = wasm_f32x4_splat(ey), ven = wasm_f32x4_splat(en), four = wasm_i32x4_splat(4);
    const int32_t o[4] = { i, i + 1, i + 2, i + 3 };
    v128_t bv = wasm_f32x4_splat(-1), bi = wasm_i32x4_splat(-1), iv = wasm_v128_load(o);
    for (; i + 4 <= i1; i += 4) {
      const v128_t dx = wasm_f32x4_sub(wasm_v128_load(qx + i), vax), dy = wasm_f32x4_sub(wasm_v128_load(qy + i), vay);
      v128_t d;
#ifdef TR_HYP4
      if (!chord) d = tr_hyp4(dx, dy);
      else
#endif
      d = wasm_f32x4_div(wasm_f32x4_abs(wasm_f32x4_sub(wasm_f32x4_mul(dx, vey), wasm_f32x4_mul(dy, vex))), ven);
      const v128_t gt = wasm_f32x4_lt(bv, d);
      bv = wasm_v128_bitselect(d, bv, gt); bi = wasm_v128_bitselect(iv, bi, gt);
      iv = wasm_i32x4_add(iv, four);
    }
    float b[4]; int32_t ix[4];
    wasm_v128_store(b, bv); wasm_v128_store(ix, bi);
    for (int k = 0; k < 4; k++) if (b[k] > bd || (b[k] == bd && ix[k] >= 0 && ix[k] < at)) { bd = b[k]; at = ix[k]; }
  }
#endif
  for (; i < i1; i++) {
    const float dx = qx[i] - ax, dy = qy[i] - ay, d = chord ? fabsf(dx * ey - dy * ex) / en : tr_hyp(dx, dy);
    if (d > bd) { bd = d; at = i; }
  }
  *dm = bd;
  return at;
}
// ai: a side's points (along, across) to a curve, robustly: from the chord between its two corners (al0, ac0) and
// ai: (al1, ac1), three rounds of a least-squares parabola over the points within a band of the curve so far (4 px or 3%
// ai: of the side, then 2 px or 1%, then 1.5 px or 0.5%): a detour of the iso-line into the picture never enters, a bow
// ai: of a few px does. u: room for n floats, each point's place along the side (-1 to 1 between the corners), made once
// ai: for the three rounds; the normal equations' nine sums are six (u^2 serves three places, u two), in the same order.
static int tr_fit(const float *al, const float *ac, int n, float al0, float ac0, float al1, float ac1, float *u, fline_t *l) {
  if (n < 8 || fabsf(al1 - al0) < 4) return 0;
  const float len = fabsf(al1 - al0), mid = 0.5f * (al0 + al1), half = 0.5f * len;
  float cf[3] = { 0, (ac1 - ac0) / (al1 - al0) * half, 0.5f * (ac0 + ac1) };   // ai: in u = (along - mid) / half
  static const float TB[3] = { 4, 2, 1.5f }, TF[3] = { 0.03f, 0.01f, 0.005f };
  int i = 0;
#ifdef OB_SIMD
  { const v128_t vm = wasm_f32x4_splat(mid), vh = wasm_f32x4_splat(half); for (; i + 4 <= n; i += 4) wasm_v128_store(u + i, wasm_f32x4_div(wasm_f32x4_sub(wasm_v128_load(al + i), vm), vh)); }
#endif
  for (; i < n; i++) u[i] = (al[i] - mid) / half;
  for (int round = 0; round < 3; round++) {
    const float tol = fmaxf(TB[round], TF[round] * len);
    double s00 = 0, s01 = 0, s02 = 0, s12 = 0, s22 = 0, s03 = 0, s13 = 0, s23 = 0;
    int m = 0;
    for (int k = 0; k < n; k++) {
      const double ud = u[k];
      if (fabsf((float)(cf[0] * ud * ud + cf[1] * ud + cf[2]) - ac[k]) > tol) continue;
      const double uu = ud * ud, a = ac[k];
      s00 += uu * uu; s01 += uu * ud; s02 += uu; s12 += ud; s22 += 1; s03 += uu * a; s13 += ud * a; s23 += a;
      m++;
    }
    if (m < 6) return 0;
    double S[3][4] = { { s00, s01, s02, s03 }, { s01, s02, s12, s13 }, { s02, s12, s22, s23 } };
    for (int col = 0; col < 3; col++) {   // ai: Gauss-Jordan on the 3 x 3
      int pv = col;
      for (int r = col + 1; r < 3; r++) if (fabs(S[r][col]) > fabs(S[pv][col])) pv = r;
      if (fabs(S[pv][col]) < 1e-12) return 0;
      for (int q = 0; q < 4; q++) { const double t = S[col][q]; S[col][q] = S[pv][q]; S[pv][q] = t; }
      for (int r = 0; r < 3; r++) if (r != col) { const double k = S[r][col] / S[col][col]; for (int q = col; q < 4; q++) S[r][q] -= k * S[col][q]; }
    }
    for (int r = 0; r < 3; r++) cf[r] = (float)(S[r][3] / S[r][r]);
  }
  // ai: back to fline_t's terms: c (along - mid)^2 + a along + b
  l->c = cf[0] / (half * half); l->a = cf[1] / half; l->b = cf[2] - l->a * mid; l->mid = mid; l->n = n;
  return 1;
}
// ai: A contour as a quad by its straight runs: Douglas-Peucker over its points (the tolerance 2% of its box's larger
// ai: side, 2 px at least; a closed one from the point farthest from its centroid), neighbouring runs within 20 degrees
// ai: joined (a bowed side), the runs at least a tenth of that side long named by the way they face from the centroid
// ai: (top, right, bottom, left; a name taken twice keeps the longer run), each side's curve fitted to its run's points
// ai: (tr_fit from the run's ends). Four names or none; the corners are where the curves meet, outside the picture or not.
// ai: px, py: the n points; rx, ry: room for n + 1 each, a closed contour unrolled from its start (its first point again
// ai: at the end), so every range is contiguous; u: room for n floats (tr_fit's).
static int tr_segs(const float *px, const float *py, int n, int closed, int w, int h, tquad_t *o, uint8_t *keep, int *st, float *rx, float *ry, float *u) {
  float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f, gx = 0, gy = 0;
  int i = 0;
#ifdef OB_SIMD
  // ai: the box four points a step (pmin and pmax keep the running value on a NaN, as fminf and fmaxf do)
  if (n >= 4) {
    v128_t a0 = wasm_f32x4_splat(1e9f), a1 = wasm_f32x4_splat(-1e9f), b0 = a0, b1 = a1;
    for (; i + 4 <= n; i += 4) {
      const v128_t vx = wasm_v128_load(px + i), vy = wasm_v128_load(py + i);
      a0 = wasm_f32x4_pmin(a0, vx); a1 = wasm_f32x4_pmax(a1, vx); b0 = wasm_f32x4_pmin(b0, vy); b1 = wasm_f32x4_pmax(b1, vy);
    }
    float t[4][4];
    wasm_v128_store(t[0], a0); wasm_v128_store(t[1], a1); wasm_v128_store(t[2], b0); wasm_v128_store(t[3], b1);
    for (int k = 0; k < 4; k++) { x0 = fminf(x0, t[0][k]); x1 = fmaxf(x1, t[1][k]); y0 = fminf(y0, t[2][k]); y1 = fmaxf(y1, t[3][k]); }
  }
#endif
  for (; i < n; i++) { x0 = fminf(x0, px[i]); x1 = fmaxf(x1, px[i]); y0 = fminf(y0, py[i]); y1 = fmaxf(y1, py[i]); }
  if (x1 - x0 < TR_MIN * w || y1 - y0 < TR_MIN * h) return 0;
  for (int k = 0; k < n; k++) { gx += px[k]; gy += py[k]; }
  gx /= n; gy /= n;
  if (!closed) {   // ai: open: both ends at the picture's edge, or it is not a cut symbol
    const int at0 = px[0] < 2 || py[0] < 2 || px[0] > w - 2 || py[0] > h - 2, at1 = px[n - 1] < 2 || py[n - 1] < 2 || px[n - 1] > w - 2 || py[n - 1] > h - 2;
    if (!at0 || !at1) return 0;
  }
  const float big = fmaxf(x1 - x0, y1 - y0), tol = fmaxf(2, 0.02f * big);
  const int m = closed ? n + 1 : n;
  const float *qx = px, *qy = py;
  if (closed) {
    float bd;
    int s0 = tr_far(px, py, 0, n, gx, gy, 0, 0, 0, &bd);
    if (s0 < 0) s0 = 0;
    memcpy(rx, px + s0, (size_t)(n - s0) * sizeof(float)); memcpy(rx + n - s0, px, (size_t)s0 * sizeof(float)); rx[n] = rx[0];
    memcpy(ry, py + s0, (size_t)(n - s0) * sizeof(float)); memcpy(ry + n - s0, py, (size_t)s0 * sizeof(float)); ry[n] = ry[0];
    qx = rx; qy = ry;
  }
  memset(keep, 0, (size_t)m);
  int sp = 0;
  keep[0] = keep[m - 1] = 1;
  st[sp++] = 0; st[sp++] = m - 1;
  while (sp) {
    const int b2 = st[--sp], a2 = st[--sp];
    if (b2 - a2 < 2) continue;
    const float ex = qx[b2] - qx[a2], ey = qy[b2] - qy[a2], en = tr_hyp(ex, ey);
    float dm;
    // ai: from the chord, or from its one point where it has none (a closed contour's first range, round to its start)
    const int im = tr_far(qx, qy, a2 + 1, b2, qx[a2], qy[a2], ex, ey, en, &dm);
    if (dm > tol) { keep[im] = 1; st[sp++] = a2; st[sp++] = im; st[sp++] = im; st[sp++] = b2; }
  }
  int vi[256], nv = 0;
  for (int k = 0; k < m && nv < 256; k++) if (keep[k]) vi[nv++] = k;
  int ra[256], rb[256], nr = 0;
  for (int k = 0; k + 1 < nv; k++) {
    if (nr) {
      const int a0 = ra[nr - 1], a1 = rb[nr - 1], b0 = vi[k], b1 = vi[k + 1];
      const float d1 = atan2f(qy[a1] - qy[a0], qx[a1] - qx[a0]), d2 = atan2f(qy[b1] - qy[b0], qx[b1] - qx[b0]);
      float dd = fabsf(d1 - d2);
      if (dd > (float)M_PI) dd = 2 * (float)M_PI - dd;
      if (dd < 0.349f) { rb[nr - 1] = vi[k + 1]; continue; }
    }
    ra[nr] = vi[k]; rb[nr] = vi[k + 1]; nr++;
  }
  int side[4] = { -1, -1, -1, -1 }, sa[4] = { 0 }, sb[4] = { 0 };
  float slen[4] = { 0, 0, 0, 0 };
  for (int r = 0; r < nr; r++) {
    const int a0 = ra[r], a1 = rb[r];
    const float len = tr_hyp(qx[a1] - qx[a0], qy[a1] - qy[a0]);
    if (len < 0.1f * big) continue;
    const float mx = 0.5f * (qx[a0] + qx[a1]) - gx, my = 0.5f * (qy[a0] + qy[a1]) - gy;
    const int nm = fabsf(mx) * (y1 - y0) > fabsf(my) * (x1 - x0) ? (mx > 0 ? 1 : 3) : (my > 0 ? 2 : 0);
    if (len > slen[nm]) { slen[nm] = len; side[nm] = r; sa[nm] = ra[r]; sb[nm] = rb[r]; }
  }
  for (int k = 0; k < 4; k++) if (side[k] < 0) return 0;
  // ai: each side's run between its corners less a twentieth at each end, read in place (x along for top and bottom)
  int good = 1;
  for (int k = 0; k < 4 && good; k++) {
    const int i0 = sa[k], i1 = sb[k], cut = (i1 - i0) / 20, q0 = i0 + cut, q = i1 - cut - q0 + 1 > 0 ? i1 - cut - q0 + 1 : 0;
    good = k & 1 ? tr_fit(qy + q0, qx + q0, q, qy[i0], qx[i0], qy[i1], qx[i1], u, &o->in[k]) : tr_fit(qx + q0, qy + q0, q, qx[i0], qy[i0], qx[i1], qy[i1], u, &o->in[k]);
  }
  return good;
}
// ai: the traced quads of a frame (at most maxq), the mean's level from the grey pyramid; min_con: the contrast floor a
// ai: start's pixel pair must cross
static int trace_quads(fctx_t *f, int min_con, tquad_t *out, int maxq) {
  const int w = f->w, h = f->h;
  int ls = 0;
  for (int l = 1; l <= TR_LEVEL && (w >> l) >= 8 && (h >> l) >= 8; l++) ls = l;
  const uint8_t *lv = fctx_level(f, ls);
  if (!lv) return 0;
  tr_exit_init();
  const int lw = w >> ls, lh = h >> ls, lw1 = lw + 1, tw = (w + TR_TILE - 1) / TR_TILE, th = (h + TR_TILE - 1) / TR_TILE;
  const size_t nvis = ((size_t)w * h + 7) / 8;
  uint8_t *cc = frame_buf(f->fb, OB_FB_CODES, (size_t)w * h);
  uint8_t *aux = frame_buf(f->fb, OB_FB_TRACE, (size_t)h * lw1 * sizeof(float) + (size_t)(w + h) * (sizeof(int32_t) + sizeof(float)) + (size_t)tw * th + nvis);
  // ai: a chase's cells and exits; its points (x, y), the unrolled copy (TR_MAXP + 1 each) and the fits' u; keep and the stack
  uint32_t *cidx = malloc((size_t)TR_MAXP * (sizeof(uint32_t) + 1 + 5 * sizeof(float) + 1 + 2 * sizeof(int)) + 128);
  int nq = 0;
  if (!cc || !aux || !cidx) goto done;
  {
    float *LyAll = (float *)aux, *ax = LyAll + (size_t)h * lw1, *ay = ax + w;
    int32_t *xi = (int32_t *)(ay + h), *yi = xi + w;
    uint8_t *tdone = (uint8_t *)(yi + h), *vis = tdone + (size_t)tw * th;
    uint8_t *ceo = (uint8_t *)(cidx + TR_MAXP);
    float *ptx = (float *)(((uintptr_t)(ceo + TR_MAXP) + 15) & ~(uintptr_t)15), *pty = ptx + TR_MAXP, *rx = pty + TR_MAXP, *ry = rx + TR_MAXP + 1, *uf = ry + TR_MAXP + 1;
    uint8_t *keep = (uint8_t *)(uf + TR_MAXP);
    int *st = (int *)(((uintptr_t)(keep + TR_MAXP + 1) + 7) & ~(uintptr_t)7);
    tctx_t c = { f->pyr[0], w, h, lv, lw, lh, ls, xi, ax, yi, ay, LyAll, lw1, cc, tdone, tw, vis, { 0 }, { 0 } };
    tr_steps(&c);
    const float sc = (float)(1 << ls);
    for (int x = 0; x < w; x++) {
      float fx = (x + 0.5f) / sc - 0.5f;
      if (fx < 0) fx = 0;
      if (fx > lw - 1.001f) fx = lw - 1.001f;
      xi[x] = (int)fx; ax[x] = fx - (int)fx;
    }
    for (int y = 0; y < h; y++) {
      float fy = (y + 0.5f) / sc - 0.5f;
      if (fy < 0) fy = 0;
      if (fy > lh - 1.001f) fy = lh - 1.001f;
      yi[y] = (int)fy; ay[y] = fy - (int)fy;
    }
    for (int y = 0; y < h; y++) tr_ly(&c, y, LyAll + (size_t)y * lw1);
    memset(tdone, 0, (size_t)tw * th);
    memset(vis, 0, nvis);
    const float cmin = 0.5f * min_con;
    const int row = h / TR_ROWS > 4 ? h / TR_ROWS : 4;
    // ai: a start row's f (fr), and its crossings with the contrast floor across them as bits (cb, x at bit x % 64 of word x / 64)
    const int nwd = (w + 63) / 64;
    float *fr = malloc((size_t)w * sizeof(float) + (size_t)nwd * sizeof(uint64_t));
    uint64_t *cb = fr ? (uint64_t *)(fr + w) : NULL;
    for (int y = 1; fr && y + 1 < h && nq < maxq; y += row) {
      const float *Ly = LyAll + (size_t)y * lw1;
      const uint8_t *rw = c.img + (size_t)y * w;
      int x = 0;
#ifdef OB_SIMD
      // ai: sixteen a step where each aligned four shares the level's pair (tr_tile's), the scalar's floats
      for (; x + 16 <= w; x += 16) {
        int same = 1;
        for (int g = 0; g < 4; g++) same &= xi[x + 4 * g] == xi[x + 4 * g + 3];
        if (!same) { for (int k = x; k < x + 16; k++) { const float a = Ly[xi[k]], b = Ly[xi[k] + 1]; fr[k] = (float)rw[k] - (a + ax[k] * (b - a)); } continue; }
        const v128_t px = wasm_v128_load(rw + x), lo = wasm_u16x8_extend_low_u8x16(px), hi = wasm_u16x8_extend_high_u8x16(px);
        const v128_t gv[4] = { wasm_u32x4_extend_low_u16x8(lo), wasm_u32x4_extend_high_u16x8(lo), wasm_u32x4_extend_low_u16x8(hi), wasm_u32x4_extend_high_u16x8(hi) };
        for (int g = 0; g < 4; g++) {
          const int j = xi[x + 4 * g];
          const float a = Ly[j], d = Ly[j + 1] - a;
          const v128_t T = wasm_f32x4_add(wasm_f32x4_splat(a), wasm_f32x4_mul(wasm_v128_load(ax + x + 4 * g), wasm_f32x4_splat(d)));
          wasm_v128_store(fr + x + 4 * g, wasm_f32x4_sub(wasm_f32x4_convert_i32x4(gv[g]), T));
        }
      }
#endif
      for (; x < w; x++) { const float a = Ly[xi[x]], b = Ly[xi[x] + 1]; fr[x] = (float)rw[x] - (a + ax[x] * (b - a)); }
      memset(cb, 0, (size_t)nwd * sizeof(uint64_t));
      x = 0;
#ifdef OB_SIMD
      // ai: four pixel pairs a step: the signs differ ((a <= 0) != (b <= 0): no NaN in f) and |a - b| reaches the floor
      { const v128_t zero = wasm_f32x4_splat(0), vmin = wasm_f32x4_splat(cmin);
        for (; x + 4 < w; x += 4) {
          const v128_t a = wasm_v128_load(fr + x), b = wasm_v128_load(fr + x + 1);
          const v128_t m = wasm_v128_and(wasm_v128_xor(wasm_f32x4_le(a, zero), wasm_f32x4_le(b, zero)), wasm_f32x4_le(vmin, wasm_f32x4_abs(wasm_f32x4_sub(a, b))));
          cb[x >> 6] |= (uint64_t)wasm_i32x4_bitmask(m) << (x & 63);
        } }
#endif
      for (; x + 1 < w; x++) { const float a = fr[x], b = fr[x + 1]; if ((a > 0) != (b > 0) && !(fabsf(a - b) < cmin)) cb[x >> 6] |= 1ull << (x & 63); }
      for (int wd = 0; wd < nwd && nq < maxq; wd++)
      for (uint64_t bits = cb[wd]; bits && nq < maxq; bits &= bits - 1) {
        const int x = wd * 64 + __builtin_ctzll(bits);
        const size_t vi = (size_t)y * w + x;
        if (vis[vi >> 3] >> (vi & 7) & 1) continue;
        // ai: chase forward, then (open) backward from the cell above, the backward run reversed in front; points only for
        // ai: a contour the size of a symbol
        int n = 0, bx[4] = { x, x, y, y };
        const int closed = tr_chase(&c, x, y, 0, cidx, ceo, &n, TR_MAXP, bx);
        if (!closed && y > 0) {
          int nb2 = 0;
          uint32_t *bi = cidx + n;
          uint8_t *be = ceo + n;
          tr_chase(&c, x, y - 1, 2, bi, be, &nb2, TR_MAXP - n, bx);
          if (nb2) {
            // ai: the backward steps reversed, each its entry edge (the step before's exit, flipped) as its exit
            uint32_t *ti = malloc((size_t)(n + nb2) * sizeof(uint32_t));
            uint8_t *te = malloc((size_t)(n + nb2));
            if (!ti || !te) { free(ti); free(te); continue; }
            for (int k = 0; k < nb2; k++) { ti[k] = bi[nb2 - 1 - k]; te[k] = k + 1 < nb2 ? (uint8_t)((be[nb2 - 2 - k] + 2) & 3) : 2; }
            memcpy(ti + nb2, cidx, (size_t)n * sizeof(uint32_t)); memcpy(te + nb2, ceo, (size_t)n);
            memcpy(cidx, ti, (size_t)(n + nb2) * sizeof(uint32_t)); memcpy(ceo, te, (size_t)(n + nb2));
            free(ti); free(te);
            n += nb2;
          }
        }
        if (n < 64 || bx[1] - bx[0] < TR_MIN * w || bx[3] - bx[2] < TR_MIN * h) continue;
        tr_pts(&c, cidx, ceo, n, ptx, pty);
        if (tr_segs(ptx, pty, n, closed, w, h, &out[nq], keep, st, rx, ry, uf)) nq++;
      }
    }
    free(fr);
  }
done:
  free(cidx);
  frame_buf_done(f->fb, cc); frame_buf_done(f->fb, aux);
  return nq;
}

// ---------------------------------------------------------------- the finder's lens
// ai: A traced quad's four sides are curves; a lens bends them and a camera's perspective does not, so how much they
// ai: bend gives the lens. The model is the registration's first radial term about the axis (ob_axis, else the image's
// ai: centre), q over R (the farthest image corner's distance from it): a point at q undistorted lies at q (1 + k |q|^2)
// ai: in the image. k is the one whose straightened sides are straightest: each side's curve sampled TRL_N times between
// ai: its corners, undistorted (Newton on the radius), the smallest variance across each side's best line summed over the
// ai: four, by a scan of k over -0.8 to 0.8 in 17 steps then golden section about the best. The quad then read is the
// ai: straight lines' corners (the homography's), the track at its points bent by k (flens_t, thin_read_both), the
// ai: corners handed back bent the same way, and the bend as the anchors' first model (find_frame's seed). On a strong
// ai: bend with the symbol filling the picture the traced quads were right and every way round scored 0.05 or less on
// ai: the track read through their homography (2026-11-23, the lens set: 1,075,029 to 1,082,015 blocks, with the seed
// ai: 1,103,529).
enum { TRL_N = 17 };
static void tr_unbend(float kk, float ax, float ay, float x, float y, float *ux, float *uy) {   // ai: kk = k / R^2
  const float dx = x - ax, dy = y - ay, rd = sqrtf(dx * dx + dy * dy);
  if (rd < 1e-3f) { *ux = x; *uy = y; return; }
  float r = rd / (1 + kk * rd * rd);   // ai: the first-order inverse, then Newton
  for (int i = 0; i < 3; i++) r -= (r + kk * r * r * r - rd) / (1 + 3 * kk * r * r);
  const float s = r / rd;
  *ux = ax + dx * s; *uy = ay + dy * s;
}
// ai: the four sides' straightness under kk; ln (given): each side's line n . p = c as { nx, ny, c }
static double tr_lens_err(float kk, float ax, float ay, const float *sx, const float *sy, float ln[4][3]) {
  double e = 0;
  // ai: the four sides' points unbent, four a step in the vector form (tr_unbend's operations lane for lane: the same
  // ai: floats), then each side's sums in order
  float uxa[4 * TRL_N], uya[4 * TRL_N];
  int j = 0;
#ifdef OB_SIMD
  {
    const v128_t vk = wasm_f32x4_splat(kk), vax = wasm_f32x4_splat(ax), vay = wasm_f32x4_splat(ay), one = wasm_f32x4_splat(1), k3 = wasm_f32x4_splat(3 * kk), eps = wasm_f32x4_splat(1e-3f);
    for (; j + 4 <= 4 * TRL_N; j += 4) {
      const v128_t x = wasm_v128_load(sx + j), y = wasm_v128_load(sy + j), dx = wasm_f32x4_sub(x, vax), dy = wasm_f32x4_sub(y, vay);
      const v128_t rd = wasm_f32x4_sqrt(wasm_f32x4_add(wasm_f32x4_mul(dx, dx), wasm_f32x4_mul(dy, dy)));
      v128_t r = wasm_f32x4_div(rd, wasm_f32x4_add(one, wasm_f32x4_mul(wasm_f32x4_mul(vk, rd), rd)));
      for (int it = 0; it < 3; it++)
        r = wasm_f32x4_sub(r, wasm_f32x4_div(wasm_f32x4_sub(wasm_f32x4_add(r, wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_mul(vk, r), r), r)), rd),
                                             wasm_f32x4_add(one, wasm_f32x4_mul(wasm_f32x4_mul(k3, r), r))));
      const v128_t sc = wasm_f32x4_div(r, rd), near = wasm_f32x4_lt(rd, eps);
      wasm_v128_store(uxa + j, wasm_v128_bitselect(x, wasm_f32x4_add(vax, wasm_f32x4_mul(dx, sc)), near));
      wasm_v128_store(uya + j, wasm_v128_bitselect(y, wasm_f32x4_add(vay, wasm_f32x4_mul(dy, sc)), near));
    }
  }
#endif
  for (; j < 4 * TRL_N; j++) tr_unbend(kk, ax, ay, sx[j], sy[j], &uxa[j], &uya[j]);
  for (int s = 0; s < 4; s++) {
    const float *ux = uxa + s * TRL_N, *uy = uya + s * TRL_N;
    double mx = 0, my = 0, xx = 0, xy = 0, yy = 0;
    for (int i = 0; i < TRL_N; i++) { mx += ux[i]; my += uy[i]; }
    mx /= TRL_N; my /= TRL_N;
    for (int i = 0; i < TRL_N; i++) { const double dx = ux[i] - mx, dy = uy[i] - my; xx += dx * dx; xy += dx * dy; yy += dy * dy; }
    const double tr = xx + yy, det = xx * yy - xy * xy, lmin = 0.5 * tr - sqrt(fmax(0, 0.25 * tr * tr - det));
    e += lmin;
    if (ln) {   // ai: the normal: the eigenvector of the smallest eigenvalue
      double nx = xy, ny = lmin - xx;
      if (fabs(nx) + fabs(ny) < 1e-12) { nx = lmin - yy; ny = xy; }
      if (fabs(nx) + fabs(ny) < 1e-12) { nx = xx < yy ? 1 : 0; ny = xx < yy ? 0 : 1; }
      const double nn = hypot(nx, ny);
      ln[s][0] = (float)(nx / nn); ln[s][1] = (float)(ny / nn); ln[s][2] = (float)((nx * mx + ny * my) / nn);
    }
  }
  return e;
}
static int tr_meet_ln(const float *a, const float *b, float *x, float *y) {
  const float det = a[0] * b[1] - a[1] * b[0];
  if (fabsf(det) < 1e-6f) return 0;
  *x = (a[2] * b[1] - a[1] * b[2]) / det; *y = (a[0] * b[2] - a[2] * b[0]) / det;
  return 1;
}
// ai: cv: the image's top, right, bottom and left curves; q: their corners (TL, TR, BR, BL), replaced by the straight
// ai: lines' corners; lk the bend. 0 where the lines do not meet.
static int tr_lens(const fline_t *const *cv, float *q, int w, int h, flens_t *lk) {
  float ax = ob_axis[0], ay = ob_axis[1];
  if (!(ax == ax) || !(ay == ay)) { ax = w * 0.5f; ay = h * 0.5f; }
  float R = 0;
  for (int c = 0; c < 4; c++) { const float d = hypotf((c & 1 ? w : 0) - ax, (c & 2 ? h : 0) - ay); if (d > R) R = d; }
  const float iR2 = 1 / (R * R), k0 = -0.8f, k1 = 0.8f;
  float sx[4 * TRL_N], sy[4 * TRL_N];
  // ai: side s from corner E0[s] to E1[s]; top and bottom curves give y from x, right and left x from y
  static const int E0[4] = { 0, 1, 3, 0 }, E1[4] = { 1, 2, 2, 3 };
  for (int s = 0; s < 4; s++) {
    const float *pa = q + 2 * E0[s], *pb = q + 2 * E1[s];
    for (int i = 0; i < TRL_N; i++) {
      const float t = (float)i / (TRL_N - 1);
      if (s & 1) { const float y = pa[1] + t * (pb[1] - pa[1]); sx[s * TRL_N + i] = line_at(cv[s], y); sy[s * TRL_N + i] = y; }
      else { const float x = pa[0] + t * (pb[0] - pa[0]); sx[s * TRL_N + i] = x; sy[s * TRL_N + i] = line_at(cv[s], x); }
    }
  }
  enum { NS = 17 };
  int bi = 0;
  double be = 1e300;
  for (int i = 0; i < NS; i++) { const double e = tr_lens_err((k0 + (k1 - k0) * i / (NS - 1)) * iR2, ax, ay, sx, sy, NULL); if (e < be) { be = e; bi = i; } }
  const float st = (k1 - k0) / (NS - 1), gr = 0.6180339887f;
  float lo = k0 + st * (bi - 1), hi = k0 + st * (bi + 1), a = hi - gr * (hi - lo), b = lo + gr * (hi - lo);
  double ea = tr_lens_err(a * iR2, ax, ay, sx, sy, NULL), eb = tr_lens_err(b * iR2, ax, ay, sx, sy, NULL);
  for (int it = 0; it < 14; it++) {
    if (ea < eb) { hi = b; b = a; eb = ea; a = hi - gr * (hi - lo); ea = tr_lens_err(a * iR2, ax, ay, sx, sy, NULL); }
    else { lo = a; a = b; ea = eb; b = lo + gr * (hi - lo); eb = tr_lens_err(b * iR2, ax, ay, sx, sy, NULL); }
  }
  const float k = 0.5f * (lo + hi);
  float ln[4][3], u[8];
  tr_lens_err(k * iR2, ax, ay, sx, sy, ln);
  if (!tr_meet_ln(ln[0], ln[3], &u[0], &u[1]) || !tr_meet_ln(ln[0], ln[1], &u[2], &u[3]) || !tr_meet_ln(ln[2], ln[1], &u[4], &u[5]) || !tr_meet_ln(ln[2], ln[3], &u[6], &u[7])) return 0;
  // ai: a fit whose unbending diverged gives lines of NaN and corners of NaN, which pass every range test after and read
  // ai: a NaN track score on the scalar and wasm readers (the vector readers clamp a NaN coordinate to pixel 0): refused
  for (int k = 0; k < 8; k++) if (!isfinite(u[k])) return 0;
  memcpy(q, u, sizeof u);
  lk->on = 1; lk->kr = k * iR2; lk->ax = ax; lk->ay = ay;
  return 1;
}
// ai: The finder's lens as the registration's model (lensm_t) about the homography through the corners it hands back
// ai: (Hb, as acquire_core makes it): the displacement from Hb to the bent straight-sided map (Hu, lk) on a 9 x 9 grid of
// ai: the layout, fitted by lensm_fit; the anchors' first round searches through it (the Garmin Virb ultrawide's and the
// ai: GoPro Hero11 wide's frames from 2 and 1 anchors and 0 blocks to 25 and 54 and 53, 2026-11-23)
static int finder_seed(const ob_layout_t *L, const homo_t *Hu, const flens_t *lk, const float *nc, const float *quad, int iw, int ih, lensm_t *seed) {
  homo_t Hb;
  if (!lk->on || !homography(nc, quad, &Hb)) return 0;
  lensm_axis(iw, ih, seed);
  enum { G = 9 };
  double qx[G * G], qy[G * G], dx[G * G], dy[G * G];
  int np = 0;
  for (int j = 0; j < G; j++) for (int i = 0; i < G; i++) {
    const float X = L->w * i / (float)(G - 1), Y = L->h * j / (float)(G - 1);
    float bu, bv, tu, tv;
    project(&Hb, X, Y, &bu, &bv);
    project(Hu, X, Y, &tu, &tv);
    flens_bend(lk, &tu, &tv);
    qx[np] = (bu - seed->ax) / seed->R; qy[np] = (bv - seed->ay) / seed->R; dx[np] = tu - bu; dy[np] = tv - bv; np++;
  }
  unsigned char drop[G * G] = { 0 };
  return lensm_fit(np, qx, qy, dx, dy, NULL, drop, 0, LENS_TERMS, seed);
}

// Finds the frame and its rotation. quad: the four lattice corners in layout order, as image points.
// Ls: the layouts the symbol may be, which differ only in how many modules a side (lizard_phy.h LIZARD_RING); *which
// comes back as the one that registered. fb: frame_buf's.
// ai: Two finders of the same four corners from two features of the format, each whole on its own and failing where the
// ai: other holds (2026-11-24, STATUS "The new C finder in liblizard"): the corner marks proposed from the grey picture
// ai: (gray_cands), and the ring traced as one loop (trace_quads); the marks fail under blur and on small symbols, the
// ai: trace past 2:1 of perspective and on lenses the symbol fills. The marks first: once a mark quad holds a rotation
// ai: (rotation_holds) the trace is not run. A traced quad is read through the lens its sides' bend gives (tr_lens), and a
// ai: winning one hands that lens on as the anchors' first model (*seed, *seeded). Before it (2026-09 to 2026-11-24) the
// ai: marks were scanned in the capture binarized at three scales and the border fitted from binarized edge points (the
// ai: line path); the new finder read the closed-ring sets level or above everywhere (front +2.7%, the lens set +1.5%,
// ai: the stress spectrum +0.1%, the stress angles +0.4%, the judge's angles -0.1%) at a third of the time on a
// ai: symbol, empty frames a third slower (one x86 core; research/build/rewrite-2026-11-23/README.md).
static int find_frame(const ob_layout_t *const *Ls, int nL, const image_t *im, ob_fbufs_t *fb, float *quad, int *orient, float *score_out, int *which, lensm_t *seed, int *seeded) {
  const int w = im->w, h = im->h;
  float best = -1, src[8], dst[8];
  int ok = 1;
  double tp = ob_now_ms();
  // The corner a layout side runs between, in the direction its modules count. Shared with the combination loop.
  static const int ENDS[4][2] = { { 0, 1 }, { 1, 2 }, { 3, 2 }, { 0, 3 } };
  fctx_t f = { fb, { im->px }, w, h, { 0 } };
  *which = 0; *seeded = 0;
  for (int k = 0; k < 8; k++) ob_finder_info[k] = k == 5 || k == 7 ? 0 : -1;
  // THE CORNER MARKS, coarse to fine, each level's candidates read as the merged square and as the core, every quad the
  // track choosing between them. Spacing cannot choose: a noisy background holds four grains at any spacing asked for,
  // and only the symbol has a track between them. The best reading of the set wins, not the first to reach 0.15: taking
  // the first lost 2 of 32 clean simulator frames of LIZARD-16 (since removed) to a mirror image of a wrong quad at
  // 0.16, ahead of the true quad at 0.49 the right way round.
  // ai: A quad's contrast gate is read near its corners (thin_levels_near): once with hypothesis 0's map, a quad that
  // ai: fails it not scored at all (a turn or flip only permutes the sides it reads), then for each hypothesis, whose ref
  // ai: and con thin_score takes.
  float msrc[8];
  cand_t *gc = malloc((size_t)2 * GRAY_MAX * sizeof(cand_t));
  float rc[GRAY_MAX];
  mquad_t mq[MARK_QUADS];
  int lv[5];
  const int nlv = gray_levels(w, h, lv);
  for (int il = 0; gc && il < nlv && !rotation_holds(best, *orient); il++) {
    const int scale = lv[il];
    int ng;
    PROF(PROF_FF_GREY, ng = gray_cands(&f, scale, gc, rc));
    cand_t *mc = gc + GRAY_MAX;
    for (int form = 1; form < 3 && !rotation_holds(best, *orient); form++) {
      gray_form(gc, rc, ng, form, mc);
      mpool_t pool;
      mark_pool(mc, ng, &pool);
      for (int li = 0; li < nL; li++) {
        const ob_layout_t *L = Ls[li];
        const int nq = mark_rect(L, &pool, ng, w, h, form, msrc, mq);
        if (form == 1 && il == 0 && li == 0) ob_finder_info[7] = (float)nq;
        unsigned qok = 0;
        for (int qi = 0; qi < nq; qi++) {
          float d0[8], r0, c0;
          homo_t H0;
          for (int k = 0; k < 4; k++) { d0[2 * k] = mq[qi].q[k].x; d0[2 * k + 1] = mq[qi].q[k].y; }
          int lev = 0;
          if (homography(msrc, d0, &H0)) PROF(PROF_FF_TRACK, lev = thin_levels_near(L, im, &H0, &r0, &c0));
          if (lev) qok |= 1u << qi;
        }
        for (int h0 = 0; h0 < 8 && !(h0 == 4 && rotation_holds(best, *orient)); h0 += 4)
        for (int qi = 0; qi < nq; qi++) for (int hyp = h0; hyp < h0 + 4; hyp++) {
          if (!(qok >> qi & 1)) continue;
          const int rot = hyp & 3, mir = hyp >> 2;
          float d2[8], ref = 0, con = 0, nc[8];
          homo_t H;
          for (int k = 0; k < 4; k++) { const int from = (rot + (mir ? 4 - k : k)) & 3; d2[2 * k] = mq[qi].q[from].x; d2[2 * k + 1] = mq[qi].q[from].y; }
          if (!homography(msrc, d2, &H)) continue;
          int lev;
          PROF(PROF_FF_TRACK, lev = thin_levels_near(L, im, &H, &ref, &con));
          if (!lev) continue;
          float sc;
          PROF(PROF_FF_TRACK, sc = thin_score(L, im, &H, ref, con, best));
          if (sc <= best) continue;
          best = sc; *orient = hyp; *which = li;
          ob_finder_info[0] = best; ob_finder_info[1] = (float)hyp; ob_finder_info[2] = (float)form; ob_finder_info[3] = (float)qi; ob_finder_info[4] = 0;
          corner_coords(L, nc);
          for (int k = 0; k < 4; k++) project(&H, nc[2 * k], nc[2 * k + 1], &quad[2 * k], &quad[2 * k + 1]);
        }
      }
    }
  }
  free(gc);
  ob_finder_info[5] = rotation_holds(best, *orient); ob_finder_info[6] = best;
  if (rotation_holds(best, *orient)) ok = 0;
  ob_prof_ms[PROF_FF_SCORE] += ob_now_ms() - tp; tp = ob_now_ms();
  // THE TRACED BORDER, only if the marks settled nothing.
  tquad_t *tq = NULL;
  int nct = 0;
  // ai: each traced quad's lens, fitted once (its passes read the same curves): 0 not yet, 1 fitted, 2 refused
  struct { int state; flens_t lk; float q[8]; } *lc = NULL;
  if (ok) {
    // Contrast floor: 0.35 of the picture's own range (5th to 95th percentile of a coarse grid), 12 at least.
    int hist[256] = { 0 }, cnt = 0, p5 = 0, p95 = 255, acc = 0;
    for (int y = 4; y < h; y += 8) for (int x = 4; x < w; x += 8) { hist[im->px[y * w + x]]++; cnt++; }
    for (int v = 0; v < 256; v++) { acc += hist[v]; if (acc * 20 < cnt) p5 = v; if (acc * 20 < cnt * 19) p95 = v; }
    const int min_con = (int)((p95 - p5) * 0.35f) < 12 ? 12 : (int)((p95 - p5) * 0.35f);
    tq = malloc((size_t)TR_MAXQ * sizeof(tquad_t));
    lc = calloc(TR_MAXQ, sizeof *lc);
    if (tq && lc) PROF(PROF_FF_TRACE, nct = trace_quads(&f, min_con, tq, TR_MAXQ));
    ok = nct > 0;
    tp = ob_now_ms();
  }
  // The quads are the image's and serve every layout; the track that judges them is each layout's own.
  for (int li = 0; ok && li < nL && !rotation_holds(best, *orient); li++) {
  const ob_layout_t *L = Ls[li];
  // One reading serves every way round on a square, where a turn or a flip lands each side on another's.
  const int npts = ob_track_cells(L->w, L->reserve);
#if defined(OB_NEON) || defined(OB_X86)
  // ai: arm64 and x86-64: past the reading's eight rows, the track cells' modules along a side (TRACK_AT4's) for thin_read_both;
  // ai: the signs are trk_signs'; each hypothesis's rows of track_s, 4 rev + map a layout side, as the loop finds them
  float *track_s = malloc((size_t)9 * npts * sizeof(float));
  if (!track_s) break;
  for (int i = 0; i < npts; i++) track_s[8 * npts + i] = (float)(L->reserve + OB_TRACK_AT(i)) + OB_CELL_MID;
  int hrow[8][4];
  for (int hyp = 0; hyp < 8; hyp++) for (int sd = 0; sd < 4; sd++) {
    const int rot = hyp & 3, mir = hyp >> 2;
    const int ia = (rot + (mir ? 4 - ENDS[sd][0] : ENDS[sd][0])) & 3, ib = (rot + (mir ? 4 - ENDS[sd][1] : ENDS[sd][1])) & 3;
    const int m = ((ia + 1) & 3) == ib ? ia : ib;
    hrow[hyp][sd] = 4 * (ia != ENDS[m][0]) + m;
  }
#else
  float *track_s = malloc((size_t)8 * npts * sizeof(float));
  int8_t *track_sign = malloc((size_t)4 * npts);
  if (!track_s || !track_sign) { free(track_s); free(track_sign); break; }
  for (int sd = 0; sd < 4; sd++) for (int i = 0; i < npts; i++) track_sign[sd * npts + i] = (int8_t)ob_thin_track(sd, i);
#endif
  // Three passes, one per depth the traced edge may run at (de: the ring's inner edge, the picture's edge once the gap has
  // blurred into the band, the ring's outer edge). Blur is the same all round the symbol, so the four sides share one
  // state. A pass runs only when the ones before settled nothing.
  //
  // Every quad is scored under every rotation, and the best of them all wins. Until 2026-09-24 the first to reach 0.15
  // fixed the rotation and the rest were scored under it alone, so a wrong quad read passably one way round shut the
  // true one out. One read serves all eight ways round, so this costs reading each side both ways, not eight reads.
  for (int pass = 0; pass < 3 && !(pass && rotation_holds(best, *orient)); pass++) {
  const float de = pass == 2 ? 1 : pass ? OB_THIN : 1 + OB_THIN_LINE;
  { const float e = de, se[8] = { e, e, L->w - e, e, L->w - e, L->h - e, e, L->h - e }; memcpy(src, se, sizeof src); }
  for (int combo = 0; ok && combo < nct; combo++) {
    const fline_t *T = &tq[combo].in[0], *Rr = &tq[combo].in[1], *B = &tq[combo].in[2], *Lf = &tq[combo].in[3];
    float q[8];
    if (!lc[combo].state) {
      meet(Lf, T, &q[0], &q[1]); meet(Rr, T, &q[2], &q[3]); meet(Rr, B, &q[4], &q[5]); meet(Lf, B, &q[6], &q[7]);
      int fin = 1;
      for (int k = 0; k < 8; k++) fin &= isfinite(q[k]);
      const fline_t *c4[4] = { T, Rr, B, Lf };
      PROF(PROF_FF_LENS, lc[combo].state = fin && tr_lens(c4, q, w, h, &lc[combo].lk) ? 1 : 2);
      memcpy(lc[combo].q, q, sizeof q);
    }
    if (lc[combo].state != 1) continue;
    const flens_t *lk = &lc[combo].lk;
    memcpy(q, lc[combo].q, sizeof q);
    if (q[2] - q[0] < TR_MIN * w || q[7] - q[1] < TR_MIN * h) continue;
    float wide = 0.5f * ((q[2] - q[0]) + (q[4] - q[6])), tall = 0.5f * ((q[7] - q[1]) + (q[5] - q[3]));
    if (wide > 2 * tall || tall > 2 * wide) continue;
    memcpy(dst, q, sizeof q);
    float pair[8], con = 0;
    int nref = 0;
    homo_t H0;
    // The track is read once, the way up the quad is and both ways along each side (thin_read_both), and each
    // hypothesis takes its sides from that reading with its own threshold and signs: the score its own map gives,
    // without eight reads. The one that wins gets its own homography for the corners it hands back.
    if (!homography(src, dst, &H0)) continue;
    PROF(PROF_FF_TRACK, con = thin_read_both(L, im, &H0, lk, track_s, pair, npts, &nref));
    if (!con) continue;
#if defined(OB_NEON) || defined(OB_X86)
    float hsc[8];
    int scored = 0;
#endif
    for (int hyp = 0; hyp < 8 && !(hyp == 4 && rotation_holds(best, *orient)); hyp++) {
#if defined(OB_NEON) || defined(OB_X86)
      const int rot = hyp & 3, mir = hyp >> 2;
      float d2[8];
      homo_t H;
      // ai: all eight at once where the hold cannot stop the loop at 4 (it held before this quad: it holds there)
      if (hyp >= scored) scored = trk_scores(track_s, npts, hrow, pair, nref, con, hyp, hyp == 0 && !rotation_holds(best, *orient) ? 2 : 1, hsc);
      const float sc = hsc[hyp];
      if (sc <= best) continue;
#else
      int rot = hyp & 3, mir = hyp >> 2, map[4];
      float d2[8], sc = 0;
      homo_t H;
      // Which image side each layout side lies on under this hypothesis: the one joining the two image corners its
      // ends map to; rev where it runs against that side's own direction.
      int rev[4];
      for (int sd = 0; sd < 4; sd++) {
        const int ia = (rot + (mir ? 4 - ENDS[sd][0] : ENDS[sd][0])) & 3, ib = (rot + (mir ? 4 - ENDS[sd][1] : ENDS[sd][1])) & 3;
        map[sd] = ((ia + 1) & 3) == ib ? ia : ib;
        rev[sd] = ia != ENDS[map[sd]][0];
      }
      {
        float trk = 0;
        for (int sd = 0; sd < 4; sd++) trk += pair[4 * rev[sd] + map[sd]];
        const float ref = 0.5f * trk / nref;
        sc = 0;
        for (int sd = 0; sd < 4; sd++) {
          const float *sm = track_s + (4 * rev[sd] + map[sd]) * npts;
          const int8_t *sg = track_sign + sd * npts;
          for (int i = 0; i < npts; i++) {
            float qq = (ref - sm[i]) / con;
            if (qq > 0.5f) qq = 0.5f; else if (qq < -0.5f) qq = -0.5f;
            sc += sg[i] ? qq : -qq;
          }
        }
        sc /= 4 * npts;
        if (sc <= best) continue;
      }
#endif
      for (int k = 0; k < 4; k++) { int from = (rot + (mir ? 4 - k : k)) & 3; d2[2 * k] = dst[2 * from]; d2[2 * k + 1] = dst[2 * from + 1]; }
      if (!homography(src, d2, &H)) continue;
      best = sc; *orient = hyp; *which = li;
      ob_finder_info[0] = best; ob_finder_info[1] = (float)hyp; ob_finder_info[4] = 1;
      // Hand back the lattice's own corners, which sit half a module further in than the line's middle, in the image:
      // bent by the lens, which then seeds the anchors' first round.
      float nc[8];
      corner_coords(L, nc);
      for (int k = 0; k < 4; k++) { project(&H, nc[2 * k], nc[2 * k + 1], &quad[2 * k], &quad[2 * k + 1]); flens_bend(lk, &quad[2 * k], &quad[2 * k + 1]); }
      *seeded = finder_seed(L, &H, lk, nc, quad, w, h, seed);
    }
  }
  }
#if defined(OB_NEON) || defined(OB_X86)
  free(track_s);
#else
  free(track_s); free(track_sign);
#endif
  }
  ob_prof_ms[PROF_FF_SCORE] += ob_now_ms() - tp;
  free(tq); free(lc);
  fctx_done(&f);
  *score_out = best;
  return best >= 0.06f;
}

// ---------------------------------------------------------------- top level

// The image the node templates read when the box has a radius (a module of 2 px or more): a copy with the thin frame's border
// band under a (2r + 1)-pixel box (box_blur_rect). The band is where every node a thin frame refines lives, the
// line of nodes at depth 1.5 modules and the corners' templates, which reach 12 modules in; taken 16 modules deep
// and 3 outside, the coarse search's module and a half included, through H and a few pixels wider. Blurring the
// whole frame instead cost 1.5 to 1.9 ms of a 3 to 4 ms acquisition at 1080; outside the band nothing reads it,
// and what is there is the image itself.
// ai: spare: a buffer of iw x ih bytes to write the copy into (arm64 and x86-64 hand it the workspace's, whose pages are
// ai: already the process's: a fresh one's first touches were most of the copy's time on an S26), or NULL for a new one
static uint8_t *border_blur(const ob_layout_t *L, const homo_t *H, const uint8_t *img, int iw, int ih, int r, uint8_t *spare) {
  uint8_t *out = spare ? spare : malloc((size_t)iw * ih);
  if (!out) return NULL;
  memcpy(out, img, (size_t)iw * ih);
  const float D = 16, Mo = 3, W = (float)L->w, Hh = (float)L->h;
  const float band[4][4] = { { -Mo, -Mo, W + Mo, D }, { -Mo, Hh - D, W + Mo, Hh + Mo }, { -Mo, -Mo, D, Hh + Mo }, { W - D, -Mo, W + Mo, Hh + Mo } };
#if defined(OB_NEON) || defined(OB_X86)
  // ai: msh_blur_rect's ring, column sums and row copy, for rectangles up to the image's width
  uint16_t *ring = r <= 7 ? malloc(((size_t)(2 * r + 4) * iw) * sizeof(uint16_t) + (size_t)iw + 2 * r) : NULL;
#endif
  for (int k = 0; k < 4; k++) {
    float lo[2] = { 1e9f, 1e9f }, hi[2] = { -1e9f, -1e9f };
    for (int c = 0; c < 4; c++) {
      float u, v;
      project(H, band[k][c & 1 ? 2 : 0], band[k][c & 2 ? 3 : 1], &u, &v);
      lo[0] = fminf(lo[0], u); hi[0] = fmaxf(hi[0], u); lo[1] = fminf(lo[1], v); hi[1] = fmaxf(hi[1], v);
    }
    const int pad = r + 4;
    int x0 = (int)floorf(lo[0]) - pad, y0 = (int)floorf(lo[1]) - pad, x1 = (int)ceilf(hi[0]) + pad, y1 = (int)ceilf(hi[1]) + pad;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > iw) x1 = iw;
    if (y1 > ih) y1 = ih;
    if (x1 <= x0 || y1 <= y0) continue;
#if defined(OB_NEON) || defined(OB_X86)
    if (ring && x1 - x0 >= 16) {
      msh_blur_rect(img, out, iw, ih, r, x0, y0, x1, y1, ring, ring + (size_t)(2 * r + 2) * iw, (uint8_t *)(ring + (size_t)(2 * r + 4) * iw));
      continue;
    }
#endif
    uint16_t *tmp = malloc((size_t)(x1 - x0) * (y1 - y0 + 2 * r) * sizeof(uint16_t));
    int *acc = malloc((size_t)(x1 - x0) * sizeof(int));
    if (tmp && acc) box_blur_rect(img, out, iw, ih, r, x0, y0, x1, y1, tmp, acc);
    free(tmp); free(acc);
  }
#if defined(OB_NEON) || defined(OB_X86)
  free(ring);
#endif
  return out;
}

void ob_reg_free(ob_reg_t *reg) { free(reg->nodes); reg->nodes = NULL; }

// in_quad: a quad found somewhere else, in image pixels, with its orientation and track score. The scan, the
// line fit and the track scoring are then all skipped and everything from the homography onwards is shared, so
// a quad found on a GPU becomes a registration on exactly the terms one found here does.
// Ls: the layouts the symbol may be (find_frame); the one that registered comes back in *which and is the one the
// registration is built on. An external quad is the first layout's.
static int acquire_core(const ob_layout_t *const *Ls, int nL, const image_t *pim, ob_reg_t *reg, ob_result_t *res,
                        const float *in_quad, int in_orient, float in_score, int *which, ob_fbufs_t *fb) {
  const ob_layout_t *L = Ls[0];
  *which = 0;
  const image_t im = *pim;
  const uint8_t *img = im.px;
  int iw = im.w, ih = im.h;
  reg->nodes = NULL;
  int got = 0, thin_orient = -1, seeded = 0;
  float thin_quad[8], thin_sc = 0;
  lensm_t seed;   // ai: the finder's lens, a traced quad's: the anchors' first model where seeded
  if (in_quad) { memcpy(thin_quad, in_quad, sizeof thin_quad); thin_orient = in_orient; thin_sc = in_score; got = 1; }
  else {
    PROF(PROF_FINDERS, got = find_frame(Ls, nL, &im, fb, thin_quad, &thin_orient, &thin_sc, which, &seed, &seeded));
    L = Ls[*which];
  }
  if (!got) return 0;
#if defined(OB_NEON) || defined(OB_X86)
  uint8_t *bin = frame_buf(fb, OB_FB_BLUR, (size_t)iw * ih);   // ai: border_blur's copy, in the workspace's buffer
#else
  uint8_t *bin = NULL;   // ai: border_blur allocates its own where it blurs
#endif

  double tm = ob_now_ms();
  float src[8];
  corner_coords(L, src);
  homo_t H;
  // The thin frame's finder has already settled rotation: it scores the coded track under every hypothesis.
  if (!homography(src, thin_quad, &H)) { frame_buf_done(fb, bin); return 0; }
  memcpy(res->quad, thin_quad, sizeof thin_quad);
  if (thin_orient < 0 || thin_sc < 0.0f) { frame_buf_done(fb, bin); return 0; }
  res->orient = thin_orient;

  // The node templates read the image under a box a module wide once a module is several pixels (border_blur). The
  // module's size is the quad's sides over the lattice's, in modules.
  image_t rim = im;
  uint8_t *blurred = NULL;
  float mpx = 0;
  for (int k = 0; k < 4; k++) {
    const int k1 = (k + 1) & 3;
    mpx += hypotf(res->quad[2 * k1] - res->quad[2 * k], res->quad[2 * k1 + 1] - res->quad[2 * k + 1]) / hypotf(src[2 * k1] - src[2 * k], src[2 * k1 + 1] - src[2 * k + 1]) / 4;
  }
  // ai: a box radius of at least one pixel, tested before the conversion (a NaN mpx blurs nothing, as before)
  // ai: on arm64 and x86-64 the copy goes into the workspace's buffer (bin)
  if (mpx / 2 >= 1 && (blurred = border_blur(L, &H, img, iw, ih, (int)(mpx / 2), bin))) rim.px = blurred;
  // Finders first, so the homography the marks are measured against is as good as four
  // points can make it.
  int nn = L->nx * L->ny;
  node_t *nodes = calloc((size_t)nn, sizeof(node_t));
  for (int c = 0; c < 4; c++) {
    int i = (c == 1 || c == 2) ? L->nx - 1 : 0, j = c >= 2 ? L->ny - 1 : 0;
    refine_node(L, &rim, &H, i, j, nodes, 0);
    node_t *nd = &nodes[j * L->nx + i];
    res->quad[2 * c] += nd->dx; res->quad[2 * c + 1] += nd->dy;
    nd->done = 0;
  }
  homography(src, res->quad, &H);
  for (int c = 0; c < 4; c++) { node_t *nd = &nodes[((c >= 2) ? L->ny - 1 : 0) * L->nx + ((c == 1 || c == 2) ? L->nx - 1 : 0)]; nd->dx = nd->dy = 0; nd->done = 1; }
  // ai: the anchors: the code registered from the anchors outward (anchor_register: a corner they disagree with
  // ai: searched again from their map), and the lens model they and the corners say, which every border node's search
  // ai: below starts from
  anchor_t anchors[OB_ANCH_K * OB_ANCH_K];
  lensm_t alm;
  const int have_alm = anchor_register(L, &im, &rim, &H, iw, ih, mpx, res->quad, nodes, anchors, &alm, seeded ? &seed : NULL);
  ob_anch_found = 0;
  for (int k = 0; k < OB_ANCH_K * OB_ANCH_K; k++) ob_anch_found += anchors[k].ok;
  // Rings outward from the corners, so each mark starts from measured neighbours.
  // ai: the marked nodes (the border's) in that order, ring by ring and in rows within one, by a counting sort on
  // ai: the ring: a pass over the lattice, not one a ring
  const int nx = L->nx, ny = L->ny;
  int start[OB_MAX_NODES + 2] = { 0 }, order[4 * OB_MAX_NODES], no = 0;
#define RING(i, j) (((i) < nx - 1 - (i) ? (i) : nx - 1 - (i)) + ((j) < ny - 1 - (j) ? (j) : ny - 1 - (j)))
  for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) if (L->node_mark[j * nx + i]) start[RING(i, j) + 1]++;
  for (int k = 1; k < OB_MAX_NODES + 2; k++) start[k] += start[k - 1];
  for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) if (L->node_mark[j * nx + i]) { order[start[RING(i, j)]++] = j * nx + i; no++; }
#undef RING
  for (int k = 0; k < no; k++) {
    if (nodes[order[k]].done) continue;
    refine_node(L, &rim, &H, order[k] % L->nx, order[k] / L->nx, nodes, have_alm ? &alm : 0);
  }
  // ai: a blur made in bin is bin's, given back with it; one made fresh (no spare) freed here
  if (blurred != bin) free(blurred);
  frame_buf_done(fb, bin);
  // the border is all there is to hang the mesh on; under the anchors' minimum (have_alm 0) none of them is used
  if (!have_alm) for (int k = 0; k < OB_ANCH_K * OB_ANCH_K; k++) anchors[k].ok = 0;
  border_fill_anchors(L, &H, iw, ih, nodes, anchors, mpx);
  float msum = 0;
  int mcount = 0;
  for (int k = 0; k < nn; k++) if (nodes[k].done) { msum += nodes[k].score; mcount++; }
  res->mark_score = msum / (mcount ? mcount : 1);
  res->found = 1;
  reg->H = H; reg->nodes = nodes;
  ob_prof_ms[PROF_MESH] += ob_now_ms() - tm;
  return 1;
}

int ob_acquire(const ob_layout_t *L, const image_t *pim, ob_reg_t *reg, ob_result_t *res, ob_fbufs_t *fb) {
  int which;
  return acquire_core(&L, 1, pim, reg, res, NULL, 0, 0, &which, fb);
}
int ob_acquire_any(const ob_layout_t *const *Ls, int nL, const image_t *pim, ob_reg_t *reg, ob_result_t *res, int *which, ob_fbufs_t *fb) {
  return acquire_core(Ls, nL, pim, reg, res, NULL, 0, 0, which, fb);
}
int ob_acquire_quad(const ob_layout_t *L, const image_t *pim, ob_reg_t *reg, ob_result_t *res,
                    const float *quad, int orient, float score, ob_fbufs_t *fb) {
  int which;
  return acquire_core(&L, 1, pim, reg, res, quad, orient, score, &which, fb);
}

// ai: the mesh cell m falls in along one axis (node ascending, nn of them): the last k of 0 to nn - 2 with k = 0 or
// ai: m >= node[k], by halving (the walk from 0 it replaces gave the same k, NaN included: 0)
static inline int node_cell(const float *node, int nn, float m) {
  int lo = 0, hi = nn - 2;
  while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (m >= node[mid]) lo = mid; else hi = mid - 1; }
  return lo;
}
void ob_map_point(const ob_layout_t *L, const ob_reg_t *reg, float mx, float my, float *pu, float *pv) {
  const int j = node_cell(L->node_y, L->ny, my), i = node_cell(L->node_x, L->nx, mx);
  float v = (my - L->node_y[j]) / (L->node_y[j + 1] - L->node_y[j]), u = (mx - L->node_x[i]) / (L->node_x[i + 1] - L->node_x[i]);
  const node_t *n00 = &reg->nodes[j * L->nx + i], *n10 = n00 + 1, *n01 = n00 + L->nx, *n11 = n01 + 1;
  float dx = (n00->dx * (1 - u) + n10->dx * u) * (1 - v) + (n01->dx * (1 - u) + n11->dx * u) * v;
  float dy = (n00->dy * (1 - u) + n10->dy * u) * (1 - v) + (n01->dy * (1 - u) + n11->dy * u) * v;
  project(&reg->H, mx, my, pu, pv);
  *pu += dx; *pv += dy;
}

// Any regular grid of points in module coordinates, through the homography and the mesh: point (xx, yy) of the
// block is at (mx0 + (x0 + xx) * step, my0 + (y0 + yy) * step) and lands in out[yy * pitch + xx]. Points up to the
// next mesh column share their four nodes and differ in mx alone, so a row is walked in such runs, four points
// to a vector. The picture is such a grid, n samples across its span, taken a strip of columns at a time.
// Which mesh column a point falls in depends on its column alone, so the runs are found once, not once a row,
// and point by point with the comparison the scalar walk makes, so no point can land in a different column.
static inline int grid_runs(const ob_layout_t *L, float mx0, float step, int x0, int nx, int *run_i, int *run_end) {
  int runs = 0;
  for (int xx = 0, i = 0; xx < nx; runs++) {
    while (i < L->nx - 2 && mx0 + (float)(x0 + xx) * step >= L->node_x[i + 1]) i++;
    int e = xx + 1;
    while (e < nx && !(i < L->nx - 2 && mx0 + (float)(x0 + e) * step >= L->node_x[i + 1])) e++;
    run_i[runs] = i; run_end[runs] = e; xx = e;
  }
  return runs;
}
void ob_sample_grid(const ob_layout_t *L, const image_t *pim, const ob_reg_t *reg, float mx0, float my0, float step, int x0, int nx, int y0, int ny, float *out, int pitch) {
  const image_t im = *pim;
  const homo_t H = reg->H;
  const node_t *nodes = reg->nodes;
  int run_i[OB_MAX_NODES + 1], run_end[OB_MAX_NODES + 1];
  const int runs = grid_runs(L, mx0, step, x0, nx, run_i, run_end);
  // ai: the rows go down the grid, so the mesh row a row falls in only moves on: the same comparisons as a search from 0
  int j = 0;
  for (int yy = 0; yy < ny; yy++) {
    float my = my0 + (float)(y0 + yy) * step;
    while (j < L->ny - 2 && my >= L->node_y[j + 1]) j++;
    float v = (my - L->node_y[j]) / (L->node_y[j + 1] - L->node_y[j]);
    // ai: Along a row, mx is the one thing that changes, and within a run the mesh's displacement is linear in it, so
    // ai: everything but the homography's division is a + b mx: dx = cx + ex mx (the cell's bilinear blend at this v,
    // ai: its slope over the column's width folded in), the homography's numerators and denominator likewise, and one
    // ai: reciprocal of the denominator for both. (2026-10-26: about 35 operations and three divisions a vector before,
    // ai: the division in u on the critical path; the same map to rounding.)
    const float kw = H.h[7] * my + 1, ku = H.h[1] * my + H.h[2], kv = H.h[4] * my + H.h[5];
#ifdef OB_SIMD
    const v128_t one = wasm_f32x4_splat(1), four = wasm_f32x4_splat(4);
    const v128_t h0 = wasm_f32x4_splat(H.h[0]), h3 = wasm_f32x4_splat(H.h[3]), h6 = wasm_f32x4_splat(H.h[6]);
    const v128_t vku = wasm_f32x4_splat(ku), vkv = wasm_f32x4_splat(kv), vkw = wasm_f32x4_splat(kw);
    const v128_t vmx0 = wasm_f32x4_splat(mx0), vstep = wasm_f32x4_splat(step);
    // ai: Each run's terms first, then the row four points at a time from its first point: a vector across a run's end
    // ai: takes each lane's own run's terms, so every point is worked out as the scalar walk works it out. (2026-11-01:
    // ai: each run took its own vectors from its own first point, and its last one, reaching past its end, was worked
    // ai: out again by the next run: about one vector a run, in a strip of 16, and a loop exit mispredicted a run.)
    float tex[OB_MAX_NODES + 1], tcx[OB_MAX_NODES + 1], tey[OB_MAX_NODES + 1], tcy[OB_MAX_NODES + 1];
    for (int r = 0; r < runs; r++) {
      const int i = run_i[r];
      const node_t *n00 = &nodes[j * L->nx + i], *n10 = n00 + 1, *n01 = n00 + L->nx, *n11 = n01 + 1;
      const float inv = 1.0f / (L->node_x[i + 1] - L->node_x[i]);
      tex[r] = ((n10->dx - n00->dx) * (1 - v) + (n11->dx - n01->dx) * v) * inv; tcx[r] = n00->dx * (1 - v) + n01->dx * v - tex[r] * L->node_x[i];
      tey[r] = ((n10->dy - n00->dy) * (1 - v) + (n11->dy - n01->dy) * v) * inv; tcy[r] = n00->dy * (1 - v) + n01->dy * v - tey[r] * L->node_x[i];
    }
    v128_t vex = wasm_f32x4_splat(tex[0]), vcx = wasm_f32x4_splat(tcx[0]), vey = wasm_f32x4_splat(tey[0]), vcy = wasm_f32x4_splat(tcy[0]);
    // The column index as floats, stepped by four: whole numbers, so exact.
    v128_t vgx = wasm_f32x4_make((float)x0, (float)(x0 + 1), (float)(x0 + 2), (float)(x0 + 3));
    for (int xx = 0, r = 0; xx < nx; xx += 4, vgx = wasm_f32x4_add(vgx, four)) {
      if (xx >= run_end[r]) {
        while (xx >= run_end[r]) r++;
        vex = wasm_f32x4_splat(tex[r]); vcx = wasm_f32x4_splat(tcx[r]); vey = wasm_f32x4_splat(tey[r]); vcy = wasm_f32x4_splat(tcy[r]);
      }
      v128_t ex4 = vex, cx4 = vcx, ey4 = vey, cy4 = vcy;
      for (int q = r; q + 1 < runs && run_end[q] < xx + 4; q++) {   // the lanes from run q's end on are run q + 1's
        const v128_t past = wasm_f32x4_le(wasm_f32x4_splat((float)(x0 + run_end[q])), vgx);
        ex4 = wasm_v128_bitselect(wasm_f32x4_splat(tex[q + 1]), ex4, past); cx4 = wasm_v128_bitselect(wasm_f32x4_splat(tcx[q + 1]), cx4, past);
        ey4 = wasm_v128_bitselect(wasm_f32x4_splat(tey[q + 1]), ey4, past); cy4 = wasm_v128_bitselect(wasm_f32x4_splat(tcy[q + 1]), cy4, past);
      }
      const v128_t vmx = wasm_f32x4_add(vmx0, wasm_f32x4_mul(vgx, vstep));
      const v128_t rw = wasm_f32x4_div(one, wasm_f32x4_add(wasm_f32x4_mul(h6, vmx), vkw));
      const v128_t pu = wasm_f32x4_mul(wasm_f32x4_add(wasm_f32x4_mul(h0, vmx), vku), rw), pv = wasm_f32x4_mul(wasm_f32x4_add(wasm_f32x4_mul(h3, vmx), vkv), rw);
      const v128_t dx = wasm_f32x4_add(cx4, wasm_f32x4_mul(ex4, vmx)), dy = wasm_f32x4_add(cy4, wasm_f32x4_mul(ey4, vmx));
      const v128_t got = sample4(&im, wasm_f32x4_add(pu, dx), wasm_f32x4_add(pv, dy));
      if (xx + 4 <= nx) { wasm_v128_store(out + yy * pitch + xx, got); continue; }
      float t[4];   // the row's last vector, past its end: those lanes are dropped
      wasm_v128_store(t, got);
      for (int l = 0; xx + l < nx; l++) out[yy * pitch + xx + l] = t[l];
    }
#else
    for (int r = 0, xx = 0; r < runs; r++) {
      const int i = run_i[r], end = run_end[r];
      const node_t *n00 = &nodes[j * L->nx + i], *n10 = n00 + 1, *n01 = n00 + L->nx, *n11 = n01 + 1;
      const float inv = 1.0f / (L->node_x[i + 1] - L->node_x[i]);
      const float ex = ((n10->dx - n00->dx) * (1 - v) + (n11->dx - n01->dx) * v) * inv, cx = n00->dx * (1 - v) + n01->dx * v - ex * L->node_x[i];
      const float ey = ((n10->dy - n00->dy) * (1 - v) + (n11->dy - n01->dy) * v) * inv, cy = n00->dy * (1 - v) + n01->dy * v - ey * L->node_x[i];
      for (; xx < end; xx++) {
        const float mx = mx0 + (float)(x0 + xx) * step, rw = 1 / (H.h[6] * mx + kw);
        const float pu = (H.h[0] * mx + ku) * rw, pv = (H.h[3] * mx + kv) * rw;
        const float dx = cx + ex * mx, dy = cy + ey * mx;
        out[yy * pitch + xx] = sample(&im, pu + dx, pv + dy);
      }
    }
#endif
  }
}
#ifdef OB_NEON
// ai: arm64, the decode's picture at gamma 1 (ob_sample_grid_pic), the same map with its read in integers (its own
// ai: values; the blocks a decode verifies are its test). Every term of a row is a line in the column: the homography's
// ai: numerators and denominator stepped by four columns a vector, the mesh's displacement D + v E tabled a column at a
// ai: time once a mesh row (D on mesh row j less the half pixel, E the step to row j + 1; D x, E x, D y, E y for four
// ai: columns side by side, two LDP a vector), one division a vector. The point's x and y to 16.16 by FCVTZU (below 0
// ai: is 0), their integer parts by UZP2, clamped on 16-bit lanes to w - 2 and h - 2 (so the reads stay in the frame),
// ai: fx the fraction's top byte (rounded: x is a 512th of a pixel on, in D), fy its top 15 bits. A point's two pixel
// ai: pairs are one word in a general register: row y's from the pixel and row y + 1's from two bytes before, merged
// ai: by BFXIL (lin_word_lo, lin_word); TBL sorts eight points' words into a b and c d; each row's pair blended on
// ai: bytes, 256 a + (b - a) fx (USHLL, UMLAL2, UMLSL: weights summing to 256, as the fraction), the two rows by
// ai: SQRDMULH on their halved difference times fy, the result (128 times the grey) to floats by one widening multiply
// ai: by 257 and a conversion by 2^-23 (1 / 32640 within 2e-5). A row is taken 16 points a step, the next step's
// ai: offsets and weights worked out before this step's reads (lin_pts, lin_read), so the reads do not wait on the
// ai: division. At most GRID_LIN columns a pass. (2026-11-02, an S26 prime core, pictures of 768 to 1536: 2.41
// ai: cycles a point in the strips, 0.51 ns at 4.74 GHz, against 3.26 for the float form before it, sample4r four
// ai: vectors at a time: its blend, clamps and fractions in floats, its words two transfers a point; 13.9 instructions
// ai: a point against 19.8. Its values within 3e-3 of the grey range of the float form's on a frame of random bytes,
// ai: 3e-4 on average; verified blocks over the eleven S26 replays and five cells +2 in 173 810, -11 to +10 a run,
// ai: where a shift of the float form's points by 4e-6 pixels moves them -2 to +8 a run. Weights summing to 255, the
// ai: fraction cut: -19.)
enum { GRID_LIN = 256 };
static void sample_grid_lin(const ob_layout_t *L, const image_t *pim, const ob_reg_t *reg, float mx0, float my0, float step, int x0, int nx, int y0, int ny, float *out, int pitch) {
  const int w = pim->w, nn = L->nx;
  const float *h = reg->H.h;
  int run_i[OB_MAX_NODES + 1], run_end[OB_MAX_NODES + 1];
  const int runs = grid_runs(L, mx0, step, x0, nx, run_i, run_end);
  // ai: a column's place in its mesh column (0 to 1), and the table, 24 columns past the end (zeros) for the step that
  // ai: works out points past the row
  enum { TW = GRID_LIN + 24 };
  float tu[TW], T[4 * TW];
  for (int r = 0, k = 0; r < runs; r++) {
    const int i = run_i[r];
    const float inv = 1.0f / (L->node_x[i + 1] - L->node_x[i]);
    for (; k < run_end[r]; k++) tu[k] = (mx0 + (float)(x0 + k) * step - L->node_x[i]) * inv;
  }
  for (int k = nx; k < nx + 4; k++) tu[k] = 0;
  for (int i = 4 * ((nx + 3) & ~3); i < 4 * (nx + 24); i++) T[i] = 0;
  const float m0 = mx0 + (float)x0 * step;
  const float32x4_t lane = { 0, 1, 2, 3 }, one = vdupq_n_f32(1);
  const float32x4_t BwL = vmulq_n_f32(lane, h[6] * step), BuL = vmulq_n_f32(lane, h[0] * step), BvL = vmulq_n_f32(lane, h[3] * step);
  // ai: the three lines' steps over four columns, and v (lane 3, a row's)
  float32x4_t C = { 4 * (h[6] * step), 4 * (h[0] * step), 4 * (h[3] * step), 0 };
  const lin_k_t K = lin_k(w, pim->h);
  const uint8_t *r0 = pim->px, *r1 = pim->px + w - 2;
  int j = 0, built = -1;
  for (int yy = 0; yy < ny; yy++) {
    const float my = my0 + (float)(y0 + yy) * step;
    while (j < L->ny - 2 && my >= L->node_y[j + 1]) j++;
    if (j != built) {
      const node_t *n0 = reg->nodes + j * nn, *n1 = n0 + nn;
      for (int r = 0, s = 0; r < runs; s = run_end[r++]) {
        const node_t *a = n0 + run_i[r], *b = n1 + run_i[r];
        const float32x4_t c0x = vdupq_n_f32(a[0].dx - (0.5f - 1.0f / 512)), c1x = vdupq_n_f32(a[1].dx - a[0].dx), e0x = vdupq_n_f32(b[0].dx - a[0].dx), e1x = vdupq_n_f32((b[1].dx - b[0].dx) - (a[1].dx - a[0].dx));
        const float32x4_t c0y = vdupq_n_f32(a[0].dy - 0.5f), c1y = vdupq_n_f32(a[1].dy - a[0].dy), e0y = vdupq_n_f32(b[0].dy - a[0].dy), e1y = vdupq_n_f32((b[1].dy - b[0].dy) - (a[1].dy - a[0].dy));
        // ai: four columns a vector from the one holding the run's first; its lanes before the run are the last run's
        // ai: (kept), and a vector past the run's end is rewritten by the next run
        for (int k = s & ~3; k < run_end[r]; k += 4) {
          const float32x4_t U = vld1q_f32(tu + k);
          float32x4_t dx = vfmaq_f32(c0x, c1x, U), ex = vfmaq_f32(e0x, e1x, U), dy = vfmaq_f32(c0y, c1y, U), ey = vfmaq_f32(e0y, e1y, U);
          if (k < s) {
            const uint32x4_t keep = vcltq_f32(vaddq_f32(vdupq_n_f32((float)k), lane), vdupq_n_f32((float)s));
            dx = vbslq_f32(keep, vld1q_f32(T + 4 * k), dx); ex = vbslq_f32(keep, vld1q_f32(T + 4 * k + 4), ex);
            dy = vbslq_f32(keep, vld1q_f32(T + 4 * k + 8), dy); ey = vbslq_f32(keep, vld1q_f32(T + 4 * k + 12), ey);
          }
          vst1q_f32(T + 4 * k, dx); vst1q_f32(T + 4 * k + 4, ex); vst1q_f32(T + 4 * k + 8, dy); vst1q_f32(T + 4 * k + 12, ey);
        }
      }
      built = j;
    }
    C = vsetq_lane_f32((my - L->node_y[j]) / (L->node_y[j + 1] - L->node_y[j]), C, 3);
    float32x4_t nw = vaddq_f32(BwL, vdupq_n_f32(h[6] * m0 + (h[7] * my + 1))), nu = vaddq_f32(BuL, vdupq_n_f32(h[0] * m0 + (h[1] * my + h[2]))), nv = vaddq_f32(BvL, vdupq_n_f32(h[3] * m0 + (h[4] * my + h[5])));
    // ai: the point's coordinates less the half pixel (in D); the steps by FMLA by element (exact as an add)
#define LIN_XY(K_, X, Y) do { \
      const float *tk = T + 4 * (K_); \
      const float32x4_t rw_ = vdivq_f32(one, nw); \
      X = vfmaq_f32(vfmaq_laneq_f32(vld1q_f32(tk), vld1q_f32(tk + 4), C, 3), nu, rw_); \
      Y = vfmaq_f32(vfmaq_laneq_f32(vld1q_f32(tk + 8), vld1q_f32(tk + 12), C, 3), nv, rw_); \
      nw = vfmaq_laneq_f32(nw, one, C, 0); nu = vfmaq_laneq_f32(nu, one, C, 1); nv = vfmaq_laneq_f32(nv, one, C, 2); \
    } while (0)
    float *o = out + (size_t)yy * pitch;
    float32x4_t xa, ya, xb, yb, xc, yc, xd, yd;
    LIN_XY(0, xa, ya); LIN_XY(4, xb, yb); LIN_XY(8, xc, yc); LIN_XY(12, xd, yd);
    lin_pts_t pa = lin_pts(&K, xa, ya, xb, yb), pb = lin_pts(&K, xc, yc, xd, yd);
    int k = 0;
    for (; k + 16 < nx; k += 16) {
      LIN_XY(k + 16, xa, ya); LIN_XY(k + 20, xb, yb); LIN_XY(k + 24, xc, yc); LIN_XY(k + 28, xd, yd);
      const lin_pts_t na = lin_pts(&K, xa, ya, xb, yb), nb = lin_pts(&K, xc, yc, xd, yd);
      lin_read(r0, r1, &K, pa, o + k); lin_read(r0, r1, &K, pb, o + k + 8);
      pa = na; pb = nb;
    }
#undef LIN_XY
    if (k + 16 <= nx) { lin_read(r0, r1, &K, pa, o + k); lin_read(r0, r1, &K, pb, o + k + 8); continue; }
    float t[16];   // the row's last points, past its end: those are dropped
    lin_read(r0, r1, &K, pa, t); lin_read(r0, r1, &K, pb, t + 8);
    for (int l = 0; k + l < nx; l++) o[k + l] = t[l];
  }
}
#elif defined(OB_X86)
// ai: x86-64, the decode's picture at gamma 1: arm64's walk above in SSE4.1 (the lines in the column stepped by four
// ai: columns a vector, the mesh tabled per column as D + v E once a mesh row, one division a vector, 16 points a step,
// ai: the next step's offsets and weights before this step's reads), its read in integers (lin_pts, lin_read: from the
// ai: same coordinates the same values). Its own values: no fused multiply-adds, so its coordinates round apart from
// ai: arm64's; the blocks a decode verifies are its test, and test/sample_grid.c holds it to ob_sample_grid.
enum { GRID_LIN = 256 };
static void sample_grid_lin(const ob_layout_t *L, const image_t *pim, const ob_reg_t *reg, float mx0, float my0, float step, int x0, int nx, int y0, int ny, float *out, int pitch) {
  const int w = pim->w, nn = L->nx;
  const float *h = reg->H.h;
  int run_i[OB_MAX_NODES + 1], run_end[OB_MAX_NODES + 1];
  const int runs = grid_runs(L, mx0, step, x0, nx, run_i, run_end);
  // ai: a column's place in its mesh column (0 to 1), and the table, 24 columns past the end (zeros) for the step that
  // ai: works out points past the row
  enum { TW = GRID_LIN + 24 };
  float tu[TW], T[4 * TW];
  for (int r = 0, k = 0; r < runs; r++) {
    const int i = run_i[r];
    const float inv = 1.0f / (L->node_x[i + 1] - L->node_x[i]);
    for (; k < run_end[r]; k++) tu[k] = (mx0 + (float)(x0 + k) * step - L->node_x[i]) * inv;
  }
  for (int k = nx; k < nx + 4; k++) tu[k] = 0;
  for (int i = 4 * ((nx + 3) & ~3); i < 4 * (nx + 24); i++) T[i] = 0;
  const float m0 = mx0 + (float)x0 * step;
  const __m128 lane = _mm_setr_ps(0, 1, 2, 3), one = _mm_set1_ps(1);
  const __m128 BwL = _mm_mul_ps(lane, _mm_set1_ps(h[6] * step)), BuL = _mm_mul_ps(lane, _mm_set1_ps(h[0] * step)), BvL = _mm_mul_ps(lane, _mm_set1_ps(h[3] * step));
  // ai: the three lines' steps over four columns
  const __m128 Cw = _mm_set1_ps(4 * (h[6] * step)), Cu = _mm_set1_ps(4 * (h[0] * step)), Cv = _mm_set1_ps(4 * (h[3] * step));
  const lin_k_t K = lin_k(w, pim->h);
  const uint8_t *r0 = pim->px, *r1 = pim->px + w - 2;
  int j = 0, built = -1;
#define LIN_MA(a, b, c) _mm_add_ps((a), _mm_mul_ps((b), (c)))
  for (int yy = 0; yy < ny; yy++) {
    const float my = my0 + (float)(y0 + yy) * step;
    while (j < L->ny - 2 && my >= L->node_y[j + 1]) j++;
    if (j != built) {
      const node_t *n0 = reg->nodes + j * nn, *n1 = n0 + nn;
      for (int r = 0, s = 0; r < runs; s = run_end[r++]) {
        const node_t *a = n0 + run_i[r], *b = n1 + run_i[r];
        const __m128 c0x = _mm_set1_ps(a[0].dx - (0.5f - 1.0f / 512)), c1x = _mm_set1_ps(a[1].dx - a[0].dx), e0x = _mm_set1_ps(b[0].dx - a[0].dx), e1x = _mm_set1_ps((b[1].dx - b[0].dx) - (a[1].dx - a[0].dx));
        const __m128 c0y = _mm_set1_ps(a[0].dy - 0.5f), c1y = _mm_set1_ps(a[1].dy - a[0].dy), e0y = _mm_set1_ps(b[0].dy - a[0].dy), e1y = _mm_set1_ps((b[1].dy - b[0].dy) - (a[1].dy - a[0].dy));
        // ai: four columns a vector from the one holding the run's first; its lanes before the run are the last run's
        // ai: (kept), and a vector past the run's end is rewritten by the next run
        for (int k = s & ~3; k < run_end[r]; k += 4) {
          const __m128 U = _mm_loadu_ps(tu + k);
          __m128 dx = LIN_MA(c0x, c1x, U), ex = LIN_MA(e0x, e1x, U), dy = LIN_MA(c0y, c1y, U), ey = LIN_MA(e0y, e1y, U);
          if (k < s) {
            const __m128 keep = _mm_cmplt_ps(_mm_add_ps(_mm_set1_ps((float)k), lane), _mm_set1_ps((float)s));
            dx = _mm_blendv_ps(dx, _mm_loadu_ps(T + 4 * k), keep); ex = _mm_blendv_ps(ex, _mm_loadu_ps(T + 4 * k + 4), keep);
            dy = _mm_blendv_ps(dy, _mm_loadu_ps(T + 4 * k + 8), keep); ey = _mm_blendv_ps(ey, _mm_loadu_ps(T + 4 * k + 12), keep);
          }
          _mm_storeu_ps(T + 4 * k, dx); _mm_storeu_ps(T + 4 * k + 4, ex); _mm_storeu_ps(T + 4 * k + 8, dy); _mm_storeu_ps(T + 4 * k + 12, ey);
        }
      }
      built = j;
    }
    const __m128 vv = _mm_set1_ps((my - L->node_y[j]) / (L->node_y[j + 1] - L->node_y[j]));
    __m128 nw = _mm_add_ps(BwL, _mm_set1_ps(h[6] * m0 + (h[7] * my + 1))), nu = _mm_add_ps(BuL, _mm_set1_ps(h[0] * m0 + (h[1] * my + h[2]))), nv = _mm_add_ps(BvL, _mm_set1_ps(h[3] * m0 + (h[4] * my + h[5])));
    // ai: the point's coordinates less the half pixel (in D)
#define LIN_XY(K_, X, Y) do { \
      const float *tk = T + 4 * (K_); \
      const __m128 rw_ = _mm_div_ps(one, nw); \
      X = LIN_MA(LIN_MA(_mm_loadu_ps(tk), _mm_loadu_ps(tk + 4), vv), nu, rw_); \
      Y = LIN_MA(LIN_MA(_mm_loadu_ps(tk + 8), _mm_loadu_ps(tk + 12), vv), nv, rw_); \
      nw = _mm_add_ps(nw, Cw); nu = _mm_add_ps(nu, Cu); nv = _mm_add_ps(nv, Cv); \
    } while (0)
    float *o = out + (size_t)yy * pitch;
    __m128 xa, ya, xb, yb, xc, yc, xd, yd;
    LIN_XY(0, xa, ya); LIN_XY(4, xb, yb); LIN_XY(8, xc, yc); LIN_XY(12, xd, yd);
    lin_pts_t pa = lin_pts(&K, xa, ya, xb, yb), pb = lin_pts(&K, xc, yc, xd, yd);
    int k = 0;
    for (; k + 16 < nx; k += 16) {
      LIN_XY(k + 16, xa, ya); LIN_XY(k + 20, xb, yb); LIN_XY(k + 24, xc, yc); LIN_XY(k + 28, xd, yd);
      const lin_pts_t na = lin_pts(&K, xa, ya, xb, yb), nb = lin_pts(&K, xc, yc, xd, yd);
      lin_read(r0, r1, &K, pa, o + k); lin_read(r0, r1, &K, pb, o + k + 8);
      pa = na; pb = nb;
    }
#undef LIN_XY
    if (k + 16 <= nx) { lin_read(r0, r1, &K, pa, o + k); lin_read(r0, r1, &K, pb, o + k + 8); continue; }
    float t[16];   // the row's last points, past its end: those are dropped
    lin_read(r0, r1, &K, pa, t); lin_read(r0, r1, &K, pb, t + 8);
    for (int l = 0; k + l < nx; l++) o[k + l] = t[l];
  }
#undef LIN_MA
}
#ifdef OB_V3_ON
// ai: x86-64-v3 (2026-11-05): the walk above, eight columns a vector, its read sixteen points a call (lin16_*, beside
// ai: the v2 read), every value as the v2 form's: the same operations in the same order (no fused multiply-adds; an
// ai: eight-lane vector holds two of the v2 form's four-column groups, so the lines step by two of its adds), so the
// ai: output is the v2 form's bit for bit (test/sample_grid.c).
OB_V3 static void sample_grid_lin_v3(const ob_layout_t *L, const image_t *pim, const ob_reg_t *reg, float mx0, float my0, float step, int x0, int nx, int y0, int ny, float *out, int pitch) {
  const int w = pim->w, nn = L->nx;
  const float *h = reg->H.h;
  int run_i[OB_MAX_NODES + 1], run_end[OB_MAX_NODES + 1];
  const int runs = grid_runs(L, mx0, step, x0, nx, run_i, run_end);
  // ai: the table eight columns a group ([dx 8, ex 8, dy 8, ey 8] at 4 k), zeros past the end for the steps past the row
  enum { TW = GRID_LIN + 32 };
  float tu[TW], T[4 * TW];
  for (int r = 0, k = 0; r < runs; r++) {
    const int i = run_i[r];
    const float inv = 1.0f / (L->node_x[i + 1] - L->node_x[i]);
    for (; k < run_end[r]; k++) tu[k] = (mx0 + (float)(x0 + k) * step - L->node_x[i]) * inv;
  }
  for (int k = nx; k < nx + 8; k++) tu[k] = 0;
  for (int i = 4 * ((nx + 7) & ~7); i < 4 * (nx + 32); i++) T[i] = 0;
  const float m0 = mx0 + (float)x0 * step;
  const __m128 lane4 = _mm_setr_ps(0, 1, 2, 3);
  const __m256 lane8 = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), one = _mm256_set1_ps(1);
  const __m128 BwL = _mm_mul_ps(lane4, _mm_set1_ps(h[6] * step)), BuL = _mm_mul_ps(lane4, _mm_set1_ps(h[0] * step)), BvL = _mm_mul_ps(lane4, _mm_set1_ps(h[3] * step));
  // ai: the v2 form's step over its four columns; two of them an eight-column step
  const __m128 Cw4 = _mm_set1_ps(4 * (h[6] * step)), Cu4 = _mm_set1_ps(4 * (h[0] * step)), Cv4 = _mm_set1_ps(4 * (h[3] * step));
  const __m256 Cw = _mm256_set1_ps(4 * (h[6] * step)), Cu = _mm256_set1_ps(4 * (h[0] * step)), Cv = _mm256_set1_ps(4 * (h[3] * step));
  const lin16_k_t K = { _mm256_set1_epi16((short)(w - 2)), _mm256_set1_epi16((short)(pim->h - 2)), _mm256_set1_epi16((short)w) };
  const uint8_t *r0 = pim->px, *r1 = pim->px + w - 2;
  int j = 0, built = -1;
#define LIN8_MA(a, b, c) _mm256_add_ps((a), _mm256_mul_ps((b), (c)))
  for (int yy = 0; yy < ny; yy++) {
    const float my = my0 + (float)(y0 + yy) * step;
    while (j < L->ny - 2 && my >= L->node_y[j + 1]) j++;
    if (j != built) {
      const node_t *n0 = reg->nodes + j * nn, *n1 = n0 + nn;
      for (int r = 0, s = 0; r < runs; s = run_end[r++]) {
        const node_t *a = n0 + run_i[r], *b = n1 + run_i[r];
        const __m256 c0x = _mm256_set1_ps(a[0].dx - (0.5f - 1.0f / 512)), c1x = _mm256_set1_ps(a[1].dx - a[0].dx), e0x = _mm256_set1_ps(b[0].dx - a[0].dx), e1x = _mm256_set1_ps((b[1].dx - b[0].dx) - (a[1].dx - a[0].dx));
        const __m256 c0y = _mm256_set1_ps(a[0].dy - 0.5f), c1y = _mm256_set1_ps(a[1].dy - a[0].dy), e0y = _mm256_set1_ps(b[0].dy - a[0].dy), e1y = _mm256_set1_ps((b[1].dy - b[0].dy) - (a[1].dy - a[0].dy));
        for (int k = s & ~7; k < run_end[r]; k += 8) {
          const __m256 U = _mm256_loadu_ps(tu + k);
          __m256 dx = LIN8_MA(c0x, c1x, U), ex = LIN8_MA(e0x, e1x, U), dy = LIN8_MA(c0y, c1y, U), ey = LIN8_MA(e0y, e1y, U);
          if (k < s) {
            const __m256 keep = _mm256_cmp_ps(_mm256_add_ps(_mm256_set1_ps((float)k), lane8), _mm256_set1_ps((float)s), _CMP_LT_OQ);
            dx = _mm256_blendv_ps(dx, _mm256_loadu_ps(T + 4 * k), keep); ex = _mm256_blendv_ps(ex, _mm256_loadu_ps(T + 4 * k + 8), keep);
            dy = _mm256_blendv_ps(dy, _mm256_loadu_ps(T + 4 * k + 16), keep); ey = _mm256_blendv_ps(ey, _mm256_loadu_ps(T + 4 * k + 24), keep);
          }
          _mm256_storeu_ps(T + 4 * k, dx); _mm256_storeu_ps(T + 4 * k + 8, ex); _mm256_storeu_ps(T + 4 * k + 16, dy); _mm256_storeu_ps(T + 4 * k + 24, ey);
        }
      }
      built = j;
    }
    const __m256 vv = _mm256_set1_ps((my - L->node_y[j]) / (L->node_y[j + 1] - L->node_y[j]));
    const __m128 w0 = _mm_add_ps(BwL, _mm_set1_ps(h[6] * m0 + (h[7] * my + 1))), u0 = _mm_add_ps(BuL, _mm_set1_ps(h[0] * m0 + (h[1] * my + h[2]))), v0 = _mm_add_ps(BvL, _mm_set1_ps(h[3] * m0 + (h[4] * my + h[5])));
    __m256 nw = _mm256_set_m128(_mm_add_ps(w0, Cw4), w0), nu = _mm256_set_m128(_mm_add_ps(u0, Cu4), u0), nv = _mm256_set_m128(_mm_add_ps(v0, Cv4), v0);
#define LIN8_XY(K_, X, Y) do { \
      const float *tk = T + 4 * (K_); \
      const __m256 rw_ = _mm256_div_ps(one, nw); \
      X = LIN8_MA(LIN8_MA(_mm256_loadu_ps(tk), _mm256_loadu_ps(tk + 8), vv), nu, rw_); \
      Y = LIN8_MA(LIN8_MA(_mm256_loadu_ps(tk + 16), _mm256_loadu_ps(tk + 24), vv), nv, rw_); \
      nw = _mm256_add_ps(_mm256_add_ps(nw, Cw), Cw); nu = _mm256_add_ps(_mm256_add_ps(nu, Cu), Cu); nv = _mm256_add_ps(_mm256_add_ps(nv, Cv), Cv); \
    } while (0)
    float *o = out + (size_t)yy * pitch;
    __m256 xa, ya, xb, yb;
    LIN8_XY(0, xa, ya); LIN8_XY(8, xb, yb);
    lin16_pts_t p = lin16_pts(&K, xa, ya, xb, yb);
    int k = 0;
    for (; k + 16 < nx; k += 16) {
      LIN8_XY(k + 16, xa, ya); LIN8_XY(k + 24, xb, yb);
      const lin16_pts_t np = lin16_pts(&K, xa, ya, xb, yb);
      lin16_read(r0, r1, p, o + k);
      p = np;
    }
#undef LIN8_XY
    if (k + 16 <= nx) { lin16_read(r0, r1, p, o + k); continue; }
    float t[16];   // the row's last points, past its end: those are dropped
    lin16_read(r0, r1, p, t);
    for (int l = 0; k + l < nx; l++) o[k + l] = t[l];
  }
#undef LIN8_MA
}
#undef OB_V3_IN
#endif
#endif
void ob_sample_grid_pic(const ob_layout_t *L, const image_t *pim, const ob_reg_t *reg, float mx0, float my0, float step, int x0, int nx, int y0, int ny, float *out, int pitch) {
#if defined(OB_NEON) || defined(OB_X86)
  if (pim->lin) {
#ifdef OB_V3_ON
    if (ob_v3()) {
      for (int c = 0; c < nx; c += GRID_LIN) sample_grid_lin_v3(L, pim, reg, mx0, my0, step, x0 + c, nx - c < GRID_LIN ? nx - c : GRID_LIN, y0, ny, out + c, pitch);
      return;
    }
#endif
    for (int c = 0; c < nx; c += GRID_LIN) sample_grid_lin(L, pim, reg, mx0, my0, step, x0 + c, nx - c < GRID_LIN ? nx - c : GRID_LIN, y0, ny, out + c, pitch);
    return;
  }
#endif
  ob_sample_grid(L, pim, reg, mx0, my0, step, x0, nx, y0, ny, out, pitch);
}

