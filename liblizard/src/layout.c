#include "layout.h"
#include "fmt.h"
#include "simd.h"
#include <stdlib.h>
#include <string.h>

// ai: The block's CRC-64 (SPEC 7.1), since 2026-11-21 (CRC-32C from 2026-11-05, IEEE 802.3's CRC-32 before): a
// ai: BCH-derived generator, G = (x + 1) m1 m3 m5 m7 m9 (x^3 + x + 1) over GF(2^12), 0xE0CB4CBE2D1C3403 with x^64,
// ai: reflected 0xC02C38B47D32D307; the register set to all ones and the result inverted; "123456789" gives
// ai: 0xBB3BA0D234061B3E. Roots alpha^0 .. alpha^10: distance at least 12 on any codeword to 4,095 bits (the block and
// ai: its CRC are 3,840), so every error of up to 11 bits, every odd-weight error and every burst to 64 bits is caught.
// ai: By carry-less multiplication where the processor has it (x86-64's PCLMULQDQ by cpuid, arm64's PMULL by the
// ai: kernel's HWCAP_PMULL, always on Apple's), over the bytes to the last whole 16; the rest, and every other target
// ai: (the wasm, OB_SCALAR), sliced by eight through crc64_tables.h. The same value for every input (test/crc_check.c
// ai: holds both to the bit loop).
#include "crc64_tables.h"
// ai: The fold (both targets, one arithmetic; research/build/crc64-clmul-2026-11-21/model.py derives the constants
// ai: and models every instruction against the bit loop). In the reflected domain a 128-bit lane D bits before the
// ai: lane it lands on is folded onto it as its low qword times x^(D + 63) mod P and its high qword times x^(D - 1) mod
// ai: P, each reflected (the reflected product carries one factor x): four lanes 64 bytes apart while 64 bytes remain,
// ai: folded into one (by 384, 256 and 128 bits), then a lane at a time; the last lane times x^64 (its low qword by
// ai: x^127 mod P, its high qword moved down), then Barrett's reduction to 64 bits: q = th + floor(th mu' / x^64), mu'
// ai: = floor(x^128 / P) - x^64, the register tl + (q P' mod x^64). n a multiple of 16, 64 or more; c the register
// ai: as it stands, returned as it ends (neither inverted).
#define CRC64_K512_LO 0xc69c7fe0129c058full
#define CRC64_K512_HI 0xa9bfcb035e069d2full
#define CRC64_K384_LO 0x37e9b58a68364616ull
#define CRC64_K384_HI 0x5af19904b0033c71ull
#define CRC64_K256_LO 0xfe845b60565108c8ull
#define CRC64_K256_HI 0xb7bf980fed556b55ull
#define CRC64_K128_LO 0x03ecaf64caf59255ull
#define CRC64_K128_HI 0xddb86ecfc0a519aeull   /* x^127 mod P: also the last lane's x^128 */
#define CRC64_MU 0x4b7010d952f90699ull
#define CRC64_P 0xc02c38b47d32d307ull
#if defined(OB_X86) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#include <wmmintrin.h>
#define OB_HAVE_CLMUL64() (__builtin_cpu_supports("pclmul"))
__attribute__((target("pclmul,sse4.1"))) static uint64_t crc64_fold(const uint8_t *p, int n, uint64_t c) {
  const __m128i k512 = _mm_set_epi64x((long long)CRC64_K512_HI, (long long)CRC64_K512_LO);
  const __m128i k384 = _mm_set_epi64x((long long)CRC64_K384_HI, (long long)CRC64_K384_LO);
  const __m128i k256 = _mm_set_epi64x((long long)CRC64_K256_HI, (long long)CRC64_K256_LO);
  const __m128i k128 = _mm_set_epi64x((long long)CRC64_K128_HI, (long long)CRC64_K128_LO);
  const __m128i bar = _mm_set_epi64x((long long)CRC64_P, (long long)CRC64_MU);
#define OB_FOLD(a, k) _mm_xor_si128(_mm_clmulepi64_si128((a), (k), 0x00), _mm_clmulepi64_si128((a), (k), 0x11))
  __m128i x0 = _mm_loadu_si128((const __m128i *)p), x1 = _mm_loadu_si128((const __m128i *)(p + 16));
  __m128i x2 = _mm_loadu_si128((const __m128i *)(p + 32)), x3 = _mm_loadu_si128((const __m128i *)(p + 48));
  x0 = _mm_xor_si128(x0, _mm_cvtsi64_si128((long long)c));
  int i = 64;
  for (; i + 64 <= n; i += 64) {
    x0 = _mm_xor_si128(OB_FOLD(x0, k512), _mm_loadu_si128((const __m128i *)(p + i)));
    x1 = _mm_xor_si128(OB_FOLD(x1, k512), _mm_loadu_si128((const __m128i *)(p + i + 16)));
    x2 = _mm_xor_si128(OB_FOLD(x2, k512), _mm_loadu_si128((const __m128i *)(p + i + 32)));
    x3 = _mm_xor_si128(OB_FOLD(x3, k512), _mm_loadu_si128((const __m128i *)(p + i + 48)));
  }
  __m128i a = _mm_xor_si128(_mm_xor_si128(OB_FOLD(x0, k384), OB_FOLD(x1, k256)), _mm_xor_si128(OB_FOLD(x2, k128), x3));
  for (; i < n; i += 16) a = _mm_xor_si128(OB_FOLD(a, k128), _mm_loadu_si128((const __m128i *)(p + i)));
#undef OB_FOLD
  a = _mm_xor_si128(_mm_clmulepi64_si128(a, k128, 0x10), _mm_srli_si128(a, 8));
  const uint64_t th = (uint64_t)_mm_cvtsi128_si64(a), tl = (uint64_t)_mm_extract_epi64(a, 1);
  const uint64_t q = th ^ ((uint64_t)_mm_cvtsi128_si64(_mm_clmulepi64_si128(a, bar, 0x00)) << 1);
  const __m128i r = _mm_clmulepi64_si128(_mm_cvtsi64_si128((long long)q), bar, 0x10);
  return tl ^ (((uint64_t)_mm_extract_epi64(r, 1) << 1) | ((uint64_t)_mm_cvtsi128_si64(r) >> 63));
}
#elif defined(OB_NEON) && defined(__clang__) && (defined(__linux__) || defined(__APPLE__))
#if defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_PMULL
#define HWCAP_PMULL (1 << 4)
#endif
#define OB_HAVE_CLMUL64() ((getauxval(AT_HWCAP) & HWCAP_PMULL) != 0)
#else
#define OB_HAVE_CLMUL64() 1
#endif
#define OB_CLMUL_TARGET __attribute__((target("aes")))
OB_CLMUL_TARGET static inline uint64x2_t clmul_ll(uint64x2_t a, uint64x2_t b) {
  return vreinterpretq_u64_p128(vmull_p64((poly64_t)vgetq_lane_u64(a, 0), (poly64_t)vgetq_lane_u64(b, 0)));
}
OB_CLMUL_TARGET static inline uint64x2_t clmul_hh(uint64x2_t a, uint64x2_t b) {
  return vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(a), vreinterpretq_p64_u64(b)));
}
OB_CLMUL_TARGET static inline uint64x2_t fold64(uint64x2_t a, uint64x2_t k) { return veorq_u64(clmul_ll(a, k), clmul_hh(a, k)); }
OB_CLMUL_TARGET static uint64_t crc64_fold(const uint8_t *p, int n, uint64_t c) {
  const uint64x2_t k512 = vcombine_u64(vcreate_u64(CRC64_K512_LO), vcreate_u64(CRC64_K512_HI));
  const uint64x2_t k384 = vcombine_u64(vcreate_u64(CRC64_K384_LO), vcreate_u64(CRC64_K384_HI));
  const uint64x2_t k256 = vcombine_u64(vcreate_u64(CRC64_K256_LO), vcreate_u64(CRC64_K256_HI));
  const uint64x2_t k128 = vcombine_u64(vcreate_u64(CRC64_K128_LO), vcreate_u64(CRC64_K128_HI));
  uint64x2_t x0 = vreinterpretq_u64_u8(vld1q_u8(p)), x1 = vreinterpretq_u64_u8(vld1q_u8(p + 16));
  uint64x2_t x2 = vreinterpretq_u64_u8(vld1q_u8(p + 32)), x3 = vreinterpretq_u64_u8(vld1q_u8(p + 48));
  x0 = veorq_u64(x0, vcombine_u64(vcreate_u64(c), vcreate_u64(0)));
  int i = 64;
  for (; i + 64 <= n; i += 64) {
    x0 = veorq_u64(fold64(x0, k512), vreinterpretq_u64_u8(vld1q_u8(p + i)));
    x1 = veorq_u64(fold64(x1, k512), vreinterpretq_u64_u8(vld1q_u8(p + i + 16)));
    x2 = veorq_u64(fold64(x2, k512), vreinterpretq_u64_u8(vld1q_u8(p + i + 32)));
    x3 = veorq_u64(fold64(x3, k512), vreinterpretq_u64_u8(vld1q_u8(p + i + 48)));
  }
  uint64x2_t a = veorq_u64(veorq_u64(fold64(x0, k384), fold64(x1, k256)), veorq_u64(fold64(x2, k128), x3));
  for (; i < n; i += 16) a = veorq_u64(fold64(a, k128), vreinterpretq_u64_u8(vld1q_u8(p + i)));
  // ai: x^64 times the lane: its low qword by x^127 mod P (k128's high), its high qword moved down
  const uint64_t lo = vgetq_lane_u64(a, 0), hi = vgetq_lane_u64(a, 1);
  const uint64x2_t t = veorq_u64(vreinterpretq_u64_p128(vmull_p64((poly64_t)lo, (poly64_t)CRC64_K128_HI)), vcombine_u64(vcreate_u64(hi), vcreate_u64(0)));
  const uint64_t th = vgetq_lane_u64(t, 0), tl = vgetq_lane_u64(t, 1);
  const uint64_t q = th ^ (vgetq_lane_u64(vreinterpretq_u64_p128(vmull_p64((poly64_t)th, (poly64_t)CRC64_MU)), 0) << 1);
  const uint64x2_t r = vreinterpretq_u64_p128(vmull_p64((poly64_t)q, (poly64_t)CRC64_P));
  return tl ^ ((vgetq_lane_u64(r, 1) << 1) | (vgetq_lane_u64(r, 0) >> 63));
}
#endif
uint64_t ob_crc64(const uint8_t *p, int n) {
  uint64_t c = ~0ull;
#ifdef OB_HAVE_CLMUL64
  if (n >= 64 && OB_HAVE_CLMUL64()) { const int m = n & ~15; c = crc64_fold(p, m, c); p += m; n -= m; }
#endif
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    const uint64_t v = c ^ ((uint64_t)p[i] | (uint64_t)p[i + 1] << 8 | (uint64_t)p[i + 2] << 16 | (uint64_t)p[i + 3] << 24 |
                            (uint64_t)p[i + 4] << 32 | (uint64_t)p[i + 5] << 40 | (uint64_t)p[i + 6] << 48 | (uint64_t)p[i + 7] << 56);
    c = CRC64_T[7][v & 0xff] ^ CRC64_T[6][(v >> 8) & 0xff] ^ CRC64_T[5][(v >> 16) & 0xff] ^ CRC64_T[4][(v >> 24) & 0xff] ^
        CRC64_T[3][(v >> 32) & 0xff] ^ CRC64_T[2][(v >> 40) & 0xff] ^ CRC64_T[1][(v >> 48) & 0xff] ^ CRC64_T[0][v >> 56];
  }
  for (; i < n; i++) c = (c >> 8) ^ CRC64_T[0][(c ^ p[i]) & 0xff];
  return ~c;
}


// j is the TRACK cell's index, not a module: the track takes every second run of OB_BAND_RUN cells of the band, the word the rest.
int ob_thin_track(int side, int j) {
  uint32_t v = (uint32_t)(side * 4099 + (j >> 1)) * 2654435761u + 40503u;
  v ^= v >> 15; v *= 2246822519u; v ^= v >> 13;
  return (int)((v ^ (uint32_t)j) & 1);   // bit for the pair of track cells, flipped in its second
}

// Cell i of the coded band spans OB_CELL modules from reserve + i * OB_CELL, and the band is read
// from reserve up to `reserve` short of the far corner, so the corners, where two sides meet, are never
// read. That dead stretch is what a corner mark is painted into (layout.h OB_THIN_RESERVE).
int ob_thin_cells(int side, int reserve) { return side > 2 * reserve ? (side - 2 * reserve) / OB_CELL : 0; }
int ob_track_cells(int side, int reserve) { return (ob_thin_cells(side, reserve) + 1) / 2; }

// The format word, over the light cells thin_init left. Only these cells move, and they are flagged CELL_WORD so
// the registration mesh skips them (layout.h), so a sender can change the word without disturbing anything
// registration depends on, at either end. Every word cell is flagged, including any past the codeword (a band whose W
// is not a multiple of 8, or past OB_FMT_BYTES_MAX), which are painted light: no template treats a word position as
// fixed.
void ob_layout_set_fmt(ob_layout_t *L, const uint8_t *cw) {
  const int w = L->w, h = L->h, side_mod = w < h ? w : h;
  const int cells = ob_fmt_side_cells(side_mod, L->reserve), bytes = ob_fmt_bytes(side_mod, L->reserve);
  for (int side = 0; side < 4; side++) for (int i = 0; i < cells; i++) {
    const uint8_t v = (ob_fmt_bit(cw, bytes, side, i) ? CELL_DARK : CELL_LIGHT) | CELL_WORD;
    for (int j = 0; j < OB_CELL; j++) for (int e = 0; e < OB_RING_DEEP; e++) {
      const int t = L->reserve + OB_WORD_AT(i) + j, d = OB_THIN_FMT + e;
      const int x = side == 0 || side == 2 ? t : side == 3 ? d : w - 1 - d;
      const int y = side == 0 ? d : side == 2 ? h - 1 - d : t;
      L->kind[y * w + x] = v;
    }
  }
}

// Thin frame: a lattice every OB_THIN_PITCH modules whose border nodes sit on OB_NODE_LINE (layout.h: not the
// border's middle), each a 5 x 5 patch of the border pattern. No mark is painted: the border is the mark.
// 1 where (x, y) falls in a corner mark: a solid block filling the square [0, S) at each corner, with
// its outermost ring light so a symbol against a dark bezel is still one dark run the scan can find
// the inner edge of, and its innermost ring light as the separator between the block and the coded rings inside it
// (but where the border's line crosses it).
static int thin_corner(const ob_layout_t *L, int x, int y, int *dark) {
  const int S = L->corner;
  // FOUR corners. Three, QR's arrangement, was tried on 2026-09-21 and reverted: the fourth corner is a MEASURED
  // point, and inferring it as a parallelogram's corner leaves the map affine until an edge mark is found to put
  // the perspective back. The edge marks are found in the easy captures, where the map was already right, and
  // missed in the hard ones, where it is not: fill 0.3 fell from 2990 B on 16 of 16 frames to 352 B on 7.
  const int cx = x < L->w - 1 - x ? x : L->w - 1 - x, cy = y < L->h - 1 - y ? y : L->h - 1 - y;
  if (cx >= S || cy >= S) return 0;
  const int in = cx >= 1 && cx <= S - 2 && cy >= 1 && cy <= S - 2;
  // The gap band stays light, so the line's inner edge, which the finder traces, is at the same depth under a mark as
  // anywhere else and the mark cannot pull the curve. The line itself (depths 1 to OB_THIN_LINE) runs through the
  // mark's separator, so it is one closed loop round the symbol (2026-11-23; a module's gap beside each mark before,
  // which made each side a contour of its own).
  const int d = cx < cy ? cx : cy;
  *dark = (in && !(d >= OB_THIN_GAP && d < OB_THIN_TRACK)) || (d >= 1 && d <= OB_THIN_LINE);
  return 1;
}

static int thin_init(ob_layout_t *L, int w, int h) {
  for (int axis = 0; axis < 2; axis++) {
    int len = axis ? h : w, n = 0;
    float *pos = axis ? L->node_y : L->node_x;
    // OB_NODE_LINE (layout.h) says where this runs and why it is not the border's middle.
    for (float c = OB_NODE_LINE; c < len - OB_NODE_LINE - OB_THIN_PITCH / 2 && n < OB_MAX_NODES - 1; c += OB_THIN_PITCH) pos[n++] = c;
    pos[n++] = len - OB_NODE_LINE;
    if (axis) L->ny = n; else L->nx = n;
  }
  for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
    int dx = x < w - 1 - x ? x : w - 1 - x, dy = y < h - 1 - y ? y : h - 1 - y, d = dx < dy ? dx : dy, dark = 0;
    if (thin_corner(L, x, y, &dark)) { L->kind[y * w + x] = dark ? CELL_DARK : CELL_LIGHT; continue; }
    if (d >= 1 && d <= OB_THIN_LINE) dark = 1;
    else if (d >= OB_THIN_TRACK && d < OB_THIN) {
      // The one coded band. Even runs are the track, odd runs are left light for the word (ob_layout_set_fmt),
      // so a configuration with no expressible word shows gaps rather than someone else's bits.
      const int along = dy <= dx ? x : y, len = dy <= dx ? w : h;
      const int side = dy <= dx ? (y < h / 2 ? 0 : 2) : (x < w / 2 ? 3 : 1);
      const int cell = (along - L->reserve) / OB_CELL, run = cell / OB_BAND_RUN;
      if (along >= L->reserve && along < len - L->reserve && !(run & 1))
        dark = ob_thin_track(side, run / 2 * OB_BAND_RUN + cell % OB_BAND_RUN);
    }
    // The guard interval is the picture's, not the frame's: DATA, so no template reads it and no sampler expects
    // it to be anything, as lizard_phy.c paints it.
    L->kind[y * w + x] = d >= L->margin - LIZARD_CP ? CELL_DATA : dark ? CELL_DARK : CELL_LIGHT;   // DATA: the picture, which the frame's templates skip
  }
  for (int j = 0; j < L->ny; j++) for (int i = 0; i < L->nx; i++) {
    int edge_x = i == 0 || i == L->nx - 1, edge_y = j == 0 || j == L->ny - 1;
    if ((edge_x || edge_y) && !(edge_x && edge_y)) L->node_mark[j * L->nx + i] = 1;
  }
  return 0;
}

int ob_layout_init(ob_layout_t *L, int side) {
  memset(L, 0, sizeof *L);
  if (side < 64) return -1;
  L->w = L->h = side;
  L->margin = OB_MARGIN; L->corner = OB_THIN_CORNER; L->reserve = OB_THIN_RESERVE;
  L->kind = calloc((size_t)side * side, 1);
  return L->kind ? thin_init(L, side, side) : -1;
}

void ob_layout_free(ob_layout_t *L) {
  free(L->kind);
  memset(L, 0, sizeof *L);
}
