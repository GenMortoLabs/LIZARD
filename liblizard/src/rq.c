// RaptorQ (RFC 6330) for the transfer: rq.h says what it is; this file is the RFC's matrix and one way to solve it.
//
// The matrix (RFC 6330 5.3.3.3), L = K' + S + H columns (the intermediate symbols):
//   S LDPC rows: source-side column i < B in rows i mod S, + a, + 2a (a = 1 + i / S), each row its own column B + row
//     and the PI columns W + (row mod P) and W + (row + 1 mod P);
//   H HDPC rows over GF(256): G_HDPC = MT GAMMA on the first K' + S columns, then the identity on the last H;
//   a row per encoding symbol: its tuple's LT columns (below W) and PI columns (W and up), binary.
// The binary rows are peeled over the columns below W, the PI columns inactive from the start; a column is inactivated
// where no unused row has one unknown left (a row with the fewest unknowns opened: all but one of its columns
// inactivated, the RFC's own rule bar its tie-break). Each peeled column is then its pivot row's right side plus a
// binary combination of the inactive columns (its projection), and every row left over, and each HDPC row, becomes a
// row of a dense system over the inactive columns. The HDPC rows go through MT GAMMA's recurrence (y_j = alpha y_(j-1)
// + z_j, each MT column adding y_j to its two rows, the last column alpha^i y to row i), so G_HDPC is never formed.
// The dense system is solved by Gauss-Jordan over GF(256), then the peeled columns in peel order.
#include "rq.h"
#include "rq_plan.h"
#include "rq_tables.h"
#include "simd.h"

#include <stdlib.h>
#include <string.h>


static int is_prime(uint32_t n) {
  if (n < 2) return 0;
  for (uint32_t i = 2; i * i <= n; i++)
    if (n % i == 0) return 0;
  return 1;
}

// The parameters of a block of K source symbols (5.3.3.3, Table 2), or -1.
static int rq_par(uint32_t K, rq_par_t *p) {
  if (K < 1 || K > RQ_KMAX) return -1;
  int i = 0;
  while (RQ_TABLE[i][0] < K) i++;
  p->K = K;
  p->Kp = RQ_TABLE[i][0], p->J = RQ_TABLE[i][1], p->S = RQ_TABLE[i][2], p->H = RQ_TABLE[i][3], p->W = RQ_TABLE[i][4];
  p->L = p->Kp + p->S + p->H, p->P = p->L - p->W, p->B = p->W - p->S;
  p->P1 = p->P;
  while (!is_prime(p->P1)) p->P1++;
  return 0;
}

// Rand[y, i, m] (5.3.5.1).
static inline uint32_t rq_rand(uint32_t y, uint32_t i, uint32_t m) {
  return (RQ_V0[(y + i) & 255] ^ RQ_V1[((y >> 8) + i) & 255] ^ RQ_V2[((y >> 16) + i) & 255] ^
          RQ_V3[((y >> 24) + i) & 255]) % m;
}

// Deg[v] (5.3.5.2).
static uint32_t rq_deg(uint32_t v, uint32_t W) {
  uint32_t d = 1;
  while (v >= RQ_DEG_F[d]) d++;
  return d < W - 2 ? d : W - 2;
}

// The columns of internal symbol `isi`'s row: its tuple (5.3.5.4) walked as Enc does (5.3.5.3). At most 30 LT columns
// and 3 PI ones, all distinct (W and P1 are prime).
enum { RQ_ROW_MAX = 33 };
static int rq_row(const rq_par_t *p, uint32_t isi, uint32_t *cols) {
  uint32_t A = 53591 + p->J * 997;
  if (!(A & 1)) A++;
  const uint32_t y = 10267 * (p->J + 1) + isi * A;
  const uint32_t d = rq_deg(rq_rand(y, 0, 1048576), p->W);
  const uint32_t a = 1 + rq_rand(y, 1, p->W - 1);
  uint32_t b = rq_rand(y, 2, p->W);
  const uint32_t d1 = d < 4 ? 2 + rq_rand(isi, 3, 2) : 2;
  const uint32_t a1 = 1 + rq_rand(isi, 4, p->P1 - 1);
  uint32_t b1 = rq_rand(isi, 5, p->P1);
  int n = 0;
  cols[n++] = b;
  for (uint32_t j = 1; j < d; j++) b = (b + a) % p->W, cols[n++] = b;
  while (b1 >= p->P) b1 = (b1 + a1) % p->P1;
  cols[n++] = p->W + b1;
  for (uint32_t j = 1; j < d1; j++) {
    b1 = (b1 + a1) % p->P1;
    while (b1 >= p->P) b1 = (b1 + a1) % p->P1;
    cols[n++] = p->W + b1;
  }
  return n;
}

// ---- symbols: GF(256) of 5.7 ----
static inline uint8_t gf_mul(uint8_t a, uint8_t b) { return a && b ? RQ_OCT_EXP[RQ_OCT_LOG[a] + RQ_OCT_LOG[b]] : 0; }
static inline uint8_t gf_inv(uint8_t a) { return RQ_OCT_EXP[255 - RQ_OCT_LOG[a]]; }

// Sixteen bytes a step where the compiler has vector types (GCC and clang: SSE2, NEON, wasm SIMD, or plain words).
// RQ_NOINLINE (2026-11-15): sym_xor and sym_mul2 stay calls. Inlined into the solve's loops, GCC loses their restrict
// and the symbol work slows (the 9950X: the HDPC right sides 1.7 against 6.8 ms forced inline; -O3, which inlines
// them, 38% slower over all the symbol work).
#if defined(__GNUC__) || defined(__clang__)
typedef uint8_t rq_v16 __attribute__((vector_size(16)));
#define RQ_V16 1
#define RQ_NOINLINE __attribute__((noinline))
#else
#define RQ_NOINLINE
#endif

static RQ_NOINLINE void sym_xor(uint8_t *restrict d, const uint8_t *restrict s, size_t n) {
  size_t i = 0;
#ifdef RQ_V16
  for (; i + 16 <= n; i += 16) {
    rq_v16 a, b;
    memcpy(&a, d + i, 16), memcpy(&b, s + i, 16);
    a ^= b;
    memcpy(d + i, &a, 16);
  }
#endif
  for (; i < n; i++) d[i] ^= s[i];
}
// c s by the products of c with each nibble (c x = c (x & 15) ^ c (x & 240)): sixteen bytes a step where a vector
// byte shuffle reads the two 16-entry tables (NEON's TBL, SSSE3's PSHUFB, wasm's swizzle; simd.h names the target),
// the scalar twin a byte at a time. The same bytes either way (2026-11-15: the scalar form only before, 11k
// multiply-adds of a symbol a 16 MiB solve).
static void gf_tables(uint8_t c, uint8_t lo[16], uint8_t hi[16]) {
  for (int k = 0; k < 16; k++) lo[k] = gf_mul(c, (uint8_t)k), hi[k] = gf_mul(c, (uint8_t)(k << 4));
}
// d ^= c s (scale 0) or d = c d (scale 1, s == d) over n bytes, the vector part; returns the bytes done
static size_t gf_vec(uint8_t *d, const uint8_t *s, const uint8_t lo[16], const uint8_t hi[16], size_t n, int scale) {
  size_t i = 0;
#if defined(OB_NEON)
  const uint8x16_t tl = vld1q_u8(lo), th = vld1q_u8(hi), m = vdupq_n_u8(15);
  for (; i + 16 <= n; i += 16) {
    const uint8x16_t v = vld1q_u8(s + i), p = veorq_u8(vqtbl1q_u8(tl, vandq_u8(v, m)), vqtbl1q_u8(th, vshrq_n_u8(v, 4)));
    vst1q_u8(d + i, scale ? p : veorq_u8(vld1q_u8(d + i), p));
  }
#elif defined(OB_X86)
  const __m128i tl = _mm_loadu_si128((const __m128i *)(const void *)lo), th = _mm_loadu_si128((const __m128i *)(const void *)hi), m = _mm_set1_epi8(15);
  for (; i + 16 <= n; i += 16) {
    const __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(s + i));
    const __m128i p = _mm_xor_si128(_mm_shuffle_epi8(tl, _mm_and_si128(v, m)), _mm_shuffle_epi8(th, _mm_and_si128(_mm_srli_epi16(v, 4), m)));
    _mm_storeu_si128((__m128i *)(void *)(d + i), scale ? p : _mm_xor_si128(_mm_loadu_si128((const __m128i *)(const void *)(d + i)), p));
  }
#elif defined(OB_SIMD)
  const v128_t tl = wasm_v128_load(lo), th = wasm_v128_load(hi), m = wasm_i8x16_splat(15);
  for (; i + 16 <= n; i += 16) {
    const v128_t v = wasm_v128_load(s + i), p = wasm_v128_xor(wasm_i8x16_swizzle(tl, wasm_v128_and(v, m)), wasm_i8x16_swizzle(th, wasm_u8x16_shr(v, 4)));
    wasm_v128_store(d + i, scale ? p : wasm_v128_xor(wasm_v128_load(d + i), p));
  }
#else
  (void)d, (void)s, (void)lo, (void)hi, (void)n, (void)scale;
#endif
  return i;
}
// d += c s
static void sym_muladd(uint8_t *restrict d, const uint8_t *restrict s, uint8_t c, size_t n) {
  if (!c) return;
  if (c == 1) { sym_xor(d, s, n); return; }
  uint8_t lo[16], hi[16];
  gf_tables(c, lo, hi);
  for (size_t i = gf_vec(d, s, lo, hi, n, 0); i < n; i++) d[i] ^= lo[s[i] & 15] ^ hi[s[i] >> 4];
}
static void sym_scale(uint8_t *d, uint8_t c, size_t n) {
  if (c == 1) return;
  uint8_t lo[16], hi[16];
  gf_tables(c, lo, hi);
  for (size_t i = gf_vec(d, d, lo, hi, n, 1); i < n; i++) d[i] = lo[d[i] & 15] ^ hi[d[i] >> 4];
}
// d = alpha d (alpha = 2: x^8 = x^4 + x^3 + x^2 + 1)
static RQ_NOINLINE void sym_mul2(uint8_t *d, size_t n) {
  size_t i = 0;
#ifdef RQ_V16
  for (; i + 16 <= n; i += 16) {
    rq_v16 a;
    memcpy(&a, d + i, 16);
    a = (a << 1) ^ ((rq_v16)(-(a >> 7)) & 0x1D);
    memcpy(d + i, &a, 16);
  }
#endif
  for (; i < n; i++) d[i] = (uint8_t)(d[i] << 1) ^ (uint8_t)(-(d[i] >> 7) & 0x1D);
}

// ---- the binary rows ----
typedef struct {
  uint32_t n;            // rows: the S LDPC rows, then a row a symbol
  uint32_t *start;       // n + 1
  uint32_t *col;
  const uint8_t **rhs;   // a row's right side, NULL for zero (the LDPC rows, the padding symbols)
} rq_rows_t;

static void rows_free(rq_rows_t *r) { free(r->start), free(r->col), free((void *)r->rhs); }

static int cmp_u32(const void *a, const void *b) {
  const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  return x < y ? -1 : x > y;
}

// The S LDPC rows and a row for each of `n` internal symbols (isi[k], right side rhs[k]).
static int rows_build(const rq_par_t *p, uint32_t n, const uint32_t *isi, const uint8_t *const *rhs, rq_rows_t *r) {
  const uint32_t S = p->S;
  r->n = S + n;
  r->start = calloc((size_t)r->n + 1, sizeof *r->start);
  r->rhs = calloc(r->n, sizeof *r->rhs);
  uint32_t *cnt = calloc(S, sizeof *cnt);
  if (!r->start || !r->rhs || !cnt) { free(cnt); return -1; }
  for (uint32_t i = 0; i < p->B; i++) {
    const uint32_t a = 1 + i / S;
    uint32_t b = i % S;
    cnt[b]++, b = (b + a) % S, cnt[b]++, b = (b + a) % S, cnt[b]++;
  }
  size_t total = 0;
  for (uint32_t i = 0; i < S; i++) r->start[i] = (uint32_t)total, total += cnt[i] + 3;
  size_t lt = 0;
  uint32_t tmp[RQ_ROW_MAX];
  for (uint32_t k = 0; k < n; k++) lt += (size_t)rq_row(p, isi[k], tmp);
  r->col = malloc((total + lt) * sizeof *r->col);
  if (!r->col) { free(cnt); return -1; }
  memset(cnt, 0, S * sizeof *cnt);
  for (uint32_t i = 0; i < p->B; i++) {
    const uint32_t a = 1 + i / S;
    uint32_t b = i % S;
    for (int t = 0; t < 3; t++) r->col[r->start[b] + cnt[b]++] = i, b = (b + a) % S;
  }
  for (uint32_t i = 0; i < S; i++) {
    uint32_t *c = r->col + r->start[i] + cnt[i];
    c[0] = p->B + i, c[1] = p->W + i % p->P, c[2] = p->W + (i + 1) % p->P;
  }
  // A column met twice in a row cancels (it never is, S and P being past the walks; kept to be the matrix exactly).
  size_t w = 0;
  for (uint32_t i = 0; i < S; i++) {
    uint32_t *c = r->col + r->start[i];
    const uint32_t m = cnt[i] + 3;
    qsort(c, m, sizeof *c, cmp_u32);
    const size_t at = w;
    for (uint32_t j = 0; j < m; j++) {
      if (j + 1 < m && c[j] == c[j + 1]) { j++; continue; }
      r->col[w++] = c[j];
    }
    r->start[i] = (uint32_t)at;
  }
  free(cnt);
  for (uint32_t k = 0; k < n; k++) {
    r->start[S + k] = (uint32_t)w;
    w += (size_t)rq_row(p, isi[k], r->col + w);
    r->rhs[S + k] = rhs[k];
  }
  r->start[r->n] = (uint32_t)w;
  return 0;
}

// ---- the solve: a plan made from the rows (rq_plan.h), then the symbol work it leaves ----
enum { COL_LIVE = 0, COL_SOLVED = 1, COL_INACTIVE = 2 };

void rq_plan_free(rq_plan_t *pl) {
  if (!pl) return;
  free(pl->lev_start), free(pl->tgt), free(pl->tgt_row), free(pl->src_start), free(pl->src_peeled), free(pl->src);
  free(pl->u_row), free(pl->u_start), free(pl->u_src), free(pl->hpair), free(pl->inact);
  free(pl->op_dst), free(pl->op_src), free(pl->row_of), free(pl->op_c);
  free(pl);
}

// The plan of a solve over rows rw (rows from sym0 on held symbols): 0, 1 where the rows do not determine the
// intermediate symbols (a rank short), -1 out of memory. Only the rows' columns are read, never their right sides.
static int plan_build(const rq_par_t *p, const rq_rows_t *rw, uint32_t sym0, rq_plan_t *pl) {
  const uint32_t L = p->L, W = p->W, H = p->H, KS = p->Kp + p->S, nr = rw->n;
  const uint32_t *rs = rw->start, *rc = rw->col;
  pl->p = *p, pl->sym0 = sym0;
  int ret = -1;
  uint32_t *cs = calloc((size_t)W + 1, sizeof *cs), *live = malloc((size_t)nr * sizeof *live);
  uint32_t *stack = malloc((size_t)nr * sizeof *stack), *order = malloc((size_t)W * sizeof *order);
  // Rows by their unknowns left (2 to RQ_ROW_MAX + 3; an LDPC row's past that waits in the last), each a stack whose
  // entries stand while the row is unused and still at that count.
  enum { NB = RQ_ROW_MAX + 4 };
  uint32_t bn[NB] = {0}, bcap[NB] = {0}, *bks[NB] = {0};
  uint32_t *ref = malloc((size_t)L * sizeof *ref), *inact = malloc((size_t)L * sizeof *inact), *cr = NULL, *lv = NULL;
  uint8_t *used = calloc(nr, 1), *state = calloc(L, 1);
  uint64_t *proj = NULL, *bits = NULL, *acc8 = NULL, *hp = NULL;
  uint8_t *coef = NULL;
  uint8_t **row = NULL;
  uint32_t *phys = NULL;
  if (!cs || !live || !stack || !order || !ref || !inact || !used || !state) goto out;

  // Each column below W: the rows holding it.
  for (uint32_t r = 0; r < nr; r++) {
    uint32_t n = 0;
    for (uint32_t k = rs[r]; k < rs[r + 1]; k++)
      if (rc[k] < W) cs[rc[k] + 1]++, n++;
    live[r] = n;
  }
  for (uint32_t c = 0; c < W; c++) cs[c + 1] += cs[c];
  cr = malloc((size_t)cs[W] * sizeof *cr);
  if (!cr) goto out;
  {
    uint32_t *fill = malloc((size_t)W * sizeof *fill);
    if (!fill) goto out;
    memcpy(fill, cs, (size_t)W * sizeof *fill);
    for (uint32_t r = 0; r < nr; r++)
      for (uint32_t k = rs[r]; k < rs[r + 1]; k++)
        if (rc[k] < W) cr[fill[rc[k]]++] = r;
    free(fill);
  }

  // Peel, inactivating where stuck.
  uint32_t sp = 0, npeel = 0, ninact = 0, remaining = W;
#define RQ_BUCKET(r)                                                                       \
  do {                                                                                     \
    const uint32_t b_ = live[r] < NB ? live[r] : NB - 1;                                   \
    if (bn[b_] == bcap[b_]) {                                                              \
      const uint32_t c_ = bcap[b_] ? bcap[b_] * 2 : 256;                                   \
      uint32_t *g_ = realloc(bks[b_], (size_t)c_ * sizeof *g_);                            \
      if (!g_) goto out;                                                                   \
      bks[b_] = g_, bcap[b_] = c_;                                                         \
    }                                                                                      \
    bks[b_][bn[b_]++] = (r);                                                               \
  } while (0)
  for (uint32_t r = 0; r < nr; r++) {
    if (live[r] == 1) stack[sp++] = r;
    else if (live[r] >= 2) RQ_BUCKET(r);
  }
#define RQ_DROP(c)                                                         \
  for (uint32_t k_ = cs[c]; k_ < cs[(c) + 1]; k_++) {                      \
    const uint32_t r_ = cr[k_];                                            \
    if (used[r_]) continue;                                                \
    if (--live[r_] == 1) stack[sp++] = r_;                                 \
    else if (live[r_] >= 2 && live[r_] < NB - 1) RQ_BUCKET(r_);            \
  }
  while (remaining) {
    while (sp) {
      const uint32_t r = stack[--sp];
      if (used[r] || live[r] != 1) continue;
      uint32_t c = UINT32_MAX;
      for (uint32_t k = rs[r]; k < rs[r + 1]; k++)
        if (rc[k] < W && state[rc[k]] == COL_LIVE) { c = rc[k]; break; }
      state[c] = COL_SOLVED, ref[c] = r, used[r] = 1, order[npeel++] = c, remaining--;
      RQ_DROP(c)
    }
    if (!remaining) break;
    // The row with the fewest unknowns (any of them: the RFC's tie-breaks only steer how many get inactivated).
    uint32_t best = UINT32_MAX;
    for (uint32_t b = 2; b < NB && best == UINT32_MAX; b++)
      while (bn[b]) {
        const uint32_t r = bks[b][--bn[b]];
        if (!used[r] && (live[r] == b || (b == NB - 1 && live[r] >= b))) { best = r; break; }
      }
    if (best == UINT32_MAX) {
      // No unused row reaches the columns left: inactive, for the dense rows to settle or to leave short.
      for (uint32_t c = 0; c < W; c++)
        if (state[c] == COL_LIVE) { state[c] = COL_INACTIVE, ref[c] = ninact, inact[ninact++] = c, remaining--; RQ_DROP(c) }
      continue;
    }
    // Keep the live column the fewest rows hold; inactivate the others.
    uint32_t keep = UINT32_MAX, kd = UINT32_MAX;
    for (uint32_t k = rs[best]; k < rs[best + 1]; k++) {
      const uint32_t c = rc[k];
      if (c < W && state[c] == COL_LIVE && cs[c + 1] - cs[c] < kd) keep = c, kd = cs[c + 1] - cs[c];
    }
    for (uint32_t k = rs[best]; k < rs[best + 1]; k++) {
      const uint32_t c = rc[k];
      if (c < W && state[c] == COL_LIVE && c != keep) { state[c] = COL_INACTIVE, ref[c] = ninact, inact[ninact++] = c, remaining--; RQ_DROP(c) }
    }
  }
#undef RQ_DROP
#undef RQ_BUCKET
  const uint32_t I = ninact + p->P, Iw = (I + 63) / 64;
  for (uint32_t c = W; c < L; c++) state[c] = COL_INACTIVE, ref[c] = ninact + (c - W), inact[ninact + (c - W)] = c;
  uint32_t Nu = 0;
  for (uint32_t r = 0; r < nr; r++) Nu += !used[r];
  const uint32_t R = Nu + H;
  if (R < I) { ret = 1; goto out; }
  pl->npeel = npeel, pl->Nu = Nu, pl->R = R, pl->I = I;
  pl->inact = malloc((size_t)I * sizeof *pl->inact);
  if (!pl->inact) goto out;
  memcpy(pl->inact, inact, (size_t)I * sizeof *inact);

  // Levels: a peeled column one past the deepest peeled column its row reads; the targets by level, in peel order
  // within one.
  lv = calloc(L, sizeof *lv);
  if (!lv) goto out;
  uint32_t nlev = 0;
  for (uint32_t k = 0; k < npeel; k++) {
    const uint32_t c = order[k], r = ref[c];
    uint32_t m = 0;
    for (uint32_t j = rs[r]; j < rs[r + 1]; j++)
      if (rc[j] != c && state[rc[j]] == COL_SOLVED && lv[rc[j]] > m) m = lv[rc[j]];
    lv[c] = m + 1;
    if (m + 1 > nlev) nlev = m + 1;
  }
  pl->nlev = nlev;
  pl->lev_start = calloc((size_t)nlev + 1, sizeof *pl->lev_start);
  pl->tgt = malloc((size_t)npeel * sizeof *pl->tgt), pl->tgt_row = malloc((size_t)npeel * sizeof *pl->tgt_row);
  pl->src_start = malloc(((size_t)npeel + 1) * sizeof *pl->src_start), pl->src_peeled = malloc((size_t)npeel * sizeof *pl->src_peeled);
  if (!pl->lev_start || !pl->tgt || !pl->tgt_row || !pl->src_start || !pl->src_peeled) goto out;
  for (uint32_t k = 0; k < npeel; k++) pl->lev_start[lv[order[k]]]++;   // level l (from 1) counted at l
  for (uint32_t l = 1; l <= nlev; l++) pl->lev_start[l] += pl->lev_start[l - 1];   // now the end of level l - 1
  {
    uint32_t *fill = malloc((size_t)nlev * sizeof *fill);
    if (!fill) goto out;
    for (uint32_t l = 0; l < nlev; l++) fill[l] = pl->lev_start[l];
    size_t nsrc = 0;
    for (uint32_t k = 0; k < npeel; k++) {
      const uint32_t c = order[k], t = fill[lv[c] - 1]++;
      pl->tgt[t] = c, pl->tgt_row[t] = ref[c];
      nsrc += rs[ref[c] + 1] - rs[ref[c]] - 1;
    }
    free(fill);
    pl->src = malloc((nsrc ? nsrc : 1) * sizeof *pl->src);
    if (!pl->src) goto out;
    size_t o = 0;
    for (uint32_t t = 0; t < npeel; t++) {
      const uint32_t c = pl->tgt[t], r = pl->tgt_row[t];
      pl->src_start[t] = (uint32_t)o;
      for (uint32_t j = rs[r]; j < rs[r + 1]; j++)
        if (rc[j] != c && state[rc[j]] == COL_SOLVED) pl->src[o++] = rc[j];
      pl->src_peeled[t] = (uint32_t)o - pl->src_start[t];
      for (uint32_t j = rs[r]; j < rs[r + 1]; j++)
        if (rc[j] != c && state[rc[j]] != COL_SOLVED) pl->src[o++] = rc[j];
    }
    pl->src_start[npeel] = (uint32_t)o;
  }

  // Each peeled column's projection on the inactive columns (peel order: a row reads only columns peeled before).
  proj = calloc((size_t)W * Iw, sizeof *proj);
  if (!proj) goto out;
  for (uint32_t k = 0; k < npeel; k++) {
    const uint32_t c = order[k], r = ref[c];
    uint64_t *pc = proj + (size_t)c * Iw;
    for (uint32_t j = rs[r]; j < rs[r + 1]; j++) {
      const uint32_t c2 = rc[j];
      if (c2 == c) continue;
      if (state[c2] == COL_SOLVED) {
        const uint64_t *p2 = proj + (size_t)c2 * Iw;
        for (uint32_t w = 0; w < Iw; w++) pc[w] ^= p2[w];
      } else {
        pc[ref[c2] >> 6] ^= (uint64_t)1 << (ref[c2] & 63);
      }
    }
  }

  // The dense system's coefficients: the unused binary rows, then the HDPC rows.
  coef = calloc((size_t)R * I, 1), bits = malloc((size_t)Iw * sizeof *bits), acc8 = calloc((size_t)8 * Iw, sizeof *acc8);
  row = malloc((size_t)R * sizeof *row), phys = malloc((size_t)R * sizeof *phys);
  pl->u_row = malloc(((size_t)Nu + 1) * sizeof *pl->u_row), pl->u_start = malloc(((size_t)Nu + 1) * sizeof *pl->u_start);
  pl->hpair = malloc((size_t)(KS > 1 ? 2 * (KS - 1) : 1) * sizeof *pl->hpair);
  if (!coef || !bits || !acc8 || !row || !phys || !pl->u_row || !pl->u_start || !pl->hpair) goto out;
  for (uint32_t k = 0; k < R; k++) row[k] = coef + (size_t)k * I, phys[k] = k;
  {
    size_t nsrc = 0;
    for (uint32_t r = 0; r < nr; r++)
      if (!used[r])
        for (uint32_t j = rs[r]; j < rs[r + 1]; j++) nsrc += state[rc[j]] == COL_SOLVED;
    pl->u_src = malloc((nsrc ? nsrc : 1) * sizeof *pl->u_src);
    if (!pl->u_src) goto out;
    uint32_t k = 0;
    size_t o = 0;
    for (uint32_t r = 0; r < nr; r++) {
      if (used[r]) continue;
      memset(bits, 0, (size_t)Iw * sizeof *bits);
      pl->u_row[k] = r, pl->u_start[k] = (uint32_t)o;
      for (uint32_t j = rs[r]; j < rs[r + 1]; j++) {
        const uint32_t c2 = rc[j];
        if (state[c2] == COL_SOLVED) {
          const uint64_t *p2 = proj + (size_t)c2 * Iw;
          for (uint32_t w = 0; w < Iw; w++) bits[w] ^= p2[w];
          pl->u_src[o++] = c2;
        } else {
          bits[ref[c2] >> 6] ^= (uint64_t)1 << (ref[c2] & 63);
        }
      }
      for (uint32_t u = 0; u < I; u++) row[k][u] = (uint8_t)(bits[u >> 6] >> (u & 63) & 1);
      k++;
    }
    pl->u_start[Nu] = (uint32_t)o;
  }
  // G_HDPC = MT GAMMA through the recurrence, its coefficients over the inactive columns; MT's two rows of each column.
  // Bit-sliced (2026-11-15): acc and the H rows held as 8 planes of the I coefficients' bits (plane b, bit u: bit b of
  // coefficient u), so a solved column's projection, binary, is one XOR into plane 0, and alpha acc renames the planes
  // (plane b the old plane b - 1, plane 0 the old plane 7) with the old plane 7 XORed into planes 2, 3 and 4 as well
  // (x^8 = x^4 + x^3 + x^2 + 1); the rows to bytes at the end. The same coefficients; before, acc in bytes took each
  // projection a bit at a time (4.8 ms of a 16 MiB solve's 62 on the S26's prime core).
  hp = calloc((size_t)8 * Iw * H, sizeof *hp);
  if (!hp) goto out;
  unsigned rot = 0;   // acc's plane b is acc8 + ((b + rot) & 7) Iw
#define RQ_PLANE(b) (acc8 + (size_t)(((b) + rot) & 7) * Iw)
#define RQ_ALPHA()                                                                                        \
  do {                                                                                                    \
    rot = (rot + 7) & 7;                                                                                  \
    const uint64_t *t_ = RQ_PLANE(0);                                                                     \
    uint64_t *p2_ = RQ_PLANE(2), *p3_ = RQ_PLANE(3), *p4_ = RQ_PLANE(4);                                  \
    for (uint32_t w_ = 0; w_ < Iw; w_++) p2_[w_] ^= t_[w_], p3_[w_] ^= t_[w_], p4_[w_] ^= t_[w_];         \
  } while (0)
#define RQ_ADD_ROW(i_)                                                                                    \
  do {                                                                                                    \
    uint64_t *r_ = hp + (size_t)(i_) * 8 * Iw;                                                            \
    for (unsigned b_ = 0; b_ < 8; b_++) {                                                                 \
      const uint64_t *a_ = RQ_PLANE(b_);                                                                  \
      uint64_t *x_ = r_ + (size_t)b_ * Iw;                                                                \
      for (uint32_t w_ = 0; w_ < Iw; w_++) x_[w_] ^= a_[w_];                                              \
    }                                                                                                     \
  } while (0)
  for (uint32_t j = 0; j < KS; j++) {
    RQ_ALPHA();
    uint64_t *z = RQ_PLANE(0);
    if (state[j] == COL_SOLVED) {
      const uint64_t *p2 = proj + (size_t)j * Iw;
      for (uint32_t w = 0; w < Iw; w++) z[w] ^= p2[w];
    } else {
      z[ref[j] >> 6] ^= (uint64_t)1 << (ref[j] & 63);
    }
    if (j + 1 < KS) {
      const uint32_t i1 = rq_rand(j + 1, 6, H), i2 = (i1 + rq_rand(j + 1, 7, H - 1) + 1) % H;
      pl->hpair[2 * j] = i1, pl->hpair[2 * j + 1] = i2;
      RQ_ADD_ROW(i1);
      RQ_ADD_ROW(i2);
    } else {
      // the last column: alpha^i acc into row i, acc taken on by alpha a row (it is not read again)
      for (uint32_t i = 0; i < H; i++) { RQ_ADD_ROW(i); RQ_ALPHA(); }
    }
  }
#undef RQ_ADD_ROW
#undef RQ_ALPHA
#undef RQ_PLANE
  for (uint32_t i = 0; i < H; i++) {
    const uint64_t *r = hp + (size_t)i * 8 * Iw;
    uint8_t *o = row[Nu + i];
    for (uint32_t u = 0; u < I; u++) {
      uint8_t v = 0;
      for (unsigned b = 0; b < 8; b++) v |= (uint8_t)((r[(size_t)b * Iw + (u >> 6)] >> (u & 63) & 1) << b);
      o[u] = v;
    }
    o[ref[KS + i]] ^= 1;
  }

  // Gauss-Jordan over GF(256), rows taken in order (the binary ones first), each step kept for the right sides: rows
  // named by where they stand in memory (phys), the swaps only moving names.
  {
    size_t cap = 1024, n = 0;
    pl->op_dst = malloc(cap * sizeof *pl->op_dst), pl->op_src = malloc(cap * sizeof *pl->op_src), pl->op_c = malloc(cap);
    if (!pl->op_dst || !pl->op_src || !pl->op_c) goto out;
#define RQ_OP(d_, s_, c_)                                                                                  \
    do {                                                                                                   \
      if (n == cap) {                                                                                      \
        cap *= 2;                                                                                          \
        uint32_t *a_ = realloc(pl->op_dst, cap * sizeof *a_); if (!a_) goto out; pl->op_dst = a_;          \
        uint32_t *b_ = realloc(pl->op_src, cap * sizeof *b_); if (!b_) goto out; pl->op_src = b_;          \
        uint8_t *e_ = realloc(pl->op_c, cap); if (!e_) goto out; pl->op_c = e_;                            \
      }                                                                                                    \
      pl->op_dst[n] = (d_), pl->op_src[n] = (s_), pl->op_c[n] = (c_), n++;                                 \
    } while (0)
    for (uint32_t u = 0; u < I; u++) {
      uint32_t pv = u;
      while (pv < R && !row[pv][u]) pv++;
      if (pv == R) { ret = 1; goto out; }
      if (pv != u) {
        uint8_t *t = row[pv]; row[pv] = row[u], row[u] = t;
        const uint32_t q = phys[pv]; phys[pv] = phys[u], phys[u] = q;
      }
      const uint8_t pc = row[u][u];
      if (pc != 1) {
        const uint8_t inv = gf_inv(pc);
        sym_scale(row[u] + u, inv, I - u);
        RQ_OP(phys[u], UINT32_MAX, inv);
      }
      for (uint32_t k = 0; k < R; k++) {
        const uint8_t f = row[k][u];
        if (k == u || !f) continue;
        sym_muladd(row[k] + u, row[u] + u, f, I - u);
        RQ_OP(phys[k], phys[u], f);
      }
    }
#undef RQ_OP
    pl->nops = (uint32_t)n;
  }
  pl->row_of = malloc((size_t)(I ? I : 1) * sizeof *pl->row_of);
  if (!pl->row_of) goto out;
  for (uint32_t u = 0; u < I; u++) pl->row_of[u] = phys[u];
  ret = 0;
out:
  free(cs), free(live), free(stack), free(order), free(ref), free(inact), free(cr), free(lv), free(used), free(state);
  free(proj), free(bits), free(coef), free(acc8), free(hp), free(row), free(phys);
  for (int b = 0; b < NB; b++) free(bks[b]);
  return ret;
}

static void plan_gj(const rq_plan_t *pl, uint8_t *rows, uint32_t T) {
  for (uint32_t k = 0; k < pl->nops; k++) {
    uint8_t *d = rows + (size_t)pl->op_dst[k] * T;
    if (pl->op_src[k] == UINT32_MAX) sym_scale(d, pl->op_c[k], T);
    else sym_muladd(d, rows + (size_t)pl->op_src[k] * T, pl->op_c[k], T);
  }
}

// The symbol work (rq_plan.h, steps 1 to 4) on the CPU: rhs[r] row r's right side (NULL zero), C the L x T result.
static int plan_exec(const rq_plan_t *pl, uint32_t T, const uint8_t *const *rhs, uint8_t *C) {
  const uint32_t H = pl->p.H, KS = pl->p.Kp + pl->p.S, Nu = pl->Nu, R = pl->R, I = pl->I;
  uint8_t *rr = calloc((size_t)R * T, 1), *as = calloc(T, 1);
  if (!rr || !as) { free(rr), free(as); return -1; }
  for (uint32_t u = 0; u < I; u++) memset(C + (size_t)pl->inact[u] * T, 0, T);
  for (uint32_t t = 0; t < pl->npeel; t++) {
    uint8_t *v = C + (size_t)pl->tgt[t] * T;
    const uint8_t *h = rhs[pl->tgt_row[t]];
    if (h) memcpy(v, h, T);
    else memset(v, 0, T);
    for (uint32_t j = pl->src_start[t], e = j + pl->src_peeled[t]; j < e; j++) sym_xor(v, C + (size_t)pl->src[j] * T, T);
  }
  for (uint32_t k = 0; k < Nu; k++) {
    uint8_t *v = rr + (size_t)k * T;
    const uint8_t *h = rhs[pl->u_row[k]];
    if (h) memcpy(v, h, T);
    for (uint32_t j = pl->u_start[k]; j < pl->u_start[k + 1]; j++) sym_xor(v, C + (size_t)pl->u_src[j] * T, T);
  }
  // The inactive columns read as zero here, so every column below K' + S is added in.
  for (uint32_t j = 0; j < KS; j++) {
    sym_mul2(as, T);
    sym_xor(as, C + (size_t)j * T, T);
    if (j + 1 < KS) {
      sym_xor(rr + (size_t)(Nu + pl->hpair[2 * j]) * T, as, T), sym_xor(rr + (size_t)(Nu + pl->hpair[2 * j + 1]) * T, as, T);
    } else {
      for (uint32_t i = 0; i < H; i++) sym_muladd(rr + (size_t)(Nu + i) * T, as, RQ_OCT_EXP[i], T);
    }
  }
  plan_gj(pl, rr, T);
  for (uint32_t u = 0; u < I; u++) memcpy(C + (size_t)pl->inact[u] * T, rr + (size_t)pl->row_of[u] * T, T);
  for (uint32_t t = 0; t < pl->npeel; t++) {
    uint8_t *v = C + (size_t)pl->tgt[t] * T;
    const uint8_t *h = rhs[pl->tgt_row[t]];
    if (h) memcpy(v, h, T);
    else memset(v, 0, T);
    for (uint32_t j = pl->src_start[t]; j < pl->src_start[t + 1]; j++) sym_xor(v, C + (size_t)pl->src[j] * T, T);
  }
  free(rr), free(as);
  return 0;
}

// The L intermediate symbols into C (L x T): 0, 1 where the rows do not determine them, -1 out of memory.
static int rq_solve(const rq_par_t *p, uint32_t T, const rq_rows_t *rw, uint8_t *C) {
  rq_plan_t *pl = calloc(1, sizeof *pl);
  if (!pl) return -1;
  int ret = plan_build(p, rw, 0, pl);
  if (!ret) ret = plan_exec(pl, T, rw->rhs, C);
  rq_plan_free(pl);
  return ret;
}

// Symbol `isi` from the intermediate symbols C.
static void rq_enc_isi(const rq_par_t *p, uint32_t T, const uint8_t *C, uint32_t isi, uint8_t *out) {
  uint32_t cols[RQ_ROW_MAX];
  const int n = rq_row(p, isi, cols);
  memcpy(out, C + (size_t)cols[0] * T, T);
  for (int k = 1; k < n; k++) sym_xor(out, C + (size_t)cols[k] * T, T);
}

uint32_t rq_symbols(size_t len, uint32_t T) {
  if (!len || !T) return 0;
  const size_t K = (len + T - 1) / T;
  return K > RQ_KMAX ? 0 : (uint32_t)K;
}

// ---- the encoder ----
struct rq_enc {
  rq_par_t p;
  uint32_t T;
  uint8_t *C;
};

rq_enc_t *rq_enc_new(const uint8_t *data, size_t len, uint32_t T) {
  const uint32_t K = rq_symbols(len, T);
  rq_enc_t *e = K ? calloc(1, sizeof *e) : NULL;
  if (!e) return NULL;
  rq_par(K, &e->p);
  e->T = T;
  const uint32_t Kp = e->p.Kp;
  uint32_t *isi = malloc((size_t)Kp * sizeof *isi);
  const uint8_t **rhs = malloc((size_t)Kp * sizeof *rhs);
  uint8_t *tail = calloc(T, 1);
  e->C = malloc((size_t)e->p.L * T);
  rq_rows_t rw = {0};
  int ok = isi && rhs && tail && e->C;
  if (ok) {
    const size_t whole = len / T;
    if (whole < K) memcpy(tail, data + whole * T, len - whole * T);
    for (uint32_t i = 0; i < Kp; i++)
      isi[i] = i, rhs[i] = i < whole ? data + (size_t)i * T : i < K ? tail : NULL;
    ok = !rows_build(&e->p, Kp, isi, rhs, &rw) && !rq_solve(&e->p, T, &rw, e->C);
  }
  rows_free(&rw), free(isi), free(rhs), free(tail);
  if (!ok) { rq_enc_free(e); return NULL; }
  return e;
}

void rq_enc_symbol(const rq_enc_t *e, uint32_t esi, uint8_t *out) {
  const uint32_t isi = esi < e->p.K ? esi : esi + (e->p.Kp - e->p.K);
  rq_enc_isi(&e->p, e->T, e->C, isi, out);
}

void rq_enc_free(rq_enc_t *e) {
  if (!e) return;
  free(e->C), free(e);
}

// ---- the decoder ----
struct rq_dec {
  rq_par_t p;
  uint32_t T, limit, n, cap;
  size_t len;
  uint64_t *seen;   // a bit an ESI
  uint32_t *isi;    // the symbols held, n of them
  uint8_t *sym;
  uint8_t *C;       // the intermediate symbols once solved
};

rq_dec_t *rq_dec_new(size_t len, uint32_t T, uint32_t esi_limit) {
  const uint32_t K = rq_symbols(len, T);
  if (!K || !esi_limit || esi_limit > (1u << 24)) return NULL;
  rq_dec_t *d = calloc(1, sizeof *d);
  if (!d) return NULL;
  rq_par(K, &d->p);
  d->T = T, d->limit = esi_limit, d->len = len;
  d->seen = calloc(((size_t)esi_limit + 63) / 64, sizeof *d->seen);
  if (!d->seen) { free(d); return NULL; }
  return d;
}

uint32_t rq_dec_count(const rq_dec_t *d) { return d->n; }

// The rows of a solve over the symbols held: the padding symbols' (zero right sides), then the held ones in order.
static int dec_rows(const rq_dec_t *d, rq_rows_t *rw) {
  const rq_par_t *p = &d->p;
  const uint32_t pad = p->Kp - p->K, n = pad + d->n;
  uint32_t *isi = malloc((size_t)n * sizeof *isi);
  const uint8_t **rhs = malloc((size_t)n * sizeof *rhs);
  int r = -1;
  if (isi && rhs) {
    for (uint32_t i = 0; i < pad; i++) isi[i] = p->K + i, rhs[i] = NULL;
    for (uint32_t k = 0; k < d->n; k++) isi[pad + k] = d->isi[k], rhs[pad + k] = d->sym + (size_t)k * d->T;
    r = rows_build(p, n, isi, rhs, rw);
  }
  free(isi), free(rhs);
  return r;
}

// Solve from the symbols held: 1 whole, 0 not yet, -1 out of memory.
static int rq_dec_try(rq_dec_t *d) {
  uint8_t *C = malloc((size_t)d->p.L * d->T);
  rq_rows_t rw = {0};
  int r = -1;
  if (C && !dec_rows(d, &rw)) {
    const int s = rq_solve(&d->p, d->T, &rw, C);
    r = s < 0 ? -1 : !s;
  }
  rows_free(&rw);
  if (r == 1) {
    d->C = C;
    free(d->sym), free(d->isi), d->sym = NULL, d->isi = NULL;
  } else {
    free(C);
  }
  return r;
}

int rq_plan_for(size_t len, uint32_t T, uint32_t n, const uint32_t *esi, rq_plan_t **out) {
  *out = NULL;
  const uint32_t K = rq_symbols(len, T);
  rq_par_t p;
  if (!K || rq_par(K, &p)) return -1;
  if (n < K) return 0;
  const uint32_t pad = p.Kp - K, m = pad + n;
  uint32_t *isi = malloc((size_t)m * sizeof *isi);
  const uint8_t **rhs = calloc(m, sizeof *rhs);
  rq_rows_t rw = {0};
  rq_plan_t *pl = calloc(1, sizeof *pl);
  int r = -1;
  if (isi && rhs && pl) {
    for (uint32_t i = 0; i < pad; i++) isi[i] = K + i;
    for (uint32_t k = 0; k < n; k++) isi[pad + k] = esi[k] < K ? esi[k] : esi[k] + pad;
    if (!rows_build(&p, m, isi, rhs, &rw)) {
      const int s = plan_build(&p, &rw, p.S + pad, pl);
      r = s < 0 ? -1 : !s;
    }
  }
  rows_free(&rw), free(isi), free(rhs);
  if (r == 1) *out = pl;
  else rq_plan_free(pl);
  return r;
}

int rq_plan_exec(const rq_plan_t *pl, uint32_t T, uint32_t n, const uint8_t *const *sym, uint8_t *C) {
  const uint32_t rows = pl->sym0 + n;
  const uint8_t **rhs = calloc(rows, sizeof *rhs);
  if (!rhs) return -1;
  for (uint32_t k = 0; k < n; k++) rhs[pl->sym0 + k] = sym[k];
  const int r = plan_exec(pl, T, rhs, C);
  free((void *)rhs);
  return r;
}

int rq_plan_recover(const rq_plan_t *pl, const uint8_t *C, uint32_t T, size_t len, uint8_t *out) {
  uint8_t *tmp = malloc(T);
  if (!tmp) return -1;
  for (uint32_t i = 0; i < pl->p.K; i++) {
    const size_t at = (size_t)i * T;
    if (len - at >= T) rq_enc_isi(&pl->p, T, C, i, out + at);
    else rq_enc_isi(&pl->p, T, C, i, tmp), memcpy(out + at, tmp, len - at);
  }
  free(tmp);
  return 0;
}

int rq_dec_hold(rq_dec_t *d, uint32_t esi, const uint8_t *sym) {
  if (d->C) return 0;
  if (esi >= d->limit) return -1;
  uint64_t *w = d->seen + (esi >> 6);
  const uint64_t bit = (uint64_t)1 << (esi & 63);
  if (*w & bit) return 0;
  if (d->n == d->cap) {
    const uint32_t cap = d->cap ? d->cap * 2 : 1024;
    uint32_t *isi = realloc(d->isi, (size_t)cap * sizeof *isi);
    if (!isi) return -1;
    d->isi = isi;
    uint8_t *s = realloc(d->sym, (size_t)cap * d->T);
    if (!s) return -1;
    d->sym = s, d->cap = cap;
  }
  *w |= bit;
  d->isi[d->n] = esi < d->p.K ? esi : esi + (d->p.Kp - d->p.K);
  memcpy(d->sym + (size_t)d->n * d->T, sym, d->T);
  d->n++;
  return 0;
}

int rq_dec_solve(rq_dec_t *d) {
  if (d->C) return 1;
  return d->n < d->p.K ? 0 : rq_dec_try(d);
}

int rq_dec_add(rq_dec_t *d, uint32_t esi, const uint8_t *sym) {
  if (d->C) return 1;
  const uint32_t n = d->n;
  if (rq_dec_hold(d, esi, sym)) return -1;
  return d->n == n ? 0 : rq_dec_solve(d);
}

int rq_dec_recover(const rq_dec_t *d, uint8_t *out) {
  if (!d->C) return -1;
  uint8_t *tmp = malloc(d->T);
  if (!tmp) return -1;
  for (uint32_t i = 0; i < d->p.K; i++) {
    const size_t at = (size_t)i * d->T, m = d->len - at < d->T ? d->len - at : d->T;
    rq_enc_isi(&d->p, d->T, d->C, i, tmp);
    memcpy(out + at, tmp, m);
  }
  free(tmp);
  return 0;
}

void rq_dec_free(rq_dec_t *d) {
  if (!d) return;
  free(d->seen), free(d->isi), free(d->sym), free(d->C), free(d);
}
