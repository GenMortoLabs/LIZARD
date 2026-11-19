#include "fmt.h"
#include "layout.h"
#include "ldpc.h"
#include "rs.h"
#include <string.h>

const int OB_TIER_RATE[OB_TIERS] = { 6, 4, 3, 2 };

int ob_block_slots(int rate) {
  const int nt = ldpc_sent_bits(rate);
  return nt ? nt + (rate == 6 ? 16 : rate == 4 ? 32 : 0) : 0;
}

int ob_fmt_layout(int version, const int count[3], int *m34) {
  if (m34) *m34 = 0;
  if (version < OB_FMT_VERSION_MIN || version > OB_FMT_VERSION_MAX) return 0;
  for (int i = 0; i < 3; i++) if (count[i] < 0 || count[i] > (i ? OB_FMT_COUNT_MAX : OB_FMT_COUNT78_MAX)) return 0;
  const long long all = (long long)version * OB_FMT_SLOTS;
  const long long used = (long long)count[0] * ob_block_slots(OB_TIER_RATE[OB_TIER_78]) + (long long)count[1] * ob_block_slots(OB_TIER_RATE[OB_TIER_23]) +
                         (long long)count[2] * ob_block_slots(OB_TIER_RATE[OB_TIER_12]);
  const long long s3 = ob_block_slots(OB_TIER_RATE[OB_TIER_34]);
  if (used + s3 > all) return 0;
  const int m = (int)((all - used) / s3);
  if (count[0] + m < 2) return 0;
  if (m34) *m34 = m;
  return 1;
}

// ai: the static word's CRC-4 (fmt.h): x^4 + x + 1 over its 28 content bits, most significant first, from 0
static uint32_t crc4(uint32_t content) {
  uint32_t r = content << 4;
  for (int i = 31; i >= 4; i--) if (r >> i & 1) r ^= 0x13u << (i - 4);
  return r & 15;
}

int ob_fmt_pack(const ob_fmt_t *f, uint8_t data[OB_FMT_DATA]) {
  if (!ob_fmt_layout(f->version, f->count, 0)) return -1;
  const uint32_t c = (uint32_t)(f->version - OB_FMT_VERSION_MIN) << 22 | (uint32_t)f->count[0] << 14 | (uint32_t)f->count[1] << 7 |
                     (uint32_t)f->count[2], x = c << 4 | crc4(c);
  data[0] = OB_FMT_MAGIC;
  for (int i = 0; i < 4; i++) data[1 + i] = (uint8_t)(x >> (24 - 8 * i));
  return 0;
}

// Whole word runs inside the band: the word's cells are the band's whole runs of word cells (the rings' bands end on
// a run boundary, so that is half of them), and a partial run at a band's end would stay light, unflagged, rather than
// run past the band over the reserve and the corner mark.
int ob_fmt_side_cells(int side, int reserve) {
  const int w = ob_thin_cells(side, reserve) / (2 * OB_BAND_RUN) * OB_BAND_RUN;
  return w >= OB_FMT_SIDE_CELLS ? w : 0;
}

// From the cells, not from a table of rings: a layout's band says the code it holds, so the painter, the reader and
// the templates that flag the word's cells agree on every ring.
int ob_fmt_bytes(int side, int reserve) {
  const int w = ob_fmt_side_cells(side, reserve), b = 4 * (w / 8);
  return b > OB_FMT_BYTES_MAX ? OB_FMT_BYTES_MAX : b;
}

// Byte k goes to frame side k % 4, eight consecutive cells at 8 * (k / 4). So a side lost to glare costs every
// fourth byte, a quarter of the word, and a smear along one side takes consecutive bytes of one quarter rather than
// of the whole word.
int ob_fmt_bit(const uint8_t *cw, int bytes, int edge, int cell) {
  const int k = 4 * (cell >> 3) + edge;
  if (edge < 0 || edge > 3 || cell < 0 || k >= bytes) return 0;
  return (cw[k] >> (7 - (cell & 7))) & 1;
}

int ob_fmt_held(const ob_fmt_t *f) {
  if (!f || f->kind != OB_FMT_STATIC || !ob_fmt_layout(f->version, f->count, 0)) return 0;
  return f->version << 22 | f->count[0] << 14 | f->count[1] << 7 | f->count[2];
}

int ob_fmt_unheld(int held, ob_fmt_t *f) {
  memset(f, 0, sizeof *f);
  if (held <= 0) return 0;
  f->kind = OB_FMT_STATIC;
  f->version = held >> 22;
  f->count[0] = held >> 14 & 255; f->count[1] = held >> 7 & 127; f->count[2] = held & 127;
  if (ob_fmt_layout(f->version, f->count, 0)) return 1;
  memset(f, 0, sizeof *f);
  return 0;
}

int ob_fmt_encode(const ob_fmt_t *f, uint8_t *cw, int bytes) {
  if (bytes < OB_FMT_BYTES || bytes > OB_FMT_BYTES_MAX) return -1;
  memset(cw, 0, (size_t)bytes);
  if (f->kind == OB_FMT_ANIM) {
    if (f->id < 0 || f->id > OB_FMT_ID_MAX) return -1;
    cw[0] = OB_FMT_MAGIC_ANIM;
    cw[1] = (uint8_t)f->id;
    memcpy(cw + 2, f->sym, 3);
  } else if (f->kind != OB_FMT_STATIC || ob_fmt_pack(f, cw)) return -1;
  rs_encode(cw, OB_FMT_DATA, bytes - OB_FMT_DATA);
  return 0;
}

// RS can miscorrect past its distance, and a wrong word is worse than none: it would send the receiver to another
// picture size and cost every block of every frame until it changed again. A magic, and the static word's counts making
// a frame of its version (ob_fmt_layout) and its CRC-4, or the animated word's id range, are the check. An
// animated word's three bytes answer to the header's CRC, so they are taken as they come.
static int accept(const uint8_t *cw, ob_fmt_t *f) {
  memset(f, 0, sizeof *f);
  if (cw[0] == OB_FMT_MAGIC) {
    const uint32_t x = (uint32_t)cw[1] << 24 | (uint32_t)cw[2] << 16 | (uint32_t)cw[3] << 8 | cw[4];
    f->kind = OB_FMT_STATIC;
    f->version = OB_FMT_VERSION_MIN + (int)(x >> 26);
    f->count[0] = (int)(x >> 18) & 255; f->count[1] = (int)(x >> 11) & 127; f->count[2] = (int)(x >> 4) & 127;
    if ((x & 15) == crc4(x >> 4) && ob_fmt_layout(f->version, f->count, 0)) return 1;
    memset(f, 0, sizeof *f);
    return 0;
  }
  if (cw[0] == OB_FMT_MAGIC_ANIM && cw[1] <= OB_FMT_ID_MAX) {
    f->kind = OB_FMT_ANIM;
    f->id = cw[1];
    memcpy(f->sym, cw + 2, 3);
    return 1;
  }
  return 0;
}

int ob_fmt_decode(const float *q, int cells, int bytes, ob_fmt_t *f) {
  const int nroots = bytes - OB_FMT_DATA;
  if (bytes < OB_FMT_BYTES || bytes > OB_FMT_BYTES_MAX || cells < 8 * ((bytes + 3) / 4)) return -1;
  uint8_t cw[OB_FMT_BYTES_MAX], got[OB_FMT_BYTES_MAX];
  float worst[OB_FMT_BYTES_MAX];
  for (int k = 0; k < bytes; k++) {
    const float *qs = q + (k & 3) * cells + 8 * (k >> 2);
    uint8_t v = 0;
    float lo = 1;
    for (int b = 0; b < 8; b++) {
      const float s = qs[b], a = s < 0 ? -s : s;
      v = (uint8_t)((v << 1) | (s > 0));
      if (a < lo) lo = a;
    }
    cw[k] = v; worst[k] = lo;
  }
  memcpy(got, cw, (size_t)bytes);
  if (rs_decode(got, bytes, nroots) >= 0 && accept(got, f)) return 0;
  // ai: Then the soft values: the least confident bytes go in as erasures, 2 more a step. rs_decode_er refuses any
  // ai: result past 2e + f <= nroots (rs.c solve), which the garbage figures below assume. The last step leaves three
  // ai: roots over its erasures (nroots - 3; every nroots here is odd: 11, 27, 59 in the three rings). Garbage (soft
  // ai: values uniform in -0.5 to 0.5) passes a magic and its check (a static word's CRC-4 and counts making a frame, an animated
  // ai: word's id to 254) at the rate test/fmt_test.c works out exactly. A faded side reads near 0 confidence and the steps
  // ai: erase its bytes: two faded sides read in every ring, three in the 128 and 256. Two glared sides are erased the
  // ai: same way (SPEC.md 5.3).
  int order[OB_FMT_BYTES_MAX];
  uint8_t used[OB_FMT_BYTES_MAX] = { 0 };
  for (int ne = 0; ne < bytes; ne++) {
    int pick = -1;
    for (int k = 0; k < bytes; k++) if (!used[k] && (pick < 0 || worst[k] < worst[pick])) pick = k;   // the lower index wins a tie
    used[pick] = 1; order[ne] = pick;
  }
  for (int ne = 2; ne <= nroots - 3; ne += 2) {
    memcpy(got, cw, (size_t)bytes);
    if (rs_decode_er(got, bytes, nroots, order, ne) >= 0 && accept(got, f)) return 0;
  }
  return -1;
}

// The carousel (fmt.h). Each of a symbol's three bytes is its own RS(255, k) codeword's byte; the header's 3k bytes
// are the codewords' data, interleaved byte by byte.
int ob_car_header(const uint8_t *content, int len, uint8_t *hdr) {
  if (len < 0) return -1;
  int k = (len + OB_CAR_CRC + OB_CAR_SYM - 1) / OB_CAR_SYM;
  if (k < OB_CAR_K_MIN) k = OB_CAR_K_MIN;
  if (k > OB_CAR_N) return -1;
  const int n = OB_CAR_SYM * k;
  memset(hdr, 0, (size_t)n);
  if (len) memcpy(hdr, content, (size_t)len);
  const uint32_t c = ob_crc32c(hdr, n - OB_CAR_CRC);
  for (int i = 0; i < OB_CAR_CRC; i++) hdr[n - OB_CAR_CRC + i] = (uint8_t)(c >> (24 - 8 * i));
  return k;
}

int ob_car_content(const ob_fmt_t *f, uint8_t *out) {
  ob_fmt_t g = *f;
  g.kind = OB_FMT_STATIC;
  if (f->fps < 0 || f->fps > 65535 || ob_fmt_pack(&g, out)) return -1;
  out[5] = (uint8_t)(f->fps >> 8); out[6] = (uint8_t)f->fps;
  return OB_CAR_CONTENT_MAX;
}

int ob_car_config(const uint8_t *hdr, int k, ob_fmt_t *f) {
  memset(f, 0, sizeof *f);
  const int len = OB_CAR_SYM * k - OB_CAR_CRC;
  ob_fmt_t g;
  if (k < OB_CAR_K_MIN || k > OB_CAR_N || len < OB_CAR_CONTENT_MAX || !accept(hdr, &g) || g.kind != OB_FMT_STATIC) return 0;
  g.fps = hdr[5] << 8 | hdr[6];
  // ai: the records: none is defined yet, so each is only walked over (a receiver skips what it does not know)
  for (int at = OB_CAR_CONTENT_MAX; at < len && hdr[at];) {
    if (at + 2 > len || at + 2 + hdr[at + 1] > len) return 0;
    at += 2 + hdr[at + 1];
  }
  *f = g;
  return 1;
}

int ob_car_symbol(const uint8_t *hdr, int k, int id, uint8_t sym[3]) {
  if (k < OB_CAR_K_MIN || k > OB_CAR_N || id < 0 || id >= OB_CAR_N) return -1;
  uint8_t cw[OB_CAR_N];
  for (int j = 0; j < OB_CAR_SYM; j++) {
    for (int p = 0; p < k; p++) cw[p] = hdr[OB_CAR_SYM * p + j];
    if (id >= k) rs_encode(cw, k, OB_CAR_N - k);
    sym[j] = cw[id];
  }
  return 0;
}

void ob_car_reset(ob_car_rx_t *r) { memset(r, 0, sizeof *r); }

// ai: k from the ids held down: with every symbol right the header stands at the first try once k are held. The
// ai: codes are nested (RS(255, k) inside RS(255, k + 1)), so past the true k the decode gives the same codeword and its
// ai: CRC, over parity bytes, fails; below it a decode fails or its CRC does. A new id tries k down to m -
// ai: OB_CAR_REACH alone (the header stands at m = k + 2e with e wrong symbols, so up to OB_CAR_REACH / 2 of them are
// ai: corrected as they come); a vote that flips tries every k. 0 where none holds.
static int car_solve(ob_car_rx_t *r, int reach) {
  int have[OB_CAR_N], eras[OB_CAR_N], m = 0, ne = 0;
  for (int p = 0; p < OB_CAR_N; p++) { if (r->cnt[p][0]) have[m++] = p; else eras[ne++] = p; }
  uint8_t cw[OB_CAR_SYM][OB_CAR_N], got[OB_CAR_N], hdr[OB_CAR_SYM * OB_CAR_N];
  for (int j = 0; j < OB_CAR_SYM; j++) {
    memset(cw[j], 0, OB_CAR_N);
    for (int i = 0; i < m; i++) cw[j][have[i]] = r->val[have[i]][0][j];
  }
  for (int k = m; k >= OB_CAR_K_MIN && k >= m - reach; k--) {
    int ok = 1;
    for (int j = 0; j < OB_CAR_SYM && ok; j++) {
      memcpy(got, cw[j], OB_CAR_N);
      if ((ne ? rs_decode_er(got, OB_CAR_N, OB_CAR_N - k, eras, ne) : rs_decode(got, OB_CAR_N, OB_CAR_N - k)) < 0) { ok = 0; break; }
      for (int p = 0; p < k; p++) hdr[OB_CAR_SYM * p + j] = got[p];
    }
    if (!ok) continue;
    const int n = OB_CAR_SYM * k;
    uint32_t c = 0;
    for (int i = 0; i < OB_CAR_CRC; i++) c = c << 8 | hdr[n - OB_CAR_CRC + i];
    if (c != ob_crc32c(hdr, n - OB_CAR_CRC)) continue;
    memcpy(r->hdr, hdr, (size_t)n);
    r->k = k;
    return k;
  }
  return 0;
}

int ob_car_add(ob_car_rx_t *r, int id, const uint8_t sym[3]) {
  if (id < 0 || id >= OB_CAR_N) return r->k;
  uint8_t (*v)[OB_CAR_SYM] = r->val[id];
  uint32_t *c = r->cnt[id];
  const int fresh = !c[0];
  int i = !c[0] || !memcmp(v[0], sym, OB_CAR_SYM) ? 0 : !c[1] || !memcmp(v[1], sym, OB_CAR_SYM) ? 1 : -1;
  if (i < 0) { i = 1; c[1] = 0; }   // ai: a third value replaces the less read
  if (!c[i]) memcpy(v[i], sym, OB_CAR_SYM);
  c[i]++;
  int flip = 0;
  if (i == 1 && c[1] > c[0]) {
    uint8_t t[OB_CAR_SYM];
    memcpy(t, v[0], OB_CAR_SYM); memcpy(v[0], v[1], OB_CAR_SYM); memcpy(v[1], t, OB_CAR_SYM);
    const uint32_t tc = c[0]; c[0] = c[1]; c[1] = tc;
    flip = 1;
  }
  if (fresh) r->held++;
  if (flip) car_solve(r, OB_CAR_N);
  else if (fresh) car_solve(r, OB_CAR_REACH);
  return r->k;
}

