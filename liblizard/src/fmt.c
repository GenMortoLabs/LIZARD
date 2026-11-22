#include "fmt.h"
#include "layout.h"
#include "ldpc.h"
#include "rs.h"
#include <string.h>

const int OB_TIER_RATE[OB_TIERS] = { 6, 4, 3, 2 };

int ob_block_slots(int rate) {
  const int nt = ldpc_sent_bits(rate);
  // ai: the code's sent bits alone: the pilots are apart from the blocks (lizard_phy.c)
  return nt;
}

int ob_fmt_layout(int version, const int count[3], int *m34) {
  if (m34) *m34 = 0;
  if (version < OB_FMT_VERSION_MIN || version > OB_FMT_VERSION_MAX) return 0;
  for (int i = 0; i < 3; i++) if (count[i] < 0 || count[i] > (i ? OB_FMT_COUNT_MAX : OB_FMT_COUNT67_MAX)) return 0;
  const long long all = (long long)version * OB_FMT_SLOTS;
  const long long used = (long long)count[0] * ob_block_slots(OB_TIER_RATE[OB_TIER_67]) + (long long)count[1] * ob_block_slots(OB_TIER_RATE[OB_TIER_23]) +
                         (long long)count[2] * ob_block_slots(OB_TIER_RATE[OB_TIER_12]);
  const long long s3 = ob_block_slots(OB_TIER_RATE[OB_TIER_34]);
  if (used + s3 > all) return 0;
  const int m = (int)((all - used) / s3);
  if (m34) *m34 = m;
  return 1;
}

// ai: The configurations in index order (fmt.h): a version's, then within it by the 6/7 count a, the 2/3 count t and the
// ai: 1/2 count c, each from 0. room: the slots left past a, t and the one 3/4 block a frame needs; c_count: the 1/2
// ai: counts that fit there (0 to the slots' quotient, at most OB_FMT_COUNT_MAX), none where it is negative.
static long long room(int version, int a, int t) {
  return (long long)version * OB_FMT_SLOTS - ob_block_slots(OB_TIER_RATE[OB_TIER_34]) - (long long)a * ob_block_slots(OB_TIER_RATE[OB_TIER_67]) -
         (long long)t * ob_block_slots(OB_TIER_RATE[OB_TIER_23]);
}
static long c_count(long long r) {
  if (r < 0) return 0;
  const long long k = r / ob_block_slots(OB_TIER_RATE[OB_TIER_12]);
  return (long)(k < OB_FMT_COUNT_MAX ? k : OB_FMT_COUNT_MAX) + 1;
}
// ai: the configurations of a version with the 6/7 count a (every t to OB_FMT_COUNT_MAX)
static long t_count(int version, int a) {
  long n = 0;
  for (int t = 0; t <= OB_FMT_COUNT_MAX; t++) { const long k = c_count(room(version, a, t)); if (!k) break; n += k; }
  return n;
}
// ai: the configurations of every version before version v (OB_FMT_BEFORE[v - OB_FMT_VERSION_MIN]), the last entry
// ai: OB_FMT_CONFIGS; each version's own count is the sum of t_count over its a (test/fmt_test.c holds the table to it)
static const long OB_FMT_BEFORE[OB_FMT_VERSION_MAX - OB_FMT_VERSION_MIN + 2] = {
  0, 73, 209, 438, 794, 1316, 2051, 3048, 4342, 6011, 8124, 10751, 13968, 17862, 22519, 28030, 34499, 42028, 50651,
  60548, 71847, 84670, 99144, 115414, 133618, 153899, 176419, 201332, 228636, 258650, 291558, 327535, 366760,
  409434, 455748, 505898, 560102, 618566, 681221, 748557, 820814, 898219, 981005, 1069429, 1163734, 1264168,
  1371006, 1484507, 1604501, 1731681, 1866344, 2008770, 2159244, 2318080, 2485573, 2662024, 2847764, 3043105,
  3247742, 3462605, 3688049, 3924406, 4172012, 4431239, 4702435 };
_Static_assert(OB_FMT_CONFIGS == 4702435, "OB_FMT_BEFORE's last entry is OB_FMT_CONFIGS");

long ob_fmt_index(int version, const int count[3]) {
  if (!ob_fmt_layout(version, count, 0)) return -1;
  long i = OB_FMT_BEFORE[version - OB_FMT_VERSION_MIN];
  for (int a = 0; a < count[0]; a++) i += t_count(version, a);
  for (int t = 0; t < count[1]; t++) i += c_count(room(version, count[0], t));
  return i + count[2];
}

int ob_fmt_config(long index, int *version, int count[3]) {
  if (index < 0 || index >= OB_FMT_CONFIGS) return -1;
  int v = OB_FMT_VERSION_MIN;
  while (OB_FMT_BEFORE[v + 1 - OB_FMT_VERSION_MIN] <= index) v++;
  long r = index - OB_FMT_BEFORE[v - OB_FMT_VERSION_MIN];
  int a = 0, t = 0;
  for (long k; r >= (k = t_count(v, a)); a++) { if (!k || a == OB_FMT_COUNT67_MAX) return -1; r -= k; }
  for (long k; r >= (k = c_count(room(v, a, t))); t++) { if (!k || t == OB_FMT_COUNT_MAX) return -1; r -= k; }
  *version = v; count[0] = a; count[1] = t; count[2] = (int)r;
  return 0;
}

// ai: the word's CRC-5 (fmt.h): x^5 + x^2 + 1 over its 35-bit index (layout.c ob_crc5, by carry-less multiplication
// ai: where the processor has it)
#define crc5 ob_crc5

int ob_fmt_pack(const ob_fmt_t *f, uint8_t data[OB_FMT_DATA]) {
  const long i = ob_fmt_index(f->version, f->count);
  if (i < 0 || f->fps < 0 || f->fps > OB_FMT_FPS_MAX || f->delay < 1 || f->delay > OB_FMT_DELAY_MAX) return -1;
  const uint64_t c = ((uint64_t)i * (OB_FMT_FPS_MAX + 1) + (uint64_t)f->fps) * OB_FMT_DELAY_MAX + (uint64_t)(f->delay - 1), x = c << 5 | crc5(c);
  data[0] = OB_FMT_MAGIC;
  for (int k = 0; k < 5; k++) data[1 + k] = (uint8_t)(x >> (32 - 8 * k));
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

int ob_fmt_encode(const ob_fmt_t *f, uint8_t *cw, int bytes) {
  if (bytes < OB_FMT_BYTES || bytes > OB_FMT_BYTES_MAX) return -1;
  memset(cw, 0, (size_t)bytes);
  if (ob_fmt_pack(f, cw)) return -1;
  rs_encode(cw, OB_FMT_DATA, bytes - OB_FMT_DATA);
  return 0;
}

// RS can miscorrect past its distance, and a wrong word is worse than none: it would send the receiver to another
// picture size and cost every block of the frame. The magic, the CRC-5 and an index under OB_FMT_CONFIGS x
// OB_FMT_CLOCKS are the check.
static int accept(const uint8_t *cw, ob_fmt_t *f) {
  memset(f, 0, sizeof *f);
  if (cw[0] != OB_FMT_MAGIC) return 0;
  uint64_t x = 0;
  for (int k = 0; k < 5; k++) x = x << 8 | cw[1 + k];
  const uint64_t c = x >> 5;
  if ((x & 31) != crc5(c) || c >= (uint64_t)OB_FMT_CONFIGS * OB_FMT_CLOCKS || ob_fmt_config((long)(c / OB_FMT_CLOCKS), &f->version, f->count)) {
    memset(f, 0, sizeof *f);
    return 0;
  }
  f->fps = (int)(c / OB_FMT_DELAY_MAX % (OB_FMT_FPS_MAX + 1)); f->delay = 1 + (int)(c % OB_FMT_DELAY_MAX);
  return 1;
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
  // ai: roots over its erasures (nroots - 3), so the steps run 1, 3, 5, ... where nroots is even (10, 26, 58 in the three
  // ai: rings since 2026-11-22; 2, 4, ... on the odd 11, 27, 59 before). Garbage (soft values uniform in -0.5 to 0.5)
  // ai: passes the magic and its checks (the CRC-5, an index under OB_FMT_CONFIGS x OB_FMT_CLOCKS) at the rate
  // ai: test/fmt_test.c works out exactly. A faded side reads near 0 confidence and the steps
  // ai: erase its bytes: two faded sides read in every ring, three in the 128 and 256. Two glared sides are erased the
  // ai: same way (SPEC.md 5.3).
  int order[OB_FMT_BYTES_MAX];
  uint8_t used[OB_FMT_BYTES_MAX] = { 0 };
  for (int ne = 0; ne < bytes; ne++) {
    int pick = -1;
    for (int k = 0; k < bytes; k++) if (!used[k] && (pick < 0 || worst[k] < worst[pick])) pick = k;   // the lower index wins a tie
    used[pick] = 1; order[ne] = pick;
  }
  for (int ne = 2 - ((nroots - 3) & 1); ne <= nroots - 3; ne += 2) {
    memcpy(got, cw, (size_t)bytes);
    if (rs_decode_er(got, bytes, nroots, order, ne) >= 0 && accept(got, f)) return 0;
  }
  return -1;
}
