#include "fmt.h"
#include "fmt_tables.h"
#include "layout.h"
#include "ldpc.h"
#include "rs.h"
#include <string.h>

const int OB_TIER_RATE[OB_TIERS] = { 6, 4, 3, 2 };

// ai: the code's sent bits alone: the pilots are apart from the blocks (lizard_phy.c)
int ob_block_slots(int rate) { return ldpc_sent_bits(rate); }

int ob_fmt_layout(int version, const int count[OB_TIERS]) {
  if (version < OB_FMT_VERSION_MIN || version > OB_FMT_VERSION_MAX) return 0;
  long long used = 0;
  for (int i = 0; i < OB_TIERS; i++) {
    if (count[i] < 0) return 0;
    used += (long long)count[i] * ob_block_slots(OB_TIER_RATE[i]);
  }
  return used <= (long long)version * OB_FMT_SLOTS;
}

// ai: The configurations in index order (fmt.h): a version's, then within it by the 6/7 count a, the 3/4 count b, the
// ai: 2/3 count t and the 1/2 count c, each from 0. In units of OB_FMT_UNIT slots (fmt_tables.h) a layout fits version v
// ai: where 7a + 8b + 9t + 12c <= R(v), R(v) = floor(10176 v / 640): the layouts with a given a number FMT_P3[R - 7a],
// ai: those with a and b FMT_P2[R - 7a - 8b], those with a, b and t floor((R - 7a - 8b - 9t) / 12) + 1, the 1/2 counts
// ai: that fit. A count that leaves no room past the version's R makes no layout.
_Static_assert(OB_FMT_SLOTS * OB_FMT_VERSION_MAX / OB_FMT_UNIT == FMT_ROOM_MAX, "fmt_tables.h is the format's");
static int room_of(int version) { return OB_FMT_SLOTS * version / OB_FMT_UNIT; }

long ob_fmt_index(int version, const int count[OB_TIERS]) {
  if (!ob_fmt_layout(version, count)) return -1;
  int r = room_of(version);
  long i = (long)FMT_BEFORE[version - OB_FMT_VERSION_MIN];
  for (int x = 0; x < count[0]; x++) i += (long)FMT_P3[r - 7 * x];
  r -= 7 * count[0];
  for (int x = 0; x < count[1]; x++) i += (long)FMT_P2[r - 8 * x];
  r -= 8 * count[1];
  for (int x = 0; x < count[2]; x++) i += (r - 9 * x) / 12 + 1;
  return i + count[3];
}

int ob_fmt_config(long index, int *version, int count[OB_TIERS]) {
  if (index < 0 || index >= OB_FMT_CONFIGS) return -1;
  int v = OB_FMT_VERSION_MIN;
  while ((long)FMT_BEFORE[v + 1 - OB_FMT_VERSION_MIN] <= index) v++;
  long q = index - (long)FMT_BEFORE[v - OB_FMT_VERSION_MIN];
  int r = room_of(v), a = 0, b = 0, t = 0;
  for (; q >= (long)FMT_P3[r - 7 * a]; a++) q -= (long)FMT_P3[r - 7 * a];
  r -= 7 * a;
  for (; q >= (long)FMT_P2[r - 8 * b]; b++) q -= (long)FMT_P2[r - 8 * b];
  r -= 8 * b;
  for (; q >= (r - 9 * t) / 12 + 1; t++) q -= (r - 9 * t) / 12 + 1;
  *version = v; count[0] = a; count[1] = b; count[2] = t; count[3] = (int)q;
  return 0;
}

int ob_fmt_pack(const ob_fmt_t *f, uint8_t data[OB_FMT_DATA]) {
  const long i = ob_fmt_index(f->version, f->count);
  if (i < 0 || f->fps < 0 || f->fps > OB_FMT_FPS_MAX || f->delay < 1 || f->delay > OB_FMT_DELAY_MAX) return -1;
  // ai: the clock k: 0 a still (its delay not stated), else fps 1 to 240 with delay 1 to 30
  const uint64_t k = f->fps ? 1 + (uint64_t)(f->fps - 1) * OB_FMT_DELAY_MAX + (uint64_t)(f->delay - 1) : 0;
  const uint64_t x = (uint64_t)i * OB_FMT_CLOCKS + k;
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
// picture size and cost every block of the frame. The magic and an index under OB_FMT_CONFIGS x OB_FMT_CLOCKS are the
// check: a false word costs that one frame (its blocks fail their CRC-64), never a byte of a file.
static int accept(const uint8_t *cw, ob_fmt_t *f) {
  memset(f, 0, sizeof *f);
  if (cw[0] != OB_FMT_MAGIC) return 0;
  uint64_t x = 0;
  for (int k = 0; k < 5; k++) x = x << 8 | cw[1 + k];
  const uint64_t c = x;
  if (c >= (uint64_t)OB_FMT_CONFIGS * OB_FMT_CLOCKS || ob_fmt_config((long)(c / OB_FMT_CLOCKS), &f->version, f->count)) {
    memset(f, 0, sizeof *f);
    return 0;
  }
  const uint64_t k = c % OB_FMT_CLOCKS;
  f->fps = k ? 1 + (int)((k - 1) / OB_FMT_DELAY_MAX) : 0; f->delay = k ? 1 + (int)((k - 1) % OB_FMT_DELAY_MAX) : 1;
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
  // ai: result past 2e + f <= nroots (rs.c solve), which the garbage figures below assume. The last step leaves one
  // ai: root over its erasures (nroots - 1; three until 2026-11-23, when the word's false reads were judged noise), so
  // ai: the steps run 1, 3, 5, ... where nroots is even (10, 26, 58 in the three rings since 2026-11-22; 2, 4, ... on the
  // ai: odd 11, 27, 59 before). Garbage (soft values uniform in -0.5 to 0.5)
  // ai: passes the magic and the index's range (under OB_FMT_CONFIGS x OB_FMT_CLOCKS) at the rate
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
  for (int ne = 2 - ((nroots - 1) & 1); ne <= nroots - 1; ne += 2) {
    memcpy(got, cw, (size_t)bytes);
    if (rs_decode_er(got, bytes, nroots, order, ne) >= 0 && accept(got, f)) return 0;
  }
  return -1;
}
