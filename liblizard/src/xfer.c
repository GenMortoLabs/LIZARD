// The transfer's blocks and its BLAKE3 tree (xfer.h has the layouts).
#include "xfer.h"
#include "blake3.h"
#include "blake3_impl.h"   // the official compression function, which the source-block-aligned subtree hash is built on
#include <stdlib.h>
#include <string.h>

static void put_le(uint8_t *p, uint64_t v, int n) { for (int i = 0; i < n; i++) p[i] = (uint8_t)(v >> 8 * i); }

uint32_t xfer_source_blocks(uint64_t length, int sb_log2) {
  if (sb_log2 < XFER_LOG2_MIN || sb_log2 > XFER_LOG2_MAX) return 0;
  const uint64_t n = (length >> sb_log2) + ((length & ((1ull << sb_log2) - 1)) != 0);
  return n > XFER_MAX_SOURCE_BLOCKS ? 0 : (uint32_t)n;
}

int xfer_sb_log2_for(uint64_t length) {
  for (int k = XFER_LOG2_DEFAULT; k <= XFER_LOG2_MAX; k++) if (!length || xfer_source_blocks(length, k)) return k;
  return -1;
}

size_t xfer_sb_len(const xfer_header_t *h, uint32_t index) {
  if (index >= h->source_blocks) return 0;
  const uint64_t size = 1ull << h->sb_log2, at = (uint64_t)index << h->sb_log2;
  return (size_t)(h->length - at < size ? h->length - at : size);
}

uint32_t xfer_symbols(size_t len) { return (uint32_t)((len + XFER_PAYLOAD - 1) / XFER_PAYLOAD); }

int xfer_header_init(xfer_header_t *h, uint64_t length, int sb_log2) {
  memset(h, 0, sizeof *h);
  const uint32_t source_blocks = xfer_source_blocks(length, sb_log2);
  if (sb_log2 < XFER_LOG2_MIN || sb_log2 > XFER_LOG2_MAX || (length && !source_blocks)) return -1;
  h->hash = XFER_HASH_BLAKE3; h->sb_log2 = (uint8_t)sb_log2;
  h->length = length; h->source_blocks = source_blocks;
  return 0;
}

// What every description is held to: the form and the length.
static int consistent(const xfer_header_t *h) {
  if (h->hash != XFER_HASH_BLAKE3) return XFER_ERR_HASH;
  if (h->sb_log2 < XFER_LOG2_MIN || h->sb_log2 > XFER_LOG2_MAX) return XFER_ERR;
  if (h->source_blocks != xfer_source_blocks(h->length, h->sb_log2) || (h->length && !h->source_blocks)) return XFER_ERR;
  return 0;
}

// ---- CBOR (RFC 8949), the description's profile (xfer.h) ----
// ai: CB_* the major types used; a head is the major type in the top three bits and the argument in the rest: below 24
// ai: itself, 24 to 27 one, two, four or eight bytes after (big-endian); 28 to 31 (reserved, indefinite) outside the
// ai: profile, as tags (6) and the simple values but false and true (7: 20, 21)
enum { CB_UINT = 0, CB_NINT = 1, CB_BYTES = 2, CB_TEXT = 3, CB_ARRAY = 4, CB_MAP = 5, CB_SIMPLE = 7, CB_DEPTH = 8 };

// ai: the head at p (n bytes): its major type into *mt, its argument into *arg; its byte count, or -1
static int cb_head(const uint8_t *p, size_t n, int *mt, uint64_t *arg) {
  if (!n) return -1;
  const int m = p[0] >> 5, ai = p[0] & 31;
  int k = 1;
  uint64_t v = (uint64_t)ai;
  if (ai >= 24) {
    if (ai > 27) return -1;
    const int w = 1 << (ai - 24);
    if (n < (size_t)(1 + w)) return -1;
    v = 0;
    for (int i = 0; i < w; i++) v = v << 8 | p[1 + i];
    k = 1 + w;
  }
  if (m == 6 || (m == CB_SIMPLE && v != 20 && v != 21)) return -1;
  *mt = m; *arg = v;
  return k;
}

// ai: the item at p (n bytes) skipped, nested to CB_DEPTH: its byte count, or -1
static long cb_skip(const uint8_t *p, size_t n, int depth) {
  int mt;
  uint64_t a;
  const int k = cb_head(p, n, &mt, &a);
  if (k < 0 || depth > CB_DEPTH) return -1;
  size_t at = (size_t)k;
  if (mt == CB_BYTES || mt == CB_TEXT) return a > n - at ? -1 : (long)(at + a);
  if (mt == CB_ARRAY || mt == CB_MAP) {
    if (a > n) return -1;   // ai: every item a byte at least
    const uint64_t items = mt == CB_MAP ? 2 * a : a;
    for (uint64_t i = 0; i < items; i++) {
      const long s = cb_skip(p + at, n - at, depth + 1);
      if (s < 0) return -1;
      at += (size_t)s;
    }
  }
  return (long)at;
}

// ai: a key at p: an unsigned or negative integer into *key; its byte count, or -1
static int cb_key(const uint8_t *p, size_t n, int64_t *key) {
  int mt;
  uint64_t a;
  const int k = cb_head(p, n, &mt, &a);
  if (k < 0 || (mt != CB_UINT && mt != CB_NINT) || a >> 31) return -1;
  *key = mt == CB_UINT ? (int64_t)a : -1 - (int64_t)a;
  return k;
}

// ai: the head of major type mt and argument v in its shortest form into out: its byte count
static size_t cb_put(uint8_t *out, int mt, uint64_t v) {
  const uint8_t t = (uint8_t)(mt << 5);
  if (v < 24) { out[0] = (uint8_t)(t | v); return 1; }
  const int w = v <= 0xff ? 1 : v <= 0xffff ? 2 : v <= 0xffffffffu ? 4 : 8;
  out[0] = (uint8_t)(t | (24 + (w == 1 ? 0 : w == 2 ? 1 : w == 4 ? 2 : 3)));
  for (int i = 0; i < w; i++) out[1 + i] = (uint8_t)(v >> 8 * (w - 1 - i));
  return (size_t)(1 + w);
}
// ai: a head's bytes in its shortest form
// ai: an integer key's head into out
static size_t cb_put_key(uint8_t *out, int key) { return key >= 0 ? cb_put(out, CB_UINT, (uint64_t)key) : cb_put(out, CB_NINT, (uint64_t)(-1 - key)); }

int xfer_session_write(uint8_t *payload, const uint8_t *desc, size_t len) {
  if (len > XFER_DESC_MAX) return XFER_ERR;
  memset(payload, 0, XFER_PAYLOAD);
  if (len) memcpy(payload, desc, len);
  return 0;
}

int xfer_same_file(const xfer_header_t *a, const xfer_header_t *b) {
  return a->hash == b->hash && a->sb_log2 == b->sb_log2 && a->length == b->length &&
         a->source_blocks == b->source_blocks && !memcmp(a->root, b->root, XFER_CV);
}

int xfer_session_parse(const uint8_t *payload, xfer_session_t *s) {
  memset(s, 0, sizeof *s);
  // ai: the description, one CBOR map from byte 0; the bytes past it are a sender's to write as 0 and a receiver's to
  // ai: ignore (2026-11-20, IP's rule: a later field there is read by receivers already in the field)
  int mt;
  uint64_t a;
  if (cb_head(payload, XFER_PAYLOAD, &mt, &a) < 0 || mt != CB_MAP) return XFER_ERR;
  const long n = cb_skip(payload, XFER_PAYLOAD, 0);
  if (n < 0) return XFER_ERR;
  s->at = 0; s->len = (size_t)n;
  return 0;
}

// ---- BLAKE3 ----

static void cv_bytes(uint8_t out[XFER_CV], const uint32_t cv[8]) { for (int i = 0; i < 8; i++) put_le(out + 4 * i, cv[i], 4); }

static void parent(const uint8_t block[BLAKE3_BLOCK_LEN], int root, uint8_t out[XFER_CV]) {
  uint32_t cv[8];
  memcpy(cv, IV, sizeof cv);
  blake3_compress_in_place(cv, block, BLAKE3_BLOCK_LEN, 0, (uint8_t)(PARENT | (root ? ROOT : 0)));
  cv_bytes(out, cv);
}

// One 1 KiB chunk, or the shorter last one. As the root it is the whole input, so its counter is 0, which is also the
// output block counter the official root output compresses with.
static void chunk(const uint8_t *in, size_t len, uint64_t counter, int root, uint8_t out[XFER_CV]) {
  uint32_t cv[8];
  memcpy(cv, IV, sizeof cv);
  const size_t blocks = len ? (len + BLAKE3_BLOCK_LEN - 1) / BLAKE3_BLOCK_LEN : 1;
  for (size_t b = 0; b < blocks; b++) {
    const size_t at = b * BLAKE3_BLOCK_LEN, n = len - at < BLAKE3_BLOCK_LEN ? len - at : BLAKE3_BLOCK_LEN;
    uint8_t pad[BLAKE3_BLOCK_LEN] = { 0 };
    if (n < BLAKE3_BLOCK_LEN && n) memcpy(pad, in + at, n);
    const uint8_t flags = (uint8_t)((b == 0 ? CHUNK_START : 0) | (b + 1 == blocks ? CHUNK_END | (root ? ROOT : 0) : 0));
    blake3_compress_in_place(cv, n == BLAKE3_BLOCK_LEN ? in + at : pad, (uint8_t)n, counter, flags);
  }
  cv_bytes(out, cv);
}

// More than one source block: the official code's own wide hash (blake3_compress_subtree_wide, as its hasher calls it), its
// vector kernels' lanes at once (4 source blocks on SSE4.1, NEON and wasm SIMD, 8 on AVX2; blake3_dispatch.c picks), which
// leaves at least two chaining values; they pair level by level, an odd last one carried up (its hasher's
// compress_subtree_to_parent_node), to the two under the top node.
static void subtree(const uint8_t *in, size_t len, uint64_t counter, int root, uint8_t out[XFER_CV]) {
  if (len <= BLAKE3_CHUNK_LEN) { chunk(in, len, counter, root, out); return; }
  uint8_t cvs[MAX_SIMD_DEGREE_OR_2 * BLAKE3_OUT_LEN];
  size_t n = blake3_compress_subtree_wide(in, len, IV, counter, 0, cvs, false);
  while (n > 2) {
    size_t m = 0;
    for (size_t i = 0; i + 1 < n; i += 2) parent(cvs + i * XFER_CV, 0, cvs + m++ * XFER_CV);
    if (n & 1) memmove(cvs + m++ * XFER_CV, cvs + (n - 1) * XFER_CV, XFER_CV);
    n = m;
  }
  parent(cvs, root, out);
}

void xfer_b3_subtree(const uint8_t *in, size_t len, uint64_t counter, int root, uint8_t out[XFER_CV]) {
  subtree(in, len, root ? 0 : counter, root, out);
}

static void stream_hasher(uint64_t id, blake3_hasher *h) {
  uint8_t seed[5];
  put_le(seed, id, 5);
  blake3_hasher_init(h);
  blake3_hasher_update(h, seed, sizeof seed);
}
void stream_fill(uint64_t id, uint8_t *out, uint32_t len) {
  blake3_hasher h;
  stream_hasher(id, &h);
  blake3_hasher_finalize(&h, out, len);
}
int stream_matches(uint64_t id, const uint8_t *payload, uint32_t len) {
  blake3_hasher h;
  stream_hasher(id, &h);
  uint8_t b[BLAKE3_BLOCK_LEN];
  for (uint32_t at = 0; at < len; at += BLAKE3_BLOCK_LEN) {
    const uint32_t n = len - at < BLAKE3_BLOCK_LEN ? len - at : BLAKE3_BLOCK_LEN;
    blake3_hasher_finalize_seek(&h, at, b, n);
    if (memcmp(b, payload + at, n)) return 0;
  }
  return 1;
}

void xfer_b3_hash(const uint8_t *in, size_t len, uint8_t out[XFER_CV]) {
  blake3_hasher h;
  blake3_hasher_init(&h);
  blake3_hasher_update(&h, in, len);
  blake3_hasher_finalize(&h, out, XFER_CV);
}

static void merge(const uint8_t *cvs, uint32_t n, int root, uint8_t out[XFER_CV]) {
  if (n == 1) { memcpy(out, cvs, XFER_CV); return; }
  const uint32_t p = (uint32_t)round_down_to_power_of_2(n - 1);
  uint8_t block[BLAKE3_BLOCK_LEN];
  merge(cvs, p, 0, block);
  merge(cvs + (size_t)p * XFER_CV, n - p, 0, block + XFER_CV);
  parent(block, root, out);
}

int xfer_root(const uint8_t *cvs, uint32_t source_blocks, uint8_t root[XFER_CV]) {
  if (source_blocks < 2 || source_blocks > XFER_MAX_SOURCE_BLOCKS) return -1;
  merge(cvs, source_blocks, 1, root);
  return 0;
}

int xfer_root_check(const uint8_t *cvs, uint32_t source_blocks, const uint8_t root[XFER_CV]) {
  uint8_t got[XFER_CV];
  return xfer_root(cvs, source_blocks, got) || memcmp(got, root, XFER_CV) ? -1 : 0;
}

int xfer_sb_cv(const uint8_t *data, size_t len, uint32_t index, int sb_log2, uint8_t cv[XFER_CV]) {
  if (sb_log2 < XFER_LOG2_MIN || sb_log2 > XFER_LOG2_MAX || !len || len > (size_t)1 << sb_log2 || index >= XFER_MAX_SOURCE_BLOCKS) return -1;
  subtree(data, len, (uint64_t)index << (sb_log2 - 10), 0, cv);
  return 0;
}

int xfer_sb_check(const xfer_header_t *h, uint32_t index, const uint8_t *data, size_t len, uint8_t *cv) {
  uint8_t got[XFER_CV];
  if (index >= h->source_blocks || len != xfer_sb_len(h, index)) return -1;
  if (h->source_blocks == 1) { xfer_b3_hash(data, len, got); return memcmp(got, h->root, XFER_CV) ? -1 : 0; }
  return cv && !xfer_sb_cv(data, len, index, h->sb_log2, cv) ? 0 : -1;
}

// ---- The object's TOI (xfer.h) ----
xfer_toi_t xfer_toi_of(const uint8_t root[32], uint64_t length, int sb_log2) {
  uint8_t in[32 + 5 + 1], out[XFER_TOI_BYTES];
  memcpy(in, root, 32);
  for (int i = 0; i < 5; i++) in[32 + i] = (uint8_t)(length >> 8 * i);
  in[37] = (uint8_t)sb_log2;
  blake3_hasher hs;
  blake3_hasher_init_derive_key(&hs, "LIZARD TOI");
  blake3_hasher_update(&hs, in, sizeof in);
  blake3_hasher_finalize(&hs, out, sizeof out);
  xfer_toi_t t = 0;
  for (int i = XFER_TOI_BYTES - 1; i >= 0; i--) t = t << 8 | out[i];
  return t ? t : 1;
}

// ---- The description (xfer.h; SPEC 7.11): the file's own map ----
int xfer_desc_write(const xfer_header_t *h, uint8_t *out) {
  if (consistent(h) || h->compression > XFER_COMP_ZSTD || h->critical) return XFER_ERR;
  // ai: the deterministic form: shortest heads, the keys in ascending order of their bytes (3 to 6, then -1); the TOI
  // ai: not written, a receiver deriving it from k, the length and the root (xfer_toi_of)
  const int pairs = 3 + (h->name_len != 0) + (h->compression != 0);   // ai: k, length, root always
  size_t n = cb_put(out, CB_MAP, (uint64_t)pairs);
  n += cb_put_key(out + n, XFER_K_LOG2); n += cb_put(out + n, CB_UINT, h->sb_log2);
  n += cb_put_key(out + n, XFER_K_LENGTH); n += cb_put(out + n, CB_UINT, h->length);
  n += cb_put_key(out + n, XFER_K_ROOT); n += cb_put(out + n, CB_BYTES, XFER_CV); memcpy(out + n, h->root, XFER_CV); n += XFER_CV;
  if (h->name_len) { n += cb_put_key(out + n, XFER_K_NAME); n += cb_put(out + n, CB_TEXT, h->name_len); memcpy(out + n, h->name, h->name_len); n += h->name_len; }
  // ai: -1 (0x20) sorts after the unsigned keys' bytes
  if (h->compression) { n += cb_put_key(out + n, XFER_K_COMPRESSION); n += cb_put(out + n, CB_UINT, h->compression); }
  return (int)n;
}

// ai: the keys of one map seen (a key twice refuses it): keys from -32 to 31 a bit each, others in a short list
typedef struct { uint64_t bits; int64_t far[8]; int nfar; } keys_t;
static int key_seen(keys_t *k, int64_t key) {
  if (key >= -32 && key < 32) {
    const uint64_t b = 1ull << (key + 32);
    if (k->bits & b) return 1;
    k->bits |= b;
    return 0;
  }
  for (int i = 0; i < k->nfar; i++) if (k->far[i] == key) return 1;
  if (k->nfar == 8) return 1;   // ai: more than eight keys outside -32 .. 31: refused as a duplicate would be
  k->far[k->nfar++] = key;
  return 0;
}

int xfer_desc_read(const uint8_t *desc, size_t len, xfer_toi_t self_toi, xfer_header_t *h, xfer_toi_t *toi) {
  memset(h, 0, sizeof *h);
  h->hash = XFER_HASH_BLAKE3;   // ai: the one hash; another would come as a key to be understood
  int mt, c = 0;
  uint64_t pairs;
  int k = cb_head(desc, len, &mt, &pairs);
  if (k < 0 || mt != CB_MAP || pairs > len) return XFER_ERR;
  size_t at = (size_t)k;
  uint32_t have = 0;
  keys_t seen = { 0 };
  for (uint64_t i = 0; i < pairs; i++) {
    int64_t key;
    if ((k = cb_key(desc + at, len - at, &key)) < 0 || key_seen(&seen, key)) return XFER_ERR;
    at += (size_t)k;
    uint64_t v;
    const int vh = cb_head(desc + at, len - at, &mt, &v);
    if (vh < 0) return XFER_ERR;
    const uint8_t *val = desc + at + vh;
    const size_t left = len - at - (size_t)vh;
    switch (key) {
      case XFER_K_LOG2: if (mt != CB_UINT || v > 63) return XFER_ERR; h->sb_log2 = (uint8_t)v; have |= 2; break;
      case XFER_K_LENGTH: if (mt != CB_UINT || v >> 40) return XFER_ERR; h->length = v; have |= 4; break;
      case XFER_K_ROOT: if (mt != CB_BYTES || v != XFER_CV || left < XFER_CV) return XFER_ERR; memcpy(h->root, val, XFER_CV); have |= 8; break;
      case XFER_K_NAME: if (mt != CB_TEXT || v > XFER_NAME_MAX || left < v) return XFER_ERR; h->name_len = (uint8_t)v; memcpy(h->name, val, (size_t)v); break;
      case XFER_K_COMPRESSION:
        if (mt != CB_UINT) return XFER_ERR;
        // ai: elective: a value this build does not know kept as it is (the file then filed as it came, its name the
        // ai: compressor's), 255 past a byte
        h->compression = v > 255 ? 255 : (uint8_t)v;
        break;
      default: if (key >= 0) c = 1;   // ai: unknown: a negative key skipped; an unsigned one (critical) keeps the file from being presented
    }
    const long s2 = cb_skip(desc + at, len - at, 0);
    if (s2 < 0) return XFER_ERR;
    at += (size_t)s2;
  }
  if (at != len || have != 14) return XFER_ERR;
  h->source_blocks = xfer_source_blocks(h->length, h->sb_log2);
  if (consistent(h)) return XFER_ERR;
  // ai: a file's TOI is its content's (xfer_toi_of), derived, never stated; the session header's own TOI is another's
  const xfer_toi_t t = xfer_toi_of(h->root, h->length, h->sb_log2);
  if (self_toi && t == (self_toi & XFER_TOI_MASK)) return XFER_ERR;
  h->critical = (uint8_t)c;
  if (toi) *toi = t;
  return 0;
}

size_t xfer_name_cut(const uint8_t *name, size_t len, uint8_t out[XFER_NAME_MAX]) {
  if (len <= XFER_NAME_MAX) { if (len) memcpy(out, name, len); return len; }
  size_t ext = 0;   // ai: the extensions' bytes with their dot, 0 for none: the last two where they fit, else the last
  for (size_t i = len, dots = 0; i-- > 1 && len - i <= XFER_EXT_MAX;)
    if (name[i] == '.') { ext = len - i; if (++dots == 2) break; }
  // ai: the stem cut at a character's start (the stem is longer than the cut, as the name is longer than XFER_NAME_MAX)
  size_t stem = XFER_NAME_MAX - ext;
  while (stem > 0 && (name[stem] & 0xc0) == 0x80) stem--;
  memcpy(out, name, stem);
  if (ext) memcpy(out + stem, name + len - ext, ext);
  return stem + ext;
}
