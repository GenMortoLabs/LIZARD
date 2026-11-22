// The transfer's C side (src/xfer.h): BLAKE3 and the source block-aligned subtree hash against the official
// test vectors, the block id, the source block size a sender picks, and files of 41 source blocks, 16, 3, 1, a block and none written
// as their file entries, read back, each source block's chaining value made and the values combined into the root, with a
// corrupted source block, a wrong chaining value, another transfer's root and entries that are not one; the key rule
// (unsigned keys critical, negative ones elective) (no fountain in this test: the wasm test exercises
// RaptorQ, liblizard/test/rq_test.c its C). From liblizard/ on x86-64, about 3 s (one
// line; the AVX2 kernel where the CPU has it; -DBLAKE3_NO_AVX2 for the SSE4.1 one, -DBLAKE3_NO_SSE41 -DBLAKE3_NO_AVX2
// for the portable code):
//   gcc -O2 -g -Wall -Wextra -fsanitize=address,undefined -Isrc -Ivendor/blake3 -DBLAKE3_NO_SSE2 -DBLAKE3_NO_AVX512
//   -o build/xfer_test test/xfer_test.c src/xfer.c vendor/blake3/blake3.c vendor/blake3/blake3_dispatch.c
//   vendor/blake3/blake3_portable.c vendor/blake3/blake3_sse41_x86-64_unix.S vendor/blake3/blake3_avx2_x86-64_unix.S
//   && build/xfer_test
#include "xfer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks, fails;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t rng = 2463534242u;
static void fill(uint8_t *p, size_t n) { for (size_t i = 0; i < n; i++) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; p[i] = (uint8_t)rng; } }
static int unhex(const char *s, uint8_t *out, int n) {
  for (int i = 0; i < n; i++) { unsigned v; if (sscanf(s + 2 * i, "%2x", &v) != 1) return -1; out[i] = (uint8_t)v; }
  return 0;
}

// The chaining values of a file's source blocks, and its root through them (or the source block's own hash for one source block).
static uint8_t *chunk_cvs(const uint8_t *file, uint64_t len, int log2, uint32_t *source_blocks) {
  *source_blocks = xfer_source_blocks(len, log2);
  uint8_t *cvs = calloc(*source_blocks ? *source_blocks : 1, XFER_CV);
  for (uint32_t i = 0; i < *source_blocks; i++) {
    const uint64_t at = (uint64_t)i << log2, n = len - at < (1ull << log2) ? len - at : 1ull << log2;
    if (xfer_sb_cv(file + at, (size_t)n, i, log2, cvs + (size_t)i * XFER_CV)) { free(cvs); return NULL; }
  }
  return cvs;
}
static int root_via_chunks(const uint8_t *file, uint64_t len, int log2, uint8_t root[XFER_CV]) {
  uint32_t source_blocks;
  uint8_t *cvs = chunk_cvs(file, len, log2, &source_blocks);
  if (!cvs) return -1;
  int rc = 0;
  if (source_blocks >= 2) rc = xfer_root(cvs, source_blocks, root);
  else xfer_b3_subtree(file, (size_t)len, 0, 1, root);
  free(cvs);
  return rc;
}

static void vectors(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { CHECK(0, "cannot open %s", path); return; }
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *js = malloc((size_t)n + 1);
  js[fread(js, 1, (size_t)n, f)] = 0;
  fclose(f);
  int cases = 0, splits = 0;
  for (const char *p = strstr(js, "\"input_len\""); p; p = strstr(p + 1, "\"input_len\"")) {
    const int len = atoi(strchr(p, ':') + 1);
    const char *h = strstr(p, "\"hash\"");
    uint8_t want[XFER_CV], got[XFER_CV];
    if (!h || unhex(strchr(h + 6, '"') + 1, want, XFER_CV)) { CHECK(0, "vector %d unreadable", len); continue; }
    uint8_t *in = malloc((size_t)len + 1);
    for (int i = 0; i < len; i++) in[i] = (uint8_t)(i % 251);
    xfer_b3_hash(in, (size_t)len, got);
    CHECK(!memcmp(got, want, XFER_CV), "official hasher, input_len %d", len);
    xfer_b3_subtree(in, (size_t)len, 0, 1, got);
    CHECK(!memcmp(got, want, XFER_CV), "subtree as root, input_len %d", len);
    // Every source block size that splits it: the source blocks' chaining values must combine to the vector's hash.
    for (int log2 = XFER_LOG2_MIN; log2 <= 17 && len; log2++) {
      if (root_via_chunks(in, (uint64_t)len, log2, got)) { CHECK(0, "chunks of 2^%d, input_len %d", log2, len); continue; }
      CHECK(!memcmp(got, want, XFER_CV), "chunks of 2^%d, input_len %d", log2, len);
      splits += xfer_source_blocks((uint64_t)len, log2) >= 2;
    }
    free(in);
    cases++;
  }
  free(js);
  printf("official vectors: %d cases, each as one buffer, as the root subtree, and at chunk sizes 2^10 to 2^17 (%d splits into 2 or more chunks)\n", cases, splits);
  CHECK(cases == 35, "expected the 35 cases of test_vectors.json, read %d", cases);
}

// Lengths either side of the source block boundaries at several source block sizes, against the official hasher on the whole file.
static void boundaries(void) {
  static const int logs[] = { 10, 11, 13, 16 };
  int n = 0;
  for (int li = 0; li < 4; li++) {
    const uint64_t c = 1ull << logs[li];
    for (uint64_t k = 1; k <= 20; k += (k < 5 ? 1 : 5)) {
      const uint64_t lens[] = { k * c - 1, k * c, k * c + 1, k * c + XFER_PAYLOAD, k * c + c / 2 };
      for (int j = 0; j < 5; j++) {
        const uint64_t len = lens[j];
        uint8_t *file = malloc((size_t)len), want[XFER_CV], got[XFER_CV];
        fill(file, (size_t)len);
        xfer_b3_hash(file, (size_t)len, want);
        CHECK(!root_via_chunks(file, len, logs[li], got) && !memcmp(got, want, XFER_CV), "length %llu at 2^%d", (unsigned long long)len, logs[li]);
        free(file);
        n++;
      }
    }
  }
  printf("chunk boundaries: %d files at 4 chunk sizes, each root through its chunks against the official hasher\n", n);
}

static void ids(void) {
  CHECK(xfer_id(0, 5) == 5, "chunk 0 keeps today's ids");
  CHECK(xfer_id(1, 0) == 1u << 24, "chunk 1 starts at 2^24");
  CHECK(xfer_id(65534, (1u << 24) - 1) == 0xfffeffffffull, "last data id");
  CHECK(XFER_ID_HEADER == 0xffff000000ull, "the header's id");
  CHECK(xfer_id_kind(0xfffeffffffull) == XFER_KIND_DATA && xfer_id_kind(XFER_ID_HEADER) == XFER_KIND_HEADER, "kinds");
  CHECK(xfer_id_kind(XFER_ID_HEADER + 1) == XFER_KIND_RESERVED && xfer_id_kind(0xffffffffffull) == XFER_KIND_RESERVED, "every other control id reserved");
  CHECK(xfer_id_sbn(xfer_id(54321, 7654321)) == 54321 && xfer_id_esi(xfer_id(54321, 7654321)) == 7654321, "fields");
  {
    // ai: the bare id in bytes 0 to 4, the TOI in 5 to 10, both little-endian, byte 11 written 0 (2026-11-21)
    uint8_t b[XFER_BLOCK];
    memset(b, 0xaa, sizeof b);
    xfer_id_put(b, 0x9a12345678ull);
    xfer_toi_put(b, 0xfedcba987654ull);
    CHECK(b[0] == 0x78 && b[3] == 0x12 && b[4] == 0x9a && b[5] == 0x54 && b[10] == 0xfe && b[11] == 0 && b[12] == 0xaa &&
          xfer_id_get(b) == 0x9a12345678ull && xfer_toi_get(b) == 0xfedcba987654ull, "the bare id in bytes 0 to 4, the TOI in 5 to 10, byte 11 zeroed");
    const xfer_id_t t = xfer_id_get(b);
    CHECK(xfer_id_sbn(t) == 0x9a12 && xfer_id_esi(t) == 0x345678, "the fields apart");
    xfer_toi_put(b, XFER_TOI_MASK);
    CHECK(xfer_toi_get(b) == XFER_TOI_MASK && xfer_id_get(b) == 0x9a12345678ull, "the widest TOI beside the id");
  }
  CHECK(xfer_source_blocks(65535ull << 24, 24) == 65535 && !xfer_source_blocks((65535ull << 24) + 1, 24), "the largest file");
  CHECK(!xfer_source_blocks(1, 9) && !xfer_source_blocks(1, 25), "chunk size range");
  xfer_header_t h;
  CHECK(xfer_header_init(&h, (65535ull << 24) + 1, 24) && !xfer_header_init(&h, 0, 10) && h.source_blocks == 0, "init limits");
  CHECK(xfer_symbols(0) == 0 && xfer_symbols(1) == 1 && xfer_symbols(XFER_PAYLOAD) == 1 && xfer_symbols(XFER_PAYLOAD + 1) == 2, "blocks of a chunk's bytes");
  // ai: the source block size a sender picks: 16 MiB, while the id holds the source blocks (16,383 of 16 MiB)
  CHECK(xfer_sb_log2_for(0) == 24 && xfer_sb_log2_for(1) == 24 && xfer_sb_log2_for(65535ull << 22) == 24 &&
        xfer_sb_log2_for((65535ull << 22) + 1) == 24 && xfer_sb_log2_for(65535ull << 24) == 24 && xfer_sb_log2_for((65535ull << 24) + 1) == -1,
        "the chunk size by length");
  CHECK(XFER_K_COMPRESSION == -1 && XFER_K_LOG2 == 3 && XFER_K_LENGTH == 4 &&
        XFER_K_ROOT == 5 && XFER_K_NAME == 6 && XFER_DESC_MAX == 460 && XFER_COMP_ZSTD == 1 &&
        XFER_DESC_LONGEST == 309 && XFER_NAME_MAX == 255 && XFER_EXT_MAX == 32,
        "the description's CBOR keys, its 460 bytes, its longest 309");
  CHECK(XFER_ID_BYTES == 12 && XFER_ID_BARE_BYTES == 5 && XFER_ID_TOI == 5 && XFER_TOI_BYTES == 6 && XFER_ID_SPARE == 11 && XFER_TOI_BITS == 48,
        "the id: 40 bits bare, then the 48-bit TOI and a reserved byte");
  {
    // ai: a file's TOI its content's (xfer_toi_of): a function of the root, length and source block size, each moving it
    uint8_t r[XFER_CV];
    for (int i = 0; i < XFER_CV; i++) r[i] = (uint8_t)i;
    const xfer_toi_t t0 = xfer_toi_of(r, 1000, 24);
    r[31] ^= 1;
    const xfer_toi_t t1 = xfer_toi_of(r, 1000, 24);
    r[31] ^= 1;
    CHECK(t0 && t0 <= XFER_TOI_MASK && t0 == xfer_toi_of(r, 1000, 24) && t1 != t0 && xfer_toi_of(r, 1001, 24) != t0 &&
          xfer_toi_of(r, 1000, 23) != t0 && xfer_toi_of(r, 1000ull | 1ull << 39, 24) != t0, "a TOI its content's, each field moving it");
    CHECK(t0 == 0x735b8ee29147ull, "the TOI's vector: root 00..1f, 1000 bytes, k 24 is 735b8ee29147 (%012llx)", (unsigned long long)t0);
  }

}

// A file through the whole C side: chaining values, root, its description written as its FDT entry and read back, each
// source block's chaining value made as a receiver decodes it and the values combined into the root (a file of one
// source block checked against the root at once), then corruption. Returns the source blocks.
// ai: a CBOR builder of the test's own (src/xfer.c's writer is what is under test): a map of key/value pairs, a value an
// ai: unsigned integer, a byte or text string, or raw bytes; wide forces an eight-byte head (a form the senders never
// ai: write, which a receiver reads all the same)
typedef struct { int64_t key; int mt; uint64_t v; const uint8_t *s; size_t sl; int wide; } kv_t;
static size_t tb_head(uint8_t *o, int mt, uint64_t v, int wide) {
  if (!wide && v < 24) { o[0] = (uint8_t)(mt << 5 | v); return 1; }
  const int w = wide ? 8 : v <= 0xff ? 1 : v <= 0xffff ? 2 : v <= 0xffffffffu ? 4 : 8;
  o[0] = (uint8_t)(mt << 5 | (24 + (w == 1 ? 0 : w == 2 ? 1 : w == 4 ? 2 : 3)));
  for (int i = 0; i < w; i++) o[1 + i] = (uint8_t)(v >> 8 * (w - 1 - i));
  return (size_t)(1 + w);
}
static size_t tb_map(uint8_t *o, const kv_t *kv, int n) {
  size_t k = tb_head(o, 5, (uint64_t)n, 0);
  for (int i = 0; i < n; i++) {
    k += kv[i].key >= 0 ? tb_head(o + k, 0, (uint64_t)kv[i].key, 0) : tb_head(o + k, 1, (uint64_t)(-1 - kv[i].key), 0);
    if (kv[i].mt == -1) { memcpy(o + k, kv[i].s, kv[i].sl); k += kv[i].sl; continue; }   // ai: raw bytes as the value
    k += tb_head(o + k, kv[i].mt, kv[i].mt == 2 || kv[i].mt == 3 ? kv[i].sl : kv[i].v, kv[i].wide);
    if (kv[i].mt == 2 || kv[i].mt == 3) { memcpy(o + k, kv[i].s, kv[i].sl); k += kv[i].sl; }
  }
  return k;
}
// ai: an entry's pairs from h under toi (the order the senders write)
static int tb_pairs(kv_t *kv, const xfer_header_t *h, xfer_toi_t toi) {
  int n = 0;
  (void)toi;   // ai: a file map states no TOI since 2026-11-22 (derived, xfer_toi_of)
  kv[n++] = (kv_t){ 3, 0, h->sb_log2, NULL, 0, 0 };
  kv[n++] = (kv_t){ 4, 0, h->length, NULL, 0, 0 };
  kv[n++] = (kv_t){ 5, 2, 0, h->root, XFER_CV, 0 };
  if (h->name_len) kv[n++] = (kv_t){ 6, 3, 0, h->name, h->name_len, 0 };
  if (h->compression) kv[n++] = (kv_t){ -1, 0, h->compression, NULL, 0, 0 };   // ai: -1 (0x20) after the unsigned keys
  return n;
}
static int transfer(const char *label, uint64_t len, int log2, const char *name) {
  uint8_t *file = malloc(len ? (size_t)len : 1), b3sum[XFER_CV];
  fill(file, (size_t)len);
  xfer_b3_hash(file, (size_t)len, b3sum);

  // Sender.
  xfer_header_t tx;
  CHECK(!xfer_header_init(&tx, len, log2), "%s: init", label);
  uint32_t source_blocks;
  uint8_t *cvs = chunk_cvs(file, len, log2, &source_blocks);
  CHECK(cvs && source_blocks == tx.source_blocks, "%s: chunk count", label);
  if (source_blocks >= 2) CHECK(!xfer_root(cvs, source_blocks, tx.root), "%s: root", label);
  else memcpy(tx.root, b3sum, XFER_CV);
  CHECK(!memcmp(tx.root, b3sum, XFER_CV), "%s: the root through the chunks is the file's b3sum", label);
  tx.name_len = (uint8_t)strlen(name); memcpy(tx.name, name, tx.name_len);
  const xfer_toi_t toi = xfer_toi_of(tx.root, tx.length, tx.sb_log2);
  uint8_t e[XFER_DESC_LONGEST];
  const int n = xfer_desc_write(&tx, e);
  {
    // ai: the senders' form: the test's builder writes the same bytes
    kv_t kv[8];
    uint8_t want[XFER_DESC_LONGEST];
    const size_t wn = tb_map(want, kv, tb_pairs(kv, &tx, toi));
    CHECK(n > 0 && (size_t)n == wn && !memcmp(e, want, wn), "%s: the entry's %d bytes, the deterministic form", label, n);
  }

  // Receiver.
  xfer_header_t rx;
  xfer_toi_t t = 0;
  CHECK(!xfer_desc_read(e, (size_t)n, 0, &rx, &t) && t == toi, "%s: the description read, its TOI derived", label);
  CHECK(rx.hash == tx.hash && rx.sb_log2 == tx.sb_log2 && rx.length == tx.length && rx.source_blocks == tx.source_blocks &&
        !memcmp(rx.root, tx.root, XFER_CV) && rx.name_len == tx.name_len &&
        !memcmp(rx.name, tx.name, tx.name_len) && !rx.compression && !rx.critical, "%s: the entry read back differs", label);
  {
    // ai: the same file under another name or compression is the same file; another root, length or source block size is not
    xfer_header_t o = rx;
    o.name_len = 3; memcpy(o.name, "zzz", 3); o.compression = XFER_COMP_ZSTD;
    CHECK(xfer_same_file(&rx, &o), "%s: renamed, the same file", label);
    o = rx; o.root[31] ^= 1; CHECK(!xfer_same_file(&rx, &o), "%s: another root, another file", label);
    o = rx; o.length++; CHECK(!xfer_same_file(&rx, &o), "%s: another length, another file", label);
    o = rx; o.sb_log2++; CHECK(!xfer_same_file(&rx, &o), "%s: another chunk size, another file", label);
  }
  // ai: each source block as a receiver decodes it: one of one checked against the root at once; of two or more its
  // ai: chaining value made (the sender's), the values combined into the root at the last
  uint8_t *got = calloc(rx.source_blocks ? rx.source_blocks : 1, XFER_CV);
  for (uint32_t i = 0; i < rx.source_blocks; i++)
    CHECK(!xfer_sb_check(&rx, i, file + ((uint64_t)i << log2), xfer_sb_len(&rx, i), rx.source_blocks >= 2 ? got + (size_t)i * XFER_CV : NULL),
          "%s: source block %u", label, i);
  if (rx.source_blocks >= 2) {
    CHECK(!memcmp(got, cvs, (size_t)source_blocks * XFER_CV), "%s: the chaining values the sender's", label);
    CHECK(!xfer_root_check(got, rx.source_blocks, rx.root), "%s: the values make the root", label);
    CHECK(xfer_sb_check(&rx, 0, file, xfer_sb_len(&rx, 0), NULL), "%s: no room for the chaining value refused", label);
  }

  // A corrupted source block: one bit of the middle one. A file of one source block refuses it at once; of two or more
  // its chaining value is another and the root not made (the file refused whole).
  if (rx.source_blocks) {
    const uint32_t bad = rx.source_blocks / 2;
    uint8_t *p = file + ((uint64_t)bad << log2) + xfer_sb_len(&rx, bad) / 2, cv[XFER_CV];
    *p ^= 0x10;
    if (rx.source_blocks == 1) CHECK(xfer_sb_check(&rx, bad, file, xfer_sb_len(&rx, bad), NULL), "%s: a corrupt source block of one passed", label);
    else {
      CHECK(!xfer_sb_check(&rx, bad, file + ((uint64_t)bad << log2), xfer_sb_len(&rx, bad), cv) && memcmp(cv, got + (size_t)bad * XFER_CV, XFER_CV),
            "%s: a corrupt source block %u's chaining value the same", label, bad);
      memcpy(got + (size_t)bad * XFER_CV, cv, XFER_CV);
      CHECK(xfer_root_check(got, rx.source_blocks, rx.root), "%s: a corrupt source block passed at the root", label);
      memcpy(got + (size_t)bad * XFER_CV, cvs + (size_t)bad * XFER_CV, XFER_CV);
    }
    *p ^= 0x10;
    CHECK(xfer_sb_check(&rx, bad, file + ((uint64_t)bad << log2), xfer_sb_len(&rx, bad) - 1, cv), "%s: a short source block passed", label);
    CHECK(xfer_sb_check(&rx, rx.source_blocks, file, xfer_sb_len(&rx, bad), cv), "%s: a source block past the file passed", label);
  }
  // A wrong chaining value (the last's) and another transfer's root: refused; the values put right, taken.
  if (rx.source_blocks >= 2) {
    const uint32_t hit = rx.source_blocks - 1;
    got[(size_t)hit * XFER_CV] ^= 1;
    CHECK(xfer_root_check(got, rx.source_blocks, rx.root), "%s: a wrong chaining value passed at the root", label);
    got[(size_t)hit * XFER_CV] ^= 1;
    CHECK(!xfer_root_check(got, rx.source_blocks, rx.root), "%s: put right", label);
    uint8_t other[XFER_CV];
    memcpy(other, rx.root, XFER_CV);
    other[3] ^= 0x80;
    CHECK(xfer_root_check(got, rx.source_blocks, other), "%s: another transfer's root passed", label);
  }
  // An entry that is not one, or one read as written though not as a sender writes it: built by the test's own encoder.
  {
    kv_t kv[10], v[10];
    uint8_t b[XFER_DESC_LONGEST + 64];
    const int np = tb_pairs(kv, &tx, toi);
    // ai: pick the value of key k in v (the copy), or -1
    #define AT(k) ({ int _i = -1; for (int _j = 0; _j < np; _j++) if (v[_j].key == (k)) _i = _j; _i; })
    #define TRY(cond, what) do { const size_t bn = tb_map(b, v, nv); const int got_ = xfer_desc_read(b, bn, 0, &rx, NULL) ? -1 : (int)bn; CHECK(cond, "%s: %s", label, what); } while (0)
    int nv;
    memcpy(v, kv, sizeof kv); nv = np; v[AT(3)].v = 25; TRY(got_ < 0, "32 MiB source blocks refused");
    memcpy(v, kv, sizeof kv); nv = np; v[AT(3)].v = 9; TRY(got_ < 0, "512-byte source blocks refused");
    memcpy(v, kv, sizeof kv); nv = np; v[AT(4)].v = 1ull << 40; TRY(got_ < 0, "a length of 2^40 refused");
    memcpy(v, kv, sizeof kv); nv = np; v[AT(5)].sl = 31; TRY(got_ < 0, "a root of 31 bytes refused");
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ 1, 0, toi, NULL, 0, 0 }; TRY(got_ == (int)bn && rx.critical, "key 1 (the TOI until 2026-11-22): unknown, critical");
    memcpy(v, kv, sizeof kv); nv = np; v[AT(3)].wide = 1; v[AT(4)].wide = 1; TRY(got_ == (int)bn && xfer_same_file(&rx, &tx), "eight-byte heads read the same");
    // ai: the key rule (RFC 9581's): an unknown negative key elective, skipped; an unknown unsigned one critical
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ -5, 0, 7, NULL, 0, 0 }; TRY(got_ == (int)bn && xfer_same_file(&rx, &tx) && !rx.critical, "an unknown elective key skipped");
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ -1, 0, 7, NULL, 0, 0 }; TRY(got_ == (int)bn && !rx.critical && rx.compression == 7, "a compression this build does not know (7): elective, read as it is, the file presented");
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ 9, 0, 7, NULL, 0, 0 }; TRY(got_ == (int)bn && rx.critical, "an unknown critical key: not to be presented");
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ 2, 0, 1, NULL, 0, 0 }; TRY(got_ == (int)bn && rx.critical, "key 2 (the compression, critical, earlier on 2026-11-22): unknown, critical");
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ 7, 3, 0, (const uint8_t *)"text/plain", 10, 0 }; TRY(got_ == (int)bn && rx.critical, "key 7 (the media type until 2026-11-22): unknown, critical");
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ -1, 0, 1, NULL, 0, 0 }; TRY(got_ == (int)bn && rx.compression == XFER_COMP_ZSTD && !rx.critical, "the zstd compression read");
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ -1, 0, 300, NULL, 0, 0 }; TRY(got_ == (int)bn && rx.compression == 255 && !rx.critical, "a compression past a byte read as 255, unknown");
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ -1, 3, 0, (const uint8_t *)"zstd", 4, 0 }; TRY(got_ < 0, "a compression not an unsigned integer refused (the schema's cut)");
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ 3, 0, tx.sb_log2, NULL, 0, 0 }; TRY(got_ < 0, "a key twice refused");
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ -9, -1, 0, (const uint8_t *)"\xc1\x01", 2, 0 }; TRY(got_ < 0, "a tag refused");
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ -9, -1, 0, (const uint8_t *)"\xf9\x3c\x00", 3, 0 }; TRY(got_ < 0, "a float refused");
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ -9, -1, 0, (const uint8_t *)"\x5f\x41\x00\xff", 4, 0 }; TRY(got_ < 0, "an indefinite length refused");
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ -9, -1, 0, (const uint8_t *)"\xf5", 1, 0 }; TRY(got_ == (int)bn && !rx.critical, "true, an unknown key's value, skipped");
    memcpy(v, kv, sizeof kv); nv = np; v[nv++] = (kv_t){ -9, -1, 0, (const uint8_t *)"\x82\xa1\x01\x02\x43xyz", 8, 0 }; TRY(got_ == (int)bn && !rx.critical, "an array holding a map, an unknown key's value, skipped");
    // ai: the pairs reversed: read the same (any order)
    for (int i = 0; i < np; i++) v[i] = kv[np - 1 - i];
    nv = np; TRY(got_ == (int)bn && xfer_same_file(&rx, &tx), "its keys in any order read the same");
    // ai: a required key gone: the root
    memcpy(v, kv, sizeof kv); nv = np; for (int i = AT(5); i < nv - 1; i++) v[i] = v[i + 1]; nv--; TRY(got_ < 0, "no root refused");
    #undef AT
    #undef TRY
    CHECK(!xfer_desc_read(e, (size_t)n, 0, &rx, NULL), "%s: the description as written still read", label);
    CHECK(xfer_desc_read(e, (size_t)n - 1, 0, &rx, NULL) < 0 && xfer_desc_read(e, (size_t)n + 1, 0, &rx, NULL) < 0, "%s: cut short or a byte more refused", label);
  }

  printf("%-9s %10llu B, source blocks of 2^%d: %u, root %02x%02x%02x%02x... = b3sum\n", label, (unsigned long long)len, log2,
         source_blocks, b3sum[0], b3sum[1], b3sum[2], b3sum[3]);
  free(file); free(cvs); free(got);
  return (int)source_blocks;
}

// The session header (SPEC 7.7): the description, one CBOR map from byte 0, the rest free.
static void session(void) {
  uint8_t p[XFER_PAYLOAD], save[XFER_PAYLOAD], desc[XFER_DESC_MAX + 1];
  xfer_session_t r;
  xfer_header_t h;
  CHECK(!xfer_header_init(&h, 5000, 24), "session: init");
  fill(h.root, XFER_CV);
  const int dn = xfer_desc_write(&h, desc);
  CHECK(dn > 0 && desc[0] == 0xa3, "session: a description of %d bytes, the file's map {3, 4, 5}", dn);
  CHECK(!xfer_session_write(p, desc, (size_t)dn) && !xfer_session_parse(p, &r) && r.at == 0 && r.len == (size_t)dn && !memcmp(p, desc, (size_t)dn) && !p[dn],
        "session: written and read back, zeros after");
  memcpy(save, p, XFER_PAYLOAD);
  p[XFER_PAYLOAD - 1] = 1;
  CHECK(!xfer_session_parse(p, &r) && r.len == (size_t)dn, "session: the bytes past the description ignored");
  memcpy(p, save, XFER_PAYLOAD);
  p[0] = 0x01; CHECK(xfer_session_parse(p, &r) == XFER_ERR, "session: not a map at its start refused"); memcpy(p, save, XFER_PAYLOAD);
  memset(p, 0, XFER_PAYLOAD); CHECK(xfer_session_parse(p, &r) == XFER_ERR, "session: an empty block refused");
  memset(p, 0, XFER_PAYLOAD); p[0] = 0xa1; p[1] = 0x09; p[2] = 0x59; p[3] = 0x01; p[4] = 0xf4;
  CHECK(xfer_session_parse(p, &r) == XFER_ERR, "session: a map running past the block refused");
  fill(desc, sizeof desc);
  CHECK(xfer_session_write(p, desc, XFER_DESC_MAX + 1), "session: 461 bytes not written");
}

// The description (SPEC 7.11): the send's one file's map, written and read back, and every refusal.
static int desc_of(uint8_t *out, uint64_t len, const char *name, int comp, xfer_header_t *h) {
  CHECK(!xfer_header_init(h, len, xfer_sb_log2_for(len)), "description: init %s", name);
  fill(h->root, XFER_CV);
  h->name_len = (uint8_t)strlen(name); memcpy(h->name, name, h->name_len);
  h->compression = (uint8_t)comp;
  const int n = xfer_desc_write(h, out);
  CHECK(n > 0, "description: %s's %d bytes", name, n);
  return n;
}
static void description(void) {
  static uint8_t d[4096], bad[4096];
  xfer_header_t h[4], got;
  xfer_toi_t t = 0;
  const struct { uint64_t len; const char *name; int comp; } fs[] = { { 10240, "server-2026-11-19.log", 0 },
                                                                       { 5u << 20, "IMG_2041.jpg.zst", XFER_COMP_ZSTD },
                                                                       { (40ull << 20) + 7, "", 0 } };
  for (int i = 0; i < 3; i++) {
    const int n = desc_of(d, fs[i].len, fs[i].name, fs[i].comp, &h[i]);
    CHECK(!xfer_desc_read(d, (size_t)n, 0x123456, &got, &t) && xfer_same_file(&got, &h[i]) && t == xfer_toi_of(h[i].root, h[i].length, h[i].sb_log2) &&
          got.name_len == h[i].name_len && !memcmp(got.name, h[i].name, got.name_len) && got.compression == h[i].compression && !got.critical,
          "description %d read back, its name, compression and derived TOI", i);
    CHECK(xfer_desc_read(d, (size_t)n, t, &got, NULL) < 0, "description %d under its own file's TOI as the session's refused", i);
  }
  // ai: refusals: empty, not a map, a key twice
  CHECK(xfer_desc_read(d, 0, 0, &got, NULL) < 0 && xfer_desc_read((const uint8_t *)"\x83\x01\x02\x03", 4, 0, &got, NULL) < 0,
        "description: empty, not a map: refused");
  {
    const int n = desc_of(d, 10240, "a", 0, &h[3]);
    memcpy(bad, d, (size_t)n); bad[0]++; bad[n] = 0x03; bad[n + 1] = 0x0a;   // ai: k twice
    CHECK(xfer_desc_read(bad, (size_t)n + 2, 0, &got, NULL) < 0, "description: a key twice refused");
    // ai: an unknown critical key (2, 7, 1, 11) flagged; an unknown elective one (-11) skipped; the old files' array (key 1) critical
    const uint8_t keys[] = { 0x02, 0x07, 0x01, 0x0b };
    for (int i = 0; i < 4; i++) {
      memcpy(bad, d, (size_t)n); bad[0]++; bad[n] = keys[i]; bad[n + 1] = 0xf5;
      CHECK(!xfer_desc_read(bad, (size_t)n + 2, 0, &got, NULL) && got.critical, "description: an unknown critical key (%d): flagged", keys[i]);
    }
    memcpy(bad, d, (size_t)n); bad[0]++; bad[n] = 0x2a; bad[n + 1] = 0xf5;
    CHECK(!xfer_desc_read(bad, (size_t)n + 2, 0, &got, NULL) && !got.critical, "description: an unknown elective key (-11): skipped");
  }
  // ai: a name of 256 bytes: refused
  {
    kv_t kv[8];
    char lon[300];
    memset(lon, 'x', sizeof lon);
    const int np = tb_pairs(kv, &h[0], 0);
    for (int i = 0; i < np; i++) if (kv[i].key == XFER_K_NAME) kv[i].sl = 256, kv[i].s = (const uint8_t *)lon;
    const size_t en = tb_map(bad, kv, np);
    CHECK(xfer_desc_read(bad, en, 0, &got, NULL) < 0, "description: a name too long refused");
  }
  // ai: the longest description: a name of 255 bytes, the compression, every key at its widest; it fits the session header
  {
    char nm[256];
    uint8_t p[XFER_PAYLOAD];
    memset(nm, 'n', 255); nm[255] = 0;
    const int dl = desc_of(bad, (1ull << 39) + 1, nm, XFER_COMP_ZSTD, &h[3]);
    CHECK(dl == XFER_DESC_LONGEST && !xfer_session_write(p, bad, (size_t)dl), "description: the longest, %d bytes, fits the session header", dl);
    h[3].compression = 2; CHECK(xfer_desc_write(&h[3], bad) < 0, "description: a compression this build does not know not written");
  }
  // ai: a name past 255 bytes cut in its stem, its extension kept (xfer_name_cut)
  {
    uint8_t in[400], out[XFER_NAME_MAX];
    memset(in, 'n', sizeof in);
    CHECK(xfer_name_cut(in, 255, out) == 255 && !memcmp(out, in, 255), "name: 255 bytes kept whole");
    memcpy(in + 296, ".txt", 4);
    CHECK(xfer_name_cut(in, 300, out) == 255 && !memcmp(out, in, 251) && !memcmp(out + 251, ".txt", 4), "name: 300 bytes, its stem cut, .txt kept");
    memset(in, 'n', sizeof in);
    CHECK(xfer_name_cut(in, 300, out) == 255 && !memcmp(out, in, 255), "name: no extension, its end cut");
    in[250] = '.';   // ai: a dot 50 bytes from the end: past XFER_EXT_MAX, no extension
    CHECK(xfer_name_cut(in, 300, out) == 255 && !memcmp(out, in, 255), "name: a dot past 32 bytes from the end, its end cut");
    memset(in, 'n', sizeof in); in[0] = '.';   // ai: a leading dot alone: no extension
    CHECK(xfer_name_cut(in, 300, out) == 255 && !memcmp(out, in, 255), "name: a leading dot, its end cut");
    memset(in, 'n', sizeof in); memcpy(in + 292, ".tar.zst", 8);
    CHECK(xfer_name_cut(in, 300, out) == 255 && !memcmp(out, in, 247) && !memcmp(out + 247, ".tar.zst", 8), "name: 300 bytes, .tar.zst kept whole");
    memset(in, 'n', sizeof in); in[260] = '.'; memcpy(in + 296, ".zst", 4);   // ai: the dot before 40 bytes from the end
    CHECK(xfer_name_cut(in, 300, out) == 255 && !memcmp(out, in, 251) && !memcmp(out + 251, ".zst", 4),
          "name: the extension before past 32 bytes with the last, the last alone kept");
    for (int k = 0; k < 100; k++) memcpy(in + 3 * k, "\xe2\x82\xac", 3);   // ai: 100 euro signs, 300 bytes
    memcpy(in + 300, ".jpg", 4);
    const size_t n = xfer_name_cut(in, 304, out);
    CHECK(n == 253 && !memcmp(out, in, 249) && !memcmp(out + 249, ".jpg", 4), "name: a UTF-8 stem cut at a character (%zu bytes), .jpg kept", n);
  }
}

int main(int argc, char **argv) {
  vectors(argc > 1 ? argv[1] : "vendor/blake3/test_vectors/test_vectors.json");
  boundaries();
  ids();
  CHECK(transfer("3 sbs", (2u << 20) + 304000, 20, "IMG_2041.jpg") == 3, "3 source blocks");
  CHECK(transfer("1 sb", 700001, 20, "notes.txt") == 1, "1 source block");
  // A last source block of one block (a source block of K = 1); a power of two of source blocks with the last one full;
  // a file of one block; an empty file.
  CHECK(transfer("41 sbs", 40 * 1024 + 100, 10, "") == 41, "41 source blocks");
  CHECK(transfer("16 sbs", 16u << 16, 16, "a") == 16, "16 source blocks");
  CHECK(transfer("1 block", XFER_PAYLOAD, 10, "tiny") == 1, "one block");
  CHECK(transfer("empty", 0, 10, "empty") == 0, "empty");
  session();
  description();
  printf("%d checks, %d failed\n", checks, fails);
  return fails != 0;
}
