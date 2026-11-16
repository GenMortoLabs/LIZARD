// The transfer in the light: the block id, the header block and the manifest blocks, so a camera alone can recover a
// file. Before this the fountain's header went from the sender page to the receiver page over the rig's network, and
// the block id was a convention of sim/phy.mjs.
//
// A file is cut into chunks of 2^k bytes (k 10 to 24; the last one shorter), each its own RaptorQ source block and each
// verified on arrival against the file's one BLAKE3 root, the way Bao and iroh stream: a chunk of 2^k bytes aligned
// at its own size is a whole subtree of BLAKE3's tree, so its chaining value comes from the chunk alone, and the
// chaining values of all the chunks combine to the root (2026-09-24).
//
// ---- The block ----
// Every LIZARD block is 476 bytes (src/lizard_phy.c block_bytes, the same in every format and at every rate; with its CRC-32C
// the 3840 data bits of the 3/4, 2/3 and 1/2 codes): the id, 40 bits little-endian in bytes 0 to 4, then XFER_PAYLOAD =
// 471 bytes of payload.
//
//   id bits 39..24  chunk   0 .. 65534 a chunk of the file, 65535 (XFER_CONTROL) a control block
//   id bits 23..0   symbol  a data block: the encoding symbol's ESI within its chunk's source block (src/rq.h)
//                           a control block: 0 the header, 1 + m manifest block m (m < XFER_MAX_MANIFEST, 4682), the rest reserved
//
// Why 24 bits of symbol: RaptorQ's whole ESI (RFC 6330 3.2). A sender that runs out of ids wraps and repeats ones a
// receiver may already hold; 2^24 is 471 times the source symbols of a 16 MiB chunk, so a receiver has to miss 99.8% of
// a chunk's blocks before the sender repeats one.
// Why 16 bits of chunk: 65,535 chunks, about 1 TiB at 16 MiB a chunk.
// Why the control range is at the top: a data block's id is then its chunk and symbol as they are, chunk 0's
// running 0 .. 2^24 - 1 from zero, and no data id has to step round a control one.
// The header's id is the same for every transfer and a manifest block's for every transfer with that many chunks, so
// a receiver that drops a block whose id it has seen misses the next file: a control block is new when its bytes are.
//
// ---- The bytes it carries ----
// A chunk's source block is the chunk's own bytes: the transfer compresses nothing. A client that compresses (the
// default client compresses a file with zstd before it enters the transfer) hands the transfer the compressed file,
// and the length, root and chaining values are of the bytes handed over. So every chunk but the last is 2^chunk_log2
// bytes and every chunk's source symbols follow from the file's length: a receiver builds any chunk's decoder from the
// header. The manifest carries the chaining values alone.
//
// ---- The header block (id XFER_ID_HEADER = 0xffff000000), 471 bytes, little-endian ----
//     0    1  hash         XFER_HASH_BLAKE3 = 1; 0 and 2..255 refused (-3)
//     1    1  chunk_log2   the chunk size is 1 << chunk_log2 bytes, 10 .. 24
//     2    1  name_len     bytes of name, 0 .. 255
//     3    1  type_len     bytes of media type, 0 .. 158
//     4    8  length       the file's bytes, u64
//    12    4  chunks       ceil(length / chunk size), u32, at most 65,535; 0 for an empty file
//    16   32  root         BLAKE3 of the whole file, what b3sum prints
//    48  255  name         the file name, UTF-8, zero padded; empty for none
//   303  158  type         the media type, ASCII, zero padded; empty for none
//   461   10  reserved, 0
// Every byte past a length is zero and the parse insists, so a block that is not a header rarely passes for one.
// The fountain: RaptorQ (RFC 6330), a chunk one source block of K = ceil(len / 471) source symbols of 471 bytes
// (T = XFER_PAYLOAD; one sub-block, Z = N = 1), the last zero padded; symbol s of a chunk is its ESI s, so symbols
// 0 .. K - 1 are the chunk's own bytes and the rest repair (src/rq.h). A chunk of one block is a block of K = 1 like
// any other.
//
// ---- The manifest blocks (id XFER_ID_MANIFEST + m, m = 0 .. xfer_manifest_blocks(chunks) - 1) ----
//     0    8  tag          the root's first 8 bytes, so a block of another transfer is refused before it is used
//     8  448  entries      the chaining values of chunks 14 m .. 14 m + 13, 32 bytes each, in order; zero past the last
//   456   15  reserved, 0
// A file of two chunks or more has ceil(chunks / 14) of them; a file of one chunk or none has none, since its root is
// the one chunk's own hash and is checked from the chunk.
//
// ---- The chunk size a sender picks (xfer_chunk_log2_for) ----
// 16 MiB (XFER_LOG2_DEFAULT, the largest power of two under RaptorQ's K'_max of 56,403 symbols: 35,621 of 471 bytes),
// for a file of up to 65,535 chunks (about 1 TiB); a file of 16 MiB or less is one chunk of its own length. A file waits on its slowest chunk, and the
// fewer and larger the chunks the less a slow chunk costs: the schedule interleaves every chunk, so a chunk x symbols
// late holds the file back by x / K of it, whatever the chunk count. RaptorQ solves at K, K + 1 or K + 2 (none past in
// 110,000 solves at 4, 8 and 16 MiB with half or 95% kept, the raptorq crate; src/rq.c the same symbols), so the size
// is set by RaptorQ's K'_max and the decoder's memory (about 36 MB at a 16 MiB solve).
//
// ---- The tree ----
// The root of chunks c[0..n), n >= 2: split at p, the largest power of two below n; left the root of c[0..p), right
// that of c[p..n); a parent node is BLAKE3's compression of left || right under the key with the PARENT flag, and the
// top one also has ROOT. A single chaining value is its own subtree. This is BLAKE3's own tree: every chunk but the
// last is 2^k bytes on a 2^k boundary, so BLAKE3's split (the largest power of two in 1 KiB chunks that leaves a byte
// for the right) always falls on a chunk boundary. test/xfer_test.c proves it on the official test vectors.
//
// ---- In the wasm (src/wasm.c) ----
// Exported as declared below: xfer_manifest_blocks, xfer_manifest_write, xfer_manifest_check, xfer_root,
// xfer_chunk_cv, xfer_b3_hash. Wrapped for JS, with flat arguments: xfer_layout, xfer_id_of, xfer_kind_of,
// xfer_hdr_write, xfer_hdr_parse, xfer_mf_parse, xfer_chunk_ok, xfer_b3_sub, xfer_log2_for. test/xfer_wasm.mjs drives
// every one.
#ifndef XFER_H
#define XFER_H
#include <stddef.h>
#include <stdint.h>

enum {
  XFER_ID_BYTES = 5, XFER_PAYLOAD = 471, XFER_BLOCK = XFER_ID_BYTES + XFER_PAYLOAD,
  XFER_SYMBOL_BITS = 24, XFER_CHUNK_BITS = 16,
  XFER_CONTROL = (1 << XFER_CHUNK_BITS) - 1, XFER_MAX_CHUNKS = XFER_CONTROL,
  XFER_HASH_BLAKE3 = 1,
  XFER_LOG2_MIN = 10, XFER_LOG2_MAX = 24, XFER_LOG2_DEFAULT = 24,
  XFER_CV = 32, XFER_ENTRY = XFER_CV,
  XFER_PER_BLOCK = 14,
  XFER_MAX_MANIFEST = (XFER_MAX_CHUNKS + XFER_PER_BLOCK - 1) / XFER_PER_BLOCK,
  XFER_NAME_MAX = 255, XFER_TYPE_MAX = 158,
  // header offsets
  XFER_H_HASH = 0, XFER_H_LOG2 = 1, XFER_H_NAME_LEN = 2, XFER_H_TYPE_LEN = 3, XFER_H_LENGTH = 4, XFER_H_CHUNKS = 12,
  XFER_H_ROOT = 16, XFER_H_NAME = 48, XFER_H_TYPE = XFER_H_NAME + XFER_NAME_MAX, XFER_H_END = XFER_H_TYPE + XFER_TYPE_MAX,
  // manifest offsets
  XFER_M_TAG = 0, XFER_M_TAG_BYTES = 8, XFER_M_ENTRIES = 8,
  // what the parses return besides 0
  XFER_ERR = -1, XFER_ERR_HASH = -3,
};
// A block's id: 40 bits in a uint64_t.
typedef uint64_t xfer_id_t;
#define XFER_ID_HEADER ((xfer_id_t)XFER_CONTROL << XFER_SYMBOL_BITS)
#define XFER_ID_MANIFEST (XFER_ID_HEADER + 1u)
#define XFER_SYMBOL_MASK ((1u << XFER_SYMBOL_BITS) - 1u)

// XFER_KIND_RESERVED: a control id past the manifest's, which a receiver ignores.
enum { XFER_KIND_RESERVED = -1, XFER_KIND_DATA = 0, XFER_KIND_HEADER = 1, XFER_KIND_MANIFEST = 2 };
static inline xfer_id_t xfer_id(uint32_t chunk, uint32_t symbol) { return (xfer_id_t)chunk << XFER_SYMBOL_BITS | (symbol & XFER_SYMBOL_MASK); }
static inline uint32_t xfer_id_chunk(xfer_id_t id) { return (uint32_t)(id >> XFER_SYMBOL_BITS); }
static inline uint32_t xfer_id_symbol(xfer_id_t id) { return (uint32_t)id & XFER_SYMBOL_MASK; }
static inline int xfer_id_kind(xfer_id_t id) {
  const uint32_t s = xfer_id_symbol(id);
  return xfer_id_chunk(id) != XFER_CONTROL ? XFER_KIND_DATA : !s ? XFER_KIND_HEADER : s <= XFER_MAX_MANIFEST ? XFER_KIND_MANIFEST : XFER_KIND_RESERVED;
}
static inline xfer_id_t xfer_id_get(const uint8_t *block) {
  xfer_id_t id = 0;
  for (int i = XFER_ID_BYTES - 1; i >= 0; i--) id = id << 8 | block[i];
  return id;
}
static inline void xfer_id_put(uint8_t *block, xfer_id_t id) { for (int i = 0; i < XFER_ID_BYTES; i++) block[i] = (uint8_t)(id >> 8 * i); }

typedef struct {
  uint8_t hash, chunk_log2, name_len, type_len;
  uint64_t length;
  uint32_t chunks;
  uint8_t root[XFER_CV];
  uint8_t name[XFER_NAME_MAX], type[XFER_TYPE_MAX];
} xfer_header_t;

// ceil(length / 2^chunk_log2), or 0 where that is past XFER_MAX_CHUNKS or chunk_log2 is out of range (an empty file
// is 0 chunks too: the caller tells them apart by length).
uint32_t xfer_chunk_count(uint64_t length, int chunk_log2);
// The chunk size's log2 a sender picks for a file of `length` bytes (above), or -1 past 65,535 chunks of 16 MiB.
int xfer_chunk_log2_for(uint64_t length);
// Bytes of chunk `index` under h, 0 past the last.
size_t xfer_chunk_len(const xfer_header_t *h, uint32_t index);
// The source symbols of `len` bytes, ceil(len / 471): a chunk's K.
uint32_t xfer_blocks(size_t len);
int xfer_manifest_blocks(uint32_t chunks);

// Fills a header for a file of `length` bytes: hash, chunk count; everything else is the caller's (the root, name
// and type). 0, or -1 for a length or chunk size the layout cannot carry.
int xfer_header_init(xfer_header_t *h, uint64_t length, int chunk_log2);
// The 471-byte payload from h, or -1 where h is inconsistent (the same checks the parse makes).
int xfer_header_write(const xfer_header_t *h, uint8_t *payload);
// 0, or XFER_ERR / XFER_ERR_HASH.
int xfer_header_parse(const uint8_t *payload, xfer_header_t *h);
// 1 where a and b describe one file's blocks: the same hash, chunk size, length, chunks and root (header bytes 0, 1
// and 4 to 47). Its name and media type are not the file's: another sender's header for the same
// file is the same transfer, and every block of the file combines whichever sender sent it. 0 otherwise.
int xfer_same_file(const xfer_header_t *a, const xfer_header_t *b);

// Manifest block m of a file of `chunks` chunks whose root is `root`: the chaining values cvs (chunks * 32 bytes).
int xfer_manifest_write(const uint8_t *cvs, uint32_t chunks, const uint8_t root[XFER_CV], uint32_t m, uint8_t *payload);
// Its entries into cvs at 14 m: 0, or -1 for a block of another transfer (tag), past the list, or with nonzero
// padding. Nothing is written on -1.
int xfer_manifest_parse(const uint8_t *payload, const xfer_header_t *h, uint32_t m, uint8_t *cvs);
// The root the chaining values of chunks >= 2 chunks combine to (the tree above). -1 for fewer than 2.
int xfer_root(const uint8_t *cvs, uint32_t chunks, uint8_t root[XFER_CV]);
// 0 where the list combines to root.
int xfer_manifest_check(const uint8_t *cvs, uint32_t chunks, const uint8_t root[XFER_CV]);

// The chaining value of chunk `index` of 2^chunk_log2 bytes, len being its bytes (the chunk size, or less for the
// last). Not the root: a file of one chunk is checked with xfer_b3_hash.
int xfer_chunk_cv(const uint8_t *data, size_t len, uint32_t index, int chunk_log2, uint8_t cv[XFER_CV]);
// 0 where chunk `index` is right: len is what h says it is, and its chaining value is cvs[index] (a list checked
// against the root first), or with one chunk its hash is the root (cvs unused, may be NULL).
int xfer_chunk_check(const xfer_header_t *h, uint32_t index, const uint8_t *data, size_t len, const uint8_t *cvs);

// BLAKE3 of a buffer, 32 bytes, through the official hasher (vendor/blake3).
void xfer_b3_hash(const uint8_t *in, size_t len, uint8_t out[XFER_CV]);
// The chaining value of the subtree BLAKE3 would make of `len` bytes starting at 1 KiB chunk `counter` (byte offset
// 1024 counter), or with root = 1 the hash, the range then being the whole input (counter ignored). Right only for a
// range that is a node of the input's tree: 2^j KiB on a 2^j KiB boundary, or all that is left of the input from
// such a boundary when that is less.
void xfer_b3_subtree(const uint8_t *in, size_t len, uint64_t counter, int root, uint8_t out[XFER_CV]);

// The test stream (SPEC 9.3): the payload of a test transfer's block id, the first len bytes of BLAKE3's extended output
// of the id as five little-endian bytes. A function of the id alone, so a receiver checks any block it decodes without
// being told which frame carried it; BLAKE3, the transfer's own hash, so the library carries one (SHAKE256 until
// 2026-11-15, a second family for no purpose: nothing here is cryptography).
void stream_fill(uint64_t id, uint8_t *out, uint32_t len);
// 1 where payload (len bytes) is id's stream, else 0: compared an output block (64 bytes, one compression) at a time
// as the stream is made, so a payload that is not the stream's (every block of a file) costs one compression.
int stream_matches(uint64_t id, const uint8_t *payload, uint32_t len);

#endif
