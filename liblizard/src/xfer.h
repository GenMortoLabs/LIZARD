// The transfer in the light: the block id and the session header (the send's description, its one file's map), so a
// camera alone can recover a file. Before this the fountain's header went from the sender page to the receiver page over the rig's network, and
// the block id was a convention of sim/phy.mjs.
//
// A file is cut into source blocks of 2^k bytes (k 10 to 24; the last one shorter), each its own RaptorQ source block,
// and the file verified against its one BLAKE3 root: a source block of 2^k bytes aligned at its own size is a whole
// subtree of BLAKE3's tree, so its chaining value comes from the source block alone, made as it decodes, in any order
// and while the rest arrive, and the chaining values of all the source blocks combine to the root (2026-09-24). Each
// block's CRC-64 is its check on the way in; the root the file's at the end (no manifest since 2026-11-21: the CRC
// leaves nothing for a per-source-block check to find, and with no back channel knowing which one failed buys nothing).
//
// ---- The block ----
// Every LIZARD block is 472 bytes (src/lizard_phy.c block_bytes, the same in every format and at every rate; with its CRC-64
// the 3840 data bits of the 3/4, 2/3 and 1/2 codes): the id, 96 bits in bytes 0 to 11, then XFER_PAYLOAD = 460 bytes of
// payload. Bytes 0 to 4 are the bare id below, 40 bits little-endian (xfer_id_get); bytes 5 to 10 the TOI, 48 bits
// little-endian (xfer_toi_get); byte 11 reserved, written 0 and ignored.
//
//   bare id bits 39..24  source block   0 .. 65534 a source block of the object, 65535 (XFER_CONTROL) a control block
//   bare id bits 23..0   symbol  a data block: the encoding symbol's ESI within its source block (src/rq.h)
//                                a control block: 0 the session header; the rest reserved
//   TOI                  the object the block is of (1 .. 2^48 - 1); 0 none (no block goes under it, a receiver ignores one)
//
// Why a TOI: every block says which object it belongs to, so a receiver that sees two senders at once (two screens in
// a camera's view, a large file sent as parts over several codes) files each block with its own, as an address does a
// Wi-Fi frame or a PID an MPEG transport packet.
// Why a file's TOI is its content's (xfer_toi_of): the first 48 bits of BLAKE3's key derivation over the file's root,
// length and source block size, so the same bytes in the same source blocks are the same object whoever sends them, in
// whichever session, and after a sender starts again: RaptorQ's symbols of a source block are a function of its bytes and
// size alone, so a receiver pools every sender's blocks of it. Two different objects share a TOI 1 time in 2^48 a pair.
// A session header's TOI is the session's own, drawn by its sender (the reference senders from the send's seed): two
// senders in one view share one 1 time in 2^48 a pair. The test stream's is drawn too.
// Why 24 bits of symbol: RaptorQ's whole ESI (RFC 6330 3.2). A sender that runs out of ids wraps and repeats ones a
// receiver may already hold; 2^24 is 460 times the source symbols of a 16 MiB source block, so a receiver has to miss 99.8% of
// a source block's blocks before the sender repeats one.
// Why 16 bits of source block: 65,535 source blocks, about 1 TiB at 16 MiB a source block; a larger file goes as parts, each its own
// object and TOI.
// Why the control range is at the top: a data block's id is then its source block and symbol as they are, source block 0's
// running 0 .. 2^24 - 1 from zero, and no data id has to step round a control one.
// The session header's id is the same under a TOI whatever it carries, so a receiver that drops a block whose id it has
// seen misses the next description under it: a control block is new when its bytes are.
//
// ---- The bytes it carries ----
// A source block is the file's own bytes: the transfer compresses nothing. A client that compresses (the
// default client compresses a file with zstd before it enters the transfer) hands the transfer the compressed file,
// and the length, root and chaining values are of the bytes handed over. So every source block but the last is 2^sb_log2
// bytes and every source block's source symbols follow from the file's length: a receiver builds any source block's decoder from the
// header.
//
// ---- The session header (bare id XFER_ID_HEADER = 0xffff000000 under the session's TOI), 460 bytes ----
// A send's one plain block, its bootstrap (2026-11-20 evening): the session's description, one CBOR map (RFC 8949) from
// byte 0, the rest of the block free (0, ignored); read from this one block. The map's keys are small integers, any
// order, each at most once: an unsigned key is critical (one a receiver does not know keeps the file from being
// presented), a negative one elective (one a receiver does not know it skips): RFC 9581's rule for its map keys
// (2026-11-21; the other way round before). The map is the send's one file's (2026-11-22; until then key 1 an array
// of file maps, the FDT, which a later FDT would bring back as a key of its own); session-wide fields to come (an
// address, a public key: none defined, the map left open) are keys beside the file's. Nothing is reserved: a field
// that comes later is a key of its own, unsigned where an older receiver must not ignore it, negative where it may
// (the extended session header and the session's instance, tabled 2026-11-20 with
// archive/extended-session-header-2026-11-20/, would come back so). The file's TOI derived (xfer_toi_of), never
// stated (key 1 a file map's until 2026-11-22, now unknown and so critical): XFER_K_LOG2 (3, k:
// source blocks of 2^k bytes), XFER_K_LENGTH (4, u40), XFER_K_ROOT (5, BLAKE3, a 32-byte string), XFER_K_NAME (6,
// UTF-8, to 255 bytes: xfer_name_cut), XFER_K_COMPRESSION (-1, elective: the bytes are a compressor's stream, the name
// its file's with the compressor's extension; XFER_COMP_ZSTD the default client's zstd frames, "<name>.zst"). A
// receiver that knows the value strips that extension and decompresses; one that does not (an unknown value, or a
// receiver without the key) keeps the file as it came, "<name>.zst", which any zstd reads (2026-11-22; key 2,
// critical, earlier that day and "encoding" from 2026-11-20). No media type (key 7 until 2026-11-22; a receiver takes
// a type from the name's extension where it needs one): unsigned keys 2 and 7 are unknown, so critical. The hash is BLAKE3:
// another would come as a key that must be understood (unsigned). The profile: unsigned and negative integers, byte and text
// strings, arrays, maps, false and true; definite lengths; no tags, floats or other simple values. Senders write RFC
// 8949's deterministic form (shortest heads, keys in ascending order of their bytes), so two senders' descriptions of one
// content are the same bytes; receivers take any order of the keys and refuse a key twice.
//
// ---- The session ----
// A sender's session is what it has on air: the one file its description describes, under its own TOI (its
// content's), and the session header under the session's TOI. A receiver files every block by its TOI alone; a file's
// TOI names its bytes, so any description of it describes the same file (xfer_same_file), and a receiver takes the
// first it holds.
// The fountain: RaptorQ (RFC 6330), each source block one RaptorQ source block of K = ceil(len / 460) source symbols of 460 bytes
// (T = XFER_PAYLOAD; one sub-block, Z = N = 1), the last zero padded; symbol s of a source block is its ESI s, so symbols
// 0 .. K - 1 are the source block's own bytes and the rest repair (src/rq.h). A source block of 460 bytes or less is a source
// block of K = 1 like any other.
//
// ---- The file's check (2026-11-21) ----
// A file of one source block: its BLAKE3 the root, checked as it decodes (xfer_sb_check). Of two or more: each source
// block's chaining value made as it decodes (xfer_sb_check), kept, and at the last the values combined and held to the
// root (xfer_root_check); a file that fails is refused whole and collected again (only a broken sender makes one). The
// manifest (the chaining values sent as a RaptorQ object under a TOI of its own, 2026-11-20; plain control blocks
// before) is gone.
//
// ---- The source block size a sender picks (xfer_sb_log2_for) ----
// 16 MiB (XFER_LOG2_DEFAULT, the largest power of two under RaptorQ's K'_max of 56,403 symbols: 36,473 of 460 bytes),
// for a file of up to 65,535 source blocks (about 1 TiB); a file of 16 MiB or less is one source block of its own length. A file waits on its slowest source block, and the
// fewer and larger the source blocks the less a slow source block costs: the schedule interleaves every source block, so a source block x symbols
// late holds the file back by x / K of it, whatever the source block count. RaptorQ solves at K, K + 1 or K + 2 (none past in
// 110,000 solves at 4, 8 and 16 MiB with half or 95% kept, the raptorq crate; src/rq.c the same symbols), so the size
// is set by RaptorQ's K'_max and the decoder's memory (about 36 MB at a 16 MiB solve).
//
// ---- The tree ----
// The root of source blocks c[0..n), n >= 2: split at p, the largest power of two below n; left the root of c[0..p), right
// that of c[p..n); a parent node is BLAKE3's compression of left || right under the key with the PARENT flag, and the
// top one also has ROOT. A single chaining value is its own subtree. This is BLAKE3's own tree: every source block but the
// last is 2^k bytes on a 2^k boundary, so BLAKE3's split (the largest power of two in 1 KiB chunks that leaves a byte
// for the right) always falls on a source block boundary. test/xfer_test.c proves it on the official test vectors.
//
// ---- In the wasm (src/wasm.c) ----
// Exported as declared below: xfer_root_check, xfer_root, xfer_sb_cv, xfer_b3_hash. Wrapped for JS, with flat
// arguments: xfer_layout, xfer_id_of, xfer_kind_of, xfer_session_write_of, xfer_session_read, xfer_desc_write_of, xfer_desc_parse, xfer_sb_ok, xfer_b3_sub,
// xfer_sb_log2_of, xfer_name_cut_of, xfer_toi_of_wasm, which test/xfer_wasm.mjs drives every one of; xfer_same_file_of,
// and the test stream's stream_fill_of and stream_matches_of.
#ifndef XFER_H
#define XFER_H
#include <stddef.h>
#include <stdint.h>

enum {
  XFER_ID_BYTES = 12, XFER_PAYLOAD = 460, XFER_BLOCK = XFER_ID_BYTES + XFER_PAYLOAD,
  XFER_ID_BARE_BYTES = 5, XFER_ID_TOI = 5, XFER_TOI_BYTES = 6, XFER_ID_SPARE = 11,   // the bare id's bytes, the TOI's first and count, the reserved byte
  XFER_ESI_BITS = 24, XFER_SBN_BITS = 16, XFER_TOI_BITS = 48,
  XFER_TOI_NONE = 0,
  XFER_CONTROL = (1 << XFER_SBN_BITS) - 1, XFER_MAX_SOURCE_BLOCKS = XFER_CONTROL,
  XFER_HASH_BLAKE3 = 1,
  XFER_LOG2_MIN = 10, XFER_LOG2_MAX = 24, XFER_LOG2_DEFAULT = 24,
  XFER_CV = 32,
  XFER_NAME_MAX = 255, XFER_EXT_MAX = 32,
  // the description (2026-11-20 evening, CBOR; the file's own map since 2026-11-22): its keys (above); the most it
  // takes, the whole payload; the longest a description is (every key at its widest, the one compression a byte)
  XFER_K_LOG2 = 3, XFER_K_LENGTH = 4, XFER_K_ROOT = 5, XFER_K_NAME = 6, XFER_K_COMPRESSION = -1,
  XFER_COMP_NONE = 0, XFER_COMP_ZSTD = 1,
  XFER_DESC_MAX = XFER_PAYLOAD,
  XFER_DESC_LONGEST = 1 + 3 + 10 + 35 + 3 + XFER_NAME_MAX + 2,
  // what the parses return besides 0
  XFER_ERR = -1,
};
// A block's bare id (bytes 0 to 4): its source block over its symbol; and its TOI (bytes 5 to 10), the object's.
typedef uint64_t xfer_id_t;
typedef uint64_t xfer_toi_t;
#define XFER_ID_HEADER ((xfer_id_t)XFER_CONTROL << XFER_ESI_BITS)
#define XFER_ESI_MASK ((1u << XFER_ESI_BITS) - 1u)
#define XFER_TOI_MASK (((xfer_toi_t)1 << XFER_TOI_BITS) - 1u)
// XFER_KIND_RESERVED: a control id other than the header's, which a receiver ignores.
enum { XFER_KIND_RESERVED = -1, XFER_KIND_DATA = 0, XFER_KIND_HEADER = 1 };
static inline xfer_id_t xfer_id(uint32_t sbn, uint32_t symbol) { return (xfer_id_t)sbn << XFER_ESI_BITS | (symbol & XFER_ESI_MASK); }
static inline uint32_t xfer_id_sbn(xfer_id_t id) { return (uint32_t)(id >> XFER_ESI_BITS) & XFER_CONTROL; }
static inline uint32_t xfer_id_esi(xfer_id_t id) { return (uint32_t)id & XFER_ESI_MASK; }
static inline int xfer_id_kind(xfer_id_t id) {
  const uint32_t s = xfer_id_esi(id);
  return xfer_id_sbn(id) != XFER_CONTROL ? XFER_KIND_DATA : !s ? XFER_KIND_HEADER : XFER_KIND_RESERVED;
}
// The block's bare id (bytes 0 to 4) and its TOI (bytes 5 to 10, written with the reserved byte 11 as 0).
static inline xfer_id_t xfer_id_get(const uint8_t *block) {
  xfer_id_t id = 0;
  for (int i = XFER_ID_BARE_BYTES - 1; i >= 0; i--) id = id << 8 | block[i];
  return id;
}
static inline void xfer_id_put(uint8_t *block, xfer_id_t id) { for (int i = 0; i < XFER_ID_BARE_BYTES; i++) block[i] = (uint8_t)(id >> 8 * i); }
static inline xfer_toi_t xfer_toi_get(const uint8_t *block) {
  xfer_toi_t t = 0;
  for (int i = XFER_TOI_BYTES - 1; i >= 0; i--) t = t << 8 | block[XFER_ID_TOI + i];
  return t;
}
static inline void xfer_toi_put(uint8_t *block, xfer_toi_t toi) {
  for (int i = 0; i < XFER_TOI_BYTES; i++) block[XFER_ID_TOI + i] = (uint8_t)(toi >> 8 * i);
  block[XFER_ID_SPARE] = 0;
}
// A file's TOI: the first six bytes, little-endian, of BLAKE3's key derivation under the context "LIZARD TOI" over its
// root (32 bytes), length (five bytes, little-endian) and source block size (sb_log2, one byte); 1 where they are 0.
xfer_toi_t xfer_toi_of(const uint8_t root[32], uint64_t length, int sb_log2);

// A file's description, as the session header's map carries it (xfer_desc_write, xfer_desc_read).
typedef struct {
  uint8_t hash, sb_log2, name_len;
  uint64_t length;
  uint32_t source_blocks;
  uint8_t root[XFER_CV];
  uint8_t name[XFER_NAME_MAX];
  uint8_t compression; // the bytes' compression on air (XFER_COMP_*; a value this build does not know kept, 255 past it), 0 none
  uint8_t critical;    // 1 where its map carries an unsigned (critical) key this build does not know
} xfer_header_t;

// ceil(length / 2^sb_log2), or 0 where that is past XFER_MAX_SOURCE_BLOCKS or sb_log2 is out of range (an empty file
// is 0 source blocks too: the caller tells them apart by length).
uint32_t xfer_source_blocks(uint64_t length, int sb_log2);
// The source block size's log2 a sender picks for a file of `length` bytes (above), or -1 past 65,535 source blocks of 16 MiB.
int xfer_sb_log2_for(uint64_t length);
// Bytes of source block `index` under h, 0 past the last.
size_t xfer_sb_len(const xfer_header_t *h, uint32_t index);
// The source symbols of `len` bytes, ceil(len / 460): a source block's K.
uint32_t xfer_symbols(size_t len);

// Fills a header for a file of `length` bytes: hash, source block count; everything else is the caller's (the root,
// the name and the compression). 0, or -1 for a length or source block size the layout cannot carry.
int xfer_header_init(xfer_header_t *h, uint64_t length, int sb_log2);
// A session header read (xfer_session_parse): the description's offset and bytes.
typedef struct {
  size_t at, len;
} xfer_session_t;
// The session header's 460 bytes: the description (desc, len bytes, at most XFER_DESC_MAX), then 0: 0, or -1 where it
// is longer.
int xfer_session_write(uint8_t *payload, const uint8_t *desc, size_t len);
// A session header read into s: the description's offset (0) and bytes, those of the CBOR map at its start. 0, or
// XFER_ERR where none is there whole or it is outside the profile (its keys and fields are xfer_desc_read's).
int xfer_session_parse(const uint8_t *payload, xfer_session_t *s);
// 1 where a and b describe one transfer's blocks: the same hash, source block size, length and root.
// The name and the compression are not the bytes': a description under the same TOI
// that differs in them alone is the same transfer. 0 otherwise.
int xfer_same_file(const xfer_header_t *a, const xfer_header_t *b);

// The root the chaining values of source blocks >= 2 source blocks combine to (the tree above). -1 for fewer than 2.
int xfer_root(const uint8_t *cvs, uint32_t source_blocks, uint8_t root[XFER_CV]);
// 0 where the chaining values of a file's source_blocks source blocks (two or more, a receiver's own) combine to root.
int xfer_root_check(const uint8_t *cvs, uint32_t source_blocks, const uint8_t root[XFER_CV]);

// ---- The session's description (SPEC 7.11): the file's own CBOR map (above) ----
// The description of h (its TOI its content's, xfer_toi_of, not written) into out (room for XFER_DESC_LONGEST bytes,
// which the session header always holds), in the deterministic form: its byte count, or -1 where h is inconsistent or
// its compression unknown.
int xfer_desc_write(const xfer_header_t *h, uint8_t *out);
// The description desc (len bytes, one CBOR map to its last byte) read into h (critical 1 where it carries an unsigned
// key this build does not know, keys 1, 2 and 7 among them since 2026-11-22; a compression it does not know read as it
// is, elective) and its TOI, derived from its k, length and root (xfer_toi_of), into *toi (may be NULL). 0, or -1 where
// it is not one map to its last byte, does not parse, carries a key twice, or (self_toi, the session header's TOI, not
// 0) names the session's own TOI as its file's.
int xfer_desc_read(const uint8_t *desc, size_t len, xfer_toi_t self_toi, xfer_header_t *h, xfer_toi_t *toi);
// The name a description carries for a name of len bytes (UTF-8) into out: at most XFER_NAME_MAX bytes, its byte count.
// A longer name is cut in its stem, at a character's start, its extensions kept whole: the bytes from the earlier of
// its last two '.' (not its first byte) that leaves at most XFER_EXT_MAX with the dot (".tar.zst" kept whole), else
// from its last; a name with no extension (or a longer one) is cut at its end, at a character's start. A file's description always fits the session header
// with a name of XFER_NAME_MAX bytes (XFER_DESC_LONGEST, 309 bytes at the most), so nothing is cut for room.
size_t xfer_name_cut(const uint8_t *name, size_t len, uint8_t out[XFER_NAME_MAX]);

// The chaining value of source block `index` of 2^sb_log2 bytes, len being its bytes (the source block size, or less for the
// last). Not the root: a file of one source block is checked with xfer_b3_hash.
int xfer_sb_cv(const uint8_t *data, size_t len, uint32_t index, int sb_log2, uint8_t cv[XFER_CV]);
// Source block `index` as it decodes: 0 where len is what h says it is and, with one source block, its hash is the
// root; with two or more its chaining value into cv (32 bytes, the caller's to keep for xfer_root_check; may be NULL
// with one). -1 otherwise.
int xfer_sb_check(const xfer_header_t *h, uint32_t index, const uint8_t *data, size_t len, uint8_t *cv);

// BLAKE3 of a buffer, 32 bytes, through the official hasher (vendor/blake3).
void xfer_b3_hash(const uint8_t *in, size_t len, uint8_t out[XFER_CV]);
// The chaining value of the subtree BLAKE3 would make of `len` bytes starting at 1 KiB chunk `counter` (byte offset
// 1024 counter), or with root = 1 the hash, the range then being the whole input (counter ignored). Right only for a
// range that is a node of the input's tree: 2^j KiB on a 2^j KiB boundary, or all that is left of the input from
// such a boundary when that is less.
void xfer_b3_subtree(const uint8_t *in, size_t len, uint64_t counter, int root, uint8_t out[XFER_CV]);

// The test stream, the reference tools' and no part of the format (since 2026-11-20): data blocks
// under a TOI the sender draws, as any send's, no header; block id i's payload the first len bytes of BLAKE3's extended
// output of its bare id (its source block and symbol, bytes 0 to 4) as five little-endian bytes, the TOI no part. A
// function of the id alone, so a receiver that knows it tells a test frame from its payloads without being told which
// frame carried it (src/xfer.h stream_matches; sim/phy.mjs blockJudge); BLAKE3, the transfer's own hash, so the library
// carries one (SHAKE256 until 2026-11-15). Its vector: id 0's first 16 bytes cdc96eca844d7912acdbb3dca677757d, id 1's
// 7784b17d1d8ff658cfe8f0e111316a96, id 0x0403020100's BLAKE3's own five-byte vector (test/vectors.mjs).
void stream_fill(uint64_t id, uint8_t *out, uint32_t len);
// 1 where payload (len bytes) is id's stream, else 0: compared an output block (64 bytes, one compression) at a time
// as the stream is made, so a payload that is not the stream's (every block of a file) costs one compression.
int stream_matches(uint64_t id, const uint8_t *payload, uint32_t len);

#endif
