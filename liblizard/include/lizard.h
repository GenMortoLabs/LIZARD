/* ai: liblizard: the LIZARD code as a library (2026-10-03): the C (NEON on arm64) and the GPU through Vulkan behind one
 * ai: portable API, with no UI. A luminance-only animated barcode: a frame is a grey picture whose FFT
 * ai: coefficients carry blocks of 472 bytes (a 12-byte id: its source block and symbol in 40 bits, then its object's
 * ai: 48-bit TOI and a reserved byte; and 460 of payload, each block under a CRC-64), inside a
 * ai: border that says what is inside; a file goes as a fountain-coded transfer verified by its BLAKE3 root.
 * ai:
 * ai: Layers, each usable alone:
 * ai:   1. format arithmetic: a format's geometry, the room it needs, the largest that fits a room;
 * ai:   2. the per-frame codec: blocks to a painted frame, a captured image to its verified blocks (the C reference,
 * ai:      NEON on arm64, SSE4.1 and AVX2 on x86-64, WebAssembly SIMD in a wasm build, scalar elsewhere);
 * ai:   3. the transfer: a file (or the test stream) to each frame's blocks, verified blocks back to the file.
 * ai: Layers 1 to 3 start no thread and keep no state between calls but their handles'. A handle is used from one
 * ai: thread at a time; different handles from different threads at once.
 * ai:
 * ai: The format is never frozen: a sender and a receiver need the same release of this library (liz_version).
 * ai:
 * ai: Errors: a function returns LIZ_OK (0) or a count, or a negative LIZ_E_*; one returning a handle returns NULL. In
 * ai: either case liz_last_error() is a message for the calling thread's last failure. Strings the library returns are
 * ai: valid until the next call on the same handle. Buffers are the caller's, sized by the functions that say how big. */
#ifndef LIZARD_H
#define LIZARD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(LIZ_BUILDING)
#define LIZ_API __declspec(dllexport)
#elif defined(LIZ_STATIC)
#define LIZ_API
#else
#define LIZ_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define LIZ_API __attribute__((visibility("default")))
#else
#define LIZ_API
#endif

LIZ_API const char *liz_version(void);
LIZ_API const char *liz_last_error(void);
/* ai: the calling thread's last failure's LIZ_E_* code (LIZ_OK after a call that succeeded): what a NULL handle meant */
LIZ_API int liz_last_error_code(void);
/* ai: 1 where the codec's vector paths are compiled in (NEON, SSE4.1, WebAssembly SIMD), else 0 */
LIZ_API int liz_simd(void);

enum {
  LIZ_OK = 0,
  LIZ_E_ARG = -1,          /* ai: an argument out of range */
  LIZ_E_NOMEM = -2,
  LIZ_E_IO = -3,           /* ai: a file could not be read or written */
  LIZ_E_UNSUPPORTED = -4,  /* ai: not in this build, or not on this device */
  LIZ_E_FORMAT = -5,       /* ai: a format the codec refuses */
  LIZ_E_TOOBIG = -6,       /* ai: a file past what the transfer or the store holds */
  LIZ_E_STATE = -7,        /* ai: not now (no file yet, no frame ready) */
  LIZ_E_INTERNAL = -8
};

enum {
  LIZ_BLOCK = 472,         /* ai: a block: the id and the payload (the codec adds its CRC-64, 8 bytes) */
  LIZ_ID_BYTES = 12,       /* ai: the id, 96 bits: bytes 0 to 4 its 16-bit source block over its 24-bit symbol, 5 to 10 the
                            * ai: object's 48-bit TOI, both little-endian, byte 11 reserved (0) (src/xfer.h) */
  LIZ_PAYLOAD = 460,
  LIZ_MIN_BLOCKS = 1,      /* ai: the smallest format's size (LIZARD-16, two blocks a symbol) */
  LIZ_MAX_BLOCKS = 67,     /* ai: the largest size (LIZARD-1072), the word's last version (src/fmt.h OB_FMT_VERSION_MAX) */
  LIZ_RINGS = 3,           /* ai: the borders: rings of 64, 128 or 256 cells a side */
  LIZ_RING_DEFAULT = -1,   /* ai: the senders' default ring, the 128 */
  LIZ_GAP_MODULES = 12,    /* ai: two codes side by side are this many modules apart */
  LIZ_QUIET = 2,           /* ai: the margin painted round a symbol, in modules */
  LIZ_STEP = 16,           /* ai: the sub-channels a size: a symbol of b blocks is LIZARD-(16 b), its word's version b */
  LIZ_BLOCKS_DEFAULT = 30  /* ai: the senders' size unless one is chosen (LIZARD-480) */
};

typedef enum {
  LIZ_GREY8 = 1,  /* ai: one byte a pixel */
  LIZ_RGBX8 = 2,  /* ai: four bytes a pixel, grey in each of the first three; the fourth 255 (painting) or ignored */
  LIZ_RGBA8 = 3,  /* ai: as RGBX8 when painting; a captured image's luma by its R, G and B */
  LIZ_BGRA8 = 4   /* ai: as RGBX8 when painting; a captured image's luma by its B, G and R */
} liz_pixfmt;

/* ---- 1. format arithmetic ---------------------------------------------------------------------------------------- */

/* ai: blocks: the format's size, 1 to 67 (LIZARD-16 to LIZARD-1072: the number is 16 x blocks), the version its word
 * ai: names; a symbol of it carries liz_blocks_for(blocks) blocks, the code's rate following the frequency (6/7 on its
 * ai: lowest frequencies, then 3/4 and 2/3, 1/2 on its highest: 52 at LIZARD-432); ring: 0 to LIZ_RINGS - 1 (the 64,
 * ai: 128 and 256 rings) or LIZ_RING_DEFAULT;
 * ai: fps: 1 to 240, the pictures a second the sender paints, which every frame's word states with its places' largest
 * ai: Refresh delay (src/fmt.h);
 * ai: codes: 1, or 2 symbols side by side (one format, one
 * ai: word, the frame's blocks split between them, the first code the first ones). */
typedef struct {
  int blocks, ring, fps, codes;
} liz_format;

/* ai: What a format paints. n: the picture's samples a side; span: the picture's modules a side (2 x the ring's
 * ai: cells); pxm: pixels a module; side: a symbol's pixels a side, its margin in; width, height: the frame's pixels
 * ai: (codes symbols and the gaps between them); gap: the pixels between two codes; frame_blocks: blocks a frame
 * ai: (codes x liz_blocks_for(blocks)). */
typedef struct {
  int n, span, pxm, side, width, height, gap, frame_blocks;
} liz_geometry;

/* ai: the cells a side of ring r (0 to LIZ_RINGS - 1; LIZ_RING_DEFAULT the default's), or LIZ_E_ARG */
LIZ_API int liz_ring_cells(int ring);
/* ai: a symbol's modules a side in ring r, its quiet margin not counted: the picture's span (2 x the ring's cells) and
 * ai: the border, 15 modules deep, on either side; or LIZ_E_ARG */
LIZ_API int liz_ring_modules(int ring);
LIZ_API int liz_geometry_of(const liz_format *format, liz_geometry *out);
/* ai: The pixels a side a symbol of this many blocks needs on a display for its outer coefficients to keep 2.7 pixels
 * ai: a cycle (the web sender's ROOM_FOR); a negative error for an argument out of range. */
LIZ_API double liz_room_for(int blocks, int ring);
/* ai: The radius, in coefficients, of the half-disc a symbol's sub-channels fill (320 coefficients each), and the
 * ai: camera pixels across a symbol it needs: 2.12 a cycle of that radius at the symbol's width over its picture's
 * ai: (fitted on the simulator, SPEC 3.5: a fact about the format, no camera's); a negative error for an argument out
 * ai: of range. */
LIZ_API double liz_radius_for(int blocks);
LIZ_API double liz_camera_for(int blocks, int ring);
/* ai: The blocks a symbol of a format's size (blocks, 1 to 67) carries: the default senders' layout's count; 0 outside. */
LIZ_API int liz_blocks_for(int blocks);
/* ai: Where block b (0 to liz_blocks_for(blocks) - 1) of a symbol of a format's size lies: its code rate (0 6/7, 1 3/4,
 * ai: 2 2/3, 3 1/2), its first data slot and its slots (636 data slots a sub-channel, its other two coefficients the
 * ai: pilots; the lowest frequencies first, the blocks packed from data slot 0 inner rates first), and the first and last
 * ai: sub-channel it spans. A block changes whole or not at all (one codeword), so a schedule is a block's, not a
 * ai: sub-channel's. */
typedef struct {
  int rate, first_slot, slots, first_subch, last_subch;
} liz_block_place;
LIZ_API int liz_block_place_of(int blocks, int b, liz_block_place *out);
/* ai: The largest size (blocks) a symbol may have, at most top_blocks, for codes symbols in a room of w x h display
 * ai: pixels (the web sender's pickVersion: each symbol's room is the width over codes and the gaps, against the
 * ai: height); at least LIZ_MIN_BLOCKS; top_blocks 1 to 67. */
LIZ_API int liz_pick(double w, double h, int codes, int ring, int top_blocks);

/* ---- 2. the per-frame codec -------------------------------------------------------------------------------------- */

typedef struct liz_encoder liz_encoder;
LIZ_API liz_encoder *liz_encoder_new(const liz_format *format);
LIZ_API void liz_encoder_free(liz_encoder *e);
LIZ_API int liz_encoder_geometry(const liz_encoder *e, liz_geometry *out);
/* ai: One frame: blocks, frame_blocks x LIZ_BLOCK bytes (each an id and a payload; the codec adds the CRC);
 * ai: picture, the frame's count since the start (its value mod 4 is what the pilots paint: count every frame shown,
 * ai: once, in order); out, width x height pixels of fmt, rows stride bytes apart. Grey on white margins. */
LIZ_API int liz_encoder_paint(liz_encoder *e, const uint8_t *blocks, uint32_t picture, uint8_t *out, int stride, liz_pixfmt fmt);

typedef struct liz_decoder liz_decoder;
/* ai: nmax: the largest picture to decode, 384 to 1536 (0: 1536, every format); smaller saves memory. */
LIZ_API liz_decoder *liz_decoder_new(int nmax);
LIZ_API void liz_decoder_free(liz_decoder *d);
/* ai: the most blocks a frame can verify: the decode's buffer is this many x LIZ_BLOCK bytes */
LIZ_API int liz_decoder_max_blocks(const liz_decoder *d);

typedef struct {
  int found;          /* ai: a symbol registered */
  int ring;           /* ai: the ring that registered, 0 to LIZ_RINGS - 1; -1 none */
  int n;              /* ai: the picture it was finished at; 0 not finished */
  int word;           /* ai: the frame's own word read */
  int blocks;         /* ai: the size of the format it was decoded at, its version (0 none) */
  int total;          /* ai: the blocks the symbol carries */
  int verified;       /* ai: the blocks verified */
  float quad[8];      /* ai: its corners in the image, x y x y ..., top left, top right, bottom right, bottom left */
  int pilot_subch;    /* ai: the sub-channels the pilots were read over (0 none) */
  float pilot_r[2];   /* ai: the pilots' readings, the even sub-channels' count pilots over the references and the
                         ai: odd's: their signs the picture's count; NaN where none was read (pilot_subch 0) */
  float pilot_sd[2];  /* ai: their standard errors (NaN with its reading) */
} liz_decoded;

/* ai: One captured frame (or one region of it: px at the region's first pixel, w x h, rows stride bytes apart): its
 * ai: verified blocks packed into verified (liz_decoder_max_blocks x LIZ_BLOCK bytes), their count returned. A frame is
 * ai: decoded at its own word's version and layout; one whose word does not read, not at all (the word before the
 * ai: frame). Every frame is decoded on its own: nothing carries from one to the next. out may be NULL. */
LIZ_API int liz_decode(liz_decoder *d, const uint8_t *px, int w, int h, int stride, liz_pixfmt fmt,
                       uint8_t *verified, liz_decoded *out);

/* ai: Where a camera frame's symbols are looked for: layout 1, the centre square; 2, the centre region twice as long as
 * ai: it is high along the frame's long side, cut into two squares (a sender's two codes). rect[k] = x, y, w, h; returns
 * ai: how many (1 or 2). */
LIZ_API int liz_layout_rects(int w, int h, int layout, int rect[2][4]);

/* ai: the test stream's payload for an id (BLAKE3 of its low five bytes, the source block and symbol; the TOI no part):
 * ai: the reference tools' test stream, no part of the format; a receiver that knows it judges a frame of it by its bytes */
LIZ_API void liz_stream_fill(uint64_t id, uint8_t payload[LIZ_PAYLOAD]);

/* ---- 3. the transfer --------------------------------------------------------------------------------------------- */

typedef struct liz_tx liz_tx;
/* ai: LIZ_TX_COPY: liz_tx_new copies the bytes (without it they must outlive the liz_tx); LIZ_TX_ZSTD: the bytes are
 * ai: the default client's zstd stream (core/tx/pack.h), the description saying compression 1 (src/xfer.h) */
enum { LIZ_TX_COPY = 1, LIZ_TX_ZSTD = 2 };
/* ai: A file: its bytes and its name as its header carries it (a name past 255 bytes cut in its stem, its extension
 * ai: kept: src/xfer.h xfer_name_cut). Returns once the file is hashed; its source blocks are made by threads of
 * ai: its own behind it, the first frames carrying the header alone until every source block is made (natively), so every
 * ai: source block starts the schedule level. The file's TOI is its content's (src/xfer.h xfer_toi_of); the session
 * ai: header's is drawn from the send's seed. */
LIZ_API liz_tx *liz_tx_new(const uint8_t *data, size_t length, const char *name, int flags);
/* ai: liz_tx_new in source blocks of 2^sb_log2 bytes (10 to 24) in place of xfer_sb_log2_for's for the length (16 MiB
 * ai: for every file the format carries): a sender's choice, the format's header saying it (a test's smaller source
 * ai: blocks, the web sender's ?chunk=). */
LIZ_API liz_tx *liz_tx_new_sized(const uint8_t *data, size_t length, const char *name, int flags, int sb_log2);
/* ai: A file read from a path (into memory); name NULL takes the path's last part; flags LIZ_TX_ZSTD as above. */
LIZ_API liz_tx *liz_tx_new_path(const char *path, const char *name, int flags);
/* ai: A file read as the transfer asks for it, never whole (2026-11-25: a file past what memory holds, the web's picked
 * ai: file): read(user, offset, n, into) puts the n bytes of the file at offset into `into` and returns 1, else 0 (the
 * ai: transfer then fails: liz_tx_new_read's error, or liz_tx_next's LIZ_E_TOOBIG); it is called from the transfer's own
 * ai: threads, several at once, a source block at a time each. spill_dir: a folder for the spill (no source block keeps
 * ai: its encoder in memory: its symbols in one file there, about 0.75 times the file's size; NULL none, every encoder
 * ai: kept, about 1.02 times the file in memory). length the file's bytes; name, flags as liz_tx_new's; sb_log2 as
 * ai: liz_tx_new_sized's, or -1 xfer_sb_log2_for's. */
typedef int (*liz_read_fn)(void *user, uint64_t offset, size_t n, uint8_t *into);
LIZ_API liz_tx *liz_tx_new_read(liz_read_fn read, void *user, uint64_t length, const char *name, int flags, int sb_log2, const char *spill_dir);
/* ai: The test stream from first_id (0: a random one), under a random TOI of its own as any send's: each block the
 * ai: stream's payload for its id. */
LIZ_API liz_tx *liz_tx_new_test(uint32_t first_id);
LIZ_API void liz_tx_free(liz_tx *t);
/* ai: A file's schedule drawn from seed in place of the random one liz_tx_new draws (each source block's first symbol,
 * ai: each round's order of the source blocks, each frame's shuffle, the session header's TOI), so a send repeats:
 * ai: before the first frame (LIZ_E_STATE after, and for the test stream), as liz_sender_seed. */
LIZ_API int liz_tx_seed(liz_tx *t, uint32_t seed);
/* ai: The next frame's n blocks (n x LIZ_BLOCK bytes): a frame asked for twice is two frames. Returns how many of
 * ai: them carry the file's data (the others its header); LIZ_E_TOOBIG where a source block could
 * ai: not be made. */
LIZ_API int liz_tx_next(liz_tx *t, int n, uint8_t *blocks);

typedef struct {
  int test;              /* ai: the test stream */
  uint64_t length;       /* ai: the file's bytes */
  uint32_t source_blocks;       /* ai: its source blocks (16 MiB each, xfer_sb_log2_for; a file to 65,535 of them, about 1 TiB) */
  uint32_t cycle;          /* ai: data blocks a cycle of the schedule (0 until every source block is made, behind liz_tx_new) */
  uint8_t root[32];      /* ai: BLAKE3 of the file (b3sum's) */
  uint32_t seed;         /* ai: the schedule's seed in use (liz_tx_seed's, else the random one) */
  uint32_t sb_log2;      /* ai: the source block size's power of two (the header's k) */
} liz_tx_info;
LIZ_API int liz_tx_info_get(const liz_tx *t, liz_tx_info *out);

typedef struct liz_rx liz_rx;
/* ai: Where a receiver keeps what it holds: dir NULL, in memory, one transfer at a time; else files under dir, a folder
 * ai: a transfer, the finished file at a path under it. A folder store keeps an unfinished transfer when another
 * ai: file's header arrives, and across receivers on the same folder: its own header read again resumes it from the
 * ai: blocks already held. Opening the folder keeps those transfers and removes anything else under it. max_bytes
 * ai: (memory only, 0 no cap) bounds the stored blocks and the file together, which peak near twice the file's size
 * ai: (its bytes and every source block's blocks before it solves); a source block's solve, about 36 MB at 16 MiB, is
 * ai: outside it. A store that cannot hold a transfer (over the cap, a write refused, a disk
 * ai: full) ends it: never done, its reason in LIZ_META_ERROR, until a new file's header begins afresh. */
typedef struct {
  const char *dir;
  uint64_t max_bytes;
} liz_store;
LIZ_API liz_rx *liz_rx_new(const liz_store *store);
LIZ_API void liz_rx_free(liz_rx *r);

/* ai: What a frame's verified blocks came to. seen: blocks taken (the header's in, repeats in); bad:
 * ai: a test frame's blocks that are not the stream's; judged: a test frame's data blocks; fresh: data blocks new to
 * ai: this receiver; test: the frame was the test stream's (none of its blocks go to a file). */
typedef struct {
  int seen, bad, judged, fresh, test;
} liz_verdict;
/* ai: One frame's verified blocks, as liz_decode packed them (of both of a frame's regions, one call each or one
 * ai: together). A source block completed by them is solved and checked inside the call (tens of ms for 16 MiB). */
LIZ_API int liz_rx_frame(liz_rx *r, const uint8_t *verified, int count, liz_verdict *out);

typedef struct {
  int header;            /* ai: the file's header read */
  int done;              /* ai: the file whole and verified by its root, and kept whole by the store */
  uint64_t length;       /* ai: its bytes */
  uint64_t bytes_in;     /* ai: how many are in: a verified source block's all, another's blocks in hand to its length */
  double fraction;       /* ai: bytes_in over length */
  uint32_t source_blocks, verified, rejected;
  double solve_ms;       /* ai: the fountain's and the hash's time so far */
  uint64_t forwarded;    /* ai: blocks the judge passed on to the transfer (its headers in, repeats out) */
  uint64_t done_at;      /* ai: forwarded when the file was whole (0 before) */
  uint32_t refused;      /* ai: session headers refused (a description that does not read) */
  uint32_t over;         /* ai: source blocks RaptorQ wanted a symbol more than K for */
} liz_progress;
LIZ_API int liz_rx_progress(liz_rx *r, liz_progress *out);
/* ai: each source block's state into per (at most cap): 0 to 99 the share in, 255
 * ai: verified; returns the source blocks */
LIZ_API int liz_rx_source_blocks(liz_rx *r, uint8_t *per, int cap);
enum { LIZ_META_NAME = 0, LIZ_META_ROOT = 1, LIZ_META_PATH = 2, LIZ_META_ERROR = 3 };
/* ai: the header's name, the root as hex, the finished file's path (a folder store's), the store's failure
 * ai: in this transfer (the reason it ended); "" where there is none */
LIZ_API const char *liz_rx_meta(liz_rx *r, int which);
/* ai: The finished file's bytes in a memory store (valid until the next call on r); LIZ_E_STATE before it is done. */
LIZ_API int liz_rx_data(liz_rx *r, const uint8_t **data, size_t *length);
/* ai: The transfer and every file received let go, so the same file still in the light is received again. */
LIZ_API void liz_rx_clear(liz_rx *r);

/* ai: The senders' bundle (each app's and the web page's): a choice of files and folders as one payload. One file
 * ai: alone at the top is itself; anything else is one POSIX tar: ustar headers (the name, and the prefix field where a
 * ai: path is long), a pax extended header (typeflag x) carrying the path where it fits neither field or is not plain
 * ai: ASCII and the size past ustar's 8 GiB; folders their own entries (typeflag 5, so an empty folder survives); files
 * ai: mode 644, folders 755, owner 0; two zero blocks at the end. It is named after its one top folder ("<folder>.tar")
 * ai: or its first item and how many more ("<first> and N more.tar"), the stem cut on a character so the name fits the
 * ai: session header's 255 bytes. Each file's folders are added where missing, a path given twice is numbered ("b
 * ai: (2).txt"), a folder given no time takes the newest of what it holds (an empty one the newest of the choice), and
 * ai: the entries go in the order of their paths (as UTF-16 code units). The archive is laid out, not written: for each
 * ai: entry the caller writes its head, then its item's bytes (size of them), then pad zero bytes, and after the last
 * ai: entry LIZ_TAR_END zero bytes, so a file's bytes never pass through the library. */
typedef struct liz_bundle liz_bundle;
enum { LIZ_TAR_END = 1024 };
LIZ_API liz_bundle *liz_bundle_new(void);
LIZ_API void liz_bundle_free(liz_bundle *b);
/* ai: An item of the choice: its path, relative, '/' between its parts (UTF-8); a folder (dir 1, size 0) or a file of
 * ai: size bytes; its time, seconds since 1970 (NaN: none given). Returns its index, from 0. */
LIZ_API int liz_bundle_add(liz_bundle *b, const char *path, int dir, int64_t size, double mtime);
typedef struct {
  int64_t size;  /* ai: the payload's bytes: the archive's, or the one file's */
  int single;    /* ai: the item sent as itself (one file alone at the top), or -1 for a tar */
  int files;     /* ai: the files chosen */
  int entries;   /* ai: the tar's entries (0 for one file alone) */
} liz_bundle_info;
/* ai: The choice made one payload; now: seconds since 1970, the time of an entry given none. LIZ_E_ARG for an empty
 * ai: choice or a path that is not relative. */
LIZ_API int liz_bundle_make(liz_bundle *b, double now, liz_bundle_info *out);
/* ai: the payload's name (valid until the next call on b), "" before liz_bundle_make */
LIZ_API const char *liz_bundle_name(liz_bundle *b);
typedef struct {
  int64_t size;  /* ai: the bytes after the head: the item's (0 for a folder) */
  int item;      /* ai: the choice's item whose bytes those are, or -1 (a folder) */
  int pad;       /* ai: zero bytes after them */
  int head_len;  /* ai: the head's bytes */
} liz_bundle_entry;
/* ai: Entry e (0 to entries - 1) in order: its head's bytes (head_len of them, valid until the next call on b), NULL
 * ai: for an e out of range. */
LIZ_API const uint8_t *liz_bundle_entry_get(liz_bundle *b, int e, liz_bundle_entry *out);

/* ai: A tar read entry by entry (the apps' receivers unpacking a bundle): plain files and folders, pax (x, g) and GNU
 * ai: long-name (L) headers read for the path and size, ustar's prefix; anything else (a link, a device, a FIFO), a
 * ai: path that is absolute, holds "..", has no part or names a drive, a damaged header or a tar cut short refused.
 * ai: The reader asks for the bytes it reads (liz_untar_want) and is handed them (liz_untar_take); an entry's own bytes
 * ai: are the caller's to read or skip from its stream. */
typedef struct liz_untar liz_untar;
enum { LIZ_UNTAR_END = 0, LIZ_UNTAR_MORE = 1, LIZ_UNTAR_ENTRY = 2 };
LIZ_API liz_untar *liz_untar_new(void);
LIZ_API void liz_untar_free(liz_untar *u);
/* ai: the bytes the reader takes next: a header's 512, an extended header's body and its padding, an entry's padding */
LIZ_API int64_t liz_untar_want(const liz_untar *u);
/* ai: The next bytes, exactly want's unless the stream ended there (fewer): LIZ_UNTAR_MORE (read on), LIZ_UNTAR_ENTRY
 * ai: (an entry, liz_untar_entry says which: the caller reads or skips its size bytes from the stream before the next
 * ai: take), LIZ_UNTAR_END (the archive's end, or the stream's at a header), or LIZ_E_FORMAT (liz_last_error why). */
LIZ_API int liz_untar_take(liz_untar *u, const uint8_t *bytes, size_t n);
/* ai: the entry the last take named: its path's parts joined by '/' (UTF-8, valid until the next call on u), dir 1
 * ai: for a folder (size 0), its bytes */
LIZ_API const char *liz_untar_entry(const liz_untar *u, int *dir, int64_t *size);

/* ---- 4. engines: threads, the GPU (Vulkan) where a device runs it, else the CPU; native builds only --------------- */
/* ai: What the native apps run: a receiver decoding camera frames on its own threads (the GPU decoder in batches, or
 * ai: the C on a pool of workers) into a transfer, and a sender painting frames ahead on its own threads (the GPU's
 * ai: encoder, or the C on several). The GPU paths read their kernels and nets from an assets folder (share/lizard
 * ai: as installed); without one, or with no Vulkan device that runs them, the CPU does the work, and the receiver's
 * ai: LIZ_DECODER_AUTO decodes on the CPU on a software Vulkan device too (Vulkan's CPU type: lavapipe, SwiftShader),
 * ai: where LIZ_DECODER_GPU still runs. A wasm build has no engines (LIZ_E_UNSUPPORTED). Their stats are JSON, the
 * ai: field names the apps' (not a stable interface). */

enum { LIZ_DECODER_AUTO = 0, LIZ_DECODER_GPU = 1, LIZ_DECODER_CPU = 2 };
typedef struct {
  const char *assets;      /* ai: the GPU's kernels and nets; NULL: the CPU decoder alone */
  const char *cache_dir;   /* ai: the GPU's pipeline cache; NULL none */
  const char *store_dir;   /* ai: where the file is kept, under store_dir/lizard-xfer (unfinished transfers kept there to resume) */
  const char *device;      /* ai: a substring of the Vulkan device's name to use; NULL the first discrete, else the first */
  const char *precision;   /* ai: the GPU decoder's nets: "auto" (NULL), "int8", "f16" or "f32" */
  /* ai: LIZ_DECODER_*: AUTO the GPU decoder where a device runs a variant the assets hold and is no software Vulkan
   * ai: device, else the C (its stats' gpuWhy says why); GPU the GPU decoder on any device that runs it, a software one
   * ai: too, else liz_receiver_new fails (LIZ_E_UNSUPPORTED; LIZ_E_ARG with no assets); CPU the C */
  int decoder;
  int cpu_threads;         /* ai: the CPU decoder's workers; 0 its own rule (one, more while frames are lost) */
  int layout;              /* ai: 1 the centre square, 2 the two squares of a 2:1 region (liz_layout_rects) */
  void (*log)(void *user, const char *line);   /* ai: NULL none; called from the receiver's threads */
  void *user;
} liz_receiver_config;

typedef struct liz_receiver liz_receiver;
/* ai: release(user, tag): a frame pushed is the caller's again (from a receiver thread, or inside the push). */
LIZ_API liz_receiver *liz_receiver_new(const liz_receiver_config *config, void (*release)(void *user, uint64_t tag));
LIZ_API void liz_receiver_free(liz_receiver *r);
/* ai: One camera frame's luma plane, w x h, rows stride bytes apart, at timestamp_ns on the camera's clock: copied (or
 * ai: dropped) before the call returns, release(tag) called by then. Never blocks on the device. */
LIZ_API int liz_receiver_push(liz_receiver *r, const uint8_t *luma, int w, int h, int stride, int64_t timestamp_ns, uint64_t tag);
/* ai: Android: a camera AHardwareBuffer (YUV), read by the GPU in place; it stays the caller's until release(tag). */
LIZ_API int liz_receiver_push_hardware_buffer(liz_receiver *r, void *ahardwarebuffer, int w, int h, int64_t timestamp_ns, uint64_t tag);
/* ai: 1 where frames must come as luma planes (the CPU decoder, or the GPU's on a device without the zero-copy ingest),
 * ai: 0 where hardware buffers are read in place */
LIZ_API int liz_receiver_wants_luma(liz_receiver *r);
LIZ_API const char *liz_receiver_stats(liz_receiver *r);
/* ai: The frames decoded since since_ms (on the camera's clock), 8 doubles each into out (at most cap doubles), after
 * ai: the last word read's configuration (its index, src/fmt.h ob_fmt_index, plus 1; 0 none; a statistic, no frame decoded by it): ms, verified blocks, new blocks, found, the pilots' r and its error over the even blocks,
 * ai: then over the odd (NaN none); returns the doubles written. For a camera's phase lock. */
LIZ_API int liz_receiver_series(liz_receiver *r, double since_ms, double *out, int cap);
/* ai: the most frames a GPU launch takes, 1 to 32 (0: the batcher's size); a launch goes as soon as a frame waits */
LIZ_API void liz_receiver_batch_cap(liz_receiver *r, int frames);
/* ai: the received file's path once whole and verified, else "" */
LIZ_API const char *liz_receiver_file(liz_receiver *r);
LIZ_API void liz_receiver_clear(liz_receiver *r);
/* ai: The camera's stream closed (Android: its reader), every frame handed back: the GPU lets go of that reader's
 * ai: buffers, and the stats' window ends with the stream (its figures 0 until the next stream's first second has
 * ai: closed, that window the next stream's alone). */
LIZ_API void liz_receiver_camera_closed(liz_receiver *r);

enum { LIZ_PAINTER_CPU = 0, LIZ_PAINTER_GPU = 1, LIZ_PAINTER_AUTO = 2 };
typedef struct liz_sender liz_sender;
/* ai: A file (data NULL: the test stream), its bytes copied with LIZ_TX_COPY, else alive as long as the sender;
 * ai: LIZ_TX_ZSTD where they are the default client's zstd stream. */
LIZ_API liz_sender *liz_sender_new(const uint8_t *data, size_t length, const char *name, int flags);
LIZ_API void liz_sender_free(liz_sender *s);
/* ai: The format to paint (a re-pick keeps the transfer), on painter (LIZ_PAINTER_*: auto takes the GPU where its first
 * ai: frames are the C's), threads CPU painters, the GPU's assets and device as the receiver's. */
LIZ_API int liz_sender_configure(liz_sender *s, const liz_format *format, int painter, int threads, const char *assets, const char *device);
LIZ_API int liz_sender_geometry(liz_sender *s, liz_geometry *out);
/* ai: 1 where the next frame is painted */
LIZ_API int liz_sender_ready(liz_sender *s);
/* ai: The next frame in order (width x height of fmt, rows stride bytes apart): 1, it is the screen's now; 0, not painted
 * ai: yet (dst untouched). Each frame taken once; show them in order, each for as long as the display rate asks. */
LIZ_API int liz_sender_take(liz_sender *s, uint8_t *dst, int stride, liz_pixfmt fmt);
LIZ_API const char *liz_sender_stats(liz_sender *s);
/* ai: Refresh: each block place of a symbol takes a new block every `every` pictures, its delay, a whole number 1 to
 * ai: LIZ_REFRESH_MAX (1, the default, every picture), so its rate is the format's fps over a whole number: picture k
 * ai: (the sender's count) shows a new block there where k is a multiple of it, the picture before's block again
 * ai: otherwise. The pilots are apart from the blocks and stay each picture's count, so a receiver and its phase lock read
 * ai: every picture as ever, whatever each place's delay: no format change, a repeated block one a receiver has. From the
 * ai: next frame made. */
enum { LIZ_REFRESH_MAX = 30 };
/* ai: per code rate r (0 6/7, 1 3/4, 2 2/3, 3 1/2; liz_block_place's rate): every block of the rate; kept across
 * ai: configures */
LIZ_API int liz_sender_refresh(liz_sender *s, const int every[4]);
/* ai: per block place b of the configured format's symbol, n = liz_blocks_for(the format's blocks), every code alike
 * ai: (LIZ_E_ARG for another count); a configure to a format of another block count ends it, each place its rate's again */
LIZ_API int liz_sender_refresh_blocks(liz_sender *s, const int *every, int n);
/* ai: The rule itself, for a sender of its own (the web's: liz_tx's ids, its own paint), the native sender's
 * ai: (core/tx/refresh.h). The places' delays of a format's `blocks` places from the four rates' (`every`: 6/7, 3/4, 2/3,
 * ai: 1/2, each held to 1 .. LIZ_REFRESH_MAX), laid by the default senders' layout (lizard_tiers_for: the tiers from the
 * ai: inside out): place[p] place p's. LIZ_E_ARG where subch is no format's or its layout lays other than `blocks`. */
LIZ_API int liz_refresh_places(int subch, int blocks, const int every[4], int *place);
/* ai: The frame painted seq-th of `total` blocks (codes x a symbol's `blocks` places, place[] their delays): held[t] 1
 * ai: where block t repeats the frame before's at its place (`before` nonzero: there is a frame before of the same
 * ai: blocks; the place's delay past 1 and seq no multiple of it), 0 where it takes a new block. The new blocks' count. */
LIZ_API int liz_refresh_held(const int *place, int blocks, int total, uint64_t seq, int before, uint8_t *held);
/* ai: A file's schedule drawn from seed in place of the random one liz_sender_new draws (each source block's first symbol, each
 * ai: round's order of the source blocks, each frame's shuffle), so a run can be repeated: before the first frame is made
 * ai: (LIZ_E_STATE after, and for the test stream). The seed in use is the stats' "seed". */
LIZ_API int liz_sender_seed(liz_sender *s, uint32_t seed);
/* ai: The transfer's data blocks a cycle of its schedule (each of the file's blocks once), 0 until every source block is made
 * ai: (and for the test stream): what sizes an export (frames = cycles x cycle / the format's blocks a frame). The source blocks
 * ai: are made once the sender is configured: an export configures, waits for the cycle, then configures again (the
 * ai: frames painted ahead meanwhile, the header alone, dropped), so its first frame carries data. */
LIZ_API int64_t liz_sender_cycle(liz_sender *s);
/* ai: A stream file (a .lizard whose ZIP comment is "LIZARD stream 1", 2026-11-18): the configured sender's next frames,
 * ai: `frames` of them as liz_sender_take gives them (grey, in order), written to path (through path.part, renamed at the
 * ai: end), with a meta.json a player shows them by (kind "stream"; fps, layout, the format; the file's name, compression,
 * ai: length, source blocks, cycle and BLAKE3 root; the colours; the seed; w, h, frames, sizes as a replay's). A player shows the
 * ai: frames in turn at fps and loops, painting nothing; any receiver's Upload decodes the file as frames.
 * ai: The frames from the first one made once every source block is (frames made before carry the header alone), `frames` of
 * ai: them (0: 2.5 cycles of the file's blocks; the test stream needs a count), made odd; each packed by zstd at `level`
 * ai: (0: 1; every frame of the file the same, meta.json's "zstd") on the machine's cores, or where the CPU paints, by
 * ai: each painter as it paints. progress(user, done, frames) as frames are written (NULL none): 0 from it stops the
 * ai: export (the .part removed, LIZ_E_STATE). Blocks until done (not on a UI thread); the sender makes its frames for
 * ai: the export alone, so a screen's sender is not the one to export from. Returns frames, or an error. */
LIZ_API int liz_sender_export(liz_sender *s, const char *path, int frames, int level, int (*progress)(void *user, int done, int frames), void *user);
/* ai: A stream file played (2026-11-18): a sender that shows path's frames in turn at its fps and loops, painting
 * ai: nothing (LIZ_E_FORMAT where path is a replay or its frames are not its format's). liz_sender_configure takes its
 * ai: format from the stream (format may be NULL; painter and threads: its frames unpacked on that many CPU threads into
 * ai: the frame ring); take, ready, geometry and stats as a painting sender's (the stats' "stream", "pictures",
 * ai: "picture", "cycle", "pass"). Seed and refresh change nothing: the frames are painted already. */
LIZ_API liz_sender *liz_sender_play(const char *path);
/* ai: a played stream's meta.json (its format, fps, layout, pictures, file), "" for a painting sender */
LIZ_API const char *liz_sender_stream(liz_sender *s);
/* ai: What a .lizard holds, by its ZIP comment: LIZ_LIZARD_REPLAY (a camera's frames: Upload decodes it) or
 * ai: LIZ_LIZARD_STREAM (a sender's: opened, it sends; Upload decodes it too), else an error */
enum { LIZ_LIZARD_REPLAY = 0, LIZ_LIZARD_STREAM = 1 };
LIZ_API int liz_lizard_kind(const char *path);

/* ---- 5. the receive session: a camera into a receiver, with what a receiving app does around it ---------------- */
/* ai: What every app's receive screen runs (core/rx/session.h): a camera of the system's (UVC through V4L2, Media
 * ai: Foundation or AVFoundation; Android's back cameras through Camera2; a phone streaming to this machine; a replay
 * ai: played as a camera, upload:<path>; the simulated camera where LIZ_CAMERA_SIM is set) feeding a receiver, and
 * ai: around them the phase lock where the camera can delay a capture (Android's), the phase a screen shows, the
 * ai: camera's settings kept per camera, a verified file taken out of the store into the data folder's kept/ for the
 * ai: app to place, the camera stopped once a file is in, one session a data folder (a lock), Save replays, the stats
 * ai: and the preview. Every call returns at once: the work runs in order on the session's own thread, and a refusal
 * ai: is the phase's error (liz_session_stats) or a line in the log; liz_session_free blocks. Strings are UTF-8; a
 * ai: function writing JSON into json (at most cap bytes, ended by a NUL) returns the whole JSON's length, so a
 * ai: return of cap or more asks for a larger buffer. The JSON's field names are the apps' (not a stable interface).
 * ai: Desktop systems and Android (since 2026-10-31): elsewhere (wasm) liz_session_new says LIZ_E_UNSUPPORTED. */
typedef struct liz_session liz_session;
typedef struct {
  const char *assets;      /* ai: the GPU's kernels and nets; NULL: the CPU decoder alone */
  const char *cache_dir;   /* ai: the GPU's pipeline cache; NULL none */
  const char *data_dir;    /* ai: the session's folder: store/ (unfinished transfers kept to resume), kept/, replays/,
                            * ai: receive.lock, cameras.json */
  const char *decoder;     /* ai: "auto" (NULL), "gpu" or "cpu", as liz_receiver_config's decoder; "pc" on Android: the
                            * ai: frames served to a PC that decodes them (liz_link) */
  const char *layout;      /* ai: "1:1" (NULL) or "2:1" (liz_layout_rects) */
  int forget_settings;     /* ai: 1: the camera settings held for this session alone, none read or written (a test's) */
  int keep_packed;         /* ai: 1: a file the default client packed (SPEC 9.6) taken out of the store as it came, for
                            * ai: the app to unpack as it files it (liz_kept's "packed"); 0 unpacked into kept/ */
  void (*log)(void *user, const char *line);   /* ai: NULL none; called from the session's threads */
  void *user;
} liz_session_config;
/* ai: NULL for a decoder or layout it does not know, or no data folder (LIZ_E_ARG). The data folder's lock is taken on
 * ai: the session's thread: held by another session, every start says so (another LIZARD window is receiving). */
LIZ_API liz_session *liz_session_new(const liz_session_config *config);
/* ai: Blocks: the calls before it done, the camera stopped and closed (its controls put back), a recording replay
 * ai: written, the receiver let go, the lock released. Not on a UI thread. */
LIZ_API void liz_session_free(liz_session *s);

/* ai: Every camera listed: {"cameras": [{id, name, backend, detail}], "notes": [why a backend saw none]}; opens
 * ai: devices to look (seconds on some systems). The stamp changes when the list may have (nothing opened). */
LIZ_API size_t liz_cameras(char *json, size_t cap);
LIZ_API size_t liz_cameras_stamp(char *out, size_t cap);
/* ai: The camera by id ("" the first listed; one that does not open and is not listed, the first listed in its place),
 * ai: its kept settings taken up. */
LIZ_API void liz_session_open(liz_session *s, const char *camera_id);
LIZ_API void liz_session_close(liz_session *s);
/* ai: zoom, exposure (us), gain or any control the camera lists, to value or the camera's automatic; kept for the
 * ai: camera and set after each start. "mode": an index into the camera's modes, kept as its mode. */
LIZ_API void liz_session_set(liz_session *s, const char *control, double value, int automatic);
/* ai: The camera's stream into the receiver: mode NULL or "" the camera's kept mode, else an index or "WxH@fps". */
LIZ_API void liz_session_start(liz_session *s, const char *mode);
LIZ_API void liz_session_stop(liz_session *s);
/* ai: Another decoder or crop: the receiver made again on the transfer in hand, which goes on. */
LIZ_API void liz_session_reconfigure(liz_session *s, const char *decoder, const char *layout);
LIZ_API void liz_session_batch_cap(liz_session *s, int frames);
/* ai: The phase lock, on (the default) or off: on, a camera that can delay a capture (liz_session_camera's "lockBar" "")
 * ai: runs the lock and at most 60 captures a second (a faster mode runs at the fastest of its size and format at or under
 * ai: 60); off, every mode the camera lists runs as asked. */
LIZ_API void liz_session_phase_lock(liz_session *s, int on);

/* ai: The receiver's stats and the session's: phase ("idle", "starting", "loading", "on", "error") and why, the camera,
 * ai: live, asked, autoRan, kept {taken, root, error}, replays, the platform's fields (session.h stats). Never waits. */
LIZ_API size_t liz_session_stats(liz_session *s, char *json, size_t cap);
/* ai: The camera open: {info, modes, pick, controls, mode, streaming, kept {mode, zoom, exposure, gain}}, {} none. */
LIZ_API size_t liz_session_camera(liz_session *s, char *json, size_t cap);
/* ai: The newest frame's crop as the decoder reads it, newer than since, area-averaged to at most side pixels on its
 * ai: long side, into out (w x h bytes, at most cap): 1 with w, h and its frame's count in seq; 0 none newer. */
LIZ_API int liz_session_preview(liz_session *s, uint64_t since, int side, uint8_t *out, size_t cap, uint32_t *w, uint32_t *h,
                                uint64_t *seq);

/* ai: An unfinished transfer let go by its root or its folder's id; the transfer in hand and its file let go. */
LIZ_API void liz_session_forget(liz_session *s, const char *root);
LIZ_API void liz_session_clear(liz_session *s);
/* ai: Save replays on or off (a run while the camera reads, data_dir/replays/<run>.lizard); meta_json the platform's
 * ai: fields for the run's meta.json. liz_session_extra: the platform's own fields (a JSON object: heat, power), merged
 * ai: into the stats and each replay row from then on. */
LIZ_API void liz_session_replays(liz_session *s, int on, const char *meta_json);
/* ai: Replay size: a run's cap, the raw bytes of the frames it keeps (the newest that fit; the oldest dropped as new ones
 * ai: come), held to 256 MiB .. 8 GiB, 1 GiB until set; the run recording takes it at once. */
LIZ_API void liz_session_replay_bytes(liz_session *s, long long bytes);
/* ai: Largest file (2026-11-15, the apps' setting; session.h fileLimit): a transfer whose file is over bytes (0: the
 * ai: format's own limit), or wants more free space than room times its file where the store is (0: none asked; 1
 * ai: where the app's filing moves the file, 2 where it copies it), refused at its header before anything is written:
 * ai: the camera stopped, the phase's error "File too large: ..." or "Not enough space: ...", the stats' file
 * ai: {refusal: "limit" | "room", limit, free}. Held for every receiver; a start or a change takes a refused transfer
 * ai: afresh at its next header. */
LIZ_API void liz_session_file_limit(liz_session *s, long long bytes, double room);
LIZ_API void liz_session_extra(liz_session *s, const char *json);

/* ai: The platform's hooks, Android's (elsewhere LIZ_E_UNSUPPORTED). The app's preview surface (an ANativeWindow*,
 * ai: the session takes its own reference; NULL drops it), each camera's second stream from then on: blocks until the
 * ai: camera writes to it or no longer does (the app releases a surface it dropped after that); size the surface to
 * ai: the mode's preview (liz_session_camera's modes, "preview") before the start. The PC link, process-wide and no
 * ai: session's (a PC sees the phone whether a session lives or not): up while a USB cable is in (up 1, its hello
 * ai: naming layout), down after (up 0); its frames packed as .lizard frames or raw; its state as JSON (up, present,
 * ai: connected, peer, port, want {seq, on}, accessory; {} elsewhere); a USB accessory's fd to its bridge, which closes
 * ai: it (LIZ_E_STATE where one is up already, the fd closed; elsewhere the fd stays the caller's). */
LIZ_API int liz_session_preview_window(liz_session *s, void *native_window);
LIZ_API int liz_link(int up, const char *layout);
LIZ_API int liz_link_pack(int on);
LIZ_API size_t liz_link_state(char *json, size_t cap);
LIZ_API int liz_usb_accessory(int fd);

/* ai: The data folder with or without a session: the verified files taken into kept/ and not yet placed, oldest first
 * ai: ([{id, name, root, path, at, packed}], packed true where path is the default client's zstd stream, name its
 * ai: <name>.zst, for the app to unpack as it files it under name less the .zst: keep_packed's, or one that did not unpack); one placed by the app (its kept/<id> deleted); the unfinished transfers,
 * ai: newest first ([{id, name, length, fraction, root, at}]); one deleted (LIZ_E_STATE while a session holds the folder:
 * ai: liz_session_forget then; a session of this process being freed is waited for, up to 10 s). */
LIZ_API size_t liz_kept(const char *data_dir, char *json, size_t cap);
LIZ_API int liz_kept_placed(const char *data_dir, const char *kept_id);
LIZ_API size_t liz_partials(const char *data_dir, char *json, size_t cap);
LIZ_API int liz_partial_delete(const char *data_dir, const char *id);

#ifdef __cplusplus
}
#endif
#endif
