// The format word: the one thing in the border that carries information, and the only thing a receiver cannot
// derive for itself. It says WHICH FORMAT the symbol is and HOW FAST the sender means to paint.
//
// It carries the version because that is what precedent carries. QR's format word is 15 bits, 5 of data (error
// level and mask) and 10 of BCH, drawn twice. Aztec's mode message is 28 bits compact, 7 of data (layers and
// codeword count) with Reed-Solomon over the rest. Neither says anything about the TRANSFER: a QR symbol's mode
// indicator and character count live in the data stream, inside the payload's own error correction. LIZARD's do
// the same. Sub-channels are 16 * version, the picture size is the smallest that holds them (sim/lizard_pick.mjs
// N_FOR), the mode is always LDPC, the code rates are the rate profile the sub-channels give (lizard_phy.h
// lizard_tiers_for), and the fountain's header is payload.
//
// It carries the display rate because that is the one thing about an ANIMATED symbol that no single frame shows.
// The version is a property of the picture in front of the camera; the rate is a property of a sequence the
// receiver only ever sees part of, and light is the only channel it has. What it buys the receiver: how long a
// frame is on screen, so a capture rate can be judged against it (a camera below the display rate cannot see
// every frame, whatever else is right), and the difference between a repeat and a frame that was missed. It is
// the rate the sender MEANS to paint at, not the rate it achieved, so a sender falling behind reads as what it
// is: the word says 60, the receiver counts 15 distinct frames a second, and the sender is the problem.
// 1 to 255 whole frames a second, and 0 for a sender that does not state one.
//
// Why it is a whole byte. 240 fps panels exist, so the field has to reach past the 60 a code table would have
// covered. A miscorrection that lands in the rate byte alone mis-states the rate, which no decode decision depends
// on, where one in the version byte costs every block.
//
// Why one code over every word cell (2026-09-24). The band has W word cells a side, B / 2 in ring B: 32, 64 and
// 128 over the three rings (2B + 30 modules a side). Whole copies of an 8-byte word summed before RS spent the extra
// cells on averaging; a word as long as the band spends them on parity, RS(W / 2, 5), so a side lost to glare is
// W / 8 erasures out of W / 2 - 5 roots: two faded sides read in every ring, three in the 128 and 256 (src/fmt.c,
// test/fmt_test.c; SPEC.md 5.4).
//
// Why five data bytes, two magics (2026-11-15). The word is its magic and four bytes. A static word (OB_FMT_MAGIC):
// the frame's whole configuration; every frame names itself. An animated word (OB_FMT_MAGIC_ANIM): one symbol of the
// header carousel below, its id and three bytes. The magic is the switch, so a receiver tells the two apart before it
// reads anything else. Two of the roots RS(n, 3) had over three data bytes (13, 29, 61) went to content: an odd count
// of roots left (11, 27, 59), so the erasure steps end on three roots as before, and the word's reads on recorded
// captures and under damage barely moved.
//
// The static word's 32 bits after its magic (2026-11-18), most significant first: the version less 4 (6 bits,
// versions 4 to 67), the rate's code in OB_FMT_RATE (5 bits), and the frame's block counts at 7/8, 2/3 and 1/2 (7 bits
// each); the 3/4 blocks fill the rest of the frame (ob_fmt_layout). The sender chooses the counts, once, as a
// broadcast's scheduler does (DVB-T2's and ATSC 3.0's signalling state each stream's blocks or cells as integers), and
// every receiver reads and checks them: no receiver works a layout out, so none has to repeat a sender's search to
// the bit. Before, the word held the version, a rate byte (1 to 255) and two extension bytes kept at 0, and every
// receiver derived the one rate profile from the version (lizard_tiers_for, now the default senders' choice).
// The rate is the display rate the sender means to paint at, from a table of the 32 whole rates to 60 that displays
// and halved cameras give (60 Hz's divisors, 50, 48, 24 and the high-refresh monitors' divisors, every one to 10);
// a sender at any other rate paints animated words, whose header states it exactly (a rate record, below).
//
// The header carousel. Content too long for a static word is a header of 3k bytes, k 2 to 255: the content, zero
// bytes, and the CRC-32C (layout.h ob_crc32c) of everything before it, its most significant byte first as the block's.
// Its bytes are the data of three Reed-Solomon codewords RS(255, k), the word's own code (rs.c, data first): byte
// 3p + j is codeword j's byte p, p below k. Symbol p, 0 to 254, is the three codewords' byte p: header bytes for p
// below k, parity past them; an animated word names its symbol's p, its id. A sender's animated words carry its
// symbols in any order, each as often as it likes; any k distinct symbols give the header (the code is MDS) and the
// receiver finds k by the CRC, so no frame states k or the header's length. LIZARD's header: a static word's five data
// bytes as they stand (its magic included, read by the same rule), then records, each a type byte, a length byte and
// that many bytes, type 0 (the zero bytes) ending them; type 1 the exact rate, two bytes most significant first, 1 to
// 65,535 a second, in place of the table's. A receiver skips a record whose type it does not know; no record changes
// how a frame is decoded, which is the five bytes' alone. The default's header: the five bytes and the CRC, nine bytes,
// k = 3. A frame whose word is animated names no configuration: it is decoded at the assembled header's.
//
// Why it is not an anchor count. Counting anchors was built and measured on 2026-09-21: the version as the
// number of edge marks a side, three corner marks QR-style, no word at all. It works in a good capture and it
// costs what the word never did. The marks have to be 16 modules to be detected (a 12-module edge mark has a
// 2-module core, under the detector's floor), which pushes the border from 15 to 19; and five a side blank a
// quarter of the timing track. Measured over the standard twelve cells: fill 0.3 fell from 2521 B on 16 of 16
// frames to 381 B on 8, 45 degrees lost 24%, the phone cell 18%. The version was read on 93 of 129 captures
// against the word's 99 of 99, and a miscount loses every block of the frame where a missed word loses nothing.
// A count has no error correction and no way to gain any; Reed-Solomon does. It could not have carried the
// display rate at all.
#ifndef OB_FMT_H
#define OB_FMT_H
#include <stdint.h>

// OB_FMT_MAGIC, OB_FMT_MAGIC_ANIM: the word's first byte, checked after Reed-Solomon, the static word's and the
// animated word's. RS can miscorrect past its distance, and a wrong word is worse than none: it would send the
// receiver to another picture size and cost every block of every frame until it changed again. 0x4c is 'L', 0xb3 its
// complement. The other check on a static word is its counts: a layout that does not fit (ob_fmt_layout) is refused.
// OB_FMT_VERSION_MIN, OB_FMT_VERSION_MAX: the versions a word states, 4 to 67, the version being sub-channels / 16
// (lizard_phy.h LIZARD_STEP): 64 to 1072 sub-channels. Versions 1 to 3 are no formats: too few slots for the pilots'
// two readings and the four codes.
// OB_FMT_SLOTS: a version's slots, 16 sub-channels of 640 (lizard_phy.h LIZARD_STEP x 2 LIZARD_SUB).
// OB_FMT_RATES: the rate table's entries (OB_FMT_RATE). OB_FMT_COUNT_MAX: a count's largest, 127.
// OB_FMT_BYTES: the shortest word, RS(16, 5), the 64 ring's, and OB_FMT_SIDE_CELLS the cells a side it needs.
// OB_FMT_BYTES_MAX: the longest a band holds, RS(64, 5), the 256 ring's (W = 128). A wider border paints its remaining
// word cells light. OB_FMT_ID_MAX: the last carousel id, 254 (RS(255, k)); an id of 255 is refused.
enum { OB_FMT_MAGIC = 0x4c, OB_FMT_MAGIC_ANIM = 0xb3, OB_FMT_DATA = 5, OB_FMT_BYTES = 16, OB_FMT_BYTES_MAX = 64,
       OB_FMT_SIDE_CELLS = 32, OB_FMT_VERSION_MIN = 4, OB_FMT_VERSION_MAX = 67, OB_FMT_ID_MAX = 254, OB_FMT_SLOTS = 10240,
       OB_FMT_RATES = 32, OB_FMT_COUNT_MAX = 127 };
enum { OB_FMT_STATIC = 0, OB_FMT_ANIM = 1 };
// ai: the four codes in the order their blocks are packed from the frame's first slot (inner first), as LDPC rate
// ai: indices (ldpc.h): 7/8, 3/4 (the filling tier), 2/3, 1/2; a word's counts are tiers 0, 2 and 3
enum { OB_TIER_78 = 0, OB_TIER_34 = 1, OB_TIER_23 = 2, OB_TIER_12 = 3, OB_TIERS = 4 };
extern const int OB_TIER_RATE[OB_TIERS];
// The rate table: code c states OB_FMT_RATE[c] a second.
extern const uint8_t OB_FMT_RATE[OB_FMT_RATES];

// A word: kind OB_FMT_STATIC (version, fps, count: the 7/8, 2/3 and 1/2 blocks) or OB_FMT_ANIM (id, sym).
typedef struct { int kind, version, fps, count[3], id; uint8_t sym[3]; } ob_fmt_t;

// A block's slots at an LDPC rate index: its code's sent bits (ldpc.h nt), then its pilots, 16 at 7/8 and 32 at 3/4 (a
// 2/3 or 1/2 block has none): 4496, 5152, 5760 and 7680; 0 for a rate with no code.
int ob_block_slots(int rate);
// The frame a version's counts make: its 3/4 blocks, m = floor((S - a s7 - t s2 - c s1) / s3) over its S = version x
// OB_FMT_SLOTS slots, into *m34; 1 where it is a frame, 0 where it is none: no room for one 3/4 block, or fewer than two
// blocks carrying pilots (a + m: the pilots' two readings need an even block and an odd one), or a version or count out
// of range.
int ob_fmt_layout(int version, const int count[3], int *m34);
// The rate code stating fps a second, or -1 where the table has none; and the table's rate nearest fps (the lower of
// two as near), what a static word states for a sender at another rate.
int ob_fmt_rate_code(int fps);
int ob_fmt_rate_near(int fps);
// A static word's five data bytes (its magic, then the 32 bits above): 0, or -1 where it is no word (a version, a rate
// the table does not hold, or counts that are no frame).
int ob_fmt_pack(const ob_fmt_t *f, uint8_t data[OB_FMT_DATA]);
// A configuration as one int, what a receiver holds for its animated frames (a carousel's header, or the last static
// word): version << 21 | count[0] << 14 | count[1] << 7 | count[2], never 0 (the version is 4 or more), 0 where f is
// none; and back: 1 with the version and counts in *f, 0 where held is no configuration.
int ob_fmt_held(const ob_fmt_t *f);
int ob_fmt_unheld(int held, ob_fmt_t *f);
// Cells a side the word takes (every word cell, W), and the codeword's length there: RS(4 floor(W / 8), 5), so 16,
// 32 and 64 bytes in the 64, 128 and 256 rings. Taken from the layout the receiver registered, which the module count
// gives before the word is read. 0 where the side is too short to hold the shortest word.
int ob_fmt_side_cells(int side, int reserve);
int ob_fmt_bytes(int side, int reserve);
// The word as `bytes` of codeword, `bytes` being ob_fmt_bytes(side): 5 data bytes and bytes - 5 roots. 0, or -1
// where a static word is none (ob_fmt_pack) or an animated word's id is past OB_FMT_ID_MAX.
int ob_fmt_encode(const ob_fmt_t *f, uint8_t *cw, int bytes);
// Bit `cell` of frame side `edge` (0..3), as the word is laid out: byte k goes to side k % 4. 0 past the codeword.
int ob_fmt_bit(const uint8_t *cw, int bytes, int edge, int cell);
// q: soft values, q[side * cells + i] in -0.5 .. 0.5, positive for dark. A word is accepted only with a magic and,
// static, counts that make a frame of its version (any ring carries any version, 2026-09-27), or, animated, an id to
// OB_FMT_ID_MAX. 0, or -1 where nothing decoded.
int ob_fmt_decode(const float *q, int cells, int bytes, ob_fmt_t *f);

// The carousel (above). OB_CAR_N symbols of OB_CAR_SYM bytes; a header of k symbols, OB_CAR_K_MIN to OB_CAR_N.
enum { OB_CAR_N = 255, OB_CAR_SYM = 3, OB_CAR_K_MIN = 2, OB_CAR_CRC = 4, OB_CAR_REACH = 8 };
// The header for `len` content bytes: 3k bytes into hdr (room for 3 OB_CAR_N), k the least that holds the content and
// its CRC (at least OB_CAR_K_MIN). Returns k, or -1 where the content does not fit.
int ob_car_header(const uint8_t *content, int len, uint8_t *hdr);
// A configuration's header content (OB_CAR_CONTENT_MAX bytes at most): a static word's five data bytes, then, where
// f->fps is a rate the table does not hold (1 to 65,535), a rate record and the five bytes stating the table's nearest
// rate (the lower of two as near). Returns its length, or -1 where f is no configuration.
enum { OB_CAR_CONTENT_MAX = 9, OB_CAR_REC_RATE = 1 };
int ob_car_content(const ob_fmt_t *f, uint8_t *out);
// The configuration an assembled header of k symbols states: 1 with it in *f (kind OB_FMT_STATIC; a rate record's
// rate in fps), 0 where its first five bytes are no static word or a record runs past the content.
int ob_car_config(const uint8_t *hdr, int k, ob_fmt_t *f);
// Symbol id (0 .. OB_CAR_N - 1) of a header of k symbols, into sym. 0, or -1 out of range.
int ob_car_symbol(const uint8_t *hdr, int k, int id, uint8_t sym[3]);
// A receiver's assembly: each id's values as read, the two most often read kept with their counts. ob_car_add takes an
// animated word's symbol and returns k once the symbols held give a header whose CRC holds (the header in hdr, its
// content the first 3k - 4 bytes), else 0: each time the most-read value of an id is new, k is tried from the ids
// held down (OB_CAR_REACH below them for a new id, to OB_CAR_K_MIN for a flipped vote), each codeword decoded with the
// ids not held erased (rs.c rs_decode_er: errors in up to (ids - k) / 2 of the rest corrected), the first k whose CRC
// holds taken. A header that stands is kept until a later solve gives another. Zero it (or ob_car_reset) for a new
// sender.
typedef struct {
  uint8_t val[OB_CAR_N][2][OB_CAR_SYM];
  uint32_t cnt[OB_CAR_N][2];
  int held, k;
  uint8_t hdr[OB_CAR_SYM * OB_CAR_N];
} ob_car_rx_t;
void ob_car_reset(ob_car_rx_t *r);
int ob_car_add(ob_car_rx_t *r, int id, const uint8_t sym[3]);

#endif
