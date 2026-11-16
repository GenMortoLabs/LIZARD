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
// the version, the rate and two extension bytes, 0 and 0 for the format as it stands; every frame names itself. An
// animated word (OB_FMT_MAGIC_ANIM): one symbol of the header carousel below, its position and three bytes. The magic
// is the switch, so a receiver tells the two apart before it reads anything else. Two of the roots RS(n, 3) had over
// three data bytes (13, 29, 61) went to content: an odd count of roots left (11, 27, 59), so the erasure steps end on
// three roots as before, and the word's reads on recorded captures and under damage barely moved.
//
// The header carousel. Content too long for a static word (an extension's parameters) is a header of 3k bytes, k 2
// to 255: the content, zero bytes, and the CRC-32C (layout.h ob_crc32c) of everything before it, its most significant
// byte first as the block's.
// Its bytes are the data of three Reed-Solomon codewords RS(255, k), the word's own code (rs.c, data first): byte
// 3p + j is codeword j's byte p, p below k. Symbol p, 0 to 254, is the three codewords' byte p: header bytes for p
// below k, parity past them. A sender's animated words carry its symbols in any order, each as often as it likes;
// any k distinct symbols give the header (the code is MDS) and the receiver finds k by the CRC, so no frame states
// k or the header's length. LIZARD's header: the version, the rate and the two extension bytes a static word would
// carry, then the extension's own bytes; k is at least 3. A frame whose word is animated names no version: it is
// decoded at the held configuration, which the assembled header sets as a static word does.
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
// complement.
// OB_FMT_VERSION_MIN, OB_FMT_VERSION_MAX: the versions a word may state, LIZARD-64's 4 to LIZARD-1024's 64, the version
// being sub-channels / 16 (lizard_phy.h LIZARD_STEP; 7 blocks at version 4, 124 at 64 under the rate profile). A byte could
// say 0 to 255; keeping the range to the formats that exist keeps it a check on a miscorrected word, with the magic: a
// version byte of 0 to 3 or 65 to 255 is refused. Versions 1 to 3 (LIZARD-16, -32, -48) are no formats: their odd
// blocks' pilots are one 7/8 block's 8 coefficients or none, too few for the phase lock's second reading; the smallest
// format holds seven blocks, two 7/8 blocks of each parity.
// OB_FMT_BYTES: the shortest word, RS(16, 5), the 64 ring's, and OB_FMT_SIDE_CELLS the cells a side it needs.
// OB_FMT_BYTES_MAX: the longest a band holds, RS(64, 5), the 256 ring's (W = 128). A wider border paints its remaining
// word cells light. OB_FMT_POS_MAX: the last carousel position, 254 (RS(255, k)); a position of 255 is refused.
enum { OB_FMT_MAGIC = 0x4c, OB_FMT_MAGIC_ANIM = 0xb3, OB_FMT_DATA = 5, OB_FMT_BYTES = 16, OB_FMT_BYTES_MAX = 64,
       OB_FMT_SIDE_CELLS = 32, OB_FMT_VERSION_MIN = 4, OB_FMT_VERSION_MAX = 64, OB_FMT_POS_MAX = 254 };
enum { OB_FMT_STATIC = 0, OB_FMT_ANIM = 1 };

// A word: kind OB_FMT_STATIC (version, fps, ext) or OB_FMT_ANIM (pos, sym). { version, fps } alone is the static word
// with extension bytes 0 and 0, the format as it stands.
typedef struct { int version, fps, kind; uint8_t ext[2]; int pos; uint8_t sym[3]; } ob_fmt_t;

// Cells a side the word takes (every word cell, W), and the codeword's length there: RS(4 floor(W / 8), 3), so 16,
// 32 and 64 bytes in the 64, 128 and 256 rings. Taken from the layout the receiver registered, which the module count
// gives before the word is read. 0 where the side is too short to hold the shortest word.
int ob_fmt_side_cells(int side, int reserve);
int ob_fmt_bytes(int side, int reserve);
// The word as `bytes` of codeword, `bytes` being ob_fmt_bytes(side): 5 data bytes and bytes - 5 roots. 0, or -1
// where a static word's version is outside OB_FMT_VERSION_MIN .. OB_FMT_VERSION_MAX or its fps outside 0 to 255 (0
// a sender that states no rate), or an animated word's position past OB_FMT_POS_MAX.
int ob_fmt_encode(const ob_fmt_t *f, uint8_t *cw, int bytes);
// Bit `cell` of frame side `edge` (0..3), as the word is laid out: byte k goes to side k % 4. 0 past the codeword.
int ob_fmt_bit(const uint8_t *cw, int bytes, int edge, int cell);
// q: soft values, q[side * cells + i] in -0.5 .. 0.5, positive for dark. A word is accepted only with a magic and,
// static, a version in OB_FMT_VERSION_MIN .. OB_FMT_VERSION_MAX (any ring carries any version, 2026-09-27), or,
// animated, a position to OB_FMT_POS_MAX. 0, or -1 where nothing decoded.
int ob_fmt_decode(const float *q, int cells, int bytes, ob_fmt_t *f);

// The carousel (above). OB_CAR_N symbols of OB_CAR_SYM bytes; a header of k symbols, OB_CAR_K_MIN to OB_CAR_N.
enum { OB_CAR_N = 255, OB_CAR_SYM = 3, OB_CAR_K_MIN = 2, OB_CAR_CRC = 4, OB_CAR_REACH = 8 };
// The header for `len` content bytes: 3k bytes into hdr (room for 3 OB_CAR_N), k the least that holds the content and
// its CRC (at least OB_CAR_K_MIN). Returns k, or -1 where the content does not fit.
int ob_car_header(const uint8_t *content, int len, uint8_t *hdr);
// Symbol pos (0 .. OB_CAR_N - 1) of a header of k symbols, into sym. 0, or -1 out of range.
int ob_car_symbol(const uint8_t *hdr, int k, int pos, uint8_t sym[3]);
// A receiver's assembly: each position's values as read, the two most often read kept with their counts. ob_car_add
// takes an animated word's symbol and returns k once the symbols held give a header whose CRC holds (the header in
// hdr, its content the first 3k - 4 bytes), else 0: each time the most-read value of a position is new, k is tried
// from the positions held down (OB_CAR_REACH below them for a new position, to OB_CAR_K_MIN for a flipped vote), each
// codeword decoded with the positions not held erased (rs.c rs_decode_er: errors in up to (positions - k) / 2 of the
// rest corrected), the first k whose CRC holds taken. A
// header that stands is kept until a later solve gives another. Zero it (or ob_car_reset) for a new sender.
typedef struct {
  uint8_t val[OB_CAR_N][2][OB_CAR_SYM];
  uint32_t cnt[OB_CAR_N][2];
  int held, k;
  uint8_t hdr[OB_CAR_SYM * OB_CAR_N];
} ob_car_rx_t;
void ob_car_reset(ob_car_rx_t *r);
int ob_car_add(ob_car_rx_t *r, int pos, const uint8_t sym[3]);

#endif
