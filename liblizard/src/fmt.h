// The format word: the one thing in the border that carries information, and the only thing a receiver cannot
// derive for itself. It says WHICH FORMAT the symbol is (its version and the layout of its blocks) and its sender's clock.
//
// It carries the version because that is what precedent carries. QR's format word is 15 bits, 5 of data (error
// level and mask) and 10 of BCH, drawn twice. Aztec's mode message is 28 bits compact, 7 of data (layers and
// codeword count) with Reed-Solomon over the rest. Neither says anything about the TRANSFER: a QR symbol's mode
// indicator and character count live in the data stream, inside the payload's own error correction. LIZARD's do
// the same. Sub-channels are 16 * version, the picture size is the smallest that holds them (sim/lizard_pick.mjs
// N_FOR), the mode is always LDPC, the code rates are the four codes at the counts the word states, and the fountain's
// header is payload.
//
// It carries the sender's clock (2026-11-22): the pictures a second it paints, a whole number 0 to 240 (0 a still
// picture: a print, or one picture with no clock), and the largest delay of its block places (Refresh: a place takes a
// new block every whole number of pictures, 1 to 30), so a receiver knows from any one frame how fast the picture
// changes and how slowly its slowest place does (fps / delay pictures a second, 8 at the top). No decode reads either;
// the phase lock measures the period itself. A whole number: a sender on a 59.94 Hz display states 60.
//
// Why one code over every word cell (2026-09-24). The band has W word cells a side, B / 2 in ring B: 32, 64 and
// 128 over the three rings (2B + 30 modules a side). Whole copies of an 8-byte word summed before RS spent the extra
// cells on averaging; a word as long as the band spends them on parity, RS(W / 2, 6), so a side lost to glare is
// W / 8 erasures out of W / 2 - 6 roots: two faded sides read in every ring, three in the 128 and 256 (src/fmt.c,
// test/fmt_test.c; SPEC.md 5.4).
//
// Why six data bytes, a magic (2026-11-15, the sixth 2026-11-22). The word is its magic and five bytes: the frame's
// whole configuration and the sender's clock, so every frame names itself. The magic names the format: the picture has
// no extension, and a format that changes it is another format with a magic of its own, which this decoder refuses like
// any byte but OB_FMT_MAGIC (SPEC 5.5). The clock took one root from each ring (10, 26 and 58 left); the erasure steps
// still end on three roots (ob_fmt_decode).
//
// The 40 bits after the magic, most significant first: the word's index (35 bits) and a CRC-5 of it (x^5 + x^2 + 1,
// USB's, most significant bit first, from 0). The index is enumerative coding of everything the word states at once:
// (configuration x 241 + fps) x 30 + delay - 1, so OB_FMT_CONFIGS x 7,230 values, 2^34.98 (99% of 35 bits); plain fields
// would take 36 (the configuration's index 23, the fps 8, the delay 5), and the bit saved is the CRC's fifth. The
// configuration is the version and the frame's block counts at 6/7, 2/3 and 1/2; the 3/4 blocks fill the rest of the
// frame (ob_fmt_layout). Its own index numbers every configuration that makes a frame, OB_FMT_CONFIGS of them (2^22.2),
// in the order of the version, then the 6/7 count, the 2/3 count and the 1/2 count, each ascending (ob_fmt_index): 23
// bits where the plain fields took 28 (6 + 8 + 7 + 7), which is what made room for the clock. An index past the last
// is refused. The sender chooses the counts, as a broadcast's scheduler does (DVB-T2's and ATSC 3.0's
// signalling state each stream's blocks or cells as integers), and every receiver reads and checks them: no receiver
// works a layout out. The CRC is the check on a word Reed-Solomon miscorrected: a garbage read lands on a random
// codeword, which any four check bits refuse 15 times in 16, but a real word damaged past the code's reach can land on
// one whose data differ from the truth in a byte or two, and a CRC over every bit refuses those where four fixed bits
// see only their own byte. Its history: until 2026-11-18 a version byte, a rate byte and two extension bytes kept at 0;
// that day the counts as plain fields (the version less 4 in 6 bits, the 6/7 count in 8, the 2/3 and 1/2 in 7 each,
// with a CRC-4 over those 28 bits, after a rate code tried and dropped); since 2026-11-22 the index and the clock.
//
// Why it is not an anchor count. Counting anchors was built and measured on 2026-09-21: the version as the
// number of edge marks a side, three corner marks QR-style, no word at all. It works in a good capture and it
// costs what the word never did. The marks have to be 16 modules to be detected (a 12-module edge mark has a
// 2-module core, under the detector's floor), which pushes the border from 15 to 19; and five a side blank a
// quarter of the timing track. Measured over the standard twelve cells: fill 0.3 fell from 2521 B on 16 of 16
// frames to 381 B on 8, 45 degrees lost 24%, the phone cell 18%. The version was read on 93 of 129 captures
// against the word's 99 of 99, and a miscount loses every block of the frame where a missed word loses nothing.
// A count has no error correction and no way to gain any; Reed-Solomon does.
#ifndef OB_FMT_H
#define OB_FMT_H
#include <stdint.h>

// OB_FMT_MAGIC: the word's first byte, checked after Reed-Solomon. RS can miscorrect past its distance, and a wrong
// word is worse than none: it would send the receiver to another picture size and cost every block of the frame. 0x4c
// is 'L'; any other byte is refused. The other checks are the CRC-5 and the index's range: an index past the last word
// (a layout that does not fit, ob_fmt_layout, has none) is refused.
// OB_FMT_VERSION_MIN, OB_FMT_VERSION_MAX: the versions a word states, 4 to 67, the version being sub-channels / 16
// (lizard_phy.h LIZARD_STEP): 64 to 1072 sub-channels. Versions 1 to 3 are no formats (the ladder starts at LIZARD-64).
// OB_FMT_SLOTS: a version's data slots, 16 sub-channels of 636 (lizard_phy.h LIZARD_STEP x 2 LIZARD_SUB, less each
// sub-channel's two pilots).
// OB_FMT_COUNT67_MAX, OB_FMT_COUNT_MAX: a count's largest, 255 for the 6/7 blocks and 127 for the 2/3 and 1/2.
// OB_FMT_CONFIGS: the configurations that make a frame (every version's counts that leave room for one 3/4 block), the
// index's range.
// OB_FMT_FPS_MAX: the clock's largest, 240 pictures a second (0 a still). OB_FMT_DELAY_MAX: the largest delay a word
// states, 30 pictures. OB_FMT_CLOCKS: the clocks a word states, 241 x 30.
// OB_FMT_BYTES: the shortest word, RS(16, 6), the 64 ring's, and OB_FMT_SIDE_CELLS the cells a side it needs.
// OB_FMT_BYTES_MAX: the longest a band holds, RS(64, 6), the 256 ring's (W = 128). A wider border paints its remaining
// word cells light.
enum { OB_FMT_MAGIC = 0x4c, OB_FMT_DATA = 6, OB_FMT_BYTES = 16, OB_FMT_BYTES_MAX = 64,
       OB_FMT_SIDE_CELLS = 32, OB_FMT_VERSION_MIN = 4, OB_FMT_VERSION_MAX = 67, OB_FMT_SLOTS = 10176,
       OB_FMT_COUNT67_MAX = 255, OB_FMT_COUNT_MAX = 127, OB_FMT_CONFIGS = 4702435,
       OB_FMT_FPS_MAX = 240, OB_FMT_DELAY_MAX = 30, OB_FMT_CLOCKS = (OB_FMT_FPS_MAX + 1) * OB_FMT_DELAY_MAX };
// ai: the four codes in the order their blocks are packed from the frame's first slot (inner first), as LDPC rate
// ai: indices (ldpc.h): 6/7, 3/4 (the filling tier), 2/3, 1/2; a word's counts are tiers 0, 2 and 3
enum { OB_TIER_67 = 0, OB_TIER_34 = 1, OB_TIER_23 = 2, OB_TIER_12 = 3, OB_TIERS = 4 };
extern const int OB_TIER_RATE[OB_TIERS];

// A word: the version and count, the 6/7, 2/3 and 1/2 blocks; the sender's clock, fps (0 to 240, 0 a still) and delay
// (1 to 30, its places' largest).
typedef struct { int version, count[3], fps, delay; } ob_fmt_t;

// A block's slots at an LDPC rate index: its code's sent bits (ldpc.h nt): 4480, 5120, 5760 and 7680; 0 for a rate with
// no code.
int ob_block_slots(int rate);
// The frame a version's counts make: its 3/4 blocks, m = floor((S - a s7 - t s2 - c s1) / s3) over its S = version x
// OB_FMT_SLOTS data slots, into *m34; 1 where it is a frame, 0 where it is none: no room for one 3/4 block, or a version
// or count out of range.
int ob_fmt_layout(int version, const int count[3], int *m34);
// A configuration's index, 0 to OB_FMT_CONFIGS - 1 (the order above), or -1 where it makes no frame (ob_fmt_layout);
// and the configuration of an index: 0, or -1 for an index past the last.
long ob_fmt_index(int version, const int count[3]);
int ob_fmt_config(long index, int *version, int count[3]);
// The word's six data bytes (its magic, then the 40 bits above): 0, or -1 where it is no word (a version or counts
// that are no frame, a clock out of range).
int ob_fmt_pack(const ob_fmt_t *f, uint8_t data[OB_FMT_DATA]);
// Cells a side the word takes (every word cell, W), and the codeword's length there: RS(4 floor(W / 8), 6), so 16,
// 32 and 64 bytes in the 64, 128 and 256 rings. Taken from the layout the receiver registered, which the module count
// gives before the word is read. 0 where the side is too short to hold the shortest word.
int ob_fmt_side_cells(int side, int reserve);
int ob_fmt_bytes(int side, int reserve);
// The word as `bytes` of codeword, `bytes` being ob_fmt_bytes(side): 6 data bytes and bytes - 6 roots. 0, or -1
// where it is no word (ob_fmt_pack).
int ob_fmt_encode(const ob_fmt_t *f, uint8_t *cw, int bytes);
// Bit `cell` of frame side `edge` (0..3), as the word is laid out: byte k goes to side k % 4. 0 past the codeword.
int ob_fmt_bit(const uint8_t *cw, int bytes, int edge, int cell);
// q: soft values, q[side * cells + i] in -0.5 .. 0.5, positive for dark. A word is accepted only with OB_FMT_MAGIC,
// its CRC-5 right and an index under OB_FMT_CONFIGS x OB_FMT_CLOCKS (any ring carries any version, 2026-09-27). 0, or -1
// where nothing decoded.
int ob_fmt_decode(const float *q, int cells, int bytes, ob_fmt_t *f);

#endif
