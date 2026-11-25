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
// changes and how slowly its slowest place does (fps / delay pictures a second, 8 at the top). A still states no delay
// (it has no refresh; read back as 1). No decode reads either; the phase lock measures the period itself. A whole
// number: a sender on a 59.94 Hz display states 60.
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
// end on one root (ob_fmt_decode; three until 2026-11-23).
//
// The 40 bits after the magic, most significant first, are the word's index alone: enumerative coding of everything the
// word states at once, configuration x 7,201 + k, the clock k 0 for a still and 1 + (fps - 1) x 30 + delay - 1 for fps 1
// to 240, so OB_FMT_CONFIGS x 7,201 values, 2^39.81 (87.8% of the 40 bits; an index past the last is refused). The
// configuration is the version and the frame's four block counts, at 6/7, 3/4, 2/3 and 1/2: every layout whose blocks
// fit the version's data slots (ob_fmt_layout), the padding any size, a frame of no blocks among them, OB_FMT_CONFIGS of
// them (2^27.0). Its own index numbers them in the order of the version, then the 6/7, 3/4, 2/3 and 1/2 counts, each
// ascending (ob_fmt_index), counted by fmt_tables.h's room tables: every block is a whole number of OB_FMT_UNIT slots.
// The sender chooses the counts, as a broadcast's scheduler does (DVB-T2's and ATSC 3.0's signalling state each stream's
// blocks or cells as integers), and every receiver reads and checks them: no receiver works a layout out. No CRC: the
// word's check is the magic, the index's range and Reed-Solomon's own (the erasure steps keep one root); a false word
// costs one frame's decode, its blocks failing their CRC-64, never a byte of a file. Its history: until 2026-11-18 a
// version byte, a rate byte and two extension bytes kept at 0; that day the counts as plain fields (the version less 4
// in 6 bits, the 6/7 count in 8, the 2/3 and 1/2 in 7 each, with a CRC-4 over those 28 bits, after a rate code tried and
// dropped); from 2026-11-22 the index and the clock, a 35-bit index under a CRC-5 with the 3/4 blocks filling the rest
// of the frame (4,702,481 configurations), the clock (configuration x 241 + fps) x 30 + delay - 1 until 2026-11-23 (a
// still taking 30 delays where it has none); since 2026-11-23 every layout, the CRC's five bits spent on the fourth
// count.
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
// is 'L'; any other byte is refused. The other check is the index's range: an index past the last word is refused.
// OB_FMT_VERSION_MIN, OB_FMT_VERSION_MAX: the versions a word states, 1 to 67, the version being sub-channels / 16
// (lizard_phy.h LIZARD_STEP): 16 to 1072 sub-channels.
// OB_FMT_SLOTS: a version's data slots, 16 sub-channels of 636 (lizard_phy.h LIZARD_STEP x 2 LIZARD_SUB, less each
// sub-channel's two pilots).
// OB_FMT_UNIT: the data slots every block size is a whole number of (fmt_tables.h): the 6/7 code's 7, the 3/4's 8, the
// 2/3's 9 and the 1/2's 12. Version 67 holds at most 152, 133, 118 and 88 blocks of one code.
// OB_FMT_CONFIGS: the configurations, every version's layouts that fit (ob_fmt_layout), the index's range.
// OB_FMT_FPS_MAX: the clock's largest, 240 pictures a second (0 a still). OB_FMT_DELAY_MAX: the largest delay a word
// states, 30 pictures. OB_FMT_CLOCKS: the clocks a word states, 1 + 240 x 30: a still, then every fps 1 to 240 with
// every delay.
// OB_FMT_BYTES: the shortest word, RS(16, 6), the 64 ring's, and OB_FMT_SIDE_CELLS the cells a side it needs.
// OB_FMT_BYTES_MAX: the longest a band holds, RS(64, 6), the 256 ring's (W = 128). A wider border paints its remaining
// word cells light.
enum { OB_FMT_MAGIC = 0x4c, OB_FMT_DATA = 6, OB_FMT_BYTES = 16, OB_FMT_BYTES_MAX = 64,
       OB_FMT_SIDE_CELLS = 32, OB_FMT_VERSION_MIN = 1, OB_FMT_VERSION_MAX = 67, OB_FMT_SLOTS = 10176,
       OB_FMT_UNIT = 640, OB_FMT_CONFIGS = 134069046,
       OB_FMT_FPS_MAX = 240, OB_FMT_DELAY_MAX = 30, OB_FMT_CLOCKS = 1 + OB_FMT_FPS_MAX * OB_FMT_DELAY_MAX };
// ai: the four codes in the order their blocks are packed from the frame's first slot (inner first), as LDPC rate
// ai: indices (ldpc.h): 6/7, 3/4, 2/3, 1/2; a word's four counts are the four tiers in this order
enum { OB_TIER_67 = 0, OB_TIER_34 = 1, OB_TIER_23 = 2, OB_TIER_12 = 3, OB_TIERS = 4 };
extern const int OB_TIER_RATE[OB_TIERS];

// A word: the version and count[4], the 6/7, 3/4, 2/3 and 1/2 blocks; the sender's clock, fps (0 to 240, 0 a still) and
// delay (1 to 30, its places' largest; a still states none, and a word read back says 1).
typedef struct { int version, count[OB_TIERS], fps, delay; } ob_fmt_t;

// A block's slots at an LDPC rate index: its code's sent bits (ldpc.h nt): 4480, 5120, 5760 and 7680; 0 for a rate with
// no code.
int ob_block_slots(int rate);
// 1 where a version's counts make a frame: each 0 or more, a s7 + b s3 + t s2 + c s1 at most its version x OB_FMT_SLOTS
// data slots (s the blocks' slots, ob_block_slots), the rest padding; 0 otherwise, or for a version out of range.
int ob_fmt_layout(int version, const int count[OB_TIERS]);
// A configuration's index, 0 to OB_FMT_CONFIGS - 1 (the order above), or -1 where it makes no frame (ob_fmt_layout);
// and the configuration of an index: 0, or -1 for an index past the last.
long ob_fmt_index(int version, const int count[OB_TIERS]);
int ob_fmt_config(long index, int *version, int count[OB_TIERS]);
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
// an index under OB_FMT_CONFIGS x OB_FMT_CLOCKS (any ring carries any version, 2026-09-27). 0, or -1
// where nothing decoded.
int ob_fmt_decode(const float *q, int cells, int bytes, ob_fmt_t *f);

#endif
